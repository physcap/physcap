"""Planner agent for PhysCaP.

Queries a VLM with a scene image and task description to decide whether the
robot has enough information to act, and if not, proposes exploration
candidates (an action on an object) for the Prioritizer or the joint mode to
choose from.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from physcap.agents.scene_knowledge import _load_google_genai, _resolve_api_key_and_model, agent_base_url


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------

# The probe types a candidate may perform. Constrained so the value can serve as
# half of a candidate's identity: free text would let "measure mass", "weigh"
# and "get mass" name the same probe three ways, which is exactly the drift the
# de-dup exists to absorb. Passed through as a JSON-schema enum, so routes that
# support structured output enforce it rather than merely suggesting it.
ACTION_VOCAB = (
    "measure_mass",
    "measure_stiffness",
    "relocate",
    "inspect",
    "other",
)

# Sentinel emitted by the cycle when exploration is done; it is not a real
# candidate and never joins the pool, so it keeps its bare name as its key.
TASK_EXECUTE_NAME = "original_task_execute"

# Output token budget for the planner's model call. A truncated response
# fails JSON parsing (see safe_load_json_from_model_text) and burns a retry
# attempt, so this should stay generous relative to typical exploration
# responses; see _call_planner_once's retry loop in planner() for the
# fallback when a response is truncated anyway.
PLANNER_MAX_OUTPUT_TOKENS = 4096


def normalize_candidate_name(name: str | None) -> str:
    """Normalize a candidate name for cross-source string comparison.

    Models write the same name in different ways (``can_0``, ``can 0``,
    ``can-0``). Comparing on a normalized form keeps the
    executed/failed filter and de-dup robust to those superficial differences.
    """
    if not isinstance(name, str):
        return ""
    return name.replace("_", " ").replace("-", " ").strip().lower()


def candidate_key(candidate: dict | str | None) -> str:
    """Identity of one exploration candidate: an action ON an object.

    ``name`` alone is the OBJECT -- the planner is instructed to use the
    canonical object id from the scene vocabulary -- so keying on it collapses
    every probe of one object into a single candidate. A task that has to trade
    two physical properties off against each other needs measuring a cube's
    mass and probing its stiffness to stay separable, or the second probe is
    discarded as a duplicate of the first.

    Accepts a bare string so entries recorded before candidates carried an
    action still compare without blowing up. They simply will not match a
    composite key, which is correct: they are a different identity.
    """
    if isinstance(candidate, str):
        return normalize_candidate_name(candidate)
    if not isinstance(candidate, dict):
        return ""

    name = normalize_candidate_name(candidate.get("name"))
    if name == normalize_candidate_name(TASK_EXECUTE_NAME):
        return name

    action = normalize_candidate_name(candidate.get("action"))
    # Fall back to the description when the action is missing or deliberately
    # unspecific. Some OpenRouter routes cannot enforce response_json_schema
    # (see _query_model below), so the field is only advisory there -- and with
    # no discriminator at all every probe of one object would collide again.
    # Free text is a weaker key than the enum, but never a silently wrong one.
    if not action or action == "other":
        action = normalize_candidate_name(candidate.get("description"))
    return f"{name}::{action}" if action else name


# ---------------------------------------------------------------------------
# JSON response schemas
# ---------------------------------------------------------------------------

def exploration_candidate_schema(require_parameters: bool = True) -> dict:
    required = ["name", "action", "description", "expected_info", "estimated_cost"]
    if require_parameters:
        required.insert(3, "parameters")

    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "action": {
                "type": "string",
                "enum": list(ACTION_VOCAB),
            },
            "description": {"type": "string"},
            "parameters": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "expected_info": {"type": "string"},
            "estimated_cost": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": required,
        "additionalProperties": False,
        "propertyOrdering": [
            "name",
            "action",
            "description",
            "parameters",
            "expected_info",
            "estimated_cost",
        ],
    }


def planner_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "task_complete": {"type": "boolean"},
            "reason": {"type": "string"},
            "central_objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "required_properties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "description", "required_properties"],
                    "additionalProperties": False,
                    "propertyOrdering": ["name", "description", "required_properties"],
                },
            },
            "exploration_candidates": {
                "type": "array",
                "items": exploration_candidate_schema(require_parameters=True),
            },
        },
        "required": ["sufficient", "task_complete", "reason", "central_objects", "exploration_candidates"],
        "additionalProperties": False,
        "propertyOrdering": [
            "sufficient",
            "task_complete",
            "reason",
            "central_objects",
            "exploration_candidates",
        ],
    }


# ---------------------------------------------------------------------------
# JSON parsing helpers (also imported by the Prioritizer)
# ---------------------------------------------------------------------------

def extract_first_json_blob(text: str) -> str:
    """Find the first balanced {...} or [...] in text."""
    starts = [(idx, char) for char in ("{", "[") if (idx := text.find(char)) != -1]
    if not starts:
        raise ValueError("No JSON object or array found in model text.")
    start, opening = min(starts, key=lambda item: item[0])
    closing = "}" if opening == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("No balanced JSON object or array found in model text.")


def extract_markdown_json_blocks(text: str) -> list[str]:
    """Return contents of fenced markdown code blocks, preferring ```json."""
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_blocks:
        return json_blocks
    return re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)


def safe_load_json_from_model_text(text: str):
    """Parse JSON from raw model output, trying several fallback strategies."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for block in extract_markdown_json_blocks(text):
        try:
            return json.loads(block.strip())
        except Exception:
            pass
    try:
        blob = extract_first_json_blob(text)
        return json.loads(blob)
    except Exception as e:
        raise ValueError(
            f"Failed to parse JSON from model output: {e}\n--- raw output ---\n{text}"
        )


def load_json_from_gemini_response(response):
    """Extract parsed JSON from a Gemini response object."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, (dict, list)):
        return parsed
    raw_text = response.text
    if not raw_text:
        raise ValueError("response.text is None.")
    return safe_load_json_from_model_text(raw_text)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _call_planner_once(
    image_path: str,
    task: str,
    system_prompt: str,
    model_name: str,
    api_key: str,
) -> dict:
    """Query the VLM once and return the parsed response dict.

    Raises ValueError if the model's response cannot be parsed as JSON, or
    RuntimeError for provider-level failures (safety block, no candidates,
    dead API). Callers decide whether/how to retry.
    """
    api_key_val, model_name, use_openrouter = _resolve_api_key_and_model(api_key, model_name)

    user_text = (
        f"TASK: {task}\n\n"
        "Return the single JSON object as specified by the system message. "
        "If the information in the image is sufficient to act, set \"sufficient\": true. "
        "If not, set \"sufficient\": false and fill central_objects + exploration_candidates. "
        "Set \"task_complete\": true only if the image and the accumulated knowledge show the "
        "task's goal has already been achieved."
    )

    # Model-scoped hedge only: see identical comment in
    # scene_knowledge.init_scene_knowledge. Do not extend to other models.
    if model_name.startswith("anthropic/claude-fable"):
        user_text += (
            "\n\n(Note: this is your own direct reasoning about a live robot workspace "
            "for an active robotics task, not a request to reproduce or extract "
            "another model's output.)"
        )

    if use_openrouter:
        import base64
        import openai

        client = openai.OpenAI(api_key=api_key_val, base_url=agent_base_url())
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        schema_text = json.dumps(planner_response_schema(), indent=2)
        full_prompt = (
            system_prompt + "\n\n" + user_text
            + f"\n\nOUTPUT FORMAT:\nYou MUST return a valid JSON object matching this JSON Schema:\n{schema_text}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ]

        def _create(**extra):
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=PLANNER_MAX_OUTPUT_TOKENS,
                **extra,
            )

        # Not every provider accepts response_format=json_object through
        # OpenRouter (e.g. some Anthropic/OpenAI routes). The schema is also
        # embedded in the prompt and the parser is tolerant, so fall back to a
        # plain call if the structured-output request is rejected.
        try:
            response = _create(response_format={"type": "json_object"})
        except Exception:
            response = _create()

        if getattr(response, "choices", None) is None:
            err_msg = getattr(response, "error", "No choices returned in OpenRouter response.")
            raise RuntimeError(
                f"OpenRouter API failed: {err_msg}\n"
                f"Check if model '{model_name}' exists and is accessible."
            )

        raw_text = response.choices[0].message.content
        if not raw_text:
            choice = response.choices[0]
            raise ValueError(
                "OpenRouter response content is None. "
                f"model={model_name!r} "
                f"finish_reason={getattr(choice, 'finish_reason', None)!r} "
                f"error={getattr(response, 'error', None)!r} "
                f"raw={response!r}"
            )
        parsed = safe_load_json_from_model_text(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Model output is not a JSON object.\nraw_text: {raw_text}")

    else:
        genai, types = _load_google_genai()
        client = genai.Client(api_key=api_key_val)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        full_prompt = system_prompt + "\n\n" + user_text

        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_json_schema=planner_response_schema(),
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            )
        except Exception:
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_json_schema=planner_response_schema(),
            )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                full_prompt,
            ],
            config=config,
        )

        if response.candidates:
            if response.candidates[0].finish_reason == "SAFETY":
                raise RuntimeError("Model blocked response due to safety concerns.")
        else:
            raise RuntimeError("No candidates generated. Check prompt/model configuration.")

        parsed = load_json_from_gemini_response(response)
        if not isinstance(parsed, dict):
            raise ValueError(f"Model output is not a JSON object.\nraw_text: {response.text}")

    return parsed


def planner(
    image_path: str,
    task: str,
    system_prompt: str,
    model_name: str = "gemini-3.1-pro-preview",
    output_base_dir: str = "output",
    api_key: str = None,
    max_retries: int = 3,
) -> str:
    """Generate exploration candidates for a scene.

    Queries a VLM with the scene image and task. If information is sufficient
    to complete the task, returns a placeholder "original_task" candidate.
    Otherwise returns a list of exploration steps needed.

    A response that cannot be parsed as JSON (e.g. truncated mid-generation)
    is retried up to ``max_retries`` times before giving up, since a single
    malformed response should not be fatal to the whole run.

    Returns:
        Path to the saved exploration_candidates.json file.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    parsed = None
    for attempt in range(1, max_retries + 1):
        try:
            parsed = _call_planner_once(image_path, task, system_prompt, model_name, api_key)
            break
        except ValueError as e:
            if attempt < max_retries:
                print(
                    f"Warning: Planner output could not be parsed as JSON on "
                    f"attempt {attempt}/{max_retries} ({e}). Retrying..."
                )
            else:
                raise

    task_complete = parsed.get("task_complete") is True
    if task_complete:
        # Nothing is left to do, so there is nothing to propose -- not even the
        # original-task stand-in, which would make codegen redo the task.
        candidates = []
    elif parsed.get("sufficient") is True:
        candidates = [{
            "name": "original_task",
            "description": task,
            "expected_info": "Complete the task as originally requested.",
            "estimated_cost": "low",
        }]
    else:
        candidates = parsed.get("exploration_candidates", [])

    # Save outputs
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(output_base_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    full_response = {
        "sufficient": parsed.get("sufficient", False),
        "task_complete": task_complete,
        "reason": parsed.get("reason", ""),
        "central_objects": parsed.get("central_objects", []),
        "exploration_candidates": candidates,
    }
    (out_dir / "exploration_full_response.json").write_text(
        json.dumps(full_response, indent=4), encoding="utf-8"
    )
    candidates_path = out_dir / "exploration_candidates.json"
    candidates_path.write_text(json.dumps(candidates, indent=4), encoding="utf-8")

    return str(candidates_path)
