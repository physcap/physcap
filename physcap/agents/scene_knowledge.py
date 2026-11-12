"""Scene knowledge management for PhysCaP.

Handles initialization, updates, and property appending for the
``scene_knowledge.txt`` file that persists across turns in a trial, plus the
canonical object-label vocabulary the Planner, Prioritizer, and code
generation stages all refer to.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from physcap.prompt_files import PROMPTS_DIR

SCENE_KNOWLEDGE_FILENAME = "scene_knowledge.txt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_google_genai():
    """Lazy-import google-genai so the rest of the package loads without it."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "The agents package requires the `google-genai` package. "
            "Install it with: uv pip install google-genai"
        ) from exc
    return genai, types


def _configure_google_api_env(api_key: str | None) -> str:
    """Set GEMINI_PAID_API_KEY in the environment if a key is provided.

    Returns the env-var name so callers can pass it to Google GenAI clients.
    """
    env_var = "GEMINI_PAID_API_KEY"
    if api_key:
        os.environ[env_var] = api_key
    return env_var


# The Planner, the Prioritizer and the scene-knowledge calls talk to an
# OpenAI-compatible endpoint: OpenRouter by default, or any server named in this
# variable (set by ``physcap.run --agent-server-url``, e.g. a local vLLM server).
AGENT_SERVER_URL_ENV = "PHYSCAP_AGENT_SERVER_URL"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def agent_base_url() -> str:
    """Base URL the Planner, Prioritizer and scene-knowledge calls are sent to."""
    return os.environ.get(AGENT_SERVER_URL_ENV) or OPENROUTER_BASE_URL


def _resolve_api_key_and_model(
    api_key: str | None, model_name: str
) -> tuple[str | None, str, bool]:
    """Resolve the API key value and normalize the model name for the backend.

    ``api_key`` may be a raw key string OR an environment-variable name.
    If ``os.environ[api_key]`` exists, that value is used; otherwise the
    argument is treated as a raw key and stored as ``GEMINI_PAID_API_KEY``
    for native Google GenAI callers.

    Model-name normalization is provider-agnostic so the agents can run on
    Gemini, Claude, or GPT via OpenRouter:

    * OpenRouter (``sk-or-*`` keys) expects ``<provider>/<model>`` ids such as
      ``google/gemini-3.1-pro-preview``, ``anthropic/claude-opus-4-5``, or
      ``openai/gpt-5.4``. Any namespaced id is passed through unchanged; a bare
      name (no ``/``) gets ``google/`` in front, e.g. ``gemini-3.1-pro-preview``
      becomes ``google/gemini-3.1-pro-preview``.
    * Native Google GenAI expects the bare Gemini model id with no provider
      namespace (e.g. ``gemini-3.1-pro-preview``).

    Returns:
        (api_key_val, normalized_model_name, use_openrouter)
    """
    if api_key and os.environ.get(api_key):
        api_key_val = os.environ[api_key].strip()
    else:
        api_key_val = api_key.strip() if api_key else None
        if api_key_val:
            os.environ["GEMINI_PAID_API_KEY"] = api_key_val

    custom_server = bool(os.environ.get(AGENT_SERVER_URL_ENV))
    # "use_openrouter" means "call an OpenAI-compatible endpoint" (agent_base_url());
    # otherwise the key is taken as a Google Gemini key and the native SDK is used.
    use_openrouter = custom_server or bool(api_key_val and api_key_val.startswith("sk-or-"))
    if custom_server and not api_key_val:
        api_key_val = "EMPTY"  # local servers usually accept any key

    # The "openrouter/" routing prefix is only understood by the local proxy;
    # neither OpenRouter's REST API nor the native Gemini SDK accept it.
    if model_name.startswith("openrouter/"):
        model_name = model_name[len("openrouter/"):]

    if custom_server:
        pass  # a custom server gets the model name exactly as given
    elif use_openrouter:
        # Pass any "<provider>/<model>" id through unchanged (google/, anthropic/,
        # openai/, ...). Only bare names default to google/ routing.
        if "/" not in model_name:
            model_name = "google/" + model_name
    else:
        # Native Google GenAI wants the bare Gemini id.
        if model_name.startswith("google/"):
            model_name = model_name[len("google/"):]

    return api_key_val, model_name, use_openrouter


# ---------------------------------------------------------------------------
# Scene image helper
# ---------------------------------------------------------------------------

def save_scene_image(image: Image.Image, output_dir: Path, filename: str) -> Path:
    """Save a PIL image to output_dir/filename and return the full path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    image.save(path)
    return path


# ---------------------------------------------------------------------------
# Knowledge file I/O
# ---------------------------------------------------------------------------

def _load_scene_knowledge(output_dir: Path) -> str | None:
    """Return the contents of scene_knowledge.txt, or None if absent."""
    scene_knowledge_path = output_dir / SCENE_KNOWLEDGE_FILENAME
    if scene_knowledge_path.exists():
        return scene_knowledge_path.read_text(encoding="utf-8")
    return None


def append_object_property_to_knowledge(
    output_dir: str | Path,
    object_name: str,
    property_type: str,
    value: float | int,
    unit: str = "",
    description: str | None = None,
) -> Path | None:
    """Append a measured object property to scene_knowledge.txt.

    Called automatically when get_mass() or get_stiffness() are invoked
    from within the generated code sandbox.

    Args:
        output_dir: Path to the trial output directory.
        object_name: Name of the object (e.g. "cup", "red_cube").
        property_type: "mass" or "stiffness".
        value: Measured value.
        unit: Unit of measurement (e.g. "kg", "level").
        description: Optional additional context.

    Returns:
        Path to the updated file, or None on failure.
    """
    try:
        output_path = Path(output_dir)
        scene_knowledge_path = output_path / SCENE_KNOWLEDGE_FILENAME

        if not scene_knowledge_path.exists():
            return None

        if property_type.lower() == "mass":
            if description:
                sentence = f"The {object_name} has a mass of {value} {unit}; {description}"
            else:
                sentence = f"The {object_name} has a mass of {value} {unit}."
        elif property_type.lower() == "stiffness":
            stiffness_names = {
                1: "ultra-soft (e.g., sponge, foam)",
                2: "soft (e.g., soft rubber, ripe fruit)",
                3: "semi-rigid (e.g., cardboard, firm fruits)",
                4: "stiff (e.g., hardwood, dense polymer)",
                5: "rigid (e.g., metal, ceramic, stone)",
            }
            stiffness_desc = stiffness_names.get(int(value), f"stiffness level {value}")
            if description:
                sentence = f"The {object_name} is {stiffness_desc}; {description}"
            else:
                sentence = f"The {object_name} is {stiffness_desc}."
        else:
            sentence = f"The {object_name} has {property_type}: {value} {unit}"
            if description:
                sentence += f"; {description}"

        with open(scene_knowledge_path, "a", encoding="utf-8") as f:
            f.write(f"\n[Object Property Measurement]\n{sentence}")

        return scene_knowledge_path

    except Exception:
        return None


MEASURED_PROPERTIES_FILENAME = "measured_properties.json"


def append_measured_property(
    output_dir: str | Path,
    object_name: str,
    property_type: str,
    turn: int | None = None,
) -> None:
    """Record a probe in machine-readable form, beside the prose knowledge file.

    ``append_object_property_to_knowledge`` writes sentences for the VLM to
    read; the reasoning cycle needs to know which (object, property) pairs have
    actually been probed so it can retire the matching candidates, and parsing
    that back out of prose would be fragile. Best-effort: losing a record costs
    accounting accuracy, never the trial.
    """
    try:
        path = Path(output_dir) / MEASURED_PROPERTIES_FILENAME
        records = []
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        records.append(
            {
                "object_name": str(object_name),
                "property_type": str(property_type),
                "turn": turn,
            }
        )
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def load_measured_properties(output_dir: str | Path) -> list[dict]:
    """Return every recorded probe, or [] if none have been written yet."""
    try:
        path = Path(output_dir) / MEASURED_PROPERTIES_FILENAME
        if not path.exists():
            return []
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


# Prefixes that identify low-level infrastructure debug lines. These come from
# the robot driver, Molmo perception backend, and motion control layers. They
# add noise to scene_knowledge.txt without helping the reasoning VLM.
_STDOUT_NOISE_PREFIXES: tuple[str, ...] = (
    "[PiperReal]",
    "[Molmo]",
    "[Franka]",
    "Molmo response:",
    "Moving to home pose",
    "Moving to cartesian",
    "Waiting for observation",
)


def _filter_stdout_for_knowledge(stdout: str) -> str:
    """Strip low-level robot/perception debug lines from stdout.

    Keeps task-relevant output (THOUGHT:, ACTION:, task-state prints, mass
    summaries from [get_mass]/[get_stiffness]) while removing verbose
    infrastructure messages that add noise without informing higher-level reasoning.
    """
    filtered = [
        line for line in stdout.splitlines()
        if not any(line.strip().startswith(p) for p in _STDOUT_NOISE_PREFIXES)
    ]
    return "\n".join(filtered).strip()


def append_stdout_to_knowledge(
    output_dir: str | Path,
    stdout: str,
    turn: int,
    candidate_name: str = "",
) -> Path | None:
    """Append filtered console stdout from a code execution turn to scene_knowledge.txt.

    Only task-relevant print output (THOUGHT:, ACTION:, task state, mass
    summaries) is kept; low-level robot/perception debug lines are stripped
    so the VLM receives clean, signal-rich knowledge on every turn.

    Called after each env.step() in the trial loop.

    Args:
        output_dir: Path to the trial output directory.
        stdout: Console stdout from the code execution (empty → no-op).
        turn: Turn index (1-based) used for the section header.
        candidate_name: Name of the candidate that ran (for context).

    Returns:
        Path to the updated file, or None if stdout was empty or on failure.
    """
    stdout = _filter_stdout_for_knowledge(stdout.strip())
    if not stdout:
        return None
    try:
        scene_knowledge_path = Path(output_dir) / SCENE_KNOWLEDGE_FILENAME
        if not scene_knowledge_path.exists():
            return None

        header = f"\n[Execution Output - Turn {turn}"
        if candidate_name:
            # Truncate long names so the header stays readable
            short_name = candidate_name[:80]
            header += f": {short_name}"
        header += "]"

        with open(scene_knowledge_path, "a", encoding="utf-8") as f:
            f.write(f"{header}\n{stdout}")

        return scene_knowledge_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Knowledge initialization
# ---------------------------------------------------------------------------

def init_scene_knowledge(
    *,
    image_path: Path,
    task: str,
    output_dir: Path,
    api_key: str | None = None,
    model_name: str = "gemini-3.1-pro-preview",
    system_prompt_path: str | None = None,
) -> Path:
    """Analyze the initial scene image and write scene_knowledge.txt.

    Supports both OpenRouter (sk-or-* keys) and native Google GenAI.

    Args:
        image_path: Path to the scene image.
        task: Task description used to condition the scene analysis.
        output_dir: Directory where scene_knowledge.txt is written.
        api_key: Raw API key string (or env-var name if already set).
        model_name: Model to use for scene understanding.
        system_prompt_path: Override path for the system prompt file.

    Returns:
        Path to the written scene_knowledge.txt.
    """
    api_key_val, model_name, use_openrouter = _resolve_api_key_and_model(api_key, model_name)
    print(f"[KNOWLEDGE INIT] Using model {model_name} (OpenRouter={use_openrouter})")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    _system_prompt_path = (
        Path(system_prompt_path)
        if system_prompt_path
        else PROMPTS_DIR / "scene_knowledge.txt"
    )
    system_prompt = _load_text(_system_prompt_path)

    user_text = (
        f"TASK: {task}\n\n"
        "Analyze the current scene and provide a comprehensive description that will help "
        "guide robot decision-making. Focus on what is visible, what is hidden, and what "
        "information is needed to complete the task."
    )

    # Model-scoped hedge only: anthropic/claude-fable-* has been observed to
    # intermittently refuse this prompt with finish_reason=content_filter
    # ("...reverse engineering or duplicating model outputs"). Append a short
    # disclaimer ONLY for that model family so every other model's prompt
    # text stays byte-identical.
    if model_name.startswith("anthropic/claude-fable"):
        user_text += (
            "\n\n(Note: this is your own direct observation of a live robot workspace "
            "for an active robotics task, not a request to reproduce or extract "
            "another model's output.)"
        )

    full_prompt = system_prompt + "\n\n" + user_text
    scene_description: str | None = None

    # Bounded retry: only retries the empty-content/refusal failure path.
    # Cannot change behavior for any call that already succeeds, so this is
    # safe to apply uniformly across all models (mirrors planner()'s retry).
    _MAX_ATTEMPTS = 3
    last_response = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if use_openrouter:
            import openai

            client = openai.OpenAI(api_key=api_key_val, base_url=agent_base_url())
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ],
                    }
                ],
                temperature=0.3,
                max_tokens=3000,
            )
            last_response = response
            scene_description = response.choices[0].message.content if getattr(response, "choices", None) else None
        else:
            genai, types = _load_google_genai()
            client = genai.Client(api_key=api_key_val)

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            try:
                gen_config = types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                )
            except Exception:
                gen_config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=3000)

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    full_prompt,
                ],
                config=gen_config,
            )
            last_response = response

            if response.candidates:
                if response.candidates[0].finish_reason == "SAFETY":
                    raise RuntimeError("Model blocked response due to safety concerns.")
            else:
                raise RuntimeError("No candidates generated. Check prompt/model configuration.")

            scene_description = response.text

        if scene_description:
            break
        if attempt < _MAX_ATTEMPTS:
            print(
                f"[KNOWLEDGE INIT] Empty response on attempt {attempt}/{_MAX_ATTEMPTS} "
                f"(model={model_name!r}). Retrying..."
            )

    if not scene_description:
        if use_openrouter:
            choice = last_response.choices[0] if getattr(last_response, "choices", None) else None
            raise ValueError(
                "Failed to generate scene description from model. "
                f"model={model_name!r} "
                f"finish_reason={getattr(choice, 'finish_reason', None)!r} "
                f"error={getattr(last_response, 'error', None)!r} "
                f"raw={last_response!r}"
            )
        raise ValueError("Failed to generate scene description from model.")

    scene_knowledge_path = output_dir / SCENE_KNOWLEDGE_FILENAME
    scene_knowledge_path.write_text(scene_description, encoding="utf-8")
    return scene_knowledge_path


# ---------------------------------------------------------------------------
# Knowledge update
# ---------------------------------------------------------------------------

def update_scene_knowledge(
    *,
    output_dir: Path,
    previous_image: Image.Image | None = None,
    current_image: Image.Image,
    action_description: str,
    visual_differencing_args: Any | None = None,
    task_description: str | None = None,
) -> Path:
    """Append an action + observation entry to scene_knowledge.txt.

    Uses visual differencing (if args provided) to describe what changed
    between the before/after images.

    Args:
        output_dir: Trial-level directory containing scene_knowledge.txt.
        previous_image: Scene before the action (None → skip visual diff).
        current_image: Scene after the action.
        action_description: Human-readable description of the action taken.
        visual_differencing_args: ModelQueryArgs for visual diff (None → skip).
        task_description: Task context for the diff prompt.

    Returns:
        Path to the updated scene_knowledge.txt.
    """
    scene_knowledge_path = output_dir / SCENE_KNOWLEDGE_FILENAME

    if not scene_knowledge_path.exists():
        print(
            f"[KNOWLEDGE UPDATE] Warning: scene_knowledge.txt not found at {scene_knowledge_path}. "
            "Ensure init_scene_knowledge() succeeded on turn 1."
        )
        return scene_knowledge_path

    update_entry = f"\n\n[Step Update]\nAction: {action_description}"

    if previous_image is not None and visual_differencing_args is not None:
        try:
            from capx.llm.client import query_model as _query_model

            def _img_to_b64(img: Image.Image) -> str:
                buf = io.BytesIO()
                img.save(buf, format="png")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

            differencing_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a robot perception assistant. "
                        "Describe what physically changed in the scene between two images. Focus only on the tabletop and objects on it."
                        "Respond ONLY in plain natural language sentences. "
                        "Do NOT output Python code, markdown, or code fences. "
                        "Be concise and factual."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Task context: {task_description or 'Complete the given action'}\n\n"
                                f"Action taken: {action_description}\n\n"
                                "Compare BEFORE and AFTER and describe in plain English what changed. "
                                "What new information was revealed? What objects moved or were uncovered? "
                                "Write only a short paragraph."
                            ),
                        },
                        {"type": "text", "text": "BEFORE:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_to_b64(previous_image)}"}},
                        {"type": "text", "text": "AFTER:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_to_b64(current_image)}"}},
                    ],
                },
            ]

            result = _query_model(visual_differencing_args, differencing_prompt)
            differencing_text = result.get("content", "")
            if differencing_text:
                update_entry += f"\nObservation: {differencing_text}"
        except Exception:
            update_entry += "\nObservation: Action completed. Visual differencing failed."
    elif visual_differencing_args is None:
        update_entry += "\nObservation: Action completed. Visual differencing not configured."
    else:
        update_entry += "\nObservation: Action completed. Previous image unavailable for comparison."

    with open(scene_knowledge_path, "a", encoding="utf-8") as f:
        f.write(update_entry)

    return scene_knowledge_path


# ---------------------------------------------------------------------------
# Canonical object labelling
# ---------------------------------------------------------------------------

OBJECT_LABELS_FILENAME = "object_labels.json"

_OBJECT_LABELS_USER_PROMPT = (
    "TASK: {task}\n\n"
    "List every task-relevant object visible on the table. For each object, assign:\n"
    "  - id: a short snake_case identifier of the form <noun>_<index>, "
    "indexed from 0 in left-to-right scene order. Two objects of the same noun "
    "share that noun (e.g. avocado_0, avocado_1, avocado_2).\n"
    "  - description: a one-line natural-language phrase that uniquely identifies "
    "the object by color and spatial position. This phrase will be passed verbatim "
    "to a pointing model, so it must be unambiguous on its own (e.g. \"the dark "
    "avocado that is second from the left\").\n\n"
    "Return ONLY a JSON object: {{\"objects\": [{{\"id\": ..., \"description\": ...}}, ...]}}.\n"
    "Do NOT include any object that is not relevant to the task."
)


def _object_labels_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["id", "description"],
                    "additionalProperties": False,
                    "propertyOrdering": ["id", "description"],
                },
            }
        },
        "required": ["objects"],
        "additionalProperties": False,
    }


def _slugify_id(raw: str) -> str:
    """Force an arbitrary id string into safe snake_case so downstream code can use it as a key."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().lower()).strip("_")
    return slug or "obj"


def _normalize_object_labels(objects: list[dict]) -> list[dict]:
    """De-duplicate ids and coerce them to safe slugs."""
    seen: set[str] = set()
    out: list[dict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        raw_id = obj.get("id", "")
        desc = obj.get("description", "")
        if not isinstance(raw_id, str) or not isinstance(desc, str):
            continue
        slug = _slugify_id(raw_id)
        # Ensure uniqueness; append a counter if the model duplicated an id.
        unique = slug
        counter = 1
        while unique in seen:
            counter += 1
            unique = f"{slug}_{counter}"
        seen.add(unique)
        out.append({"id": unique, "description": desc.strip()})
    return out


def init_object_labels(
    *,
    image_path: Path,
    task: str,
    output_dir: Path,
    api_key: str | None = None,
    model_name: str = "gemini-3.1-pro-preview",
) -> Path:
    """Produce a canonical id/description for every task-relevant object in the scene.

    Writes ``object_labels.json`` at ``output_dir`` and returns its path. The
    canonical list is later injected into the Planner, Prioritizer, and code
    generation prompts so every stage refers to objects by the same ids
    (e.g. ``avocado_1``) instead of inventing free-form names each call.
    """
    api_key_val, model_name, use_openrouter = _resolve_api_key_and_model(api_key, model_name)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    user_prompt = _OBJECT_LABELS_USER_PROMPT.format(task=task)
    schema = _object_labels_schema()
    objects_raw: list[dict] = []

    if use_openrouter:
        import openai

        client = openai.OpenAI(api_key=api_key_val, base_url=agent_base_url())
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        schema_text = json.dumps(schema, indent=2)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You label objects in a robot workspace scene. "
                        f"Return ONLY a JSON object matching this schema:\n{schema_text}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.1,
        )
        raw_text = response.choices[0].message.content or ""
        # Reuse the planner's tolerant parser to handle ```json fences etc.
        from physcap.agents.planner import safe_load_json_from_model_text
        parsed = safe_load_json_from_model_text(raw_text)
        if isinstance(parsed, dict):
            objects_raw = parsed.get("objects", []) or []
    else:
        genai, types = _load_google_genai()
        client = genai.Client(api_key=api_key_val)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_json_schema=schema,
            )
        except Exception:
            config = None

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                user_prompt,
            ],
            config=config,
        )
        from physcap.agents.planner import load_json_from_gemini_response
        parsed = load_json_from_gemini_response(response)
        if isinstance(parsed, dict):
            objects_raw = parsed.get("objects", []) or []
        elif isinstance(parsed, list):
            objects_raw = parsed

    if not isinstance(objects_raw, list):
        objects_raw = []
    objects = _normalize_object_labels(objects_raw)

    path = output_dir / OBJECT_LABELS_FILENAME
    path.write_text(json.dumps(objects, indent=2), encoding="utf-8")
    return path


def load_object_labels(output_dir: Path) -> list[dict]:
    """Return the persisted object labels, or [] if none have been written yet."""
    path = Path(output_dir) / OBJECT_LABELS_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def format_object_labels_block(objects: list[dict]) -> str:
    """Render a canonical object list as a prompt fragment.

    Returns an empty string when no labels are available so callers can
    concatenate the result unconditionally.
    """
    if not objects:
        return ""
    rows = "\n".join(
        f"- {obj.get('id', '')}: {obj.get('description', '')}"
        for obj in objects
        if isinstance(obj, dict)
    )
    return (
        "\n\n## Canonical Object Vocabulary\n"
        "When you refer to an object in this scene you MUST use ONE OF THE "
        "CANONICAL IDS below verbatim (e.g. the `name` field of a candidate, "
        "or the `object_id` used in generated code). Do NOT invent new ids "
        "or rephrase existing ones across turns.\n"
        "For natural-language pointing prompts (e.g. arguments to "
        "`get_object_pose` or `get_stiffness`), use the canonical description "
        "verbatim, since a downstream pointing model will consume it.\n\n"
        f"{rows}\n"
    )
