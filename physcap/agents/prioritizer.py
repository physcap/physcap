"""Prioritizer agent for PhysCaP.

Scores exploration candidates with a VLM, given the scene image and the
accumulated scene knowledge, and selects the highest-scoring one.

The VLM emits an integer ``priority_score`` in [0, 10] per candidate and the
caller takes the argmax (``cycle.select_scored_winner``). Read as an
unnormalized belief over which candidate is the task answer, that argmax
coincides with greedy maximization of expected information gain: for a
single-target task with an absolute decision threshold, the information gain of
probing candidate i is the binary entropy H2(b_i), which is monotone in b_i.

Execution cost enters only as a tie-break stated in the prompt; it is not
computed here. Nothing in this module normalizes the scores or maintains an
explicit belief state.
"""

import json
from collections import Counter
from pathlib import Path

from physcap.agents.planner import (
    candidate_key,
    exploration_candidate_schema,
    load_json_from_gemini_response,
    safe_load_json_from_model_text,
)
from physcap.agents.scene_knowledge import _load_google_genai, _resolve_api_key_and_model, agent_base_url


def _canonical_candidate(candidate: dict) -> str:
    """Normalize a candidate dict to a stable JSON string for set comparisons."""
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def build_prioritizer_user_text(
    model_candidates: list[dict],
    scene_knowledge: str | None = None,
    has_image: bool = False,
) -> str:
    """Assemble the user turn for one Prioritizer call.

    Accumulated knowledge goes here rather than in the system prompt: it is
    per-turn data, not policy, and it changes on every call. Without it the
    ranking can only weigh static properties -- spatial layout and action type --
    so it cannot demote a candidate that an earlier measurement has already made
    irrelevant, and its output barely differs from turn to turn.
    """
    knowledge_block = ""
    if scene_knowledge and scene_knowledge.strip():
        knowledge_block = (
            "\n\n## Accumulated Scene Knowledge\n"
            "Everything learned so far this trial: the initial scene description, the "
            "physical properties measured, and the output of each executed step. Use "
            "the measured values to judge which remaining candidates can still change "
            "the outcome — a candidate whose result can no longer change the answer "
            "ranks last.\n"
            f"{scene_knowledge.strip()}"
        )

    image_context_note = (
        "\n\nA scene image is attached showing the current robot workspace. "
        "Use the visual context (sizes, occlusions, and spatial layout) "
        "to inform your ranking of the candidates."
        if has_image else ""
    )

    return (
        f"Here are the exploration candidates:\n{json.dumps(model_candidates, indent=2)}"
        f"{knowledge_block}"
        f"{image_context_note}"
    )


def _build_original_candidate_map(candidates: list[dict]) -> dict[str, dict]:
    """Build a map for restoring original candidate objects by name variants.

    Keyed by candidate key first, because `name` is the object id: two probes of
    one object share it, and a name-only map would silently keep whichever came
    last. The plain name variants stay as a fallback for callers that echo back
    a name without its action.
    """
    original_map: dict[str, dict] = {}
    for cand in candidates:
        name = cand.get("name", "")
        if isinstance(name, str):
            # setdefault: first writer wins, so an ambiguous bare name resolves
            # to the same candidate the model was shown first rather than
            # flipping to the last one merged in.
            original_map.setdefault(name, cand)
            original_map.setdefault(name.replace("_", " "), cand)
        key = candidate_key(cand)
        if key:
            original_map[key] = cand
    return original_map


def _validate_prioritized_candidates(
    prioritized: list,
    original_candidates: list[dict],
    original_map: dict[str, dict],
) -> bool:
    """Verify the model returned the same set of candidates in a different order."""
    if not isinstance(prioritized, list):
        return False
    if len(prioritized) != len(original_candidates):
        return False

    original_counter = Counter(_canonical_candidate(c) for c in original_candidates)
    restored_counter = Counter()

    for item in prioritized:
        if not isinstance(item, dict):
            return False
        name_key = item.get("name")
        if not isinstance(name_key, str):
            return False
        original = original_map.get(name_key)
        if original is None:
            return False
        restored_counter[_canonical_candidate(original)] += 1

    return restored_counter == original_counter


def prioritized_candidates_schema() -> dict:
    return {
        "type": "array",
        "items": exploration_candidate_schema(require_parameters=False),
    }


def prioritized_candidates_with_reasons_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "prioritized_candidates": prioritized_candidates_schema(),
            "reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "reason"],
                },
            },
        },
        "required": ["prioritized_candidates", "reasons"],
    }


def scored_candidates_schema() -> dict:
    """Schema for score-based candidate ranking (mode='scores')."""
    return {
        "type": "object",
        "properties": {
            "scored_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        # Without this a scored entry cannot say WHICH probe of
                        # the object it ranked, so the winner cannot be mapped
                        # back to its full candidate.
                        "action": {"type": "string"},
                        "reason": {"type": "string"},
                        "priority_score": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                    "required": ["name", "action", "reason", "priority_score"],
                },
            },
        },
        "required": ["scored_candidates"],
    }


def _validate_prioritized_with_reasons(
    prioritized: dict,
    original_candidates: list[dict],
    original_map: dict[str, dict],
) -> bool:
    if not isinstance(prioritized, dict):
        return False

    prioritized_candidates = prioritized.get("prioritized_candidates")
    reasons = prioritized.get("reasons")
    if not isinstance(prioritized_candidates, list) or not isinstance(reasons, list):
        return False

    if not _validate_prioritized_candidates(prioritized_candidates, original_candidates, original_map):
        return False

    if len(reasons) != len(original_candidates):
        return False

    # Keyed on the candidate, not the name: `name` is the object id, so two
    # probes of one object share it and a name-keyed set would reject a
    # perfectly good ordering as containing duplicates.
    expected = {candidate_key(c) for c in original_candidates}
    seen: set[str] = set()
    for entry in reasons:
        if not isinstance(entry, dict):
            return False
        name = entry.get("name")
        reason_text = entry.get("reason")
        if not isinstance(name, str) or not isinstance(reason_text, str):
            return False
        key = candidate_key(entry)
        if key in seen:
            return False
        # original_map holds both keyed and bare-name entries, so a model that
        # omitted `action` degrades to the name lookup instead of hard-failing.
        if key not in expected and name not in original_map:
            return False
        seen.add(key)

    return len(seen) == len(original_candidates)


def _validate_scored_candidates(
    scored: dict,
    original_candidates: list[dict],
    original_map: dict[str, dict],
) -> bool:
    """Validate score-based output: all candidates present with valid scores (0-10)."""
    if not isinstance(scored, dict):
        return False

    scored_list = scored.get("scored_candidates")
    if not isinstance(scored_list, list):
        return False

    if len(scored_list) != len(original_candidates):
        return False

    # Keyed on the candidate, not the name. With one probe per (object, action)
    # a name-keyed set makes this unsatisfiable: 8 candidates over 4 objects
    # always trip the duplicate check, so every scores-mode run burned its
    # retries and fell back to the original order.
    expected = {candidate_key(c) for c in original_candidates}
    seen: set[str] = set()
    for entry in scored_list:
        if not isinstance(entry, dict):
            return False
        name = entry.get("name")
        reason_text = entry.get("reason")
        score = entry.get("priority_score")
        if not isinstance(name, str) or not isinstance(reason_text, str):
            return False
        if not isinstance(score, int) or score < 0 or score > 10:
            return False
        key = candidate_key(entry)
        if key in seen:
            return False
        if key not in expected and name not in original_map:
            return False
        seen.add(key)

    return len(seen) == len(original_candidates)


def prioritizer(
    input_json_path: str,
    system_prompt: str,
    model_name: str = "gemini-3.1-pro-preview",
    api_key: str = None,
    image_path: str | None = None,
    max_retries: int = 3,
    mode: str | None = None,
    scene_knowledge: str | None = None,
) -> list | dict:
    """Reorder exploration candidates by efficiency.

    Args:
        input_json_path: Path to the JSON file containing exploration candidates.
        system_prompt: System prompt for the Prioritizer model.
        model_name: VLM model name to use for prioritization.
        api_key: Environment variable name (or raw value) holding the API key.
        image_path: Optional scene image — when provided the VLM uses visual
            context (spatial layout, occlusions) to inform its ranking.
        max_retries: Number of times to retry if the model output is invalid.
        scene_knowledge: Optional accumulated scene knowledge (the contents of
            scene_knowledge.txt). Without it the ranking can only reason about
            static properties -- spatial layout and action type -- so it cannot
            tell that a candidate has already been ruled out by an earlier
            measurement, and its ordering is near identical from turn to turn.
        mode: Prioritization mode. Options:
            - None (default): Return only reordered candidates as a list.
            - "reasons": Return dict with prioritized_candidates and reasons.
            - "scores": Return dict with scored_candidates (each with name, reason, priority_score 0-10).

    Returns:
        List of reordered candidates (mode=None), or dict with explanations/scores.
    """
    input_path = Path(input_json_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    if not isinstance(candidates, list):
        raise ValueError("Exploration candidates must be a JSON array.")

    if not candidates:
        print("Warning: Candidates list is empty. Nothing to prioritize.")
        p_path = input_path.parent / "prioritized_candidates.json"
        p_path.write_text(json.dumps([], indent=4), encoding="utf-8")
        return []

    api_key_val, model_name, use_openrouter = _resolve_api_key_and_model(api_key, model_name)

    # Clean candidate names for the model; keep a map to restore originals.
    original_map = _build_original_candidate_map(candidates)
    model_candidates = []
    for cand in candidates:
        cand_copy = cand.copy()
        name = cand_copy.get("name", "")
        if isinstance(name, str):
            cand_copy["name"] = name.replace("_", " ")
        model_candidates.append(cand_copy)

    has_image = image_path is not None and Path(image_path).exists()
    user_text = build_prioritizer_user_text(
        model_candidates,
        scene_knowledge=scene_knowledge,
        has_image=has_image,
    )

    def _run_prioritizer_once() -> list | dict:
        """Execute a single Prioritizer model call and return raw parsed output."""
        if mode == "reasons":
            schema = prioritized_candidates_with_reasons_schema()
            explanation_note = (
                "\n\nFor each candidate, provide a short reason explaining why it should be prioritized. "
                "Return ONLY a valid JSON object with keys \"prioritized_candidates\" and \"reasons\"."
            )
        elif mode == "scores":
            schema = scored_candidates_schema()
            explanation_note = (
                "\n\nFor each candidate, provide a reason and assign a priority_score from 0 to 10 "
                "(10=highest priority, 0=lowest). Score each candidate independently. "
                "Return ONLY a valid JSON object with key \"scored_candidates\"."
            )
        else:
            schema = prioritized_candidates_schema()
            explanation_note = ""

        if use_openrouter:
            import base64
            import openai

            client = openai.OpenAI(api_key=api_key_val, base_url=agent_base_url())
            schema_text = json.dumps(schema, indent=2)
            full_prompt = (
                system_prompt
                + explanation_note
                + f"\n\nOUTPUT FORMAT:\nYou MUST return a valid JSON response matching this JSON Schema:\n{schema_text}"
            )

            user_content: list[dict] = [{"type": "text", "text": user_text}]
            if has_image:
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.2,
            )

            raw_text = response.choices[0].message.content
            if not raw_text:
                raise ValueError("OpenRouter response content is None.")
            prioritized_output = safe_load_json_from_model_text(raw_text)

        else:
            genai, types = _load_google_genai()
            client = genai.Client(api_key=api_key_val)

            full_prompt = system_prompt + explanation_note + "\n\n" + user_text

            try:
                config = types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                )
            except Exception:
                config = None

            contents: list = []
            if has_image:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            contents.append(full_prompt)

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            if response.candidates:
                if response.candidates[0].finish_reason == "SAFETY":
                    raise RuntimeError("Model blocked response due to safety concerns.")
            else:
                raise RuntimeError("No candidates generated. Check prompt/model configuration.")

            prioritized_output = load_json_from_gemini_response(response)

        if mode == "reasons" or mode == "scores":
            if not isinstance(prioritized_output, dict):
                raise ValueError(f"Model output must be a JSON object when mode={mode}.")
        else:
            if not isinstance(prioritized_output, list):
                raise ValueError(f"Model output is not a JSON list.\nraw_text: {prioritized_output}")

        return prioritized_output

    prioritized = None
    for attempt in range(1, max_retries + 1):
        try:
            prioritized = _run_prioritizer_once()
        except ValueError as e:
            print(
                f"Warning: Prioritizer output could not be parsed as JSON on "
                f"attempt {attempt}/{max_retries} ({e})."
            )
            valid = False
        else:
            if mode == "scores":
                valid = _validate_scored_candidates(prioritized, candidates, original_map)
            elif mode == "reasons":
                valid = _validate_prioritized_with_reasons(prioritized, candidates, original_map)
            else:
                valid = _validate_prioritized_candidates(prioritized, candidates, original_map)

        if valid:
            break
        if attempt < max_retries:
            print(
                f"Warning: Prioritizer returned invalid candidate ordering on attempt {attempt}/{max_retries}. Retrying..."
            )
        else:
            print(
                f"Warning: Prioritizer output did not preserve candidate count/details after {max_retries} attempts. "
                "Falling back to original candidate order."
            )
            # `action` must be carried through: without it these synthesized
            # entries cannot be keyed back to their candidates, so the caller
            # restores nothing and codegen receives an object id with no
            # instruction -- a silent failure worse than the one being handled.
            if mode == "scores":
                prioritized = {
                    "scored_candidates": [
                        {
                            "name": cand.get("name", ""),
                            "action": cand.get("action", ""),
                            "reason": "Fallback: prioritizer failed to produce valid output.",
                            "priority_score": 5,
                        }
                        for cand in candidates
                    ],
                }
            elif mode == "reasons":
                prioritized = {
                    "prioritized_candidates": candidates,
                    "reasons": [
                        {
                            "name": cand.get("name", ""),
                            "action": cand.get("action", ""),
                            "reason": "Fallback original order after prioritizer failed to produce valid output.",
                        }
                        for cand in candidates
                    ],
                }
            else:
                prioritized = candidates

    def _restore_original_ordered_candidates(ordered_candidates: list) -> list:
        final = []
        for item in ordered_candidates:
            # Try the action-aware key first so a reordered pair of probes on one
            # object resolves to the right member rather than collapsing.
            restored = original_map.get(candidate_key(item)) if isinstance(item, dict) else None
            if restored is None:
                name_key = item.get("name") if isinstance(item, dict) else (item if isinstance(item, str) else None)
                restored = original_map.get(name_key, item) if name_key else item
            final.append(restored)
        return final

    def _canonical_name(spaced_or_underscored: str | None) -> str | None:
        """Map a model-returned name back to its original (underscored) form."""
        if not isinstance(spaced_or_underscored, str):
            return spaced_or_underscored
        original = original_map.get(spaced_or_underscored)
        if original is not None:
            canonical = original.get("name")
            if isinstance(canonical, str):
                return canonical
        return spaced_or_underscored

    if mode == "scores" and isinstance(prioritized, dict):
        # Restore canonical (underscored) names so downstream string-matching
        # against all_candidates / executed_plans uses a consistent key.
        for entry in prioritized.get("scored_candidates", []):
            if isinstance(entry, dict):
                entry["name"] = _canonical_name(entry.get("name"))
        output_file = input_path.parent / "prioritized_candidates.json"
        output_file.write_text(json.dumps(prioritized, indent=4), encoding="utf-8")
        return prioritized

    if mode == "reasons" and isinstance(prioritized, dict):
        prioritized["prioritized_candidates"] = _restore_original_ordered_candidates(
            prioritized["prioritized_candidates"]
        )
        for entry in prioritized.get("reasons", []):
            if isinstance(entry, dict):
                entry["name"] = _canonical_name(entry.get("name"))
        output_file = input_path.parent / "prioritized_candidates.json"
        output_file.write_text(json.dumps(prioritized, indent=4), encoding="utf-8")
        return prioritized

    final = _restore_original_ordered_candidates(prioritized)
    output_file = input_path.parent / "prioritized_candidates.json"
    output_file.write_text(json.dumps(final, indent=4), encoding="utf-8")
    return final
