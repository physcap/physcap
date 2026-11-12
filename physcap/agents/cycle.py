"""Per-turn orchestration of the Planner and Prioritizer for PhysCaP.

Provides ``run_reasoning_turn``, a unified function that runs the Planner
(and optionally the Prioritizer) for a single turn, tracks interaction
history across turns, and returns the selected candidate for code
generation. Two modes are supported: plain Planner (take the top pending
candidate), Planner+Prioritizer (rank pending candidates with a second VLM
call), and PhysCaP-joint (the Planner re-ranks its own candidates every turn;
``follow_fresh_ordering`` makes the cycle follow that ordering instead of the
accumulated pool's insertion order).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from physcap.agents.planner import candidate_key, planner, normalize_candidate_name
from physcap.agents.prioritizer import prioritizer
from physcap.agents.scene_knowledge import (
    OBJECT_LABELS_FILENAME,
    load_measured_properties,
    _configure_google_api_env,
    _load_scene_knowledge,
    load_object_labels,
    _load_text,
    format_object_labels_block,
    init_scene_knowledge,
    init_object_labels,
)
from physcap.prompt_files import PROMPTS_DIR, for_robot

DEFAULT_PLANNER_PROMPT_PATH = PROMPTS_DIR / "planner.txt"
DEFAULT_PRIORITIZER_PROMPT_PATH = PROMPTS_DIR / "prioritizer.txt"


# Candidate identity (candidate_key) lives in planner.py, next to the schema
# that defines it, so the Prioritizer can share it without importing this module.


def merge_new_candidates(all_candidates: list[dict], new_candidates: list[dict]) -> int:
    """Append candidates not already in the pool. Returns how many were added.

    Keyed by :func:`candidate_key`, so two probes of one object are distinct
    entries while a genuinely re-proposed probe is still absorbed.
    """
    existing = {candidate_key(c) for c in all_candidates}
    added = 0
    for cand in new_candidates:
        key = candidate_key(cand)
        if key and key not in existing:
            all_candidates.append(cand)
            existing.add(key)
            added += 1
    return added


def filter_pending_candidates(
    all_candidates: list[dict], executed: list, failed: list
) -> list[dict]:
    """Drop candidates already attempted, so nothing is re-selected.

    Compares on the candidate key rather than the name: the Prioritizer can echo
    names back with different spacing, and -- more importantly -- measuring an
    object's mass must not mark its stiffness probe as done as well.
    """
    done = {candidate_key(n) for n in list(executed) + list(failed)}
    return [c for c in all_candidates if candidate_key(c) not in done]


# Actions whose success is judged by a recorded measurement, not by the sandbox
# exiting cleanly: code that refuses to move still exits 0.
_MEASURE_ACTIONS = ("measure_mass", "measure_stiffness")
_MAX_UNRECORDED_ATTEMPTS = 2


def probed_candidate_keys(
    measured_properties: list[dict], object_labels: list[dict]
) -> set[str]:
    """Candidate keys for probes that were actually performed.

    Generated code can batch two probes into one step -- measuring a cube's
    mass and reading its stiffness in the same trip is locally sensible even
    when the selected candidate asked for one. Retiring only the selected
    candidate would then leave the other probe in the pool to be offered and
    re-run.

    Object names are matched against the canonical labels (by description, which
    generated code is instructed to pass verbatim, or by id). An unmatched name
    retires nothing: over-retiring would hide work the agent still owes.
    """
    by_label: dict[str, str] = {}
    for obj in object_labels:
        if not isinstance(obj, dict):
            continue
        oid = obj.get("id")
        if not oid:
            continue
        for field in ("description", "id"):
            token = normalize_candidate_name(obj.get(field))
            if token:
                by_label.setdefault(token, oid)

    keys: set[str] = set()
    for record in measured_properties:
        if not isinstance(record, dict):
            continue
        name = normalize_candidate_name(record.get("object_name"))
        prop = record.get("property_type")
        oid = by_label.get(name)
        if not oid or not prop:
            continue
        keys.add(candidate_key({"name": oid, "action": f"measure_{prop}"}))
    return keys


def select_from_fresh_ordering(
    new_candidates: list[dict], pending_candidates: list[dict]
) -> dict | None:
    """Pick the highest-ranked still-pending candidate from THIS turn's Planner call.

    PhysCaP-joint re-generates and re-orders its candidates every turn, but
    ``pending_candidates`` is the accumulated pool in *insertion* order and
    ``merge_new_candidates`` appends novel entries at the end. Taking the pool's
    head would therefore follow turn 1's ranking for the rest of the trial and
    silently discard every later re-prioritization.

    This honours the freshest ordering instead, while still refusing anything
    already executed. Falls back to the pool when the Planner re-proposed
    nothing that is still outstanding.
    """
    by_key = {candidate_key(c): c for c in pending_candidates}
    for cand in new_candidates or []:
        pooled = by_key.get(candidate_key(cand))
        if pooled is not None:
            # Merge the way select_scored_winner does: this turn's wording wins,
            # the pooled copy backfills any field the re-emission dropped.
            # Codegen reads description/parameters off this, and has improvised
            # its own action before when handed a candidate stripped of them.
            return {**pooled, **cand}

    if not pending_candidates:
        return None

    # Nothing proposed this turn is still outstanding, so the pool holds only
    # candidates the Planner has stopped re-proposing. Take the most recently
    # merged one rather than the head: pending_candidates[0] is the
    # insertion-ordered front, and returning it would defeat the point of
    # following the fresh ordering.
    #
    # Known limitation: "most recently merged" only means "most recently
    # proposed" when candidates entered the pool across different turns. If they
    # all arrived on turn 1 -- the usual case -- [-1] is that turn's
    # LOWEST-ranked survivor. The warning below exists to find out whether this
    # path is ever actually taken; if it is, the honest fix is to record each
    # candidate's insertion turn rather than lean on list position.
    print(
        "[Cycle] Fresh ordering matched no pending candidate; falling back to the "
        f"most recently merged of {len(pending_candidates)} pending."
    )
    return pending_candidates[-1]


def select_scored_winner(
    scored_list: list[dict], pending_candidates: list[dict]
) -> dict | None:
    """Pick the highest-scored candidate and restore its full definition.

    A scored entry carries only {name, action, reason, priority_score}. Codegen
    reads description/parameters/expected_info off the selected candidate, so
    returning the scored entry verbatim leaves it with an object id and no
    instruction -- it then improvises its own action, and the Prioritizer's plan
    is not what actually runs. Merging the winner onto its original candidate
    keeps `reason`/`priority_score` for the artifacts while restoring the rest.
    (mode="reasons" gets this for free via _restore_original_ordered_candidates.)

    A winner that matches nothing in the pool falls through to the bare scored
    entry, which is the pre-existing behaviour.
    """
    if not scored_list:
        return None
    winner = max(scored_list, key=lambda x: x.get("priority_score", 0))

    by_key = {candidate_key(c): c for c in pending_candidates}
    # Name-only fallback (first writer wins). A scored entry that lost its
    # `action` would otherwise key to the bare object id, match nothing, and
    # hand codegen an instruction-less candidate. Resolving to *some* probe of
    # the right object is strictly better than resolving to nothing.
    by_name: dict[str, dict] = {}
    for c in pending_candidates:
        by_name.setdefault(normalize_candidate_name(c.get("name")), c)

    original = (
        by_key.get(candidate_key(winner))
        or by_name.get(normalize_candidate_name(winner.get("name")))
        or {}
    )
    return {**original, **winner}


# ---------------------------------------------------------------------------
# Interaction history helpers
# ---------------------------------------------------------------------------

def _load_interaction_history(output_dir: Path) -> dict[str, Any]:
    """Load persisted interaction history from the trial output directory."""
    history_file = output_dir / "interaction_history.json"
    if history_file.exists():
        return json.loads(history_file.read_text(encoding="utf-8"))
    return {
        "executed_plans": [],
        "failed_plans": [],
        "all_candidates": [],
        "turn": 0,
    }


def _save_interaction_history(output_dir: Path, history: dict[str, Any]) -> None:
    """Persist interaction history for the next turn."""
    (output_dir / "interaction_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Planner prompt builder
# ---------------------------------------------------------------------------

def _build_incremental_planner_prompt(
    system_prompt: str,
    task: str,
    interaction_history: dict[str, Any],
    user_task_prompt: str | None = None,
    scene_knowledge: str | None = None,
    output_dir: Path | None = None,
    object_labels: list[dict] | None = None,
) -> str:
    """Build a Planner prompt enriched with scene knowledge and past attempts.

    Args:
        system_prompt: Base system prompt text.
        task: Task description.
        interaction_history: Dict with executed_plans, failed_plans lists.
        user_task_prompt: Optional task-specific prefix to prepend.
        scene_knowledge: Pre-loaded knowledge text (loads from output_dir if None).
        output_dir: Trial-level dir to load scene_knowledge.txt from.
        object_labels: Canonical object list to inject. When provided, every
            ``name`` in the Planner's output must come from this list.
    """
    if scene_knowledge is None and output_dir is not None:
        scene_knowledge = _load_scene_knowledge(output_dir)

    executed = interaction_history.get("executed_plans", [])
    failed = interaction_history.get("failed_plans", [])

    incremental_context = ""

    if object_labels:
        incremental_context += format_object_labels_block(object_labels)
        incremental_context += (
            "\nIMPORTANT: Use the canonical id (e.g. `avocado_1`) as the `name` "
            "field of every exploration_candidates entry. Do not invent new "
            "object names or rephrase the canonical ids.\n"
        )

    if scene_knowledge:
        incremental_context += "\n\n## Current Scene Knowledge\n"
        incremental_context += "Based on initial scene analysis and previous interactions:\n"
        incremental_context += scene_knowledge

    plan_outputs = interaction_history.get("plan_outputs", {})

    if executed or failed:
        incremental_context += "\n\n## Previous Interaction Attempts\n"
        if executed:
            incremental_context += "\nAlready executed:\n"
            for plan_name in executed:
                snippet = plan_outputs.get(plan_name, "")
                if snippet:
                    incremental_context += f"- {plan_name}\n  Output: {snippet}\n"
                else:
                    incremental_context += f"- {plan_name}\n"
        if failed:
            incremental_context += f"\nFailed to achieve goal (need new strategies): {', '.join(failed)}\n"
        incremental_context += (
            "\nFor new exploration candidates, avoid repeating the above strategies. "
            "Only propose NEW exploration actions that haven't been attempted yet."
        )

    prompt = ""
    if user_task_prompt:
        prompt += user_task_prompt.strip() + "\n\n"
    prompt += system_prompt
    prompt += incremental_context
    return prompt


# ---------------------------------------------------------------------------
# Reasoning summary writer
# ---------------------------------------------------------------------------

def _write_reasoning_summary(
    turn_dir: Path,
    iteration: int,
    image_path: Path | str,
    candidates_json_path: Path,
    prioritized_json_path: Path,
    selected: dict | None,
    interaction_history: dict,
    execution_decision: str = "standard",
) -> None:
    summary = {
        "iteration": iteration,
        "image_path": str(image_path),
        "candidates_json_path": str(candidates_json_path),
        "prioritized_json_path": str(prioritized_json_path),
        "selected_candidate": selected,
        "execution_decision": execution_decision,
        "interaction_history": interaction_history,
    }
    (turn_dir / "reasoning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Unified reasoning turn
# ---------------------------------------------------------------------------

def run_reasoning_turn(
    *,
    image_path: Path | str,
    task: str,
    output_dir: Path,
    turn_dir: Path,
    iteration: int,
    execution_result: dict[str, Any] | None = None,
    api_key: str | None = None,
    planner_model: str = "gemini-3.1-pro-preview",
    prioritizer_model: str = "gemini-3.1-pro-preview",
    use_prioritizer: bool = True,
    follow_fresh_ordering: bool = False,
    planner_system_prompt_path: str | None = None,
    prioritizer_system_prompt_path: str | None = None,
    user_task_prompt_path: str | None = None,
    robot: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run one turn of the Planner / Prioritizer pipeline.

    Tracks interaction history across turns. Scene knowledge is maintained
    in ``output_dir``; per-turn artefacts go to ``turn_dir``.

    Args:
        image_path: Path to the current scene image.
        task: Task description.
        output_dir: Trial-level directory (scene_knowledge.txt lives here).
        turn_dir: Per-turn output directory for candidates and summaries.
        iteration: Current turn index (0-based).
        execution_result: Result from the previous turn's code execution.
            Expected keys: 'executed_plan_name', 'success', 'sandbox_rc',
            'stdout', 'stderr'. stdout is already appended to scene_knowledge.txt
            before this function is called; it is used here only for history tracking.
        api_key: Raw API key (or env-var name if already exported).
        planner_model: Model name for the Planner.
        prioritizer_model: Model name for the Prioritizer.
        use_prioritizer: If True, run the Prioritizer on accumulated candidates
            (Planner+Prioritizer mode). If False, pick a candidate directly
            from the pool (plain Planner mode, or PhysCaP-joint below).
        follow_fresh_ordering: PhysCaP-joint mode. When ``use_prioritizer`` is
            False, the plain-Planner branch selects the head of the
            accumulated pool, which is insertion-ordered and so freezes turn
            1's ranking. Set True to follow the ordering the Planner produced
            *this* turn instead -- the Planner is expected to re-rank its
            candidates every call in this mode.
        planner_system_prompt_path: Override path for the Planner system prompt.
        prioritizer_system_prompt_path: Override path for the Prioritizer prompt.
        user_task_prompt_path: Optional task-specific user prompt file.
        robot: The robot the task runs on; replaces the AgileX PiPER named in
            the Planner and Prioritizer prompts (``physcap.prompt_files.for_robot``).

    Returns:
        Dict with keys: selected_candidate, candidates_json_path,
            prioritized_json_path, iteration, interaction_history, timing.
    """
    api_env_var = _configure_google_api_env(api_key)

    _planner_prompt_path = (
        Path(planner_system_prompt_path)
        if planner_system_prompt_path
        else DEFAULT_PLANNER_PROMPT_PATH
    )
    _prioritizer_prompt_path = (
        Path(prioritizer_system_prompt_path)
        if prioritizer_system_prompt_path
        else DEFAULT_PRIORITIZER_PROMPT_PATH
    )

    planner_system_prompt = for_robot(_load_text(_planner_prompt_path), robot)
    prioritizer_system_prompt = for_robot(_load_text(_prioritizer_prompt_path), robot)
    user_task_prompt = _load_text(Path(user_task_prompt_path)) if user_task_prompt_path else None

    turn_dir.mkdir(parents=True, exist_ok=True)

    # --- Load interaction history ---
    interaction_history = _load_interaction_history(output_dir)
    interaction_history["turn"] = iteration
    if "plan_outputs" not in interaction_history:
        interaction_history["plan_outputs"] = {}  # plan_name -> stdout snippet

    # --- Ensure scene knowledge exists ---
    if not (output_dir / "scene_knowledge.txt").exists():
        try:
            init_scene_knowledge(
                image_path=Path(image_path),
                task=task,
                output_dir=output_dir,
                api_key=api_env_var,
                model_name=planner_model,
            )
        except Exception as e:
            print(f"[TRIAL] Warning: Failed to initialize scene knowledge: {e}")

    # --- Ensure canonical object labels exist ---
    # Produced once per trial on the first turn; all subsequent Planner /
    # Prioritizer / codegen calls reference these ids so naming stays stable.
    if not (output_dir / OBJECT_LABELS_FILENAME).exists():
        try:
            init_object_labels(
                image_path=Path(image_path),
                task=task,
                output_dir=output_dir,
                api_key=api_env_var,
                model_name=planner_model,
            )
        except Exception as exc:
            print(f"[Cycle] Warning: init_object_labels failed ({exc}); continuing without canonical vocabulary.")
    object_labels = load_object_labels(output_dir)
    _probed = probed_candidate_keys(
        load_measured_properties(output_dir), object_labels
    )

    # --- Record the previous step ---
    if execution_result:
        # Prefer the full candidate: its key distinguishes measuring an
        # object's mass from probing its stiffness, which the bare name
        # cannot. The name is still accepted for callers that only have that.
        _executed_candidate = execution_result.get("executed_candidate")
        executed_plan = (
            candidate_key(_executed_candidate)
            if _executed_candidate
            else execution_result.get("executed_plan_name")
        )
        stdout = execution_result.get("stdout", "").strip()
        _action = (_executed_candidate or {}).get("action")
        if executed_plan:
            if not execution_result.get("success"):
                if executed_plan not in interaction_history["failed_plans"]:
                    interaction_history["failed_plans"].append(executed_plan)
            elif _action in _MEASURE_ACTIONS and executed_plan not in _probed:
                # The code exited cleanly but never took the measurement (e.g. it
                # refused over a precondition, or measured another object). Keep
                # the candidate pending; retire it as failed after repeated misses
                # so a model that keeps refusing cannot burn every turn on it.
                attempts = interaction_history.setdefault("unrecorded_attempts", {})
                attempts[executed_plan] = attempts.get(executed_plan, 0) + 1
                print(
                    f"[Cycle] {executed_plan!r} ran without recording its measurement "
                    f"(attempt {attempts[executed_plan]}/{_MAX_UNRECORDED_ATTEMPTS}); "
                    "keeping it pending."
                )
                if (
                    attempts[executed_plan] >= _MAX_UNRECORDED_ATTEMPTS
                    and executed_plan not in interaction_history["failed_plans"]
                ):
                    interaction_history["failed_plans"].append(executed_plan)
            elif executed_plan not in interaction_history["executed_plans"]:
                interaction_history["executed_plans"].append(executed_plan)
            # Record a short snippet of the printed output so the Planner can
            # reason about what each plan revealed without re-reading scene_knowledge.
            if stdout and executed_plan not in interaction_history["plan_outputs"]:
                _MAX_SNIPPET = 500
                interaction_history["plan_outputs"][executed_plan] = (
                    stdout[:_MAX_SNIPPET] + "..." if len(stdout) > _MAX_SNIPPET else stdout
                )

    # --- Retire candidates for probes that were actually performed ---
    # Recomputed in full each turn rather than diffed, so this is idempotent and
    # self-healing: a record that arrives late still lands, and re-running a turn
    # cannot double-count. Covers probes the generated code performed beyond the
    # one it was asked for.
    for _key in sorted(_probed):
        if _key not in interaction_history["executed_plans"]:
            interaction_history["executed_plans"].append(_key)

    # --- Build enriched Planner prompt ---
    enhanced_planner_prompt = _build_incremental_planner_prompt(
        planner_system_prompt,
        task,
        interaction_history,
        user_task_prompt,
        output_dir=output_dir,
        object_labels=object_labels,
    )

    # --- Run Planner ---
    if debug:
        print("=" * 80)
        print("Inferencing planner module")
    _planner_start = time.time()
    candidates_json_path = Path(
        planner(
            image_path=str(image_path),
            task=task,
            system_prompt=enhanced_planner_prompt,
            model_name=planner_model,
            output_base_dir=str(turn_dir),
            api_key=api_env_var,
        )
    )
    _planner_elapsed = time.time() - _planner_start
    if debug:
        print(f"Time taken to complete planner module: {_planner_elapsed:.2f} seconds.")

    # --- Check sufficiency / completion flags ---
    sufficient_information = False
    task_complete = False
    completion_reason = ""
    full_response_path = candidates_json_path.parent / "exploration_full_response.json"
    if full_response_path.exists():
        try:
            full_response = json.loads(full_response_path.read_text(encoding="utf-8"))
            sufficient_information = full_response.get("sufficient", False)
            task_complete = full_response.get("task_complete", False) is True
            completion_reason = full_response.get("reason", "")
        except Exception:
            sufficient_information = False
            task_complete = False

    # The goal is already reached, so stop instead of mapping "sufficient" to
    # original_task_execute, whose codegen redoes the whole task. Only trusted
    # once something has executed: before that, a "done" claim has no basis.
    if task_complete and not interaction_history.get("executed_plans"):
        print(
            "[Cycle] Warning: planner reported task_complete before any step executed; "
            "ignoring it."
        )
        task_complete = False
    if task_complete:
        selected = {
            "name": "task_complete",
            "description": "Task goal already achieved; no further action.",
            "reason": completion_reason,
        }
        prioritized_json_path = turn_dir / "no_prioritization_task_complete.json"
        prioritized_json_path.write_text(json.dumps([selected], indent=2), encoding="utf-8")
        _save_interaction_history(output_dir, interaction_history)
        _write_reasoning_summary(
            turn_dir, iteration, image_path, candidates_json_path,
            prioritized_json_path, selected, interaction_history,
            execution_decision="task_complete",
        )
        return {
            "selected_candidate": selected,
            "candidates_json_path": str(candidates_json_path),
            "prioritized_json_path": str(prioritized_json_path),
            "iteration": iteration,
            "interaction_history": interaction_history,
            "timing": {"planner_s": _planner_elapsed, "prioritizer_s": 0.0},
        }

    if sufficient_information:
        selected: dict = {
            "name": "original_task_execute",
            "description": "Execute task directly: sufficient information already gathered.",
            "expected_info": "Complete the task as originally requested.",
            "estimated_cost": "low",
        }
        prioritized_json_path = turn_dir / "no_prioritization_sufficient.json"
        prioritized_json_path.write_text(json.dumps([selected], indent=2), encoding="utf-8")
        _save_interaction_history(output_dir, interaction_history)
        _write_reasoning_summary(
            turn_dir, iteration, image_path, candidates_json_path,
            prioritized_json_path, selected, interaction_history,
            execution_decision="direct_execution_sufficient",
        )
        return {
            "selected_candidate": selected,
            "candidates_json_path": str(candidates_json_path),
            "prioritized_json_path": str(prioritized_json_path),
            "iteration": iteration,
            "interaction_history": interaction_history,
            "timing": {"planner_s": _planner_elapsed, "prioritizer_s": 0.0},
        }

    # --- Merge new candidates into accumulated history ---
    new_candidates = json.loads(candidates_json_path.read_text(encoding="utf-8"))
    if not isinstance(new_candidates, list):
        new_candidates = []
    merge_new_candidates(interaction_history["all_candidates"], new_candidates)

    # Filter out already-executed and failed candidates so the Prioritizer/
    # plain-Planner selector never re-selects an action already attempted.
    # Compare on the candidate key (case/underscore/hyphen-insensitive, and
    # action-aware) because the Prioritizer can echo back names with different
    # spacing than the canonical entries in all_candidates -- and because
    # measuring an object's mass must not mark its stiffness probe as done too.
    pending_candidates = filter_pending_candidates(
        interaction_history["all_candidates"],
        interaction_history.get("executed_plans", []),
        interaction_history.get("failed_plans", []),
    )
    if debug:
        _skipped = len(interaction_history["all_candidates"]) - len(pending_candidates)
        print(
            f"[Cycle] Candidate pool: {len(interaction_history['all_candidates'])} total, "
            f"{_skipped} already-done filtered out → {len(pending_candidates)} pending"
        )

    pending_candidates_path = turn_dir / "pending_candidates.json"
    pending_candidates_path.write_text(
        json.dumps(pending_candidates, indent=2), encoding="utf-8"
    )

    # --- All candidates exhausted: force task execution ---
    if not pending_candidates:
        print(
            "[Cycle] All known exploration candidates have been attempted. "
            "Forcing direct task execution."
        )
        selected = {
            "name": "original_task_execute",
            "description": "Execute task directly: all exploration candidates exhausted.",
            "expected_info": "Complete the task with the information gathered so far.",
            "estimated_cost": "low",
        }
        prioritized_json_path = turn_dir / "no_prioritization_exhausted.json"
        prioritized_json_path.write_text(json.dumps([selected], indent=2), encoding="utf-8")
        _save_interaction_history(output_dir, interaction_history)
        _write_reasoning_summary(
            turn_dir, iteration, image_path, candidates_json_path,
            prioritized_json_path, selected, interaction_history,
            execution_decision="direct_execution_exhausted",
        )
        return {
            "selected_candidate": selected,
            "candidates_json_path": str(candidates_json_path),
            "prioritized_json_path": str(prioritized_json_path),
            "iteration": iteration,
            "interaction_history": interaction_history,
            "timing": {"planner_s": _planner_elapsed, "prioritizer_s": 0.0},
        }

    # --- Prioritize or pick first ---
    _prioritizer_elapsed = 0.0
    if use_prioritizer:
        if object_labels:
            prioritizer_system_prompt += format_object_labels_block(object_labels)
            prioritizer_system_prompt += (
                "\nDo NOT change the `name` field of any candidate. Each `name` is "
                "already a canonical id from the vocabulary above.\n"
            )
        if debug:
            print("=" * 80)
            print("Inferencing prioritizer module")
        _prioritizer_start = time.time()
        prioritized = prioritizer(
            input_json_path=str(pending_candidates_path),
            system_prompt=prioritizer_system_prompt,
            model_name=prioritizer_model,
            api_key=api_env_var,
            image_path=str(image_path),
            mode="scores",
            # Without this the ranking can only weigh static properties, so it
            # cannot demote a candidate that earlier measurements have already
            # made irrelevant -- and its output barely changes between turns.
            scene_knowledge=_load_scene_knowledge(output_dir),
        )
        _prioritizer_elapsed = time.time() - _prioritizer_start
        if debug:
            print(f"Time taken to complete prioritizer module: {_prioritizer_elapsed:.2f} seconds.")
        prioritized_json_path = turn_dir / "prioritized_candidates.json"
        prioritized_json_path.write_text(json.dumps(prioritized, indent=2), encoding="utf-8")
        if isinstance(prioritized, dict):
            # mode="scores" or mode="reasons" - extract prioritized candidates
            if "scored_candidates" in prioritized:
                # mode="scores": highest priority_score wins, restored to its
                # full candidate so codegen gets the action, not just the object.
                scored_list = prioritized["scored_candidates"]
                selected = select_scored_winner(scored_list, pending_candidates)
            elif "prioritized_candidates" in prioritized:
                # mode="reasons": already ordered, just select first
                selected = prioritized["prioritized_candidates"][0] if prioritized["prioritized_candidates"] else None
            else:
                selected = None
        else:
            # mode=None (default): prioritized is already a list
            selected = prioritized[0] if prioritized else None
    elif follow_fresh_ordering:
        # PhysCaP-joint: honour the ordering the Planner produced this turn.
        selected = select_from_fresh_ordering(new_candidates, pending_candidates)
        # Record the ordering that actually drove the choice. Pointing this at
        # the pending pool would file the insertion order as the run's ranking --
        # misrepresenting the trial in exactly the way this mode exists to fix.
        prioritized_json_path = turn_dir / "prioritized_candidates.json"
        prioritized_json_path.write_text(
            json.dumps(new_candidates, indent=2), encoding="utf-8"
        )
    else:
        # Plain Planner mode: pick the top pending (un-executed) candidate from
        # the insertion-ordered pool.
        selected = pending_candidates[0] if pending_candidates else None
        prioritized_json_path = pending_candidates_path

    _save_interaction_history(output_dir, interaction_history)
    _write_reasoning_summary(
        turn_dir, iteration, image_path, candidates_json_path,
        prioritized_json_path, selected, interaction_history,
    )

    return {
        "selected_candidate": selected,
        "candidates_json_path": str(candidates_json_path),
        "prioritized_json_path": str(prioritized_json_path),
        "iteration": iteration,
        "interaction_history": interaction_history,
        "timing": {"planner_s": _planner_elapsed, "prioritizer_s": _prioritizer_elapsed},
    }
