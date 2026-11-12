"""Builds the code-generation prompt for a selected reasoning candidate.

Once the Planner (and, in Planner+Prioritizer mode, the Prioritizer) picks
the next candidate, this module turns that decision into the prompt handed to
the code-generation model, folding in scene knowledge, execution history, and
the canonical object vocabulary.

The text itself is in ``prompts/coding_agent/``: ``single_step.txt`` (one
exploration step), ``full_task.txt`` / ``full_task_with_knowledge.txt`` (the
Planner says there is enough information), ``state_change.txt``,
``previous_step.txt`` and ``object_labels_note.txt``.
"""

from __future__ import annotations

import copy
from typing import Any

from capx.llm.client import collapse_text_image_inputs
from physcap.agents.scene_knowledge import format_object_labels_block
from physcap.prompt_files import coding_agent_prompt


def build_candidate_code_prompt(
    obs: dict[str, Any],
    selected_candidate: dict[str, Any],
    *,
    original_task: str,
    executed_code: str = "",
    console_stdout: str = "",
    console_stderr: str = "",
    differencing_feedback: str = "",
    scene_knowledge: str | None = None,
    object_labels: list[dict] | None = None,
) -> list[dict]:
    """Build a code-generation prompt from a selected reasoning candidate.

    Instead of planning the whole task from the image, the coding agent writes
    code for the one step the Planner or Prioritizer selected; the task and
    tool prompt stay as they are.

    When selected_candidate["name"] == "original_task_execute", the Planner
    has determined that sufficient information is already available. In that
    case, scene_knowledge (the accumulated scene knowledge) is injected
    prominently so the LLM can complete the whole task in one shot.
    """
    prompt = copy.deepcopy(obs["full_prompt"])
    for message in prompt:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item for item in content
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
    candidate_name = selected_candidate.get("name", "unnamed_candidate")
    candidate_desc = selected_candidate.get("description", "")
    candidate_info = selected_candidate.get("expected_info", "")
    candidate_cost = selected_candidate.get("estimated_cost", "unknown")
    candidate_action = selected_candidate.get("action") or "(unspecified)"
    candidate_params = selected_candidate.get("parameters", {})

    is_full_task = candidate_name == "original_task_execute"

    if is_full_task and scene_knowledge:
        guidance = coding_agent_prompt("full_task_with_knowledge").format(
            scene_knowledge=scene_knowledge, original_task=original_task
        )
    elif is_full_task:
        guidance = coding_agent_prompt("full_task").format(original_task=original_task)
    else:
        guidance = coding_agent_prompt("single_step").format(
            candidate_name=candidate_name,
            candidate_action=candidate_action,
            candidate_desc=candidate_desc,
            candidate_info=candidate_info,
            candidate_cost=candidate_cost,
            candidate_params=candidate_params,
            original_task=original_task,
        )

    if differencing_feedback:
        guidance += "\n\n" + coding_agent_prompt("state_change").format(
            differencing_feedback=differencing_feedback
        )

    if executed_code:
        guidance += "\n\n" + coding_agent_prompt("previous_step").format(
            executed_code=executed_code,
            console_stdout=console_stdout,
            console_stderr=console_stderr,
        )

    if object_labels:
        guidance += format_object_labels_block(object_labels)
        guidance += "\n" + coding_agent_prompt("object_labels_note")

    prompt[-1]["content"].append({"type": "text", "text": guidance.strip()})
    prompt[-1]["content"] = collapse_text_image_inputs(prompt[-1]["content"])
    return prompt
