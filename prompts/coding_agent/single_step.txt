Selected next high-level step from the reasoning pipeline:
- name (the object to act on): {candidate_name}
- action (what to do to it): {candidate_action}
- description: {candidate_desc}
- expected_info: {candidate_info}
- estimated_cost: {candidate_cost}
- parameters: {candidate_params}

Original task:
{original_task}

Your job is to generate Python code for ONLY the selected next high-level step.
Perform exactly this one action on exactly this one object. Do not measure any
additional property, and do not act on any other object, even when doing so in
the same code block would be more efficient — the reasoning pipeline schedules
those as their own steps, and doing them here corrupts its accounting.
Do not plan the whole task from scratch unless the selected candidate is explicitly the original task.
Use the available APIs from the prompt above.
The code should be executable Python only.
