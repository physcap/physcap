# Prompts

Every prompt the agents use is a plain-text file in this folder. Edit a file
here to change what an agent is told; no code change is needed. The paper
section each file corresponds to is in brackets.

## The Planner and the Prioritizer

| File | Used by | What it is |
| --- | --- | --- |
| `planner.txt` | Planner (CaP+PhysX+Planner, PhysCaP) | Decides whether more information is needed and lists the possible next interactions (exploration candidates). [Appendix D.2] |
| `prioritizer.txt` | Prioritizer (PhysCaP) | Scores and ranks the Planner's candidates; includes the Ranking Ladder. [Appendix E.2] |
| `joint.txt` | Merged Planner and Prioritizer (PhysCaP-joint) | Does both jobs in one call. [Appendix D.3] |
| `scene_knowledge.txt` | Scene knowledge update | Turns each turn's results into the running description of the scene that both agents read. Not printed in the paper. |

You can point a run at a different file with `--planner-prompt` or
`--prioritizer-prompt`.

## The coding agent (`coding_agent/`)

| File | When it is used |
| --- | --- |
| `task_template.txt` | The coding agent's first message: the robot, the task prompt and the task's coding rules (below) are put into it. |
| `robot.txt` | Which robot the agents control (default: the AgileX PiPER). A task can name another robot with its own `robot.txt`. |
| `single_step.txt` | The Planner picked one interaction: write code for that step only. [Appendix F.2] |
| `full_task_with_knowledge.txt` | The Planner says there is enough information: finish the whole task, with the scene knowledge included. [Appendix F.2] |
| `full_task.txt` | Same, when there is no scene knowledge yet. |
| `state_change.txt` | Added when the Visual Differencing Module described what changed since the last step. |
| `previous_step.txt` | Added after the first step: the code that just ran and its output. |
| `object_labels_note.txt` | Added when objects have fixed names, so the code uses the same names. |
| `multi_turn.txt` | CaP and CaP+PhysX only: CaP-X's message asking the coding agent to write more code or finish. |

Words in `{curly brackets}` in these files are filled in by the code; keep them when you edit a file.

## Tasks (`tasks/<task>/`)

Each task has its own folder, named in the task's config with `prompt_dir:`.

| File | What it is |
| --- | --- |
| `task.txt` | The task prompt: environment, objective, steps and constraints. The Planner and Prioritizer read it, and it is the core of the coding agent's first message. [Appendix A.2] |
| `coding_rules.txt` | Extra rules for the coding agent (how to move around the objects, when to print). [Appendix F.2] |
| `short.txt` | A one-line summary of the task, used by the Visual Differencing Module. |
| `robot.txt` | (optional) The robot this task runs on, if not the AgileX PiPER. It replaces "an AgileX PiPER 6-DOF robot arm" in the coding agent's prompt and in `planner.txt`, `prioritizer.txt` and `joint.txt` when they are loaded; the files themselves keep the paper's wording. |

## Robots (`robots/`)

A task can reuse another task's prompts on a different robot: its config keeps
that task's `prompt_dir:` and names a file here with `robot_prompt:`. The file
replaces the robot in the same places as a task's `robot.txt`.

| File | Used by |
| --- | --- |
| `libero_franka.txt` | The LIBERO simulation of Identify Empty Can, which uses `tasks/identify_empty_can/` unchanged. |
