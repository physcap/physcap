# Method

This page describes what happens in one PhysCaP trial and where each part is in the code. Section numbers refer to the [paper](https://arxiv.org/abs/2608.21031).

## The parts

| Part | What it does | Paper | Code | Prompt |
| --- | --- | --- | --- | --- |
| PhysX modules | Measure an object's mass and stiffness with the robot | Section 3.1, Appendix B | not in this repository (see below) | — |
| Scene knowledge | A text file with everything known about the scene so far | Section 3.2 | `physcap/agents/scene_knowledge.py` | `prompts/scene_knowledge.txt` |
| Planner | Decides whether more information is needed, and lists the possible next interactions (exploration candidates) | Section 3.2, Appendix D | `physcap/agents/planner.py` | `prompts/planner.txt` |
| Prioritizer | Scores and ranks the Planner's candidates | Section 3.3, Appendix E | `physcap/agents/prioritizer.py` | `prompts/prioritizer.txt` |
| Merged Planner and Prioritizer | Does both jobs in one call (PhysCaP-joint) | Appendix D.3 | `physcap/agents/planner.py` | `prompts/joint.txt` |
| Coding agent | Writes Python code for the chosen step | Section 3.1.3, Appendix F | `physcap/agents/codegen_prompt.py` | `prompts/coding_agent/`, `prompts/tasks/<task>/` |

`physcap/trial.py` runs one trial. `physcap/agents/cycle.py` runs the Planner and the Prioritizer for one turn and keeps track of the candidates between turns.

## The tools the coding agent can call

The coding agent controls the robot by writing code that calls these tools. (In the code and in CaP-X, a set of tools is called an API: the folder `physcap/apis/` and the config key `apis`.) Their descriptions, which is all the agent sees of them, are in `physcap/apis/piper_tools.py` and match the paper (Appendix B.3 and C.2).

| Tool | What it does |
| --- | --- |
| `get_object_pose(name)` | Finds an object from a text description and returns its position. |
| `goto_pose(position, z_approach=...)` | Moves the gripper to a position, optionally from above. |
| `open_gripper()`, `close_gripper()` | Opens or closes the gripper. |
| `home_pose()` | Moves the arm to its rest position. |
| `get_mass(name)` | PhysX: grasps and lifts the object and returns its mass in kg (Section 3.1.1). |
| `get_stiffness(name)` | PhysX: squeezes the object with the gripper and returns a stiffness level from 1 (soft) to 5 (rigid) (Section 3.1.2). |

The CaP method gets the first five tools only; every other method also gets `get_mass` and `get_stiffness`.

On the real robot, `get_mass` and `get_stiffness` measure the object with the arm's own joint and gripper readings. That code is not part of this repository yet. In dummy mode, the motion tools do nothing and the two measurements return values from a ground-truth table; see [Dummy mode](dummy_mode.md). In the [LIBERO simulation](libero.md), a simulated arm grasps and lifts the object and `get_mass` reads its mass from the simulator. Either way, each measured value is written into the scene knowledge.

## One trial

`physcap.run` reads the task config and the method (`METHODS` in `physcap/run.py`), then runs `physcap/trial.py` once per trial.

### Setup

1. Reset the environment and take the first camera image (in dummy mode, the first scene photo).
2. Write the first scene knowledge: a model (the Planner's model) looks at the image and the task and describes the scene in `scene_knowledge.txt`. It also gives each object a fixed name that all agents use from then on.

### Each turn (up to `max_turns`)

1. **Look.** Take a new image.
2. **Planner** (Section 3.2, Appendix D.1).
   - It reads: the task prompt, the image, the scene knowledge, and what happened in the previous turn.
   - It answers in JSON. The main fields are `sufficient` (is there enough information to finish the task?), `task_complete` (is the goal already reached?), and `exploration_candidates`. Each candidate names one object, one action and the information it would give. The action is a PhysX measurement (`measure_mass`, `measure_stiffness`) or one of `relocate`, `inspect` and `other`.
   - If `sufficient` is true, the coding agent is asked to finish the whole task this turn instead of doing one more exploration step. If `task_complete` is true, the trial ends.
   - New candidates are added to a list of pending candidates. A candidate leaves the list once it has been carried out.
3. **Choose the next step** (one candidate, carried out this turn). This is where the methods differ:
   - **PhysCaP:** the **Prioritizer** (Section 3.3, Appendix E.1) reads the pending candidates, the image and the scene knowledge. It drops implausible or repeated candidates and gives each remaining one a priority score from 0 to 10, using visual cues first, never re-measuring an object already measured, and preferring cheaper actions. The highest score is chosen.
   - **PhysCaP-joint:** the Planner, using `joint.txt`, already returns its candidates in priority order. The highest-ranked one that is still pending is chosen.
   - **CaP+PhysX+Planner:** there is no ranking. The oldest pending candidate is chosen.
4. **Visual Differencing Module** (CaP-X setting M3, when `use_img_differencing` is on). From the second turn on, a model describes in text what changed between the last two images. The coding agent gets this text, not the images.
5. **Coding agent** (Section 3.1.3, Appendix F). It gets: the task, the chosen step, the change description, the code and printed output of the previous step, the object names, and, when finishing the task, the scene knowledge. It returns Python code that calls the tools above, and the code is run.
6. **Update the scene knowledge.** What the code printed is added first (this is how measured values enter). Then a model describes how the scene changed.
7. **Stop** if the step was the whole task, if the code printed `FINISH`, or if the environment ended. Otherwise start the next turn.

### Methods without the Planner

CaP and CaP+PhysX do not use the loop above. They run CaP-X's own multi-turn loop with the coding agent alone: it writes code, sees the result, and decides whether to write more code or finish (`prompts/coding_agent/multi_turn.txt`). CaP has no `get_mass` or `get_stiffness`.

## What a trial writes

Each trial folder `trial_NN/` contains:

- `reasoning/`: the scene knowledge, and the Planner's and Prioritizer's input and output for every turn.
- `measurement_report.json`: every measurement in order, and the agent's final answer with its grade. The number of measurements is OI.
- `time_log.txt`: time spent in each step.
- CaP-X's usual files: the generated code, logs and images.

## Models

All agents use the model given by `model:` in the config or `--model`. The paper uses Gemini 3.1 Pro for every agent (Section 4.1). To use other models, see [Customizing](customizing.md#switch-models).
