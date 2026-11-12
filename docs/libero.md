# LIBERO simulation

A simulated version of Identify Empty Can in [LIBERO](https://libero-project.github.io/), so the full method, including physical measurement, can be run without a robot. It uses the real task's objects and the real task's prompts.

## The task

A simulated Franka arm, four drink cans and a flat wooden tray on a table. One can is empty; the agent must find it and put it on the tray.

| Can | Mass (kg) | Opened |
| --- | --- | --- |
| red | 0.015 (empty) | yes: an opening in the lid and a straw |
| green | 0.208 (still holds some liquid) | yes: an opening in the lid and a straw |
| blue | 0.388 | no |
| black | 0.392 | no |

- These are the real cans' masses (`configs/identify_empty_can/gt-mass.json`), and the same cans are opened as in the real scene.
- A can counts as empty when its mass is below 0.1 kg.
- Each trial places the cans in seven areas of the table, with the tray to their right. The trial number fixes the layout, so trial 1 has the same layout in every run and for every method.
- A trial succeeds when the red can is on the tray (LIBERO's own check).

## How it relates to the real task

| | Real robot | LIBERO |
| --- | --- | --- |
| Robot | AgileX PiPER | simulated Franka Emika Panda |
| Objects | black, green, blue and red cans; a wooden tray | the same, as simulated models |
| Prompts | `prompts/tasks/identify_empty_can/` and the agent prompts in `prompts/` | the same files, unchanged; only the robot they name differs (`prompts/robots/libero_franka.txt`) |
| Tools | `get_object_pose`, `goto_pose`, `open_gripper`, `close_gripper`, `home_pose`, `get_mass`, `get_stiffness` | the same, without `get_stiffness` (no stiffness in this task) |
| Finding objects | Molmo2 points at the object in the camera image | the simulator gives the object's exact position |
| `get_mass` | grasps and lifts the object 15 cm and computes the mass from the arm's joint torques | grasps and lifts the can 15 cm; the simulator reveals its mass once the can has been held for 0.5 s while lifted at least 3 cm for 0.5 s |

The tool descriptions are in `physcap/apis/libero_api.py`. They use the real tools' wording where the hardware does not matter.

## Install

LIBERO is an optional part of the install:

```bash
uv sync --extra libero
```

It uses the LIBERO copy that CaP-X includes (`third_party/cap-x/capx/third_party/LIBERO-PRO`). PhysCaP points LIBERO at a settings file inside the repository (`.libero/`, created automatically), so your own `~/.libero` is neither needed nor changed. The simulator renders without a display (`MUJOCO_GL=egl`); this needs a GPU with EGL support.

To check the install without any model calls, run the scripted test. It measures every can and puts the red one on the tray (a few minutes):

```bash
uv run --extra libero python tests/libero/test_01_scripted_task.py
```

## Run

```bash
uv run --extra libero python -m physcap.run --config-path configs/libero_identify_empty_can/sim.yaml --method physcap
```

All five methods work (`cap`, `cap_physx`, `cap_physx_planner`, `physcap_joint`, `physcap`); CaP gets no `get_mass`. Add `--total-trials 10` for trials 1–10. Results, `measurement_report.json` and the summary work as for the other tasks ([Running experiments](reproducing_results.md)). The report also records `sim_steps`, the number of simulation steps the trial took.

## Files

| Path | What it is |
| --- | --- |
| `configs/libero_identify_empty_can/sim.yaml` | Settings for a run. |
| `configs/libero_identify_empty_can/aliases.json` | Other names the agent may use for each object. |
| `prompts/robots/libero_franka.txt` | The robot line used instead of the AgileX PiPER. |
| `physcap/envs/libero/can_task.py` | The scene and the simulated environment. |
| `physcap/envs/libero/cans.py` | Registers the cans and the tray with LIBERO. |
| `physcap/envs/libero/mass_reveal.py` | Reveals a can's mass after a verified grasp-and-lift. |
| `physcap/envs/libero/assets/` | The can and tray models, the scene file (`bddl/put_empty_can_on_tray.bddl`), and `make_models.py`, which writes the model files. |
| `physcap/apis/libero_api.py` | The tools the coding agent can call. |
| `tests/libero/test_01_scripted_task.py` | The scripted check above. |

## Credits

The can model is robosuite's can mesh, stretched to the size of a slim drink can, and the tray uses LIBERO's wood texture; both are under the MIT License (robosuite: Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab; LIBERO: Lifelong Robot Learning).
