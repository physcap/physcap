# Customizing PhysCaP

Four common changes:

- [Add a task](#add-a-task)
- [Add a measurement](#add-a-measurement)
- [Edit the prompts](#edit-the-prompts)
- [Switch models](#switch-models) (other OpenRouter models, or your own server)

PhysCaP is built on CaP-X; this repository uses a pinned copy of it ([physcap/cap-x](https://github.com/physcap/cap-x), in `third_party/cap-x`). Where a feature belongs to CaP-X, this page links to CaP-X's own guide instead of repeating it.

## Add a task

A task has two folders: its settings and data in `configs/<task>/`, and its prompts in `prompts/tasks/<task>/`. Copy a shipped task as a starting point: `identify_empty_can` if the agent must find one object, `pack_grocery_bag` if it must choose one object.

```
configs/my_task/
  dummy.yaml          settings for a dummy-mode run
  gt-mass.json        ground-truth mass of each object       (if the task uses mass)
  gt-stiffness.json   ground-truth stiffness of each object  (if the task uses stiffness)
  aliases.json        other names the agent may use for each object
  scenes.json         which photo to show when
  scenes/             the photos
  README.md

prompts/tasks/my_task/
  task.txt            the task prompt (read by the Planner, the Prioritizer and the coding agent)
  coding_rules.txt    extra rules for the coding agent
  short.txt           a one-line summary of the task
```

### The settings file (`dummy.yaml`)

It uses CaP-X's config format, where `_target_` names the Python class to build and the other keys are its arguments. [CaP-X's configuration guide](https://github.com/physcap/cap-x/blob/main/docs/configuration.md#yaml-config-format) explains the format. The fields you will change:

| Field | What it is |
| --- | --- |
| `env.cfg.low_level` | What stands in for the robot hardware: here the dummy environment `PiperDummyLowLevel` ("Piper" is the AgileX PiPER arm used in the paper), and its settings: `scenes_path`, `scenario`, `aliases_path`, `gt_mass_path`, `gt_stiffness_path`, `auto`, and the grading fields below. |
| `env.cfg.apis` | The tools the coding agent gets: `DummyPhysXApi` (with `get_mass` and `get_stiffness`). `--method cap` replaces it with `DummyControlApi`, which has no measurement tools. |
| `prompt_dir` | The task's prompt folder, e.g. `prompts/tasks/my_task`. |
| `robot_prompt` | (optional) A file naming the robot, if not the AgileX PiPER, e.g. `prompts/robots/libero_franka.txt` ([prompts/README.md](../prompts/README.md#robots-robots)). |
| `task_name`, `model`, `max_turns`, `trials`, `output_dir` | Name used in results, the model, turn limit, default number of trials, results folder. |
| `use_visual_feedback`, `use_img_differencing` | The visual setting ([Running experiments](reproducing_results.md#settings-used)). |

### Ground truth and object names

`gt-mass.json` and `gt-stiffness.json` give each object a short name and a value, for example `{"red can": 0.015, "blue can": 0.388}`. Mass is in kg; stiffness is a level from 1 (soft) to 5 (rigid). Entries whose name starts with `_` are ignored, so you can add notes such as `"_units": "kg"`.

`aliases.json` lists, for each short name, other ways the agent may refer to the object: colour, brand, position ("red can", "coca-cola can", "leftmost can"). Names the table does not cover are matched by a short model call ([Dummy mode](dummy_mode.md#how-object-names-are-matched)). If a name keeps needing that call, add it to the table.

### Grading

For a task with one correct object, set on `low_level`:

- `target_key`: the short name of the correct object.
- `decision_label`: the text before the colon of the answer line, for example `EMPTY CAN` or `FIRST ITEM`.

`task.txt` must ask the agent to print exactly `<decision_label>: <object>` on its own line. If the two differ, no answer is ever recorded and every trial fails. The agent never sees `target_key`.

For a task without a declared answer, leave `target_key` out. A trial then succeeds when a measurement finds the target: set `empty_mass_threshold` (the target's mass is below it; default 0.1 kg) or `ripe_stiffness_threshold` (the target's stiffness level is at or below it).

### Photos

`scenes.json` maps a scenario name to its photos:

```json
{
  "tray_right": {
    "initial": "configs/my_task/scenes/start.png",
    "final":   "configs/my_task/scenes/done.png"
  }
}
```

`final` is either one photo (the task done) or one photo per possible answer, keyed by short name (as in Pack Grocery Bag). Paths are from the repository folder. `scenario:` in `dummy.yaml` picks the scenario. When each photo is shown is described in [Dummy mode](dummy_mode.md#scene-photos).

### Check it

Run one trial with `--total-trials 1` and look at `measurement_report.json` in the trial folder: the measurements should name your objects, and `decision` should hold the agent's answer.

## Add a measurement

A measurement is a tool the coding agent can call, like `get_mass`. In dummy mode a new one takes four steps.

1. **Describe the tool.** In `physcap/apis/piper_tools.py`, add a method to `PiperPhysXTools` with a docstring, and add it to `functions()`. The docstring is all the coding agent sees of the tool, so state what it does, the unit, the range, and what a failed measurement returns. ([CaP-X's guide to adding tools](https://github.com/physcap/cap-x/blob/main/docs/adding-apis.md) explains how docstrings become the tool list.)
2. **Implement it for dummy mode.** In `physcap/apis/dummy_api.py`, add the same method to `DummyPhysXApi` without a docstring (it inherits the one from step 1), returning `self._measure_property(object_name, "<property>", "_gt_<property>", "<unit>", float)`. This looks up the ground-truth value, writes it into the scene knowledge, and records it in `measurement_report.json`.
3. **Load its ground truth.** In `physcap/envs/piper_dummy.py`, add a `gt_<property>_path` argument and load it into `self._gt_<property>`, the same way `gt_mass_path` is loaded. Then set `gt_<property>_path` in your task's `dummy.yaml`.
4. **Let the agents ask for it.** The Planner chooses from a fixed list of actions:
   - add `measure_<property>` to `ACTION_VOCAB` (the action names the Planner may use) in `physcap/agents/planner.py`, and to `_MEASURE_ACTIONS` (the actions that count as measurements) in `physcap/agents/cycle.py`;
   - add it wherever the prompts list the measurements: `prompts/planner.txt`, `prompts/joint.txt` and `prompts/prioritizer.txt` (search for `measure_stiffness`).

New values appear in the scene knowledge as "The <object> has <property>: <value> <unit>".

On the real robot, the measurement itself is a separate module that moves the arm and reads its sensors; that code is not in this repository yet.

## Edit the prompts

Every prompt is a text file in `prompts/`; [prompts/README.md](../prompts/README.md) says what each one does and which paper section it matches. Edit a file and the next run uses it; no code changes are needed.

- **Task wording:** `prompts/tasks/<task>/task.txt` and `coding_rules.txt`.
- **Planner, Prioritizer, PhysCaP-joint:** `prompts/planner.txt`, `prompts/prioritizer.txt`, `prompts/joint.txt`. To try a variant without changing the original, save it under a new name and pass `--planner-prompt my_planner.txt` or `--prioritizer-prompt my_prioritizer.txt` (or `--task-prompt` for the task text the Planner and Prioritizer read).
- **Coding agent:** `prompts/coding_agent/`.

Two things to keep when editing:

- In `prompts/coding_agent/`, words in `{curly brackets}` are filled in by the code. Removing one drops that information; adding a new one causes an error. (The other prompt files are used as they are, so curly brackets there are just text.)
- The Planner and the Prioritizer must answer in the JSON format their prompt describes. The code reads fields such as `sufficient`, `task_complete` and `exploration_candidates` (Planner) and `priority_score` (Prioritizer), so keep those names.

Use `--run-label NAME` to keep results from an edited prompt apart from the originals.

## Switch models

There are five kinds of model calls. Each can use a different model:

| Agent | Flag | Sent to |
| --- | --- | --- |
| Coding agent | `--model` (or `model:` in the config) | `--server-url` |
| Visual Differencing Module | `--visual-differencing-model` | `--visual-differencing-model-server-url` |
| Planner (and scene knowledge) | `--planner-model` (default: `--model`) | `--agent-server-url` |
| Prioritizer | `--prioritizer-model` (default: `--model`) | `--agent-server-url` |
| Name matching in dummy mode (the resolver) | `resolver_model:` in the task config (default `google/gemini-2.5-flash-lite`) | `--server-url` |

The defaults send everything to OpenRouter: `--server-url` points at the small local proxy that `physcap.run` starts for you (CaP-X's OpenRouter proxy, port 8110), and `--agent-server-url` defaults to OpenRouter itself.

### Other OpenRouter models

Pass OpenRouter's model name, `<provider>/<model>`. The paper's Section 5.3 (Figure 8) runs Identify Empty Can with three models:

```bash
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap \
  --model anthropic/claude-fable-5 --run-label physcap_claude
```

Other names from the paper: `google/gemini-3.1-pro-preview` (the default) and `openai/gpt-6-astra`. Any model OpenRouter offers can be used, as long as it accepts images: the Planner, the Prioritizer, the scene-knowledge update and the Visual Differencing Module are all sent images.

The coding agent and the Visual Differencing Module get images only for models listed in `VLM_MODELS` in `third_party/cap-x/capx/llm/client.py`; with image differencing on, a Visual Differencing model missing from that list stops the run. To use a new vision model there, add its name to the list.

### Your own server (local models)

Start any server that offers an OpenAI-compatible `/chat/completions` endpoint, for example [vLLM](https://docs.vllm.ai/) (CaP-X's [notes on vLLM](https://github.com/physcap/cap-x/blob/main/docs/configuration.md#option-b-vllm-local-models)). Then point the agents at it:

```bash
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --server-url http://127.0.0.1:8000/v1/chat/completions \
  --agent-server-url http://127.0.0.1:8000/v1 \
  --visual-differencing-model Qwen/Qwen2.5-VL-72B-Instruct \
  --visual-differencing-model-server-url http://127.0.0.1:8000/v1/chat/completions
```

- `--agent-server-url` takes the server's base address (ending in `/v1`); the other two take the full `/chat/completions` address.
- `physcap.run` starts the OpenRouter proxy only when `--server-url` is on this machine and nothing answers there; pass `--no-start-proxy` to turn that off.
- With a local server, also set `resolver_model:` under `low_level` in the task config to a model your server offers; otherwise names the alias table does not cover cannot be matched.
- The model should accept images and follow the JSON answer format of the Planner and Prioritizer prompts. The paper only reports hosted models (Gemini 3.1 Pro, Claude Fable 5, GPT-6 Astra); results with local models may differ a lot.
