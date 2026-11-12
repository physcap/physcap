# Dummy mode: run PhysCaP without a robot

Dummy mode runs the whole PhysCaP pipeline (the Planner, the Prioritizer and the coding agent) with no robot arm and no camera. Only the hardware is replaced:

- A **photo** stands in for the camera image.
- The motion tools (`get_object_pose`, `goto_pose`, `open_gripper`, `close_gripper`, `home_pose`) do nothing. They print a line and return.
- `get_mass` and `get_stiffness` return the **ground-truth value** for the object from a table, or ask a person to type one.

Use it to try the agents, test a changed prompt, or compare models. It shows how the agents reason: which objects they decide to measure, in what order, and when they stop. It does not reproduce measurement noise or the time a real robot spends moving.

## Run it

```bash
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap
```

| Task | Config |
| --- | --- |
| Identify Empty Can | `configs/identify_empty_can/dummy.yaml` |
| Pack Grocery Bag | `configs/pack_grocery_bag/dummy.yaml` |

Useful flags: `--method` (`cap`, `cap_physx`, `cap_physx_planner`, `physcap_joint`, `physcap`), `--total-trials N`, `--max-turns N`, `--model NAME`. Run `uv run python -m physcap.run --help` for the full list. Every trial makes paid model calls.

## Automatic and interactive mode

The `auto` setting in the config chooses the mode. Both shipped configs use automatic mode.

**Automatic (`auto: true`).** Nothing asks for input, so you can run many trials in a row (`--total-trials N`). When the coding agent calls `get_mass("the red can on the left")`, the dummy environment:

1. Matches the name the agent used to an entry in `gt-mass.json` or `gt-stiffness.json` ("gt" stands for ground truth: the true values; see below).
2. Returns that entry's value.
3. Switches the photo when the rules below say so.

**Interactive (`auto: false`).** A person plays the robot. Each turn, a menu lists the scene photos and you can type one to change the image (or press Enter to keep it). For each measurement: if a ground-truth table is loaded, you type which table entry the agent meant; if there is none, you type a value yourself. To use it, set `auto: false` in the task's `dummy.yaml`.

## How object names are matched

The coding agent names objects in its own words. The ground-truth tables use short names such as `red can`. They are matched in two steps:

1. **Alias table.** `aliases.json` in the task folder lists known ways of naming each object. Names are lower-cased first, with punctuation and filler words such as "the" and "of" removed. An exact match wins; a partial match counts only if it fits one object.
2. **Model fallback (the "resolver").** If the alias table gives no answer, one short model call (`google/gemini-2.5-flash-lite`) picks the object or says it cannot tell. The answer is remembered for the rest of the run.

If neither step finds the object, the trial stops with an error rather than guess, because a wrong value could look like a right answer.

The fallback call uses the same server as the coding agent (`--server-url`). To send it elsewhere, set `PHYSCAP_RESOLVER_SERVER_URL`; its key comes from `PHYSCAP_RESOLVER_API_KEY`, `OPENROUTER_API_KEY` or `.openrouterkey`. To keep the names the fallback learned, set `PHYSCAP_ALIAS_CACHE` to a file path; the updated alias table is saved there and can be used as the task's `aliases_path`.

## Scene photos

`scenes.json` in the task folder lists the photos. A scenario is one arrangement of the scene (for example, where the tray is). It has an `initial` photo, shown from the start, and `final`: the scene after the task, either one photo or one photo per possible answer. The config chooses the scenario with `scenario:`.

- **Identify Empty Can** has one scenario, `tray_right`. `final` has one photo per can: that can on the tray.
- **Pack Grocery Bag** has one scenario, `bag_right`. `final` has one photo per cube: that cube in the bag.

When the photo changes:

| Task | When the photo switches to `final` |
| --- | --- |
| Identify Empty Can | When the agent declares its answer, right or wrong: the photo shows the can it chose on the tray. |
| Pack Grocery Bag | When the agent declares its first item, right or wrong: the photo shows the cube it chose in the bag. |

For your own task without a declared answer, the photo can instead switch when a measurement crosses a threshold: mass below `empty_mass_threshold` (default 0.1 kg) or stiffness at or below `ripe_stiffness_threshold` (named after the paper's Pick Ripe Avocado task, which is not in this repository). See [Customizing](customizing.md#add-a-task).

## Declared answers and grading

Each task asks the agent to print its answer on a line of its own:

```
EMPTY CAN: red can
FIRST ITEM: green cube
```

The text before the colon must match the config's `decision_label` exactly, including capital letters. The agent may print more answers, but only the first one is graded. The object can be written in any form the alias table knows (`green` and `green cube` both match the entry `green`). The object name is matched like any other name (above) and compared with the config's `target_key`, the correct object. The agent never sees `target_key`.

| Task | `decision_label` | Correct answer (`target_key`) |
| --- | --- | --- |
| Identify Empty Can | `EMPTY CAN` | `red can` (mass 0.015 kg; the other cans are 0.208–0.392 kg) |
| Pack Grocery Bag | `FIRST ITEM` | `green` (the only cube that is both heavy and rigid) |

## Output

Each trial writes `measurement_report.json`:

- `totals`: number of measurements, by property, and how many were repeats of the same object.
- `order` and `events`: every measurement in order, with its turn, property, matched object, value and time.
- `decision`: the declared answer, the matched object, whether it is `correct`, and how many measurements came before it. Empty if the agent never answered.
- `times`: trial length and time spent waiting for models.

To summarise a folder of runs (success rate, objects interacted, time):

```bash
uv run python -m physcap.summarize outputs/identify_empty_can
```

[Running experiments](reproducing_results.md) explains the summary.

## Tests

These run offline, with no model calls:

```bash
uv run --extra dev python -m pytest tests/dummy -q
```

Each file in `tests/dummy/` covers one part: photos, ground-truth lookup, automatic mode, the measurement report, grading and the summary.
