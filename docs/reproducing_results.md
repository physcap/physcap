# Running experiments

This page lists the commands for every task and method, where the results go, and how to summarise them.

## What this repository reproduces

The numbers in the paper's Table 1 (SR, OI and Time) come from a real robot: an AgileX PiPER arm with a ZED 2i camera. The real-robot code is not part of this repository yet.

This repository runs the same tasks in [dummy mode](dummy_mode.md), with the same prompts and agents. Measurements return ground-truth values instead of real readings, and nothing moves. So dummy mode reproduces how the agents decide (which objects they measure, in what order, and when they stop), and the success rate and number of interactions that follow from it. It does not reproduce measurement noise, failed grasps, or the time the robot spends moving, so dummy-mode times are shorter than the paper's.

The [LIBERO simulation](libero.md) of Identify Empty Can runs the same methods with a simulated arm that grasps and lifts each can to measure it. Its commands are on that page.

## Commands

Run from the repository folder, with your key in `.openrouterkey` (see [Install](install.md)). Replace `<method>` with `cap`, `cap_physx`, `cap_physx_planner`, `physcap_joint` or `physcap`.

```bash
# Identify Empty Can
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method <method> --total-trials 10

# Pack Grocery Bag
uv run python -m physcap.run --config-path configs/pack_grocery_bag/dummy.yaml --method <method> --total-trials 10
```

The paper runs 10 trials per task and method (Section 4.1). The configs run 1 trial unless you pass `--total-trials`. To compare all five methods, run the command once per method; each method writes to its own folder.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--total-trials N` | Number of trials. |
| `--max-turns N` | Most turns per trial. The configs set 8 (Identify Empty Can) and 10 (Pack Grocery Bag). |
| `--model NAME` | Model for every agent. See [Customizing](customizing.md#switch-models). |
| `--planner-model NAME`, `--prioritizer-model NAME` | Separate models for the Planner and the Prioritizer. |
| `--run-label NAME` | Name of the results folder. Default: the method name. Use it to keep two runs of the same method apart. |
| `--output-dir PATH` | Where results go. Default: `outputs/<task>`. |

## Settings used

Every task uses CaP-X's high-level tools (setting S2 in CaP-X) and the visual setting the paper used for that task:

| Task | `use_visual_feedback` | `use_img_differencing` |
| --- | --- | --- |
| Identify Empty Can | false | true |
| Pack Grocery Bag | true | true |

- `use_visual_feedback: true`: the CaP and CaP+PhysX coding agents also see the image after each turn. The other methods ignore it.
- `use_img_differencing: true`: the Visual Differencing Module (CaP-X setting M3) describes in text what changed since the last image, and the coding agent of every method gets that text (for CaP and CaP+PhysX, through CaP-X's own loop).
- The Planner and the Prioritizer always see the scene image.

All agents use Gemini 3.1 Pro (`google/gemini-3.1-pro-preview`), as in the paper (Section 4.1).

## Where results go

```
outputs/<task>/<run label>/<model>/<timestamp>/trial_NN/
```

For example: `outputs/identify_empty_can/physcap/google_gemini-3.1-pro-preview/2026-10-27_13-37-26/trial_01/` (the `/` in the model name becomes `_`).

Each trial folder has `measurement_report.json`, `time_log.txt`, a `reasoning/` folder with the Planner's and Prioritizer's outputs, and CaP-X's usual files (generated code, logs, images). [Method](method.md#what-a-trial-writes) describes them.

## Summarising

`physcap.run` prints a summary when it ends. To summarise any folder of results, including several methods at once:

```bash
uv run python -m physcap.summarize outputs/identify_empty_can
```

It reads every `measurement_report.json` under the folder and prints one row per task and method:

| Column | Meaning |
| --- | --- |
| `n` | Number of trials. |
| `SR` | Success rate: the share of trials that succeeded. |
| `OI` | Objects interacted: the number of measurements in the trial. Measuring the same object again counts again. |
| `Time (s)` | Trial length in seconds. |

As in the paper, OI and Time are averaged over successful trials only, shown as mean ± standard deviation.

A trial succeeds when the first answer the agent declares is correct: the red can for Identify Empty Can (`EMPTY CAN: red can`), the green cube for Pack Grocery Bag (`FIRST ITEM: green`) (see [Dummy mode](dummy_mode.md#declared-answers-and-grading)). To overrule the grade for a trial, add `"success": true` or `"success": false` to its `measurement_report.json`; the summary then uses your value.

## Cost

Each turn makes several model calls: the Planner, the Prioritizer, the Visual Differencing Module, the coding agent and the scene knowledge update. A dummy-mode trial takes a few minutes. Use `--total-trials 1` and a lower `--max-turns` for a cheap first test.
