# Identify Empty Can

Paper task 1 (Section 4.2, Appendix A.1). Four soft drink cans stand on a table; one is empty. The agent must find the empty can and move it to a wooden tray. Some cans are sealed and some are open, which is a visual hint about which ones could be empty.

**Correct answer:** the red can. A can counts as empty when its mass is below 0.1 kg.

| Can (name in `gt-mass.json`) | Mass (kg) |
| --- | --- |
| `red can` | 0.015 (empty) |
| `green can` | 0.208 |
| `blue can` | 0.388 |
| `black can` | 0.392 |

The agent must print `EMPTY CAN: <can>` on a line of its own; the trial is graded on that line.

## Files

| File | What it is |
| --- | --- |
| `dummy.yaml` | Settings for a dummy-mode run. |
| `gt-mass.json` | The mass of each can (table above). |
| `aliases.json` | Other names the agent may use for each can. |
| `scenes.json`, `scenes/` | The scene photos: a starting photo, and for each can a photo of that can on the tray (scenario `tray_right`). The photo shown at the end is the can the agent chose. |

The prompts are in `prompts/tasks/identify_empty_can/`.

## Run it

```bash
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap
```

`--method` picks one of the five methods (`cap`, `cap_physx`, `cap_physx_planner`, `physcap_joint`, `physcap`). See [Running experiments](../../docs/reproducing_results.md).
