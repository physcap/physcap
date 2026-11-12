# Pack Grocery Bag

Paper task 3 (Section 4.2, Appendix A.1; task 2, Pick Ripe Avocado, is not in this repository). Four cubes that look alike apart from their colour stand next to a paper grocery bag. The agent must decide which cube should go into the bag first, put it in, and stop. The prompt does not say what makes a good first item, and does not mention mass or stiffness.

**Correct answer:** the green cube, the only one that is both heavy and rigid, so it makes a stable base.

| Cube (name in the tables) | Mass (kg) | Stiffness (1 soft – 5 rigid) |
| --- | --- | --- |
| red | 0.052 | 3 |
| blue | 0.053 | 5 |
| green | 0.418 | 5 |
| yellow | 0.419 | 3 |

Mass alone points to yellow, and stiffness alone cannot separate green from blue; only both together single out green. The agent must print `FIRST ITEM: <cube>` on a line of its own; the trial is graded on that line.

## Files

| File | What it is |
| --- | --- |
| `dummy.yaml` | Settings for a dummy-mode run. |
| `gt-mass.json`, `gt-stiffness.json` | The mass and stiffness of each cube (table above). |
| `aliases.json` | Other names the agent may use for each cube. |
| `scenes.json`, `scenes/` | The scene photos: one starting photo, and one photo per cube showing that cube in the bag. The photo shown at the end is the cube the agent chose. |

The prompts are in `prompts/tasks/pack_grocery_bag/`.

## Run it

```bash
uv run python -m physcap.run --config-path configs/pack_grocery_bag/dummy.yaml --method physcap
```

`--method` picks one of the five methods (`cap`, `cap_physx`, `cap_physx_planner`, `physcap_joint`, `physcap`). See [Running experiments](../../docs/reproducing_results.md).
