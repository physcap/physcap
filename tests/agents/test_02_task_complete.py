"""
test_02_task_complete.py

The Planner's "task already complete" exit. On the real arm, a PhysCaP step
measured the empty can's mass, declared it, and placed it on the tray in one
code block. The next turn the Planner said the goal was achieved, but
`sufficient` was the only outcome it had, and `sufficient` maps to
original_task_execute -- whose codegen redoes the whole task, so the robot
measured and moved the can again.

`task_complete` is a separate outcome: once something has executed, it ends
the turn with a `task_complete` selection and no Prioritizer or codegen call.

No network, no hardware, no model calls: the Planner is monkeypatched, and
scene knowledge / object labels are pre-created so their init calls are skipped.

Tests:
  1. The Planner schema requires `task_complete`
  2. task_complete after a step -> `task_complete` selection, no prioritizer
  3. task_complete before any step -> ignored, falls back to sufficient
  4. sufficient without task_complete -> unchanged original_task_execute

Usage:
    uv run python tests/agents/test_02_task_complete.py
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import physcap.agents.cycle as cycle  # noqa: E402
from physcap.agents.planner import candidate_key, planner_response_schema  # noqa: E402
from physcap.agents.scene_knowledge import MEASURED_PROPERTIES_FILENAME  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


WEIGH_RED = {
    "name": "can_3",
    "action": "measure_mass",
    "description": "Measure the mass of the red can",
    "parameters": {"object_name": "string"},
    "expected_info": "mass of the red can",
    "estimated_cost": "medium",
}


def fake_planner(response: dict):
    """Stand-in for planner(): write the files the real one writes."""
    def _planner(*, image_path, task, system_prompt, model_name, output_base_dir, api_key):
        out_dir = Path(output_base_dir) / "0000_0000"
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = response.get("exploration_candidates", [])
        (out_dir / "exploration_full_response.json").write_text(json.dumps(response))
        path = out_dir / "exploration_candidates.json"
        path.write_text(json.dumps(candidates))
        return str(path)
    return _planner


def prioritizer_must_not_run(*args, **kwargs):
    raise AssertionError("prioritizer was called")


def run_turn(response: dict, executed: bool) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="task_complete_test_"))
    (tmp / "scene_knowledge.txt").write_text("The red can was placed on the wooden tray.\n")
    (tmp / cycle.OBJECT_LABELS_FILENAME).write_text(
        json.dumps([{"id": "can_3", "description": "the red can on the far right"}])
    )
    image = tmp / "scene.png"
    image.write_bytes(b"")
    if executed:
        # A measuring step only counts as done once its probe is recorded
        # (test_03), so record it the way a real trial does.
        (tmp / MEASURED_PROPERTIES_FILENAME).write_text(json.dumps([
            {"object_name": "the red can on the far right", "property_type": "mass", "turn": 1},
        ]))
    cycle.planner = fake_planner(response)
    cycle.prioritizer = prioritizer_must_not_run
    execution_result = (
        {"executed_candidate": WEIGH_RED, "success": True, "stdout": "EMPTY CAN: red can"}
        if executed else None
    )
    out = cycle.run_reasoning_turn(
        image_path=image, task="Find the empty can and put it on the tray.",
        output_dir=tmp, turn_dir=tmp / "turn_02", iteration=1,
        execution_result=execution_result, use_prioritizer=True,
    )
    summary = json.loads((tmp / "turn_02" / "reasoning_summary.json").read_text())
    return {"out": out, "summary": summary}


DONE = {
    "sufficient": True, "task_complete": True,
    "reason": "The empty can is already on the tray.",
    "central_objects": [], "exploration_candidates": [],
}

print("=== Test 02: planner task_complete exit ===\n")

# ── 1. Schema ───────────────────────────────────────────────────────────────
print("[1] The planner schema requires task_complete")
schema = planner_response_schema()
check("task_complete is a boolean property",
      schema["properties"].get("task_complete") == {"type": "boolean"})
check("task_complete is required", "task_complete" in schema["required"])
check("it is ordered right after sufficient",
      schema["propertyOrdering"][:2] == ["sufficient", "task_complete"])

# ── 2. Complete after a step ────────────────────────────────────────────────
print("\n[2] task_complete after a step ends the turn without the prioritizer")
r = run_turn(DONE, executed=True)
sel = r["out"]["selected_candidate"]
check("selection is task_complete", sel.get("name") == "task_complete", str(sel.get("name")))
check("the planner's reason is carried", sel.get("reason") == DONE["reason"])
check("summary records execution_decision=task_complete",
      r["summary"]["execution_decision"] == "task_complete")
check("the executed step was recorded first",
      candidate_key(WEIGH_RED) in r["out"]["interaction_history"]["executed_plans"])

# ── 3. Complete before any step ─────────────────────────────────────────────
print("\n[3] task_complete before any step is ignored")
r = run_turn(DONE, executed=False)
sel = r["out"]["selected_candidate"]
check("falls back to original_task_execute", sel.get("name") == "original_task_execute",
      str(sel.get("name")))
check("summary shows the sufficient path",
      r["summary"]["execution_decision"] == "direct_execution_sufficient")

# ── 4. Plain sufficient ─────────────────────────────────────────────────────
print("\n[4] sufficient without task_complete is unchanged")
r = run_turn({**DONE, "task_complete": False}, executed=True)
check("still original_task_execute",
      r["out"]["selected_candidate"].get("name") == "original_task_execute")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 02 SUMMARY")
print("=" * 60)
failed = [n for n, ok in results.items() if not ok]
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
