"""
test_03_step_accounting.py

A measurement step counts as done only when its measurement was recorded.

A measure_* step is retired only if measured_properties.json records its
measurement; otherwise it stays pending, and after repeated misses it is marked
failed.
Other actions keep the clean-exit rule. The Planner prompts also forbid
conditional candidates.

No network, no hardware, no model calls: the Planner is monkeypatched, and
scene knowledge / object labels are pre-created so their init calls are skipped.

Tests:
  1. A refused measuring step stays pending and its attempt is counted
  2. A second refusal of the same step marks it failed
  3. A measuring step whose probe was recorded is executed
  4. An inspect step with a clean exit is executed as before
  5. Both Planner prompts forbid conditional candidates

Usage:
    uv run python tests/agents/test_03_step_accounting.py
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import physcap.agents.cycle as cycle  # noqa: E402
from physcap.agents.planner import candidate_key  # noqa: E402
from physcap.agents.scene_knowledge import MEASURED_PROPERTIES_FILENAME  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


LABELS = [
    {"id": "can_0", "description": "the black can on the far left"},
    {"id": "can_1", "description": "the green can second from the left"},
    {"id": "tray_0", "description": "the wooden tray"},
]


def cand(name: str, action: str, desc: str) -> dict:
    return {"name": name, "action": action, "description": desc,
            "parameters": {"object_name": "string"}, "expected_info": desc, "estimated_cost": "medium"}


WEIGH_GREEN = cand("can_1", "measure_mass", "Measure the mass of the green can")
WEIGH_BLACK = cand("can_0", "measure_mass", "Measure the mass of the black can")
INSPECT_TRAY = cand("tray_0", "inspect", "Inspect the wooden tray")


def fake_planner(*, image_path, task, system_prompt, model_name, output_base_dir, api_key):
    out_dir = Path(output_base_dir) / "0000_0000"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [WEIGH_BLACK, WEIGH_GREEN, INSPECT_TRAY]
    full = {"sufficient": False, "task_complete": False, "reason": "keep exploring",
            "central_objects": [], "exploration_candidates": candidates}
    (out_dir / "exploration_full_response.json").write_text(json.dumps(full))
    path = out_dir / "exploration_candidates.json"
    path.write_text(json.dumps(candidates))
    return str(path)


def new_trial_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="step_accounting_test_"))
    (tmp / "scene_knowledge.txt").write_text("Four cans and a wooden tray.\n")
    (tmp / cycle.OBJECT_LABELS_FILENAME).write_text(json.dumps(LABELS))
    (tmp / "scene.png").write_bytes(b"")
    return tmp


def run_turn(tmp: Path, executed: dict, turn: int, success: bool = True) -> dict:
    cycle.planner = fake_planner
    out = cycle.run_reasoning_turn(
        image_path=tmp / "scene.png", task="Find the empty can.",
        output_dir=tmp, turn_dir=tmp / f"turn_{turn:02d}", iteration=turn - 1,
        execution_result={"executed_candidate": executed, "success": success,
                          "stdout": "no robot motion performed"},
        use_prioritizer=False,
    )
    return out["interaction_history"]


print("=== Test 03: measurement steps count only when measured ===\n")
key_green = candidate_key(WEIGH_GREEN)

# ── 1. Refused measuring stays pending ──────────────────────────────────────
print("[1] A measuring step that recorded no probe stays pending")
trial = new_trial_dir()
hist = run_turn(trial, WEIGH_GREEN, turn=2)
check("not marked executed", key_green not in hist["executed_plans"], str(hist["executed_plans"]))
check("not marked failed yet", key_green not in hist["failed_plans"])
check("attempt counted", hist.get("unrecorded_attempts", {}).get(key_green) == 1,
      str(hist.get("unrecorded_attempts")))

# ── 2. Second refusal -> failed ─────────────────────────────────────────────
print("\n[2] Refusing the same step again marks it failed")
hist = run_turn(trial, WEIGH_GREEN, turn=3)
check("attempt count reaches 2", hist.get("unrecorded_attempts", {}).get(key_green) == 2,
      str(hist.get("unrecorded_attempts")))
check("now in failed_plans", key_green in hist["failed_plans"], str(hist["failed_plans"]))
check("still not executed", key_green not in hist["executed_plans"])

# ── 3. Recorded probe -> executed ──────────────────────────────────────────
print("\n[3] A measuring step whose probe was recorded is executed")
trial = new_trial_dir()
(trial / MEASURED_PROPERTIES_FILENAME).write_text(json.dumps([
    {"object_name": "the green can second from the left", "property_type": "mass", "turn": 1},
]))
hist = run_turn(trial, WEIGH_GREEN, turn=2)
check("marked executed", key_green in hist["executed_plans"], str(hist["executed_plans"]))
check("no unrecorded attempt", not hist.get("unrecorded_attempts"))

# ── 4. Non-measurement action keeps the clean-exit rule ────────────────────
print("\n[4] An inspect step with a clean exit is executed as before")
trial = new_trial_dir()
hist = run_turn(trial, INSPECT_TRAY, turn=2)
check("inspect marked executed", candidate_key(INSPECT_TRAY) in hist["executed_plans"],
      str(hist["executed_plans"]))
trial = new_trial_dir()
hist = run_turn(trial, INSPECT_TRAY, turn=2, success=False)
check("a crashed step is still marked failed", candidate_key(INSPECT_TRAY) in hist["failed_plans"])

# ── 5. Prompt rule ──────────────────────────────────────────────────────────
print("\n[5] Planner prompts forbid conditional candidates")
for name in ("planner.txt", "joint.txt"):
    text = (REPO_ROOT / "prompts" / name).read_text(encoding="utf-8")
    check(f"{name} has the self-contained rule", "self-contained and executable now" in text)

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 03 SUMMARY")
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
