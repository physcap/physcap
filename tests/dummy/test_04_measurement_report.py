"""
test_04_measurement_report.py

Per-trial measurement tracking. In dummy auto mode the agent finds the empty
can by weighing candidates one at a time, so WHICH objects it measured and in
WHAT ORDER is the efficiency metric these trials are compared on. The env
records every get_mass()/get_stiffness() call and the trial loop writes it to
measurement_report.json next to code.py.

Events are keyed by the resolved ground-truth key, not the agent's free-text
name, so weighing one can twice under two different phrasings shows up as a
repeat instead of looking like two distinct objects.

The alias file is COPIED to a temp dir first, so your real
configs/identify_empty_can/aliases.json is never modified.

Tests:
  1. Measurements are recorded in call order, keyed by resolved GT key
  2. Two phrasings of one can count as one distinct object, one repeat
  3. target_object.measurement_index marks the measurement that found it
  4. reset() clears the tracker, so trial 2 does not inherit trial 1
  5. A measurement with no GT table serializes as a null key
  6. Stiffness calls are counted too, and never flip the scene
  7. The report is JSON-serializable and non-tracking envs are skipped cleanly

Usage:
    uv run python tests/dummy/test_04_measurement_report.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASK_DIR = REPO_ROOT / "configs/identify_empty_can"
SCENES = TASK_DIR / "scenes.json"
DEAD_PORT = "http://127.0.0.1:9/chat/completions"  # discard port: any call fails fast

os.environ["PHYSCAP_GT_MASS_PATH"] = str(TASK_DIR / "gt-mass.json")

from physcap.envs.piper_dummy import PiperDummyLowLevel  # noqa: E402
from physcap.apis.dummy_api import DummyPhysXApi  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# Work on a throwaway copy so the repo's alias file is never touched.
tmp_dir = Path(tempfile.mkdtemp(prefix="piper_dummy_test_"))
aliases_copy = tmp_dir / "aliases.json"
shutil.copy(TASK_DIR / "aliases.json", aliases_copy)


def build(**overrides):
    kwargs = dict(
        scenes_path=str(SCENES),
        scenario="tray_right",
        aliases_path=str(aliases_copy),
        auto=True,
        empty_mass_threshold=0.1,
        # As in configs/identify_empty_can/dummy.yaml: the answer is graded.
        target_key="red can",
        decision_label="EMPTY CAN",
        # Resolver disabled: every name used below is already an alias, so a
        # network call here would mean the alias layer silently regressed.
        resolver_model=None,
        resolver_server_url=DEAD_PORT,
    )
    kwargs.update(overrides)
    return PiperDummyLowLevel(**kwargs)


print("=== Test 04: Per-trial measurement report ===\n")
print(f"alias file (temp copy): {aliases_copy}")
print("gt-mass: blue/green/black cans are full, 'red can' is the empty one (0.015 kg)")

# ── 1. Order and keys ───────────────────────────────────────────────────────
print("\n[1] Three cans weighed in order -- the report records the sequence")
env = build()
api = DummyPhysXApi(env)
for name in ["blue monster can", "green sprite can", "red coca-cola can"]:
    api.get_mass(name)

report = env.measurement_report()
print(f"    order          = {report['order']}")
print(f"    totals         = {report['totals']}")

check("three measurements recorded", report["totals"]["measurements"] == 3)
check("order is the resolved GT key sequence", report["order"] == ["blue can", "green can", "red can"],
      str(report["order"]))
check("counted as mass measurements", report["totals"]["by_property"] == {"mass": 3},
      str(report["totals"]["by_property"]))
check("events are indexed from 1",
      [e["index"] for e in report["events"]] == [1, 2, 3])
check("the agent's own wording is kept",
      report["events"][0]["object_name"] == "blue monster can")
check("values come from ground truth", report["events"][0]["value"] == 0.388,
      str(report["events"][0]["value"]))
check("unit recorded", report["events"][0]["unit"] == "kg")

# ── 2. The empty can ────────────────────────────────────────────────────────
print("\n[2] target_object marks WHICH measurement found it -- the headline metric")
print("    ('red can' was weighed 3rd, so it took 3 measurements to find the empty can)")
check("empty object found", report["target_object"]["found"] is True)
check("empty key recorded", report["target_object"]["key"] == "red can")
check("found on the 3rd measurement", report["target_object"]["measurement_index"] == 3,
      str(report["target_object"]["measurement_index"]))
check("scenario carried through", report["scenario"] == "tray_right")
check("threshold carried through", report["empty_mass_threshold"] == 0.1)

# ── 3. Redundant re-measurement ─────────────────────────────────────────────
print("\n[3] Weighing ONE can under two phrasings = 1 distinct object, 1 repeat")
print("    This is what the prompt's 'prevent redundant loops' rule tries to stop,")
print("    and it is invisible if you key events by the agent's free-text name.")
env2 = build()
api2 = DummyPhysXApi(env2)
api2.get_mass("blue monster can")
api2.get_mass("The Blue Monster Can!")

r2 = env2.measurement_report()
print(f"    order = {r2['order']}, totals = {r2['totals']}")
check("two measurements", r2["totals"]["measurements"] == 2)
check("but one distinct object", r2["totals"]["distinct_objects"] == 1,
      str(r2["totals"]["distinct_objects"]))
check("flagged as one repeat", r2["totals"]["repeat_measurements"] == 1)
check("empty not found here", r2["target_object"]["found"] is False)
check("measurement_index is null when not found",
      r2["target_object"]["measurement_index"] is None)

# ── 4. reset() clears the tracker ───────────────────────────────────────────
print("\n[4] reset() clears the tracker -- THE regression this feature can hit")
print("    One env instance is reused for every trial (_run_trial_batch), so")
print("    without this trial 2's report would still contain trial 1's cans.")
env2.reset()
r3 = env2.measurement_report()
check("events cleared", r3["totals"]["measurements"] == 0, str(r3["totals"]["measurements"]))
check("order cleared", r3["order"] == [])
check("empty-found state cleared", r3["target_object"]["found"] is False)

api2.get_mass("green sprite can")
r4 = env2.measurement_report()
check("next trial re-indexes from 1", r4["events"][0]["index"] == 1)
check("next trial only has its own measurement", r4["order"] == ["green can"], str(r4["order"]))

# ── 5. No ground-truth table -> null key ────────────────────────────────────
print("\n[5] Interactive measurement with no GT table records a null key")
print("    (must serialize as JSON null, not crash, and must still count)")
env3 = PiperDummyLowLevel(
    scenes_path=str(SCENES), scenario="tray_right", auto=False, resolver_model=None,
)
env3._gt_mass = None  # force the free-typed-value path
api3 = DummyPhysXApi(env3)

import builtins  # noqa: E402

_real_input = builtins.input
builtins.input = lambda *_a, **_k: "0.25"
try:
    api3.get_mass("some unlabeled jar")
finally:
    builtins.input = _real_input

r5 = env3.measurement_report()
check("measurement still recorded", r5["totals"]["measurements"] == 1)
check("key is None", r5["events"][0]["key"] is None)
check("order carries the null", r5["order"] == [None], str(r5["order"]))
check("unkeyed object still counts as distinct", r5["totals"]["distinct_objects"] == 1)
check("serializes with a null key", '"key": null' in json.dumps(r5, indent=2))

# ── 6. Stiffness is tracked too ─────────────────────────────────────────────
print("\n[6] Stiffness calls are counted, and never flip the scene")
os.environ["PHYSCAP_GT_STIFFNESS_PATH"] = str(tmp_dir / "gt-stiffness.json")
(tmp_dir / "gt-stiffness.json").write_text(json.dumps({"_units": "level", "blue can": 3, "red can": 1}))
env4 = build()
api4 = DummyPhysXApi(env4)
api4.get_stiffness("blue monster can")
api4.get_mass("green sprite can")

r6 = env4.measurement_report()
print(f"    by_property = {r6['totals']['by_property']}")
check("both property types counted",
      r6["totals"]["by_property"] == {"stiffness": 1, "mass": 1},
      str(r6["totals"]["by_property"]))
check("stiffness value from GT", r6["events"][0]["value"] == 3)
check("stiffness unit is 'level'", r6["events"][0]["unit"] == "level")
check("a low stiffness did not flip the scene", env4._current_key == "initial")

# ── 7. Shape and the non-tracking guard ─────────────────────────────────────
print("\n[7] Report shape is stable, and envs without a tracker are skipped")
check("schema_version present", report.get("schema_version") == 2)
check("whole report is JSON-serializable", isinstance(json.dumps(report), str))
for field in ("totals", "target_object", "order", "events"):
    check(f"'{field}' present", field in report)


class _PlainEnv:
    """Stands in for any non-dummy low-level env (franka, libero, piper_real)."""


check(
    "non-tracking env exposes no measurement_report",
    not callable(getattr(_PlainEnv(), "measurement_report", None)),
)

# ── Summary ─────────────────────────────────────────────────────────────────
os.environ.pop("PHYSCAP_GT_MASS_PATH", None)
shutil.rmtree(tmp_dir, ignore_errors=True)
os.environ.pop("PHYSCAP_GT_STIFFNESS_PATH", None)
print("\n" + "=" * 60)
print("TEST 04 SUMMARY")
print("=" * 60)
failed = [n for n, ok in results.items() if not ok]
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
print(f"repo alias file untouched: {TASK_DIR / 'aliases.json'}")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
