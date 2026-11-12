"""
test_02_gt_mass.py

Ground-truth mass lookup in dummy mode. Instead of estimating mass from joint
torques (impossible with no arm), get_mass() reads a known value out of
gt-mass.json. This is the same PHYSCAP_GT_MASS_PATH mechanism the real-robot
script uses -- in dummy mode there's simply no torque measurement to bypass.

Runs fully offline; keyboard input is scripted, so nothing blocks.

Tests:
  1. gt-mass.json loads and `_`-prefixed metadata keys are stripped
  2. Interactive mode: picking a key returns that object's mass
  3. An invalid key re-prompts instead of aborting the trial
  4. With no GT table, a free-typed float is accepted
  5. A non-numeric entry re-prompts instead of crashing

Usage:
    uv run python tests/dummy/test_02_gt_mass.py
"""

import builtins
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASK_DIR = REPO_ROOT / "configs/identify_empty_can"
GT_MASS = TASK_DIR / "gt-mass.json"

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


@contextlib.contextmanager
def scripted_input(answers: list[str]):
    """Feed a canned sequence of answers to input() so nothing blocks."""
    it = iter(answers)
    original = builtins.input

    def fake_input(prompt: str = "") -> str:
        value = next(it)
        print(f"{prompt}{value}   <-- scripted")
        return value

    builtins.input = fake_input
    try:
        yield
    finally:
        builtins.input = original


def build_env(with_gt: bool):
    """Fresh env with or without a ground-truth mass table."""
    if with_gt:
        os.environ["PHYSCAP_GT_MASS_PATH"] = str(GT_MASS)
    else:
        os.environ.pop("PHYSCAP_GT_MASS_PATH", None)
    from physcap.envs.piper_dummy import PiperDummyLowLevel

    return PiperDummyLowLevel(
        scenes_path=str(TASK_DIR / "scenes.json"),
        scenario="tray_right",
        auto=False,  # interactive: a human normally types the key
    )


print("=== Test 02: Ground-truth mass ===\n")

from physcap.apis.dummy_api import DummyPhysXApi  # noqa: E402

# ── 1. Loading ──────────────────────────────────────────────────────────────
print("[1] Loading gt-mass.json via PHYSCAP_GT_MASS_PATH")
env = build_env(with_gt=True)
print(f"    parsed table: {env._gt_mass}")
print("    note: the '_units' metadata key was dropped (the `_` prefix convention)")
check("mass table loaded", env._gt_mass is not None and len(env._gt_mass) == 4)
check("metadata keys stripped", all(not k.startswith("_") for k in env._gt_mass))
check("values are floats", all(isinstance(v, float) for v in env._gt_mass.values()))

# ── 2. Picking a key ────────────────────────────────────────────────────────
print("\n[2] Interactive lookup -- the human picks the key for the object")
api = DummyPhysXApi(env)
with scripted_input(["red can"]):
    mass = api.get_mass("red coca-cola can")
check("key 'red can' returned the empty can's mass", mass == 0.015, f"got {mass}")

# ── 3. Invalid key re-prompts ───────────────────────────────────────────────
print("\n[3] A bad key re-prompts; returning 0.0 kg would read as")
print("    'this can is empty'")
with scripted_input(["99", "blue can"]):
    mass = api.get_mass("blue monster can")
check("re-prompted, then accepted a valid key", mass == 0.388, f"got {mass}")

# ── 4. Free-typed mass when no GT table exists ────────────────────────────
print("\n[4] With no GT table configured, you type the value directly")
env_nogt = build_env(with_gt=False)
api_nogt = DummyPhysXApi(env_nogt)
print(f"    env._gt_mass = {env_nogt._gt_mass}")
with scripted_input(["0.275"]):
    mass = api_nogt.get_mass("some unlisted object")
check("free-typed float accepted", mass == 0.275, f"got {mass}")

# ── 5. Non-numeric entry re-prompts ─────────────────────────────────────────
print("\n[5] A non-numeric entry re-prompts rather than crashing")
with scripted_input(["heavy", "0.5"]):
    mass = api_nogt.get_mass("another object")
check("re-prompted, then accepted a number", mass == 0.5, f"got {mass}")

# ── Summary ─────────────────────────────────────────────────────────────────
os.environ.pop("PHYSCAP_GT_MASS_PATH", None)
print("\n" + "=" * 60)
print("TEST 02 SUMMARY")
print("=" * 60)
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
failed = [n for n, ok in results.items() if not ok]
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
