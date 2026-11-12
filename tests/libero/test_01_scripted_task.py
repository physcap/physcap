"""
test_01_scripted_task.py

Runs the LIBERO Identify Empty Can scene with scripted tool calls instead of an
agent: measure every can with get_mass, then carry the red (empty) can to the
tray the way the coding rules describe. No model calls. Takes a few minutes.
Skipped when LIBERO is not installed (uv sync --extra libero).

Tests:
  1. The scene builds with four cans and the wooden tray
  2. get_mass returns each can's mass (the real cans' values)
  3. No can tips over while being measured
  4. Putting the red can on the tray counts as success

Usage:
    uv run --extra libero python tests/libero/test_01_scripted_task.py [layout ...]
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LIBERO_INSTALLED = importlib.util.find_spec("libero") is not None
EXPECTED = {"black can": 0.392, "green can": 0.208, "blue can": 0.388, "red can": 0.015}  # gt-mass.json

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def upright(env, key: str) -> bool:
    sim_env = env.handle.env.env
    q = sim_env.sim.data.body_xquat[sim_env.obj_body_id[env.sim_name(key) + "_1"]]
    return abs(1 - 2 * (q[1] ** 2 + q[2] ** 2)) > 0.95  # can axis within ~18 degrees of vertical


def run(layouts: list[int]) -> None:
    import physcap  # noqa: F401  (points LIBERO at the repository's settings first)
    from physcap.apis.libero_api import LiberoPhysXApi
    from physcap.envs.libero.can_task import LiberoCanEnv

    print("=== Test 01: LIBERO can scene, scripted ===\n")
    env = LiberoCanEnv(
        aliases_path=str(REPO_ROOT / "configs/libero_identify_empty_can/aliases.json"),
        target_key="red can",
        resolver_model=None,  # names below match the aliases exactly; no model calls
    )
    api = LiberoPhysXApi(env)
    for layout in layouts:
        print(f"\n[layout {layout}]")
        env.reset(seed=layout)
        check(f"layout {layout}: four cans and the tray", set(env.object_keys) == set(EXPECTED) | {"wooden tray"},
              ", ".join(sorted(env.object_keys)))

        start = env._sim_step_count
        masses = {key: api.get_mass(key) for key in EXPECTED}
        check(f"layout {layout}: get_mass returns each can's mass",
              all(np.isclose(masses[k], v, atol=1e-3) for k, v in EXPECTED.items()),
              ", ".join(f"{k} {m:.3f}" for k, m in masses.items()))
        check(f"layout {layout}: no can tipped over", all(upright(env, k) for k in EXPECTED))

        # Carry the red can to the tray: lift 20 cm, move over the tray, lower to the
        # can's grasp height + 5 cm (the coding rules), let go, go home.
        can, quat = api.get_object_pose("red can")
        tray, _ = api.get_object_pose("wooden tray")
        api.open_gripper()
        api.goto_pose(can, quat, z_approach=0.10)
        api.close_gripper()
        api.goto_pose(can + [0, 0, 0.20], quat)
        api.goto_pose(np.array([tray[0], tray[1], can[2] + 0.20]), quat)
        api.goto_pose(np.array([tray[0], tray[1], can[2] + 0.05]), quat)
        api.open_gripper()
        api.goto_pose(np.array([tray[0], tray[1], can[2] + 0.20]), quat)
        api.home_pose()
        steps = env._sim_step_count - start
        check(f"layout {layout}: red can on the tray (LIBERO's success check)", env.check_success(), f"{steps} steps")
    env.close()


if LIBERO_INSTALLED:
    run([int(a) for a in sys.argv[1:] if a.isdigit()] or [1])
else:
    print("LIBERO is not installed (uv sync --extra libero); skipping.")

print("\n" + "=" * 60)
print("TEST 01 SUMMARY")
print("=" * 60)
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
failed = [n for n, ok in results.items() if not ok]
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")


def test_checks_passed():
    if not LIBERO_INSTALLED:
        import pytest

        pytest.skip("LIBERO is not installed (uv sync --extra libero)")
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
