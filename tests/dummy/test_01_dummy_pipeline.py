"""
test_01_dummy_pipeline.py

Walks through the no-hardware "dummy" Piper pipeline: the env that serves a
still photo instead of a live ZED frame, paired with the API whose control
calls are no-ops. Nothing here touches a robot, a camera, piper_daemon.py, or
the network.

Tests:
  1. Scenario-pair env builds and serves the scenario's `initial` photo
  2. Observation matches the schema CaP-X expects from a real camera
  3. render() returns the same frame and never prompts
  4. reward / task_completed are the documented no-op values
  5. Control APIs (goto_pose, gripper, home_pose, get_object_pose) are no-ops
  6. Single-photo mode (initial_image_path) works too

Usage:
    uv run python tests/dummy/test_01_dummy_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from physcap.envs.piper_dummy import PiperDummyLowLevel
from physcap.apis.dummy_api import DummyPhysXApi

TASK_DIR = REPO_ROOT / "configs/identify_empty_can"
SCENES = TASK_DIR / "scenes.json"

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# ── 1. Build the env ────────────────────────────────────────────────────────
print("=== Test 01: Dummy Piper pipeline ===\n")
print("[1] Building PiperDummyLowLevel from scenes.json (scenario 'tray_right')")

env = PiperDummyLowLevel(
    scenes_path=str(SCENES),
    scenario="tray_right",
    auto=True,  # auto=True so nothing ever blocks on input()
)
obs, info = env.reset()
print(f"    reset() -> current frame key: {env._current_key!r}")
check("env builds and starts on the 'initial' frame", env._current_key == "initial")

# ── 2. Observation schema ───────────────────────────────────────────────────
print("\n[2] Observation schema (this is what the agent's code sees)")
view = obs["robot0_robotview"]
print(f"    obs['robot0_robotview'] keys : {sorted(view)}")
print(f"    images.rgb   shape : {view['images']['rgb'].shape}  dtype={view['images']['rgb'].dtype}")
print(f"    images.depth shape : {view['images']['depth'].shape}  (zeros -- placeholder)")
print(f"    intrinsics         :\n{view['intrinsics']}")

check("rgb is an HxWx3 uint8 image", view["images"]["rgb"].ndim == 3 and view["images"]["rgb"].shape[2] == 3)
check("depth matches the rgb resolution", view["images"]["depth"].shape == view["images"]["rgb"].shape[:2])
check("intrinsics / pose_mat present", view["intrinsics"].shape == (3, 3) and view["pose_mat"].shape == (4, 4))

# ── 3. render() ─────────────────────────────────────────────────────────────
print("\n[3] render() -- used by trial.py to capture visual feedback")
frame = env.render()
print(f"    render() shape: {frame.shape}")
check("render() returns the current frame", np.array_equal(frame, view["images"]["rgb"]))

# ── 4. Reward / completion ──────────────────────────────────────────────────
print("\n[4] Reward and completion are human-evaluated, never auto-scored")
print(f"    compute_reward() = {env.compute_reward()}   task_completed() = {env.task_completed()}")
check("reward is 0.0 and task_completed is False", env.compute_reward() == 0.0 and env.task_completed() is False)

# ── 5. Control APIs are no-ops ──────────────────────────────────────────────
print("\n[5] Control APIs -- these log and return, nothing physically moves")
api = DummyPhysXApi(env)
pos, quat = api.get_object_pose("blue monster can")
api.goto_pose(np.array([0.4, 0.2, 0.15]))
api.close_gripper()
api.goto_pose(np.array([0.3, 0.2, 0.25]), z_approach=0.05)
api.open_gripper()
api.home_pose()
print(f"    get_object_pose() -> pos={pos}, quat={quat}")
check("get_object_pose returns the dummy fixed pose", np.array_equal(pos, [0, 0, 0]) and np.array_equal(quat, [1, 0, 0, 0]))
check("control calls left the frame untouched", np.array_equal(env.render(), frame))

# ── 6. Single-photo mode ────────────────────────────────────────────────────
print("\n[6] The simpler single-photo mode (initial_image_path)")
single = PiperDummyLowLevel(
    initial_image_path=str(TASK_DIR / "scenes/0_initial.jpeg"),
    auto=True,
)
single.reset()
check("single-photo env serves its photo", single.render().shape == frame.shape)

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 01 SUMMARY")
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
