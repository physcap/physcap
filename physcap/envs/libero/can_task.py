"""LIBERO Identify Empty Can: the scene and the low-level environment.

The real task's objects on a LIBERO table, in front of a simulated Franka arm:
black, green, blue and red drink cans and a flat wooden tray (``cans.py``). The
red can is empty; green and red have been opened (a straw), black and blue have
not. Each trial places the cans in seven areas of the table and the tray to
their right (``assets/bddl/put_empty_can_on_tray.bddl``). A can's mass is only
revealed after a verified grasp-and-lift (``mass_reveal.py``); ``get_mass`` in
``physcap/apis/libero_api.py`` performs one.

``LiberoCanEnv`` is CaP-X's ``FrankaLiberoEnv`` built from that scene file,
plus PhysCaP's measurement log (``physcap/measurement.py``), so a trial writes
the same ``measurement_report.json`` as dummy mode. Success is LIBERO's own
check: the red can is on the tray.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import physcap.envs.libero.cans  # noqa: F401  (registers the cans and the tray with LIBERO)
from capx.envs.base import BaseEnv
from capx.envs.simulators.libero import FrankaLiberoEnv
from capx.integrations.libero import LiberoHandle, _extract_language_from_bddl
from libero.envs import OffScreenRenderEnv
from libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
from libero.envs.regions import REGION_SAMPLERS
import libero.envs.problems.libero_tabletop_manipulation  # noqa: F401

from physcap.envs.libero.mass_reveal import MassRevealTracker, object_names_from_observation
from physcap.envs.piper_env import PiperEnv
from physcap.measurement import _DEFAULT_RESOLVER_MODEL, MeasurementMixin

SCENE_FILE = Path(__file__).resolve().parent / "assets" / "bddl" / "put_empty_can_on_tray.bddl"

# A trial is not limited in simulation steps: it ends when the agent finishes or
# after max_turns turns. LIBERO and CaP-X still need a step count to compare
# against, so they get one no trial reaches. The report records the steps used.
_NO_STEP_LIMIT = 10**9


class Empty_Can_Mass_Sensing(TASK_MAPPING["libero_tabletop_manipulation"]):
    """The problem named in the scene file (``empty_can_mass_sensing``)."""


register_problem(Empty_Can_Mass_Sensing)
REGION_SAMPLERS["empty_can_mass_sensing"] = REGION_SAMPLERS["libero_tabletop_manipulation"]


class LiberoCanEnv(MeasurementMixin, FrankaLiberoEnv):
    """Franka in the LIBERO can scene; masses are revealed by grasp-and-lift."""

    def __init__(
        self,
        *,
        scene_file: str | Path | None = None,
        control_freq: int = 20,
        seed: int | None = None,
        grip_duration_s: float = 0.5,
        lift_duration_s: float = 0.5,
        lift_threshold_m: float = 0.03,
        empty_mass_threshold: float = 0.1,
        aliases_path: str | None = None,
        target_key: str | None = None,
        decision_label: str = "EMPTY CAN",
        resolver_model: str | None = _DEFAULT_RESOLVER_MODEL,
        privileged: bool = True,
        enable_render: bool = True,
        suite_name: str = "physcap",
        task_id: int = 0,
    ) -> None:
        # FrankaLiberoEnv.__init__ loads a task from a LIBERO benchmark suite. This
        # scene is not in a suite, so the same setup is done here from the scene
        # file instead; the rest of FrankaLiberoEnv (motion, poses, observations,
        # video) is used unchanged.
        BaseEnv.__init__(self)
        scene = Path(scene_file) if scene_file else SCENE_FILE
        self.privileged = privileged
        self.max_steps = _NO_STEP_LIMIT
        self.seed = seed
        self.enable_render = enable_render
        self.segmentation_level = "instance"
        self._render_width = 800
        self._render_height = 512

        env = OffScreenRenderEnv(
            bddl_file_name=os.fspath(scene.expanduser().resolve()),
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller="JOINT_POSITION",
            horizon=_NO_STEP_LIMIT,
            ignore_done=True,
            control_freq=control_freq,
            camera_depths=True,
        )
        env.seed(0)
        self.handle = LiberoHandle(
            env=env,
            suite_name=suite_name,
            task_id=task_id,
            task_language=_extract_language_from_bddl(os.fspath(scene)) or "put the empty can on the wooden tray",
            init_states=None,
        )

        self._step_count = 0
        self._sim_step_count = 0
        self._control_freq = control_freq
        self._rng = np.random.default_rng(seed)
        self._current_obs = None
        self._current_info = None
        self._current_reward = None
        self._current_done = None

        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = 4
        self._full_viser_rate = 20

        sim = self.handle.env.sim
        self.gripper_metric_length = 0.04
        self.base_link_idx = sim.model.body_name2id("robot0_base")
        self.gripper_link_idx = sim.model.body_name2id("gripper0_eef")
        self.base_link_wxyz_xyz = np.concatenate([sim.data.xquat[self.base_link_idx], sim.data.xpos[self.base_link_idx]])
        self.gripper_link_wxyz_xyz = np.concatenate(
            [sim.data.xquat[self.gripper_link_idx], sim.data.xpos[self.gripper_link_idx]]
        )
        self._panda_joint_qpos_addrs = []
        for i in range(1, 8):
            addr = sim.model.get_joint_qpos_addr(f"robot0_joint{i}")
            self._panda_joint_qpos_addrs.append(int(addr[0] if isinstance(addr, tuple) else addr))
        self.home_joint_position: np.ndarray | None = None
        self.viser_debug = False
        self.viser_server = None

        self._empty_mass_threshold = float(empty_mass_threshold)
        self._gt_mass: dict[str, float] = {}  # filled from the simulator after reset
        self._mass_tracker: MassRevealTracker | None = None
        self._init_measurement(
            aliases_path=aliases_path,
            target_key=target_key,
            resolver_model=resolver_model,
            decision_label=decision_label,
        )
        self.reset(seed=seed)
        self._mass_tracker = MassRevealTracker(
            self.handle.env,
            object_names_from_observation(self._current_obs),
            grip_duration_s=grip_duration_s,
            lift_duration_s=lift_duration_s,
            lift_threshold_m=lift_threshold_m,
        )
        self._refresh_object_table()

    # -- objects ----------------------------------------------------------------

    def _refresh_object_table(self) -> None:
        """Key every object by a short name ("red can", "wooden tray"), for name matching."""
        names = object_names_from_observation(self._current_obs)  # e.g. "red_can_1"
        self.object_keys = {n.removesuffix("_1").replace("_", " "): n.removesuffix("_1") for n in names}
        if self._mass_tracker is not None:
            masses = {s.name: s.mass for s in self._mass_tracker.states.values()}
            self._gt_mass = {k: masses[f"{v}_1"] for k, v in self.object_keys.items() if f"{v}_1" in masses}

    def object_key(self, object_name: str) -> str:
        """Match a free-text name ("the red can with a straw") to a short name ("red can")."""
        return self.resolve_object_key(object_name, self.object_keys.keys())

    def sim_name(self, key: str) -> str:
        """LIBERO's name for an object, as used by ``_get_object_pose`` ("red_can")."""
        return self.object_keys[key]

    def revealed_mass(self, key: str) -> float | None:
        """The object's mass in kg if a grasp-and-lift has revealed it, else None."""
        if self._mass_tracker is None:
            return None
        return self._mass_tracker.masses().get(f"{self.sim_name(key)}_1")

    # -- trial lifecycle --------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if self._mass_tracker is not None:
            self._mass_tracker.refresh(object_names_from_observation(self._current_obs))
            self._refresh_object_table()
        self._reset_measurement()
        return obs, info

    def move_to_joints_blocking(self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 120) -> None:
        # Same as FrankaLiberoEnv, plus a mass-reveal check after every step.
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target
        for step in range(max_steps):
            current = np.array(self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs], dtype=np.float64)
            if np.linalg.norm(current - target) < tolerance and step > 0:
                break
            action = np.concatenate([(target - current) * self._control_freq, [self._gripper_fraction]])
            action[-1] = 1.0 - action[-1] * 2.0
            self._current_obs, self._current_reward, self._current_done, self._current_info = self.handle.step(action)
            self._after_step()

    def _step_once(self) -> None:
        action = np.concatenate([np.zeros_like(self._current_joints), [self._gripper_fraction]])
        action[-1] = 1.0 - action[-1] * 2.0
        self._current_obs, self._current_reward, self._current_done, self._current_info = self.handle.step(action)
        self._after_step()

    def _after_step(self) -> None:
        self._sim_step_count += 1
        sim = self.handle.env.sim
        self.gripper_link_wxyz_xyz = np.concatenate(
            [sim.data.xquat[self.gripper_link_idx], sim.data.xpos[self.gripper_link_idx]]
        )
        if self._mass_tracker is not None:
            self._mass_tracker.update()
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    # -- measurement log --------------------------------------------------------

    def on_property_measured(self, property_type: str, key: str | None, value: Any, **kwargs: Any) -> None:
        super().on_property_measured(property_type, key, value, **kwargs)
        if property_type == "mass" and key is not None and float(value) < self._empty_mass_threshold:
            self._tracker.mark_target_found(key)

    def check_success(self) -> bool:
        return bool(self.handle.env.check_success())

    def task_completed(self) -> bool:
        return self.check_success()

    def measurement_report(self) -> dict[str, Any]:
        report = super().measurement_report()
        # Success is LIBERO's own check (the empty can is on the tray), as in
        # the paper; physcap.summarize uses this field when it is present.
        report["success"] = self.check_success()
        report["sim_steps"] = self._sim_step_count
        report["empty_mass_threshold"] = self._empty_mass_threshold
        return report

    def close(self) -> None:
        self.handle.env.close()


class LiberoTaskEnv(PiperEnv):
    """Runs the generated code; same turn bookkeeping as the PiPER tasks."""

    prompt = "You are controlling a simulated Franka Emika Panda robot arm in the LIBERO simulator."


__all__ = ["LiberoCanEnv", "LiberoTaskEnv", "SCENE_FILE"]
