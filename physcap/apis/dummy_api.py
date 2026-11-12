"""Dummy Piper APIs: a human (or ground truth) stands in for the robot.

``DummyControlApi`` and ``DummyPhysXApi`` turn every control call
(goto_pose, gripper, home_pose, get_object_pose) into a silent no-op, so the
reasoning/code-generation pipeline can run against a Piper task with no arm
attached. ``DummyPhysXApi`` additionally resolves get_mass/get_stiffness from
ground truth (auto mode) or by prompting a human (interactive mode).
``DummyControlApi`` drops those two functions entirely, for tasks where the
agent is not supposed to measure anything at all.

The tool descriptions the coding agent reads are in
``physcap/apis/piper_tools.py``; the methods below only implement them.

Used with ``physcap.envs.piper_dummy.PiperDummyLowLevel`` (a photo instead of
a camera, no hardware).
"""

from typing import Any
import numpy as np

from capx.envs.base import BaseEnv

from physcap.apis.knowledge import save_property_to_knowledge
from physcap.apis.piper_tools import PiperControlTools, PiperPhysXTools


class _DummyControlMixin:
    """Shared no-op control surface for both dummy APIs."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self.camera_name = "robot0_robotview"

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = np.array([0, 0, 0]), np.array([1, 0, 0, 0])
        print(f"Robot action: Getting object pose for {object_name} (dummy).")
        return pos, quat

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray = None, z_approach: float = 0.0
    ) -> None:
        print("Robot action: goto_pose (dummy).")

    def open_gripper(self) -> None:
        print("Robot action: open_gripper (dummy).")

    def close_gripper(self) -> None:
        print("Robot action: close_gripper (dummy).")

    def home_pose(self) -> None:
        print("Robot action: home_pose (dummy).")


class DummyControlApi(_DummyControlMixin, PiperControlTools):
    """Dummy PiPER control API with no physical-property measurement.

    Exposes the same control surface as :class:`DummyPhysXApi` minus
    ``get_mass``/``get_stiffness``, for the CaP method, where the agent
    is not meant to measure anything.
    """


class DummyPhysXApi(_DummyControlMixin, PiperPhysXTools):
    """Dummy PiPER control + physical-property measurement API.

    Control/perception calls are silent no-ops. get_mass/get_stiffness either
    resolve from a ground-truth table (auto mode) or prompt the human to type
    the measured value directly (interactive mode).
    """

    def _measure_property(
        self,
        object_name: str,
        property_type: str,
        gt_attr: str,
        unit: str,
        parse: Any,
    ) -> Any:
        """Resolve a physical property from ground truth (auto) or from a human.

        Auto mode never prompts and never guesses: the object name is mapped
        onto a ground-truth key by the env, and a name that cannot be resolved
        aborts the trial rather than returning a bogus value (a silent 0.0 kg
        would read as "this can is empty" and quietly corrupt the run).
        """
        env = self._env
        gt = getattr(env, gt_attr, None)

        if getattr(env, "_auto", False):
            if not gt:
                raise RuntimeError(
                    f"auto mode is on but no ground-truth {property_type} table is loaded; "
                    f"cannot measure '{object_name}'."
                )
            key = env.resolve_object_key(object_name, gt.keys())
            value = parse(gt[key])
            print(f"[auto] {property_type} of '{object_name}' resolved to GT key '{key}'")
        elif gt:
            print(f"Available ground-truth {property_type} keys: {sorted(gt.keys())}")
            while True:
                key = input(
                    f"Robot action: Measuring {property_type} of '{object_name}'. Enter the GT key: "
                ).strip()
                if key in gt:
                    break
                print(f"  '{key}' is not a valid key. Choose one of {sorted(gt.keys())}.")
            value = parse(gt[key])
        else:
            key = None
            while True:
                raw = input(
                    f"Robot action: Measuring {property_type} of '{object_name}'. "
                    f"Please enter the {property_type} ({unit}): "
                ).strip()
                try:
                    value = parse(raw)
                    break
                except ValueError:
                    print(f"  '{raw}' is not a valid {property_type} value.")

        save_property_to_knowledge(env, object_name, property_type, value, unit)
        if hasattr(env, "on_property_measured"):
            env.on_property_measured(
                property_type, key, value, object_name=object_name, unit=unit
            )
        return value

    def get_mass(self, object_name: str) -> float:
        mass = self._measure_property(object_name, "mass", "_gt_mass", "kg", float)
        print(f"[get_mass] '{object_name}' has mass {mass:.4f} kg\n")
        return mass

    def get_stiffness(self, object_name: str) -> int:
        stiffness = self._measure_property(object_name, "stiffness", "_gt_stiffness", "level", int)
        print(f"[get_stiffness] '{object_name}' stiffness level: {stiffness}/5\n")
        return stiffness


__all__ = ["DummyControlApi", "DummyPhysXApi"]
