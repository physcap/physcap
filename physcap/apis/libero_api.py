"""The tools the coding agent can call in the LIBERO simulation.

Same tool names and nearly the same descriptions as on the real robot
(``piper_tools.py``), so the prompts are shared. Only the parts that describe
the hardware differ: a simulated Franka arm instead of the AgileX PiPER, and
the simulator locating objects instead of Molmo2.

    LiberoControlApi   get_object_pose, goto_pose, open/close_gripper, home_pose,
                       breakpoint_code_block -- the CaP method
    LiberoPhysXApi     the same plus get_mass -- every other method

Motion comes from CaP-X's ``FrankaLiberoPrivilegedApi`` (inverse kinematics on
the simulator's exact object poses). There is no stiffness in this task, so
there is no ``get_stiffness``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.integrations.franka.libero_privileged import FrankaLiberoPrivilegedApi

from physcap.apis.knowledge import save_property_to_knowledge

GRIPPER_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz: gripper pointing down
LIFT_HEIGHT_M = 0.15  # same lift as the real get_mass (paper Section 3.1.1)
HOLD_S = 0.6  # held at the top, longer than the 0.5 s the reveal needs
# A can's pose is the centre of its body (the cans are 12 cm tall). Grasping
# 3 cm above it keeps the hand clear of the can's top while the fingers hold
# the can.
CAN_GRASP_HEIGHT_M = 0.03
# A full can held above its centre swings if it is lifted or lowered in one
# fast move, and can land tilted and fall over. Vertical moves are made through
# waypoints this far apart, with a pause to settle before letting go.
VERTICAL_STEP_M = 0.05
SETTLE_S = 0.3
# Simulation steps spent at each in-between waypoint of a vertical move. With a
# full can in hand the controller never quite meets its tolerance, so without a
# cap every waypoint would run to its full step budget; only the last waypoint
# of a move settles fully. Simulation steps count against the task's limit.
WAYPOINT_MAX_STEPS = 20


class LiberoControlApi(FrankaLiberoPrivilegedApi):
    """Simulated Franka tools without physical measurement."""

    def _grasp_position(self, object_name: str) -> tuple[str, np.ndarray]:
        """Match the name to an object; return its key and where to grasp it."""
        key = self._env.object_key(object_name)
        pos, _ = self._env._get_object_pose(self._env.sim_name(key))
        pos = np.asarray(pos, dtype=np.float64).copy()
        if key.endswith("can"):
            pos[2] += CAN_GRASP_HEIGHT_M
        return key, pos

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "home_pose": self.home_pose,
            "breakpoint_code_block": self.breakpoint_code_block,
        }

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object.
        The simulator locates the object from its name; it will only take in the current object name as input.
        Make sure that the given object name is clear and will not cause any ambiguity.
        It is advised that positional descriptions (left-most, second from right, etc.), color, and the object's name (if possible) are provided in the object name.
        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion pointing the gripper down; pass it to goto_pose.
        """
        _, pos = self._grasp_position(object_name)
        return pos, GRIPPER_DOWN.copy()

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray | None = None, z_approach: float = 0.0
    ) -> None:
        """Go to pose using inverse kinematics on the simulated Franka arm.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Example:
        goto_pose(np.array([0.1, 0.2, 0.3])) # This controls the arm directly to position [0.1, 0.2, 0.3]
        goto_pose(np.array([0.1, 0.2, 0.3]), z_approach=0.05) # This controls the arm to position [0.1, 0.2, 0.3] + [0, 0, 0.05] and then moves to position [0.1, 0.2, 0.3]
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion. Default: gripper pointing down.
            z_approach: (float) Z-axis distance offset for the goto_pose insertion approach motion. Will first arrive at position + z_approach meters in the Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        """
        quat = GRIPPER_DOWN if quaternion_wxyz is None else quaternion_wxyz
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if z_approach:
            # Reach the point above first, then come straight down in short steps,
            # so a reach correction never slides the hand sideways into an object.
            above = position + np.array([0.0, 0.0, z_approach])
            self._reach(above, quat)
            self._move_vertically(above, position, quat)
        else:
            self._reach(position, quat)

    def _move_vertically(self, start: np.ndarray, end: np.ndarray, quat: np.ndarray = GRIPPER_DOWN) -> None:
        """Move straight up or down through waypoints, so a held can does not swing."""
        n = max(1, int(np.ceil(abs(end[2] - start[2]) / VERTICAL_STEP_M)))
        for i in range(1, n):
            self._pass_through(start + (end - start) * i / n, quat)
        self._reach(end, quat)

    def _pass_through(self, position: np.ndarray, quat: np.ndarray) -> None:
        """Head for a waypoint for a fixed number of steps, without waiting to settle there."""
        q = np.asarray(quat, dtype=np.float64)
        rot = SciRotation.from_quat([q[1], q[2], q[3], q[0]])
        hand_target = np.asarray(position, dtype=np.float64).reshape(3) + rot.apply(self._TCP_OFFSET)
        solve = self._pks.solve_ik if self.cfg is None else self._pks.solve_ik_vel_cost
        kwargs = {} if self.cfg is None else {"prev_cfg": self.cfg}
        self.cfg = solve(
            robot=self._robot, target_link_name=self._target_link_name,
            target_position=hand_target, target_wxyz=q, **kwargs,
        )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints, max_steps=WAYPOINT_MAX_STEPS)

    def _reach(self, position: np.ndarray, quat: np.ndarray) -> None:
        super().goto_pose(position, quat)
        self._correct_reach(position, quat)

    def _correct_reach(self, position: np.ndarray, quat: np.ndarray) -> None:
        """Re-solve from scratch if the hand stopped more than 1 cm sideways from the target.

        CaP-X's IK prefers small joint changes from the previous pose; near the
        edge of the arm's reach that can leave the hand short of the target, and an
        object grasped off-centre slips. A fresh solve reaches it.
        """
        q = np.asarray(quat, dtype=np.float64)
        rot = SciRotation.from_quat([q[1], q[2], q[3], q[0]])
        hand_target = np.asarray(position, dtype=np.float64).reshape(3) + rot.apply(self._TCP_OFFSET)
        hand = np.asarray(self._env.get_observation()["robot_cartesian_pos"][:3], dtype=np.float64)
        # Only the sideways error counts: stopping a little high (fingers resting on
        # an object) is expected and not worth a fresh solve.
        if np.linalg.norm(hand[:2] - hand_target[:2]) > 0.01:
            self.cfg = None
            super().goto_pose(position, q)

    def open_gripper(self) -> None:
        """Open gripper fully."""
        super().open_gripper()

    def close_gripper(self) -> None:
        """Close gripper fully."""
        super().close_gripper()

    def home_pose(self) -> None:
        """Return the arm to its rest pose."""
        home = np.asarray(self._env.home_joint_position, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(home)
        self.cfg = np.concatenate([home, [0.0]])

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint."""
        return None


class LiberoPhysXApi(LiberoControlApi):
    """Simulated Franka tools plus mass measurement."""

    def functions(self) -> dict[str, Any]:
        return {**super().functions(), "get_mass": self.get_mass}

    def get_mass(self, object_name: str) -> float:
        """Measure an object's mass, in kilograms.

        Grasps the object, lifts it, reads its mass, and returns the object to
        where it found it. Gripper must be empty before calling.
        Make sure nothing is in the gripper before calling this.

        Args:
            object_name: The object to weigh. The simulator locates it from this
                name alone, so make it unambiguous — give its colour
                (e.g. "the left-most red cube"), not just "cube".

        Returns:
            Mass in kilograms. nan means no reading was taken (the grasp or
            lift failed) — reweigh in that case.

        Example:
            mass = get_mass("the left-most red cube")
            print(f"left-most red cube: {mass:.3f} kg")
        """
        env = self._env
        key, pos = self._grasp_position(object_name)
        above = pos + np.array([0.0, 0.0, LIFT_HEIGHT_M])

        # Grasp, lift 15 cm and hold: the simulator reveals the mass once the can
        # has been held for 0.5 s while lifted at least 3 cm for 0.5 s.
        super().open_gripper()
        super().goto_pose(pos, GRIPPER_DOWN, z_approach=0.10)
        super().close_gripper()
        self._move_vertically(pos, above)
        self._settle(HOLD_S)
        mass = env.revealed_mass(key)

        # Put it back where it was, let it settle, let go, and return to the rest pose.
        self._move_vertically(above, pos + np.array([0.0, 0.0, 0.005]))
        self._settle(SETTLE_S)
        super().open_gripper()
        super().goto_pose(above, GRIPPER_DOWN)
        self.home_pose()

        if mass is None:
            print(f"Warning: Failed to measure mass of '{object_name}': the grasp-and-lift was not verified.")
            return float("nan")
        mass = float(mass)
        print(f"[get_mass] '{object_name}' has mass {mass:.4f} kg")
        save_property_to_knowledge(env, object_name, "mass", mass, "kg")
        env.on_property_measured("mass", key, mass, object_name=object_name, unit="kg")
        return mass


    def _settle(self, seconds: float) -> None:
        for _ in range(int(seconds * self._env._control_freq)):
            self._env._step_once()


__all__ = ["LiberoControlApi", "LiberoPhysXApi"]
