"""The tools the coding agent can call on the PiPER robot, and their descriptions.

The coding agent never sees this code, only each tool's name, arguments and
docstring (CaP-X builds the tool list from them). The docstrings here are the
ones printed in the paper (Appendix B.3 and C.2), so every setup shows the
model the same text:

    PiperControlTools   get_object_pose, goto_pose, open/close_gripper, home_pose,
                        breakpoint_code_block -- the CaP method
    PiperPhysXTools     the same plus get_mass and get_stiffness -- every other method

The classes only define names and descriptions. Each setup subclasses them and
fills in the bodies without repeating the docstrings (Python's ``inspect.getdoc``
then finds the description here):

    physcap/apis/dummy_api.py   dummy mode: no robot, values from ground truth
    physcap/apis/piper_api.py   the real PiPER (not in this release yet)

To add a tool, add a method with its docstring here, list it in ``functions()``,
and implement it in each subclass.
"""

from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase


class PiperControlTools(ApiBase):
    """PiPER perception and motion tools, without physical measurement."""

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
        This function will use Molmo2 for object detection; it will only take in the current object name as input.
        Make sure that the given object name is clear and will not cause any ambiguity.
        It is advised that positional descriptions (left-most, second from right, etc.), color, and the object's name (if possible) are provided in the object name.
        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion (often unused for 3DOF setups).
        """
        raise NotImplementedError

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray = None, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Cartesian IK provided natively by the AgileX firmware.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Example:
        goto_pose(np.array([0.1, 0.2, 0.3])) # This controls the arm directly to position [0.1, 0.2, 0.3]
        goto_pose(np.array([0.1, 0.2, 0.3]), z_approach=0.05) # This controls the arm to position [0.1, 0.2, 0.3] + [0, 0, 0.05] and then moves to position [0.1, 0.2, 0.3]
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion. Ignored in 3DOF positioning.
            z_approach: (float) Z-axis distance offset for the goto_pose insertion approach motion. Will first arrive at position + z_approach meters in the Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        """
        raise NotImplementedError

    def open_gripper(self) -> None:
        """Open gripper fully."""
        raise NotImplementedError

    def close_gripper(self) -> None:
        """Close gripper fully."""
        raise NotImplementedError

    def home_pose(self) -> None:
        """Return the arm to its rest pose."""
        raise NotImplementedError

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint."""
        return None


class PiperPhysXTools(PiperControlTools):
    """PiPER tools plus the PhysX measurements, get_mass and get_stiffness."""

    def functions(self) -> dict[str, Any]:
        return {
            **super().functions(),
            "get_mass": self.get_mass,
            "get_stiffness": self.get_stiffness,
        }

    def get_mass(self, object_name: str) -> float:
        """Measure an object's mass, in kilograms.

        Grasps the object, lifts it, infers mass from joint load, averages a
        few readings, and returns the object to where it found it. Gripper
        must be empty before calling.
        Make sure nothing is in the gripper before calling this.

        Args:
            object_name: The object to weigh. Molmo2 locates it from this name
                alone, so make it unambiguous — give colour and position
                (e.g. "the left-most red cube"), not just "cube".

        Returns:
            Mass in kilograms, +/-30 g noise. Slightly-negative readings on
            very light objects are noise, not failure. A large negative value
            means the grasp failed; nan means no reading was taken — reweigh
            in either case.

        Example:
            mass = get_mass("the left-most red cube")
            print(f"left-most red cube: {mass:.3f} kg")
        """
        raise NotImplementedError

    def get_stiffness(self, object_name: str) -> int:
        """Measure the stiffness of an object by probing its surface.

        This function executes a controlled probing motion on the object,
        measures force and displacement, and classifies the stiffness level.

        Make sure nothing is in gripper before calling this function.

        Returns a stiffness level from 1 (soft) to 5 (rigid):
            1 = Ultra-Soft   (e.g., sponges, foams, soft plush)
            2 = Soft         (e.g., soft rubbers, ripe fruit, silicone)
            3 = Semi-Rigid   (e.g., cardboard, ripe fruit)
            4 = Stiff        (e.g., hardwood, dense polymers)
            5 = Rigid        (e.g., metal, ceramics, stone)
        Returns 0 if measurement fails. You may need to test is again.

        Args:
            object_name: Name of the object to measure (e.g., "red_cube", "apple")
            This function will use Molmo2 for object detecting, it will only take in the current object name as input.
            Make sure that the given object name is clear and will not cause any ambiguity.
            It is adviced that the positional description (left-most, second from right, etc.), color (and object's name if possible) is provided in the object name.

        Returns:
            stiffness_level: Integer 1-5 indicating stiffness. Returns 0 if measurement fails.

        Example:
            stiffness = get_stiffness("apple")
            if stiffness == 2:
                print("Apple stiffness is level 2:soft — likely ripe")
            elif stiffness == 3:
                print("Apple stiffness is level 3:firm — handle carefully")
            elif stiffness == 0:
                print("Apple stiffness is level 0:measurement failed")
        """
        raise NotImplementedError


__all__ = ["PiperControlTools", "PiperPhysXTools"]
