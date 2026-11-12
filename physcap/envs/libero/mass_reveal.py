"""Reveal an object's mass only after a verified grasp-and-lift (paper Appendix A.4).

An object's mass becomes known once the gripper has held it for at least
``grip_duration_s`` (0.5 s) while it is lifted at least ``lift_threshold_m``
(3 cm) above where it started for ``lift_duration_s`` (0.5 s). Before that its
mass reads ``None``. ``MassRevealTracker.update()`` runs after every simulation
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


GRIPPER_GEOM_TOKENS = ("finger", "gripper", "leftpad", "rightpad", "pad")


@dataclass
class MassProbeState:
    name: str
    root_body_id: int
    geom_ids: set[int]
    reference_z: float
    mass: float
    gripped_duration_s: float = 0.0
    lifted_duration_s: float = 0.0
    revealed: bool = False


class MassRevealTracker:
    """Tracks when object mass should be revealed after a grip-and-lift probe."""

    def __init__(
        self,
        env: Any,
        object_names: Iterable[str] = (),
        *,
        grip_duration_s: float = 0.5,
        lift_duration_s: float = 0.5,
        lift_threshold_m: float = 0.03,
    ) -> None:
        self.env = env
        self.grip_duration_s = float(grip_duration_s)
        self.lift_duration_s = float(lift_duration_s)
        self.lift_threshold_m = float(lift_threshold_m)
        self._gripper_geom_ids: set[int] = set()
        self.states: dict[str, MassProbeState] = {}
        self.refresh(object_names)

    def refresh(self, object_names: Iterable[str] | None = None) -> None:
        names = list(object_names or self.object_names_from_env())
        self._gripper_geom_ids = self._collect_gripper_geom_ids()
        self.states = {}

        for name in names:
            body_id = self._find_object_body_id(name)
            if body_id is None:
                continue

            root_body_id = self._find_root_body_id(body_id)
            geom_ids = self._collect_body_subtree_geom_ids(root_body_id)
            if not geom_ids:
                continue

            sim = self.env.sim
            self.states[name] = MassProbeState(
                name=name,
                root_body_id=root_body_id,
                geom_ids=geom_ids,
                reference_z=float(sim.data.xpos[root_body_id][2]),
                mass=self._body_subtree_mass(root_body_id),
            )

    def update(self) -> None:
        if not self.states:
            self.refresh()

        dt_s = 1.0 / float(self._nested_attr("control_freq", 20))
        sim = self.env.sim
        for state in self.states.values():
            gripped = self._object_is_gripped(state)
            lifted = (
                float(sim.data.xpos[state.root_body_id][2]) - state.reference_z
                >= self.lift_threshold_m
            )
            state.gripped_duration_s = state.gripped_duration_s + dt_s if gripped else 0.0
            state.lifted_duration_s = state.lifted_duration_s + dt_s if lifted else 0.0
            if (
                not state.revealed
                and state.gripped_duration_s >= self.grip_duration_s
                and state.lifted_duration_s >= self.lift_duration_s
            ):
                state.revealed = True

    def masses(self) -> dict[str, float | None]:
        return {
            name: (float(state.mass) if state.revealed else None)
            for name, state in self.states.items()
        }

    def object_names_from_env(self) -> list[str]:
        task_env = self._unwrap_to_attr("objects_dict")
        objects_dict = getattr(task_env, "objects_dict", {})
        return sorted(str(name) for name in objects_dict)

    def _nested_attr(self, attr: str, default: Any = None) -> Any:
        obj = self.env
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, attr):
                return getattr(obj, attr)
            obj = getattr(obj, "env", None)
        return default

    def _unwrap_to_attr(self, attr: str) -> Any:
        obj = self.env
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, attr):
                return obj
            obj = getattr(obj, "env", None)
        return self.env

    def _find_object_body_id(self, object_name: str) -> int | None:
        sim = self.env.sim
        candidates = (
            f"{object_name}_1_main",
            f"{object_name}_main",
            f"{object_name}_1_object",
            f"{object_name}_object",
            f"{object_name}_1",
            object_name,
        )
        for candidate in candidates:
            try:
                return int(sim.model.body_name2id(candidate))
            except Exception:
                continue

        query = object_name.lower()
        matches: list[tuple[int, str]] = []
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id) or ""
            lower_name = body_name.lower()
            if query in lower_name and not lower_name.startswith("robot"):
                matches.append((body_id, lower_name))

        if not matches:
            return None

        matches.sort(
            key=lambda item: (
                0 if item[1].endswith("_main") else 1,
                0 if "_1" in item[1] else 1,
                len(item[1]),
            )
        )
        return int(matches[0][0])

    def _find_root_body_id(self, body_id: int) -> int:
        sim = self.env.sim
        current = int(body_id)
        root_with_joint = current

        while current > 0:
            if int(sim.model.body_jntnum[current]) > 0:
                root_with_joint = current
            parent = int(sim.model.body_parentid[current])
            if parent <= 0:
                break
            current = parent

        return int(root_with_joint)

    def _body_subtree_ids(self, root_body_id: int) -> set[int]:
        sim = self.env.sim
        subtree = {int(root_body_id)}

        for body_id in range(sim.model.nbody):
            current = body_id
            while current > 0:
                if current == root_body_id:
                    subtree.add(body_id)
                    break
                current = int(sim.model.body_parentid[current])

        return subtree

    def _collect_body_subtree_geom_ids(self, root_body_id: int) -> set[int]:
        sim = self.env.sim
        body_ids = self._body_subtree_ids(root_body_id)
        contactable: set[int] = set()
        fallback: set[int] = set()

        for geom_id in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[geom_id]) not in body_ids:
                continue
            fallback.add(int(geom_id))
            if (
                int(sim.model.geom_contype[geom_id]) != 0
                or int(sim.model.geom_conaffinity[geom_id]) != 0
            ):
                contactable.add(int(geom_id))

        return contactable or fallback

    def _body_subtree_mass(self, root_body_id: int) -> float:
        sim = self.env.sim
        return float(
            sum(
                float(sim.model.body_mass[body_id])
                for body_id in self._body_subtree_ids(root_body_id)
            )
        )

    def _collect_gripper_geom_ids(self) -> set[int]:
        sim = self.env.sim
        gripper_ids: set[int] = set()

        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id) or ""
            if any(token in geom_name.lower() for token in GRIPPER_GEOM_TOKENS):
                gripper_ids.add(int(geom_id))

        try:
            contact_geoms = self.env.robots[0].robot_model.contact_geoms
        except Exception:
            contact_geoms = ()

        for geom_name in contact_geoms:
            if not any(token in geom_name.lower() for token in GRIPPER_GEOM_TOKENS):
                continue
            try:
                gripper_ids.add(int(sim.model.geom_name2id(geom_name)))
            except Exception:
                continue

        return gripper_ids

    def _object_is_gripped(self, state: MassProbeState) -> bool:
        if not self._gripper_geom_ids:
            self._gripper_geom_ids = self._collect_gripper_geom_ids()

        sim = self.env.sim
        for contact_idx in range(sim.data.ncon):
            contact = sim.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 in state.geom_ids
                and geom2 in self._gripper_geom_ids
                or geom2 in state.geom_ids
                and geom1 in self._gripper_geom_ids
            ):
                return True

        return False


def object_names_from_observation(obs: dict[str, Any]) -> list[str]:
    return sorted(
        {
            key.rsplit("_pos", 1)[0]
            for key in obs
            if key.endswith("_pos")
            and "_to_" not in key
            and not key.startswith("robot")
        }
    )
