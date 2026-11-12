"""Shared helper: write a measured physical property into scene knowledge.

Used by both ``piper_api.py`` and ``dummy_api.py`` so a mass or stiffness
measurement leaves the same trace in ``scene_knowledge.txt`` regardless of
which API measured it.
"""

from pathlib import Path

from capx.envs.base import BaseEnv


def save_property_to_knowledge(
    env: BaseEnv, object_name: str, property_type: str, value: float | int, unit: str
) -> None:
    """Write a measured physical property to scene_knowledge.txt if it is active.

    A no-op when the env has no ``knowledge_dir`` set (e.g. no reasoning
    pipeline in front of it).
    """
    knowledge_dir = getattr(env, "knowledge_dir", None)
    if not knowledge_dir:
        return
    try:
        from physcap.agents.scene_knowledge import (
            append_measured_property,
            append_object_property_to_knowledge,
        )

        append_object_property_to_knowledge(
            output_dir=Path(knowledge_dir),
            object_name=object_name,
            property_type=property_type,
            value=value,
            unit=unit,
        )
        # Machine-readable twin of the sentence above. The reasoning cycle reads
        # it to retire the candidates that were actually probed -- without it
        # the planner keeps re-proposing objects the agent has already measured.
        append_measured_property(
            output_dir=Path(knowledge_dir),
            object_name=object_name,
            property_type=property_type,
            turn=getattr(env, "current_turn", None),
        )
    except Exception as exc:
        print(f"[PhysAPI] Warning: could not update scene knowledge with {property_type}: {exc}")


__all__ = ["save_property_to_knowledge"]
