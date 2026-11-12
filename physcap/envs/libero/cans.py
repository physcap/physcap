"""The four cans and the wooden tray of the LIBERO can scene.

The same objects as the real Identify Empty Can task: black, green, blue and red
drink cans and a flat wooden tray. The masses are the real cans'
(``configs/identify_empty_can/gt-mass.json``): red is empty (0.015 kg), green
still holds some liquid (0.208 kg), blue and black are full (0.388 and 0.392 kg).
The green and red cans have been opened (an opening in the lid and a straw); the
black and blue cans are unopened, as in the real scene.

Each can is 12 cm tall and 5.5 cm wide; its mesh is robosuite's can, stretched.
The tray is 20 x 14 cm with a 2 cm rim. The model files are under ``assets/``
and are written by ``assets/make_models.py``.

Importing this module registers the objects with LIBERO so the scene file
(``assets/bddl/put_empty_can_on_tray.bddl``) can use them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from libero.envs.base_object import OBJECTS_DICT, register_object
from robosuite.models.objects import MujocoXMLObject

ASSET_DIR = Path(__file__).resolve().parent / "assets"


class _XmlObject(MujocoXMLObject):
    """An object loaded from ``assets/<folder>/<category>.xml``."""

    folder = ""
    category = ""

    def __init__(self, name: str | None = None, joints: list[dict[str, Any]] | None = None) -> None:
        super().__init__(
            os.fspath(ASSET_DIR / self.folder / f"{self.category}.xml"),
            name=name or self.category,
            joints=joints if joints is not None else [dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(re.sub(r"([A-Z])", r" \1", type(self).__name__).split()).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class BlackCan(_XmlObject):
    folder = category = "black_can"


@register_object
class GreenCan(_XmlObject):
    folder = category = "green_can"


@register_object
class BlueCan(_XmlObject):
    folder = category = "blue_can"


@register_object
class RedCan(_XmlObject):
    folder = category = "red_can"


@register_object
class FlatWoodenTray(_XmlObject):
    folder = category = "flat_wooden_tray"


OBJECTS_DICT.update(
    {
        "black_can": BlackCan,
        "green_can": GreenCan,
        "blue_can": BlueCan,
        "red_can": RedCan,
        "flat_wooden_tray": FlatWoodenTray,
    }
)
