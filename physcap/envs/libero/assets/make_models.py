"""Write the MuJoCo model files of the LIBERO can scene.

    python physcap/envs/libero/assets/make_models.py

Writes ``<colour>_can/<colour>_can.xml`` for the four cans and
``flat_wooden_tray/flat_wooden_tray.xml`` next to this script. The files are
already in the repository; run this only after changing a value below.

The cans copy the real Identify Empty Can scene: black, green, blue and red
cans, with the real cans' masses (red is empty, green still holds some liquid). Green and red have been opened:
a dark opening in the lid and a bent straw. Black and blue are unopened. The
lid, opening and straw are drawn only; they have no collision, so the gripper
cannot catch on them. Each can collides as a plain cylinder.

Sources: the can mesh (``cans/can.obj``) is robosuite's can model
(robosuite/models/assets/objects/meshes/can.obj, MIT License, Stanford Vision
and Learning Lab and UT Robot Perception and Learning Lab), stretched here. The
tray's wood texture (``flat_wooden_tray/wood.png``) is LIBERO's wooden_tray
texture (MIT License, Lifelong Robot Learning).
"""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent

# Can size (m). The mesh is 5 cm wide and 8 cm tall; it is scaled to this.
RADIUS, HEIGHT = 0.0275, 0.12
HALF = HEIGHT / 2
TOP = 0.0403 * HEIGHT / 0.08 + 0.0008  # top of the silver lid, just above the mesh's own top
HOLE_Y = 0.011  # the opening sits off-centre along y, so in the camera image the straw bends sideways
STRAW_Y = 0.018  # the straw stands at the outer end of the opening, so part of the opening stays visible
STRAW_RADIUS = 0.0042

# colour: (RGBA, mass in kg, straw RGBA or None if unopened). The masses are the
# real cans' (configs/identify_empty_can/gt-mass.json): black and blue unopened and
# full, green opened but still holding liquid, red opened and empty.
CANS = {
    "black": ("0.06 0.06 0.06 1", 0.392, None),
    "green": ("0.16 0.55 0.22 1", 0.208, "1.0 0.55 0.1 1"),
    "blue": ("0.22 0.42 0.85 1", 0.388, None),
    "red": ("0.80 0.08 0.08 1", 0.015, "1.0 0.85 0.1 1"),
}

# Tray (m): 20 x 14 cm, base 8 mm thick, 2 cm rim, 4 mm walls; mass in kg.
TRAY_X, TRAY_Y, TRAY_BASE, TRAY_RIM, TRAY_WALL, TRAY_MASS = 0.10, 0.07, 0.004, 0.010, 0.004, 0.8


def _xyz(p: tuple[float, float, float]) -> str:
    return " ".join(f"{v:.4f}" for v in p)


def can_xml(colour: str, rgba: str, mass: float, straw: str | None) -> str:
    ixx = mass * (3 * RADIUS**2 + HEIGHT**2) / 12
    izz = mass * RADIUS**2 / 2
    opened = ""
    if straw is not None:
        bend = (0, STRAW_Y, TOP + 0.055)  # the straw rises straight, then bends outwards
        tip = (0, STRAW_Y + 0.028, TOP + 0.075)
        opened = f'''
        <geom name="opening" type="box" size="0.0058 0.008 0.0006" pos="0 {HOLE_Y:.4f} {TOP:.4f}" rgba="0.02 0.02 0.02 1" group="1" contype="0" conaffinity="0"/>
        <geom name="straw" type="capsule" fromto="{_xyz((0, STRAW_Y, TOP - 0.02))} {_xyz(bend)}" size="{STRAW_RADIUS}" rgba="{straw}" group="1" contype="0" conaffinity="0"/>
        <geom name="straw_bend" type="capsule" fromto="{_xyz(bend)} {_xyz(tip)}" size="{STRAW_RADIUS}" rgba="{straw}" group="1" contype="0" conaffinity="0"/>'''
    state = "Opened: a dark opening in the lid and a bent straw." if straw else "Unopened."
    return f'''<mujoco model="{colour}_can">
  <!-- A {colour} drink can, {HEIGHT*100:.0f} cm tall and {2*RADIUS*100:.1f} cm wide, {mass} kg, with a silver lid.
       {state}
       Written by make_models.py. Mesh: robosuite's can (MIT License), stretched. -->
  <asset>
    <mesh name="{colour}_can_visual" file="../cans/can.obj" scale="1.1 1.1 {HEIGHT/0.08:.3f}"/>
    <material name="{colour}_can_mat" rgba="{rgba}" reflectance="0.2" specular="0.3" shininess="0.3"/>
    <material name="{colour}_can_silver" rgba="0.80 0.81 0.84 1" reflectance="0.5" specular="1" shininess="0.9"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx:.7f} {ixx:.7f} {izz:.7f}"/>
        <geom type="mesh" mesh="{colour}_can_visual" material="{colour}_can_mat" group="1" contype="0" conaffinity="0"/>
        <geom name="lid" type="cylinder" size="0.0262 0.003" pos="0 0 {TOP-0.003:.4f}" material="{colour}_can_silver" group="1" contype="0" conaffinity="0"/>
        <geom name="base" type="cylinder" size="0.0262 0.003" pos="0 0 {-HALF+0.003:.4f}" material="{colour}_can_silver" group="1" contype="0" conaffinity="0"/>{opened}
        <geom type="cylinder" size="{RADIUS} {HALF}" group="0" friction="1.0 0.3 0.1" solimp="0.998 0.998 0.001" solref="0.001 1"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {-HALF}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {HALF}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{RADIUS} 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
'''


def tray_xml() -> str:
    X, Y, T, R, W = TRAY_X, TRAY_Y, TRAY_BASE, TRAY_RIM, TRAY_WALL
    boxes = [(f"{X} {Y} {T}", f"0 0 {T}")]
    boxes += [(f"{X} {W} {R}", f"0 {s * (Y - W)} {R}") for s in (1, -1)]
    boxes += [(f"{W} {Y} {R}", f"{s * (X - W)} 0 {R}") for s in (1, -1)]
    collision = "\n".join(
        f'        <geom type="box" size="{size}" pos="{pos}" material="tray_wood" group="0" '
        f'friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001" solref="0.001 1"/>'
        for size, pos in boxes
    )
    visual = "\n".join(
        f'        <geom type="box" size="{size}" pos="{pos}" material="tray_wood" group="1" contype="0" conaffinity="0"/>'
        for size, pos in boxes
    )
    return f'''<mujoco model="flat_wooden_tray">
  <!-- A flat wooden tray, {2*X*100:.0f} x {2*Y*100:.0f} cm with a {2*R*100:.0f} cm rim, like the real task's tray.
       Written by make_models.py. Texture: LIBERO's wooden_tray (MIT License).
       contain_region is the box LIBERO checks for "the can is on the tray". -->
  <asset>
    <texture name="tray_wood_tex" file="wood.png" type="2d"/>
    <material name="tray_wood" texture="tray_wood_tex" texrepeat="1 1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <inertial pos="0 0 0.006" mass="{TRAY_MASS}" diaginertia="0.0013 0.0027 0.0040"/>
{collision}
{visual}
        <site type="box" pos="0 0 0.08" quat="1 0 0 0" size="{X-W} {Y-W} 0.075" group="0" rgba="0.8 0 0 0" name="contain_region"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {2*R}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{X} {Y} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
'''


def main() -> None:
    for colour, (rgba, mass, straw) in CANS.items():
        path = ASSETS / f"{colour}_can" / f"{colour}_can.xml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(can_xml(colour, rgba, mass, straw))
    (ASSETS / "flat_wooden_tray" / "flat_wooden_tray.xml").write_text(tray_xml())
    print(f"Wrote the can and tray models under {ASSETS}")


if __name__ == "__main__":
    main()
