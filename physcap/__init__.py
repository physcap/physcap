"""PhysCaP: Grounding Code-as-Policy Agents with Physics-Informed Exploration.

This package adds PhysCaP to CaP-X (in ``third_party/cap-x``):

    physcap.run          command-line entry point
    physcap.trial        one trial of any method in the paper
    physcap.agents       the Planner, the Prioritizer and the scene knowledge they share
    physcap.apis         the tools the coding agent may call (get_object_pose, get_mass, ...)
    physcap.envs         PiPER environments ("dummy": a photo instead of a robot) and the LIBERO simulation
    physcap.measurement  records each measurement and grades the final decision
    physcap.summarize    success rate, objects interacted and time, as reported in the paper

The prompts every agent uses are plain-text files in ``prompts/``.

Importing ``physcap`` registers its environments and APIs with CaP-X's
registries, so YAML configs can refer to them by name.
"""

import importlib.util

# Must run before anything imports LIBERO (CaP-X does on import).
from physcap.envs.libero.paths import configure_libero_paths

configure_libero_paths()

from capx.envs.base import register_env  # noqa: E402
from capx.integrations.base_api import register_api  # noqa: E402

from physcap.apis.dummy_api import DummyControlApi, DummyPhysXApi  # noqa: E402
from physcap.envs.piper_dummy import PiperDummyLowLevel  # noqa: E402

register_api("DummyControlApi", DummyControlApi)
register_api("DummyPhysXApi", DummyPhysXApi)
register_env("piper_dummy_low_level", PiperDummyLowLevel)

# Real-robot code (physcap.envs.piper_real, physcap.apis.piper_api, physcap.physx,
# physcap.hardware) is optional: it is not part of every copy of this package, and
# it needs the robot's own Python packages.
if importlib.util.find_spec("physcap.envs.piper_real") is not None:
    try:
        from physcap.apis.piper_api import PiperControlApi, PiperPhysXApi
        from physcap.envs.piper_real import PiperRealLowLevel

        register_api("PiperControlApi", PiperControlApi)
        register_api("PiperPhysXApi", PiperPhysXApi)
        register_env("piper_real_low_level", PiperRealLowLevel)
    except ImportError as exc:
        print(f"[physcap] real PiPER env not available ({exc}); dummy mode still works.")

# The LIBERO simulation needs the optional LIBERO install (`uv sync --extra libero`).
if importlib.util.find_spec("libero") is not None:
    try:
        from physcap.apis.libero_api import LiberoControlApi, LiberoPhysXApi
        from physcap.envs.libero.can_task import LiberoCanEnv

        register_api("LiberoControlApi", LiberoControlApi)
        register_api("LiberoPhysXApi", LiberoPhysXApi)
        register_env("libero_can_low_level", LiberoCanEnv)
    except ImportError as exc:
        print(f"[physcap] LIBERO simulation not available ({exc}).")
