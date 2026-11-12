"""Point LIBERO at the copy inside this repository, without asking anything.

LIBERO reads its file locations (assets, scene files) from
``$LIBERO_CONFIG_PATH/config.yaml`` (default ``~/.libero``) when it is first
imported, and asks questions on the terminal if that file does not exist. This
module writes a config for the LIBERO copy in ``third_party/cap-x`` into
``<repository>/.libero/`` and sets ``LIBERO_CONFIG_PATH`` to it, so runs never
depend on, or change, the user's own ``~/.libero``.

It must run before ``libero`` is imported; ``physcap/__init__.py`` calls it
first. A ``LIBERO_CONFIG_PATH`` the user has already set is left alone.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_PACKAGE = REPO_ROOT / "third_party" / "cap-x" / "capx" / "third_party" / "LIBERO-PRO" / "libero" / "libero"
CONFIG_DIR = REPO_ROOT / ".libero"


def configure_libero_paths() -> None:
    """Set ``LIBERO_CONFIG_PATH`` to a repository-local config, writing it if needed.

    Also renders without a display by default (``MUJOCO_GL=egl``).
    """
    if not LIBERO_PACKAGE.is_dir():
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("LIBERO_CONFIG_PATH"):
        return
    config = {
        "benchmark_root": LIBERO_PACKAGE,
        "bddl_files": LIBERO_PACKAGE / "bddl_files",
        "init_states": LIBERO_PACKAGE / "init_files",
        "datasets": LIBERO_PACKAGE.parent / "datasets",
        "assets": LIBERO_PACKAGE / "assets",
    }
    text = "".join(f"{key}: {path}\n" for key, path in config.items())
    config_file = CONFIG_DIR / "config.yaml"
    try:
        if not config_file.exists() or config_file.read_text() != text:
            CONFIG_DIR.mkdir(exist_ok=True)
            config_file.write_text(text)
    except OSError:
        return  # read-only checkout: fall back to LIBERO's own default
    os.environ["LIBERO_CONFIG_PATH"] = str(CONFIG_DIR)
