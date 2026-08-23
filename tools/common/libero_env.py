"""LIBERO environment construction, extracted for reuse.

Physical copy (not a re-export) of ``make_env`` from
``tools/openvla/collect_official_activations.py`` lines 125-138. The function
body is unchanged; the only addition is the ``os.environ.setdefault(...)``
block below, which the source file relied on getting for free because it is
always imported *after* ``run_single_vanilla_rollout.py`` has already set
these variables at its own module top (see that file, lines 30-38). Since this
module must be self-contained and must not import anything from
``tools/openvla``, the same three ``setdefault`` calls are reproduced here so
`make_env` behaves identically however it is imported.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

# Force headless EGL rendering and project-local LIBERO config before any
# libero import, mirroring tools/openvla/run_single_vanilla_rollout.py lines
# 30-38. setdefault() is a no-op if the caller (or a shell/micromamba
# activation hook) already set these.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)


_ORIGINAL_FILE_HANDLER = logging.FileHandler
_ROBO_LOG_REDIRECT: str | None = None


def configure_robosuite_logging(log_path: str | os.PathLike | None = None) -> str | None:
    """Keep robosuite's fixed ``/tmp/robosuite.log`` from blocking imports.

    Robosuite 1.4 hard-codes that path.  On this host an old log can be owned by
    another user, making even a plain import fail.  The fallback is activated
    only when that exact file cannot be opened for append, and redirects only
    that filename to a caller-provided, writable run/test log.  Simulator and
    policy behavior are unchanged.
    """
    global _ROBO_LOG_REDIRECT
    target = "/tmp/robosuite.log"
    try:
        with open(target, "a", encoding="utf-8"):
            pass
        return None
    except OSError:
        pass

    replacement = Path(log_path or os.environ.get("VLA_ROBOSUITE_LOG_PATH", "/tmp/vla-spatial-diagnostics/robosuite.log"))
    replacement.parent.mkdir(parents=True, exist_ok=True)
    _ROBO_LOG_REDIRECT = str(replacement)

    class RedirectedRobosuiteFileHandler(_ORIGINAL_FILE_HANDLER):
        """Preserve FileHandler's class contract for libraries that subclass it."""

        def __init__(self, filename: str, *args: Any, **kwargs: Any) -> None:
            if os.path.abspath(filename) == target:
                filename = _ROBO_LOG_REDIRECT or filename
            super().__init__(filename, *args, **kwargs)

    logging.FileHandler = RedirectedRobosuiteFileHandler
    return _ROBO_LOG_REDIRECT


def make_env(bddl_path: str, resolution: int, camera_depths: bool = False) -> Any:
    """Build the env the official evaluation builds, from an explicit BDDL path.

    ``env.seed(0)`` mirrors ``get_libero_env``. Re-seeding it with the run seed
    was measured to change the trajectory away from the validated baseline, so
    the run seed governs numpy/torch only.
    """
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=resolution,
        camera_widths=resolution, camera_depths=camera_depths,
    )
    env.seed(0)
    return env
