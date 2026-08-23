"""Repeatable, writable cache locations for simulator-dependent test imports."""

import os
import tempfile
from pathlib import Path
import sys


_TEST_CACHE_ROOT = Path(tempfile.gettempdir()) / "vla-spatial-diagnostics-pytest"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_TEST_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_TEST_CACHE_ROOT / "matplotlib"))

_COMMON = Path(__file__).resolve().parents[1] / "tools" / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from libero_env import configure_robosuite_logging  # noqa: E402

configure_robosuite_logging(_TEST_CACHE_ROOT / "robosuite.log")
