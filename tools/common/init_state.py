"""Pin the initial simulator state of every (suite, task, condition, seed).

Physical copy (not a re-export) of the following from
``tools/openvla/init_state_freezer.py``, verified pure (numpy/stdlib/libero
only) before copying:

- ``sha256_bytes``            -- source lines 44-45
- ``sha256_array``            -- source lines 48-49
- ``FrozenInitState``         -- source lines 52-68
- ``official_init_state_path``-- source lines 74-89
- ``load_init_states``        -- source lines 92-102
- ``frozen_init_state_path``  -- source lines 108-115
- ``capture_init_state``      -- source lines 118-141
- ``freeze_init_state``       -- source lines 144-228
- ``resolve_init_state``      -- source lines 234-276
- ``compare_states``          -- source lines 282-294

``FROZEN_ROOT`` uses the same ``parents[2]`` depth as the original
(``tools/openvla/init_state_freezer.py`` -> repo root is two parents up from
``tools/openvla/``; ``tools/common/init_state.py`` -> repo root is likewise two
parents up from ``tools/common/``), so it resolves to the same
``assets/frozen_init_states/`` directory without adjustment.

**Deliberately NOT copied: ``verify_reproducibility``.** Re-reading its body
found it is not actually pure: it does a local
``from changed_entity_detector import extract_object_poses``, and
``changed_entity_detector.extract_object_poses`` itself does a local
``from spatial_task_resolver import get_entity_world_position, unwrap_base_env``
-- ``spatial_task_resolver`` is one of the four explicit
diagnostics-only "single source of truth" modules
(``tools/openvla/README.md`` "Shared modules" section /
``tools/CLAUDE.md``) that must not be duplicated and that
``tools/common`` must never import from, even transitively. Copying
``verify_reproducibility`` here would either (a) silently break at import time
since ``tools/common`` cannot reach ``tools/openvla``, or (b) require
duplicating ``spatial_task_resolver`` logic, which is explicitly prohibited.
Reproducibility checking for action-generalization work can still use
``compare_states`` directly on ``env.sim.get_state()`` vectors, which is the
part of ``verify_reproducibility`` that was actually pure.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SOURCE_OFFICIAL = "official_pruned_init"
SOURCE_FROZEN = "frozen_locally"

#: Where locally frozen states live. Content is committed so runs are reproducible.
FROZEN_ROOT = Path(__file__).resolve().parents[2] / "assets" / "frozen_init_states"

#: Two runs of the same seed must land within this of each other, per element.
REPRODUCIBILITY_TOLERANCE = 1e-9


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array, dtype=np.float64).tobytes())


@dataclass
class FrozenInitState:
    suite: str
    condition: str
    task_id: int
    task_name: str
    seed: int
    init_state_id: int
    source: str
    path: str
    sha256: str
    state_dim: int
    num_available: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Official init states
# -----------------------------------------------------------------------------
def official_init_state_path(bddl_path: str | os.PathLike) -> Optional[Path]:
    """``.../bddl_files/<folder>/<task>.bddl`` -> ``.../init_files/<folder>/<task>.pruned_init``.

    Returns None when neither ``.pruned_init`` nor ``.init`` is present, which is
    the honest answer for a runtime-generated swap BDDL.
    """
    from libero.libero import get_libero_path

    bddl_path = Path(bddl_path)
    folder = bddl_path.parent.name
    root = Path(get_libero_path("init_states")) / folder
    for suffix in (".pruned_init", ".init"):
        candidate = root / (bddl_path.stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def load_init_states(path: str | os.PathLike) -> np.ndarray:
    """LIBERO stores these as torch pickles; return a plain 2-D float array."""
    import torch

    states = torch.load(str(path), map_location="cpu")
    if hasattr(states, "numpy"):
        states = states.numpy()
    array = np.asarray(states, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


# -----------------------------------------------------------------------------
# Local freezing
# -----------------------------------------------------------------------------
def frozen_init_state_path(
    suite: str, condition: str, task_id: int, seed: int, init_state_id: int = 0,
    root: Path = FROZEN_ROOT,
) -> Path:
    return (
        root / suite / condition
        / f"task_{task_id:02d}__seed{seed}__init{init_state_id:03d}.npy"
    )


def capture_init_state(bddl_path: str, resolution: int, init_state_id: int) -> np.ndarray:
    """Deterministically capture the ``init_state_id``-th placement draw.

    LIBERO samples object placements inside their regions on every reset, so
    successive resets of a seeded env give successive independent draws -- the
    same thing the shipped ``.pruned_init`` files are: a list of sampled
    placements. Index ``k`` is therefore the state after ``k + 1`` resets of a
    freshly seeded env, which is reproducible from the seed alone.
    """
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution
    )
    try:
        env.seed(0)
        for _ in range(init_state_id + 1):
            env.reset()
        return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    finally:
        try:
            env.close()
        except Exception:
            pass


def freeze_init_state(
    env: Any,
    suite: str,
    condition: str,
    task_id: int,
    task_name: str,
    seed: int,
    init_state_id: int = 0,
    bddl_path: Optional[str] = None,
    resolution: int = 256,
    root: Path = FROZEN_ROOT,
    allow_create: bool = True,
) -> FrozenInitState:
    """Return the frozen state for this key, capturing it only if absent.

    ``env`` must already be reset. Nothing is written when the file exists, so
    calling this on every episode is safe and is in fact the intended usage.
    """
    path = frozen_init_state_path(suite, condition, task_id, seed, init_state_id, root)
    notes: List[str] = []

    if path.is_file():
        state = np.load(path)
        notes.append("reused existing frozen state; regeneration is refused by design")
    else:
        if not allow_create:
            raise FileNotFoundError(
                f"no frozen init state at {path} and creation is disabled. "
                "Run the freezer once before collecting."
            )
        if init_state_id == 0 or bddl_path is None:
            state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        else:
            state = capture_init_state(bddl_path, resolution, init_state_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        # np.save appends ".npy" unless the *handle* form is used, which would
        # silently write a differently-named file and make the rename fail.
        with open(temporary, "wb") as handle:
            np.save(handle, state)
        os.replace(temporary, path)
        digest = sha256_array(state)
        path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "suite": suite,
                    "condition": condition,
                    "task_id": task_id,
                    "task_name": task_name,
                    "seed": seed,
                    "init_state_id": init_state_id,
                    "sha256": digest,
                    "state_dim": int(state.size),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        notes.append("captured once under the fixed seed; will never be regenerated")

    digest = sha256_array(state)
    recorded = path.with_suffix(".sha256")
    if recorded.is_file():
        expected = recorded.read_text(encoding="utf-8").strip()
        if expected != digest:
            raise RuntimeError(
                f"frozen init state at {path} does not match its recorded hash "
                f"({digest} != {expected}); the file has been modified"
            )

    return FrozenInitState(
        suite=suite,
        condition=condition,
        task_id=task_id,
        task_name=task_name,
        seed=seed,
        init_state_id=init_state_id,
        source=SOURCE_FROZEN,
        path=str(path),
        sha256=digest,
        state_dim=int(state.size),
        num_available=1,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Unified resolution
# -----------------------------------------------------------------------------
def resolve_init_state(
    env: Any,
    bddl_path: str | os.PathLike,
    suite: str,
    condition: str,
    task_id: int,
    task_name: str,
    seed: int,
    init_state_id: int = 0,
    resolution: int = 256,
    root: Path = FROZEN_ROOT,
) -> tuple[np.ndarray, FrozenInitState]:
    """Official state when one ships; otherwise the locally frozen one."""
    official = official_init_state_path(bddl_path)
    if official is not None:
        states = load_init_states(official)
        if not 0 <= init_state_id < len(states):
            raise IndexError(
                f"init_state_id {init_state_id} out of range for {official} "
                f"({len(states)} available)"
            )
        state = states[init_state_id]
        return state, FrozenInitState(
            suite=suite,
            condition=condition,
            task_id=task_id,
            task_name=task_name,
            seed=seed,
            init_state_id=init_state_id,
            source=SOURCE_OFFICIAL,
            path=str(official),
            sha256=sha256_array(state),
            state_dim=int(state.size),
            num_available=int(len(states)),
            notes=["official evaluation init state, used verbatim"],
        )

    record = freeze_init_state(
        env=env, suite=suite, condition=condition, task_id=task_id, task_name=task_name,
        seed=seed, init_state_id=init_state_id, bddl_path=str(bddl_path),
        resolution=resolution, root=root,
    )
    return np.load(record.path), record


# -----------------------------------------------------------------------------
# Reproducibility check
# -----------------------------------------------------------------------------
def compare_states(
    first: np.ndarray, second: np.ndarray, tolerance: float = REPRODUCIBILITY_TOLERANCE
) -> Dict[str, Any]:
    first, second = np.asarray(first, float).ravel(), np.asarray(second, float).ravel()
    if first.shape != second.shape:
        return {"identical": False, "reason": f"shape {first.shape} != {second.shape}"}
    delta = np.abs(first - second)
    return {
        "identical": bool(delta.max() <= tolerance),
        "max_abs_diff": float(delta.max()),
        "num_differing": int((delta > tolerance).sum()),
        "state_dim": int(first.size),
    }
