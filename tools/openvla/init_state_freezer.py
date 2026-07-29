"""Pin the initial simulator state of every (suite, task, condition, seed).

Two sources exist, and which one applies is decided by looking on disk rather
than by the condition name:

1. **Official init states.** LIBERO ships ``init_files/<problem_folder>/<task>.pruned_init``
   next to each BDDL, and the position-offset conditions
   (``libero_object_temp_x0.1`` and friends) ship their own. When one exists it
   is used verbatim -- that is the official evaluation condition and we do not
   get to replace it.

2. **Frozen locally.** ``libero_spatial`` swap BDDLs are generated at runtime and
   ship no init states. For those, the state is captured **once** under a fixed
   seed, written to ``assets/frozen_init_states/`` with its SHA-256, and reused
   forever after. Regeneration is refused, because a second capture under a
   different RNG state would silently change what "the same episode" means.

Either way the manifest records ``init_state_source``, the file path and the
hash, so an episode collected today can be proven to have started where an
episode collected next month starts.
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
    suite: str, condition: str, task_id: int, seed: int, root: Path = FROZEN_ROOT
) -> Path:
    return root / suite / condition / f"task_{task_id:02d}__seed{seed}.npy"


def freeze_init_state(
    env: Any,
    suite: str,
    condition: str,
    task_id: int,
    task_name: str,
    seed: int,
    root: Path = FROZEN_ROOT,
    allow_create: bool = True,
) -> FrozenInitState:
    """Return the frozen state for this key, capturing it only if absent.

    ``env`` must already be reset. Nothing is written when the file exists, so
    calling this on every episode is safe and is in fact the intended usage.
    """
    path = frozen_init_state_path(suite, condition, task_id, seed, root)
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
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        np.save(temporary, state)
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
        init_state_id=0,
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

    record = freeze_init_state(env, suite, condition, task_id, task_name, seed, root)
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


def verify_reproducibility(
    make_env: Any,
    apply_state: Any,
    entities: Sequence[str],
    seed: int,
    repeats: int = 2,
) -> Dict[str, Any]:
    """Reset twice under the same seed and confirm robot state and poses agree.

    ``make_env`` returns a fresh reset env; ``apply_state`` sets the pinned init
    state on it. Object poses are compared alongside the raw state vector so a
    mismatch points at *what* drifted, not just *that* something did.
    """
    from changed_entity_detector import extract_object_poses

    observed: List[Dict[str, Any]] = []
    for _ in range(repeats):
        env = make_env()
        apply_state(env)
        poses = extract_object_poses(env, entities)
        observed.append({
            "state": np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
            "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
            "poses": {name: pose.xyz for name, pose in poses.items()},
        })
        try:
            env.close()
        except Exception:
            pass

    reference = observed[0]
    comparisons = []
    for index, other in enumerate(observed[1:], start=1):
        pose_deltas = {
            name: (
                None
                if reference["poses"].get(name) is None or other["poses"].get(name) is None
                else float(
                    np.linalg.norm(
                        np.asarray(reference["poses"][name]) - np.asarray(other["poses"][name])
                    )
                )
            )
            for name in entities
        }
        comparisons.append({
            "repeat": index,
            "sim_state": compare_states(reference["state"], other["state"]),
            "robot_qpos": compare_states(reference["qpos"], other["qpos"]),
            "object_pose_delta_m": pose_deltas,
            "max_object_pose_delta_m": max(
                (v for v in pose_deltas.values() if v is not None), default=0.0
            ),
        })

    return {
        "seed": seed,
        "repeats": repeats,
        "comparisons": comparisons,
        "passed": all(
            c["sim_state"]["identical"]
            and c["robot_qpos"]["identical"]
            and c["max_object_pose_delta_m"] <= 1e-9
            for c in comparisons
        ),
    }
