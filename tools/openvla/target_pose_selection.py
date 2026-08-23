"""Evaluation-side selection of init states with distinct target poses.

This module runs before OpenVLA inference.  It uses simulator state only to
choose a repeat-evaluation sample set; the selected target XYZ is never passed
to the grounding prediction path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


class TargetPoseSelectionError(RuntimeError):
    """No valid set of genuinely distinct target poses can be selected."""


@dataclass(frozen=True)
class TargetPoseCandidate:
    init_state_id: int
    target_xyz: tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_distinct_target_poses(
    candidates: Iterable[TargetPoseCandidate], tolerance_m: float, max_samples: int,
) -> List[TargetPoseCandidate]:
    """Keep the first representative of each target-XYZ equivalence class."""
    if tolerance_m <= 0:
        raise ValueError("target pose tolerance_m must be positive")
    if max_samples < 2:
        raise ValueError("target pose selection must request at least two samples")
    selected: List[TargetPoseCandidate] = []
    for candidate in candidates:
        position = np.asarray(candidate.target_xyz, dtype=np.float64)
        if not np.isfinite(position).all() or position.shape != (3,):
            raise TargetPoseSelectionError(f"invalid target pose for init {candidate.init_state_id}: {candidate.target_xyz}")
        if all(np.linalg.norm(position - np.asarray(existing.target_xyz)) > tolerance_m for existing in selected):
            selected.append(candidate)
        if len(selected) == max_samples:
            return selected
    if len(selected) < 2:
        raise TargetPoseSelectionError(
            "target pose variation is absent: fewer than two distinct target XYZ values were found"
        )
    return selected


def _runtime_object_position(env: Any, target_object: str) -> np.ndarray:
    base = env
    seen = set()
    while hasattr(base, "env") and id(base) not in seen:
        seen.add(id(base))
        base = base.env
    states = getattr(base, "object_states_dict", {})
    if target_object in states:
        position = np.asarray(states[target_object].get_geom_state()["pos"], dtype=np.float64)
        if position.shape == (3,) and np.isfinite(position).all():
            return position
    names = [name for name in env.sim.model.body_names if name.startswith(target_object)]
    if names:
        return np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(names[0])], dtype=np.float64)
    raise TargetPoseSelectionError(f"could not read runtime target position for {target_object!r}")


def scan_all_target_pose_candidates(config: Dict[str, Any]) -> List[TargetPoseCandidate]:
    """Apply every official init state and return target XYZ after dummy steps."""
    from init_state import load_init_states, official_init_state_path, resolve_init_state
    from libero_env import make_env
    from openvla_model import get_libero_dummy_action
    from seeding import episode_seed
    from task_resolution import resolve_task_condition

    target_object = str(config["evaluation_target_object"])
    resolution = resolve_task_condition(config["suite"], int(config["task_id"]), config["condition"])
    official_path = official_init_state_path(resolution.resolved_bddl_path)
    if official_path is None:
        raise TargetPoseSelectionError("cannot enumerate all init states: no official init-state file")
    count = len(load_init_states(official_path))
    env = make_env(resolution.resolved_bddl_path, int(config["resolution"]))
    candidates: List[TargetPoseCandidate] = []
    try:
        env.seed(0)
        for init_state_id in range(count):
            np.random.seed(episode_seed(int(config["seed"]), resolution.suite, resolution.task_id, init_state_id))
            env.reset()
            init_state, _ = resolve_init_state(
                env=env, bddl_path=resolution.resolved_bddl_path, suite=resolution.suite,
                condition=resolution.requested_condition, task_id=resolution.task_id, task_name=resolution.task_name,
                seed=int(config["seed"]), init_state_id=init_state_id, resolution=int(config["resolution"]),
            )
            env.set_init_state(init_state)
            for _ in range(int(config.get("num_steps_wait", 10))):
                _, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    raise TargetPoseSelectionError(f"environment terminated during dummy steps for init {init_state_id}")
            xyz = _runtime_object_position(env, target_object)
            candidates.append(TargetPoseCandidate(init_state_id, tuple(float(value) for value in xyz)))
    finally:
        env.close()
    return candidates
