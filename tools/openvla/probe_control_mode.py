"""Opt-in control condition: same robot start pose, perturbation untouched.

Why this exists
---------------
The official assets ship a separate ``.pruned_init`` per perturbation condition,
and those files do not agree on where the *robot* starts. Measured over all 135
matched (task, init state, condition) pairs of the y-curve collection, the arm
joints differ in **every** pair -- up to 5.44e-2 rad, which puts the end-effector
as much as 29.6 mm from where the vanilla episode began. Against a 70 mm
perturbation that is 42% of the signal, so a bias measured between those two
episodes confounds "the object moved" with "the arm started somewhere else".

This module removes that confound without touching anything official. The
perturbed init state is applied exactly as the official pipeline applies it, and
then the **robot's** 9 degrees of freedom -- 7 arm joints plus 2 gripper joints --
are overwritten from the vanilla init state of the same task and index. Object,
fixture and camera state are never written, so the perturbation survives intact.

That the robot slice can be spliced out at all is a property of the model layout,
verified rather than assumed: ``robot0_joint1..7`` occupy qpos 0-6 and
``gripper0_finger_joint1..2`` occupy qpos 7-8, all single-DOF, and the first
object free joint starts at qpos 9. qvel is indexed the same way over that
prefix because every one of those nine joints has one velocity DOF.

What this is and is not
-----------------------
This is a **control condition for probe reliability**, not an official
evaluation. Episodes collected this way carry ``evaluation_mode=probe_control``
in their manifest and live under their own root. Nothing here changes the
official mode, which remains the default.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MODE_OFFICIAL = "official"
MODE_PROBE_CONTROL = "probe_control"

#: 7 arm joints + 2 gripper joints, contiguous at the head of qpos and qvel.
ROBOT_QPOS_DOF = 9
ROBOT_QVEL_DOF = 9

#: Two states agreeing to better than this are "identical up to numerical noise".
POSE_TOLERANCE_RAD = 1e-9
POSE_TOLERANCE_M = 1e-9


def robot_slice_is_valid(env: Any) -> Tuple[bool, str]:
    """Confirm the model really does put the robot in the first nine DOF.

    Called before any splice. If a future LIBERO version reorders joints, this
    fails loudly instead of silently overwriting an object's pose.
    """
    base = env.env if hasattr(env, "env") else env
    model = base.sim.model
    robot = base.robots[0]
    arm = list(robot._ref_joint_pos_indexes)
    grip = list(robot._ref_gripper_joint_pos_indexes)
    expected = list(range(ROBOT_QPOS_DOF))
    if arm + grip != expected:
        return False, f"robot qpos indices {arm + grip} != {expected}"
    # The first non-robot joint must start at or after the robot block.
    starts = [int(model.jnt_qposadr[j]) for j in range(model.njnt)]
    non_robot = [s for j, s in enumerate(starts) if j >= len(arm) + len(grip)]
    if non_robot and min(non_robot) < ROBOT_QPOS_DOF:
        return False, f"a non-robot joint starts inside the robot block at qpos {min(non_robot)}"
    return True, f"robot occupies qpos/qvel 0..{ROBOT_QPOS_DOF - 1}, objects from {ROBOT_QPOS_DOF}"


def splice_robot_pose(
    perturbed_state: np.ndarray,
    vanilla_state: np.ndarray,
    nq: int,
    include_velocity: bool = True,
) -> np.ndarray:
    """Perturbed scene, vanilla robot.

    The flattened state is ``[time, qpos(nq), qvel(nv)]``. Only the first
    ``ROBOT_QPOS_DOF`` entries of qpos -- and optionally the matching qvel
    prefix -- are taken from ``vanilla_state``; every object, fixture and the
    simulator time come from ``perturbed_state`` unchanged.
    """
    perturbed = np.asarray(perturbed_state, dtype=np.float64).copy()
    vanilla = np.asarray(vanilla_state, dtype=np.float64)
    if perturbed.shape != vanilla.shape:
        raise ValueError(
            f"state shapes differ: perturbed {perturbed.shape} vs vanilla {vanilla.shape}"
        )
    qpos_start = 1
    perturbed[qpos_start:qpos_start + ROBOT_QPOS_DOF] = \
        vanilla[qpos_start:qpos_start + ROBOT_QPOS_DOF]
    if include_velocity:
        qvel_start = 1 + nq
        perturbed[qvel_start:qvel_start + ROBOT_QVEL_DOF] = \
            vanilla[qvel_start:qvel_start + ROBOT_QVEL_DOF]
    return perturbed


def vanilla_init_state(
    vanilla_bddl_path: str, init_state_id: int
) -> np.ndarray:
    """The official vanilla init state this perturbed episode should borrow from."""
    from init_state_freezer import load_init_states, official_init_state_path

    path = official_init_state_path(vanilla_bddl_path)
    if path is None:
        raise FileNotFoundError(
            f"no official init state beside {vanilla_bddl_path}; probe_control needs one"
        )
    states = load_init_states(path)
    if not 0 <= init_state_id < len(states):
        raise IndexError(f"init_state_id {init_state_id} out of range for {path}")
    return np.asarray(states[init_state_id], dtype=np.float64)


# -----------------------------------------------------------------------------
# State snapshots for verification
# -----------------------------------------------------------------------------
@dataclass
class StateSnapshot:
    """Everything the control condition promises to hold or to preserve."""

    label: str
    arm_qpos: List[float]
    gripper_qpos: List[float]
    eef_pos: List[float]
    eef_quat: List[float]
    entity_xyz: Dict[str, Optional[List[float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def snapshot(env: Any, entities: Sequence[str], label: str) -> StateSnapshot:
    from changed_entity_detector import extract_object_poses

    base = env.env if hasattr(env, "env") else env
    sim = base.sim
    obs = base._get_observations()
    poses = extract_object_poses(env, entities)
    return StateSnapshot(
        label=label,
        arm_qpos=[float(v) for v in sim.data.qpos[:7]],
        gripper_qpos=[float(v) for v in sim.data.qpos[7:9]],
        eef_pos=[float(v) for v in obs["robot0_eef_pos"]],
        eef_quat=[float(v) for v in obs["robot0_eef_quat"]],
        entity_xyz={name: (None if p.xyz is None else [float(v) for v in p.xyz])
                    for name, p in poses.items()},
    )


def compare(a: StateSnapshot, b: StateSnapshot) -> Dict[str, Any]:
    def worst(x, y):
        if x is None or y is None:
            return None
        return float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max())

    entities = {}
    for name in set(a.entity_xyz) | set(b.entity_xyz):
        entities[name] = worst(a.entity_xyz.get(name), b.entity_xyz.get(name))
    return {
        "arm_qpos_max_rad": worst(a.arm_qpos, b.arm_qpos),
        "gripper_qpos_max": worst(a.gripper_qpos, b.gripper_qpos),
        "eef_pos_max_m": worst(a.eef_pos, b.eef_pos),
        "eef_quat_max": worst(a.eef_quat, b.eef_quat),
        "entity_displacement_m": entities,
    }


def robot_pose_matches(comparison: Dict[str, Any]) -> bool:
    """Did the control actually equalise the robot?"""
    return (
        (comparison["arm_qpos_max_rad"] or 0.0) <= POSE_TOLERANCE_RAD
        and (comparison["gripper_qpos_max"] or 0.0) <= POSE_TOLERANCE_RAD
        and (comparison["eef_pos_max_m"] or 0.0) <= POSE_TOLERANCE_M
    )
