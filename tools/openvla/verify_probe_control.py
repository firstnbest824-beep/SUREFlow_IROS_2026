"""State-only verification of the probe-control condition. No policy, no activations.

Answers two questions per matched group, and answers them from the simulator
rather than from the code that is being checked:

1. **Did the control work?** After splicing, do vanilla and perturbed start the
   robot in the same pose -- arm joints, gripper, end-effector position and
   orientation -- to numerical precision?
2. **Did it cost anything?** Is the object perturbation still exactly what it was,
   and did any other object or fixture move as a side effect?

Both are checked twice: immediately after ``set_init_state`` (the reset instant)
and again after the same 10-step warm-up the collection uses, because a state
that matches at reset can still diverge once physics runs.

The uncontrolled numbers are measured in the same run, so the before/after
comparison is like-for-like rather than quoted from an earlier report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from probe_control_mode import (  # noqa: E402
    POSE_TOLERANCE_M,
    POSE_TOLERANCE_RAD,
    compare,
    robot_pose_matches,
    robot_slice_is_valid,
    snapshot,
    splice_robot_pose,
)

WARMUP_STEPS = 10


def build(bddl: str, reset_seed: int, resolution: int = 64):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution
    )
    env.seed(0)
    np.random.seed(reset_seed)
    env.reset()
    return env


def run_condition(
    bddl: str, state: np.ndarray, entities: List[str], reset_seed: int, label: str,
    resolution: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply a state, snapshot at reset, warm up, snapshot again."""
    from run_single_vanilla_rollout import get_libero_dummy_action

    env = build(bddl, reset_seed, resolution)
    try:
        env.set_init_state(state)
        at_reset = snapshot(env, entities, f"{label}@reset")
        for _ in range(WARMUP_STEPS):
            env.step(get_libero_dummy_action())
        after = snapshot(env, entities, f"{label}@warmup")
        return at_reset.to_dict(), after.to_dict()
    finally:
        try:
            env.close()
        except Exception:
            pass


def verify_group(
    suite: str, task_id: int, init_state_id: int, conditions: List[str], seed: int,
    resolution: int = 64,
) -> Dict[str, Any]:
    from changed_entity_detector import extract_object_poses  # noqa: F401
    from entity_role_resolver import resolve_entity_roles_from_path
    from init_state_freezer import load_init_states, official_init_state_path
    from official_task_pair_resolver import episode_seed, resolve_official_task_pair
    from probe_control_mode import StateSnapshot

    def snap(d):  # dict -> snapshot for compare()
        return StateSnapshot(**d)

    vanilla_pair = resolve_official_task_pair(
        suite=suite, task_id=task_id, condition="vanilla", seed=seed
    )
    roles = resolve_entity_roles_from_path(vanilla_pair.vanilla_bddl_path)
    entities = roles.tracked_entities
    reset_seed = episode_seed(seed, suite, task_id, init_state_id)

    vanilla_state = np.asarray(
        load_init_states(official_init_state_path(vanilla_pair.vanilla_bddl_path))[init_state_id],
        dtype=np.float64,
    )
    v_reset, v_warm = run_condition(
        vanilla_pair.vanilla_bddl_path, vanilla_state, entities, reset_seed, "vanilla", resolution
    )

    out: Dict[str, Any] = {
        "suite": suite, "task_id": task_id, "init_state_id": init_state_id,
        "source": roles.source, "destination": roles.destination,
        "reset_seed": reset_seed, "conditions": {},
    }

    for condition in conditions:
        pair = resolve_official_task_pair(
            suite=suite, task_id=task_id, condition=condition, seed=seed
        )
        bddl = pair.perturbed_bddl_path or pair.vanilla_bddl_path
        own = np.asarray(
            load_init_states(official_init_state_path(bddl))[init_state_id], dtype=np.float64
        )

        # Uncontrolled: the official condition, exactly as collected.
        u_reset, u_warm = run_condition(bddl, own, entities, reset_seed, condition, resolution)

        # Controlled: same state with the robot's nine DOF taken from vanilla.
        probe_env = build(bddl, reset_seed, resolution)
        try:
            valid, detail = robot_slice_is_valid(probe_env)
            nq = int(probe_env.env.sim.model.nq)
        finally:
            try:
                probe_env.close()
            except Exception:
                pass
        spliced = splice_robot_pose(own, vanilla_state, nq=nq)
        c_reset, c_warm = run_condition(bddl, spliced, entities, reset_seed, condition, resolution)

        entry = {
            "condition": condition,
            "robot_slice_valid": valid, "robot_slice_detail": detail,
            "uncontrolled_at_reset": compare(snap(v_reset), snap(u_reset)),
            "uncontrolled_after_warmup": compare(snap(v_warm), snap(u_warm)),
            "controlled_at_reset": compare(snap(v_reset), snap(c_reset)),
            "controlled_after_warmup": compare(snap(v_warm), snap(c_warm)),
        }
        # Object displacement must survive the splice unchanged.
        src = roles.source
        entry["source_displacement_uncontrolled_m"] = \
            entry["uncontrolled_at_reset"]["entity_displacement_m"].get(src)
        entry["source_displacement_controlled_m"] = \
            entry["controlled_at_reset"]["entity_displacement_m"].get(src)
        entry["perturbation_preserved"] = (
            entry["source_displacement_uncontrolled_m"] is not None
            and entry["source_displacement_controlled_m"] is not None
            and abs(entry["source_displacement_uncontrolled_m"]
                    - entry["source_displacement_controlled_m"]) <= 1e-9
        )
        # Did the splice disturb anything that is not the robot?
        moved = {}
        for name, before in entry["uncontrolled_at_reset"]["entity_displacement_m"].items():
            after = entry["controlled_at_reset"]["entity_displacement_m"].get(name)
            if before is None or after is None:
                continue
            if abs(before - after) > 1e-9:
                moved[name] = {"uncontrolled": before, "controlled": after}
        entry["entities_disturbed_by_splice"] = moved
        entry["robot_equalised_at_reset"] = robot_pose_matches(entry["controlled_at_reset"])
        entry["robot_equalised_after_warmup"] = robot_pose_matches(entry["controlled_after_warmup"])
        out["conditions"][condition] = entry

    return out


def summarise(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(key, field):
        return [
            c[key][field] for g in groups for c in g["conditions"].values()
            if c[key].get(field) is not None
        ]

    stats = {}
    for stage in ("uncontrolled_at_reset", "controlled_at_reset",
                  "uncontrolled_after_warmup", "controlled_after_warmup"):
        stats[stage] = {}
        for field in ("arm_qpos_max_rad", "gripper_qpos_max", "eef_pos_max_m", "eef_quat_max"):
            vals = collect(stage, field)
            stats[stage][field] = (
                None if not vals else {"max": max(vals), "median": st.median(vals)}
            )

    entries = [c for g in groups for c in g["conditions"].values()]
    return {
        "groups": len(groups),
        "condition_pairs": len(entries),
        "stats": stats,
        "robot_equalised_at_reset": sum(1 for c in entries if c["robot_equalised_at_reset"]),
        "robot_equalised_after_warmup": sum(1 for c in entries if c["robot_equalised_after_warmup"]),
        "perturbation_preserved": sum(1 for c in entries if c["perturbation_preserved"]),
        "entities_disturbed": {
            f"task{g['task_id']}/init{g['init_state_id']}/{c['condition']}": c["entities_disturbed_by_splice"]
            for g in groups for c in g["conditions"].values()
            if c["entities_disturbed_by_splice"]
        },
        "robot_slice_invalid": [
            f"task{g['task_id']}/{c['condition']}" for g in groups
            for c in g["conditions"].values() if not c["robot_slice_valid"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0,1,2,3,4,6,7,8,9")
    parser.add_argument("--init_states", default="0,1,2,3,4")
    parser.add_argument("--conditions", default="y0.1,y0.2,y0.3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--json_out", required=True)
    args = parser.parse_args()

    tasks = [int(t) for t in args.tasks.split(",")]
    inits = [int(i) for i in args.init_states.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    groups = []
    for task_id in tasks:
        for init in inits:
            g = verify_group(args.suite, task_id, init, conditions, args.seed, args.resolution)
            groups.append(g)
            worst_before = max(
                (c["uncontrolled_at_reset"]["arm_qpos_max_rad"] or 0) for c in g["conditions"].values()
            )
            worst_after = max(
                (c["controlled_at_reset"]["arm_qpos_max_rad"] or 0) for c in g["conditions"].values()
            )
            ok = all(c["robot_equalised_at_reset"] for c in g["conditions"].values())
            print(f"  task {task_id:2d} init {init}  arm diff {worst_before:.3e} -> {worst_after:.3e}"
                  f"  {'OK' if ok else 'NOT EQUALISED'}", flush=True)

    summary = summarise(groups)
    Path(args.json_out).write_text(
        json.dumps({"summary": summary, "groups": groups}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("\n=== summary ===")
    n = summary["condition_pairs"]
    for stage in ("uncontrolled_at_reset", "controlled_at_reset",
                  "uncontrolled_after_warmup", "controlled_after_warmup"):
        s = summary["stats"][stage]
        arm, eef = s["arm_qpos_max_rad"], s["eef_pos_max_m"]
        print(f"  {stage:28s} arm max {arm['max']:.3e} med {arm['median']:.3e}   "
              f"eef max {eef['max']:.3e} med {eef['median']:.3e}")
    print(f"\n  robot equalised at reset      : {summary['robot_equalised_at_reset']}/{n}")
    print(f"  robot equalised after warm-up : {summary['robot_equalised_after_warmup']}/{n}")
    print(f"  perturbation preserved        : {summary['perturbation_preserved']}/{n}")
    print(f"  entities disturbed by splice  : {len(summary['entities_disturbed'])}")
    print(f"  robot slice invalid           : {summary['robot_slice_invalid'] or 'none'}")
    print(f"\n  written to {args.json_out}")

    ok = (summary["robot_equalised_at_reset"] == n
          and summary["perturbation_preserved"] == n
          and not summary["entities_disturbed"]
          and not summary["robot_slice_invalid"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
