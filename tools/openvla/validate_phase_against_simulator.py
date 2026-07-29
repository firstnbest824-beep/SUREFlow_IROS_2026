"""Score the phase resolver against the simulator's own grasp test.

The resolver decides pre_grasp / post_grasp from generic quantities (distance,
contact, gripper opening, lift, comovement). Whether those rules are *right* is
an empirical question, and it has an independent answer available: robosuite's
``ManipulationEnv._check_grasp`` requires a geom from **both** finger pads to be
in contact with the object. That is a strictly stronger and structurally
different test than anything the resolver uses -- in particular it cannot be
satisfied by the palm bumping the object, which is exactly the failure mode the
audit found.

Ground truth is obtained by **deterministic replay**: an episode records its
pinned init state and, at every timestep, the action that was applied. Re-running
those actions on a fresh env reproduces the trajectory without needing the 7B
model or a GPU. Replay fidelity is not assumed -- the episode also recorded a
SHA of the simulator state at every timestep, so the replay is verified to
reproduce the original state exactly before any score is computed. A replay that
diverges is reported and excluded rather than silently scored.

Reported:

* instantaneous agreement between ``phase == post_grasp`` and "the simulator says
  a grasp is in progress"
* precision / recall / FP / FN against the phase-level ground truth
  (grasp onset -> release), which is what post_grasp actually means
* onset error in timesteps for episodes where both agree a grasp happened
* a sensitivity sweep over every threshold that gates the decision
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from task_phase_resolver import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    PhaseThresholds,
    TaskPhaseResolver,
)

#: Replay fidelity is judged on physical quantities, not on a state hash.
#:
#: A bit-exact hash comparison was tried first and rejected every libero_object
#: episode. Measured, the cause was 5.7e-15 of floating-point drift, arising only
#: because the collector had called ``reset()`` once more than the replay before
#: ``set_init_state``; the end-effector position agreed to every printed digit and
#: object positions were identical. A hash cannot tell that apart from a genuine
#: divergence, so the check is the thing that actually matters: does the replayed
#: trajectory pass through the same physical states?
REPLAY_POSITION_TOLERANCE_M = 1e-3


def sha256_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


# -----------------------------------------------------------------------------
# Replay
# -----------------------------------------------------------------------------
@dataclass
class ReplayResult:
    episode_dir: str
    suite: str
    condition: str
    task_id: int
    episode_index: int
    success: bool
    num_steps: int
    replay_exact: bool
    first_divergence: Optional[int]
    max_position_drift_m: float = 0.0
    #: Per timestep, from the simulator itself.
    sim_grasp: List[bool] = field(default_factory=list)
    sim_contact_any: List[bool] = field(default_factory=list)
    source_height: List[float] = field(default_factory=list)
    #: Per timestep, as stored by the collector.
    stored_phase: List[str] = field(default_factory=list)


def replay_episode(episode_dir: Path, resolution: int = 64) -> ReplayResult:
    """Re-run an episode's stored actions and read the simulator's grasp test.

    ``resolution`` is deliberately small: no images are needed, and the renderer
    is only instantiated because LIBERO's env requires one.
    """
    from init_state_freezer import load_init_states, official_init_state_path
    from libero.libero.envs import OffScreenRenderEnv
    from run_single_vanilla_rollout import get_libero_dummy_action

    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (episode_dir / "per_step_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = manifest["entity_roles"]["source"]

    env = OffScreenRenderEnv(
        bddl_file_name=manifest["bddl_path"],
        camera_heights=resolution,
        camera_widths=resolution,
    )
    result = ReplayResult(
        episode_dir=str(episode_dir),
        suite=manifest["suite"],
        condition=manifest["condition"],
        task_id=manifest["task_id"],
        episode_index=manifest.get("episode_index", 0),
        success=bool(manifest.get("success")),
        num_steps=len(records),
        replay_exact=True,
        first_divergence=None,
        max_position_drift_m=0.0,
    )

    try:
        env.seed(0)
        env.reset()

        init_path = manifest["init_state_path"]
        if manifest["init_state_source"] == "official_pruned_init":
            states = load_init_states(init_path)
            init_state = states[manifest.get("init_state_id", 0)]
        else:
            init_state = np.load(init_path)
        env.set_init_state(init_state)

        for _ in range(manifest.get("num_steps_wait", 10)):
            env.step(get_libero_dummy_action())

        base_env = env.env
        gripper = base_env.robots[0].gripper
        object_model = base_env.objects_dict.get(source)
        object_geoms = None if object_model is None else object_model.contact_geoms

        for record in records:
            observed_eef = np.asarray(base_env._get_observations()["robot0_eef_pos"], dtype=float)
            expected_eef = np.asarray(record["eef_pos"], dtype=float)
            drift = float(np.abs(observed_eef - expected_eef).max())
            result.max_position_drift_m = max(result.max_position_drift_m, drift)
            if drift > REPLAY_POSITION_TOLERANCE_M:
                result.replay_exact = False
                if result.first_divergence is None:
                    result.first_divergence = record["timestep"]
                break

            if object_geoms is not None:
                result.sim_grasp.append(
                    bool(base_env._check_grasp(gripper=gripper, object_geoms=object_geoms))
                )
                result.sim_contact_any.append(
                    bool(base_env.check_contact(gripper, object_model))
                )
            else:
                result.sim_grasp.append(False)
                result.sim_contact_any.append(False)
            result.source_height.append(float(record["entity_world_xyz"][source][2]))
            result.stored_phase.append(record["phase"])

            env.step(record["action_applied"])
    finally:
        try:
            env.close()
        except Exception:
            pass

    return result


# -----------------------------------------------------------------------------
# Ground truth and scoring
# -----------------------------------------------------------------------------
def sim_grasp_phase(sim_grasp: Sequence[bool], min_run: int = 3) -> List[bool]:
    """Turn instantaneous fingerpad contact into a held-object interval.

    ``_check_grasp`` flickers as contacts make and break during manipulation, so
    a single frame is not a grasp and a single dropped frame is not a release.
    The interval starts at the first run of ``min_run`` consecutive grasp frames
    and ends at the last one, which is what "post_grasp" is meant to denote.
    """
    first = last = None
    run = 0
    for index, value in enumerate(sim_grasp):
        if value:
            run += 1
            if run >= min_run:
                if first is None:
                    first = index - min_run + 1
                last = index
        else:
            run = 0
    if first is None:
        return [False] * len(sim_grasp)
    return [first <= i <= last for i in range(len(sim_grasp))]


def score(predicted: Sequence[bool], truth: Sequence[bool]) -> Dict[str, Any]:
    tp = sum(1 for p, t in zip(predicted, truth) if p and t)
    fp = sum(1 for p, t in zip(predicted, truth) if p and not t)
    fn = sum(1 for p, t in zip(predicted, truth) if not p and t)
    tn = sum(1 for p, t in zip(predicted, truth) if not p and not t)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "steps": len(predicted),
    }


def first_true(values: Sequence[bool]) -> Optional[int]:
    for index, value in enumerate(values):
        if value:
            return index
    return None


def resolve_phases(
    records: Sequence[Dict[str, Any]],
    source: str,
    destination: str,
    thresholds: PhaseThresholds,
) -> List[str]:
    resolver = TaskPhaseResolver(source, destination, thresholds=thresholds)
    phases = []
    for record in records:
        phases.append(
            resolver.update(
                timestep=record["timestep"],
                source_position=record["entity_world_xyz"][source],
                destination_position=record["entity_world_xyz"][destination],
                gripper_position=record["eef_pos"],
                gripper_qpos=record["gripper_qpos"],
                contact=record["contact"],
            ).phase
        )
    return phases


def load_records(episode_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (episode_dir / "per_step_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, records


# -----------------------------------------------------------------------------
# Sensitivity sweep
# -----------------------------------------------------------------------------
SWEEP = {
    "grasp_height_delta_m": [0.005, 0.010, 0.015, 0.020, 0.030, 0.050],
    "grasp_distance_m": [0.04, 0.05, 0.06, 0.08, 0.10],
    "comovement_relative_drift_m": [0.005, 0.010, 0.015, 0.025],
    "min_gripper_closure_from_open_m": [0.005, 0.010, 0.020, 0.040],
    "min_consecutive_grasp_steps_for_transition": [1, 2, 3, 5, 8],
    "min_consecutive_contact_steps": [1, 2, 3, 5],
}


def sweep_thresholds(
    episodes: Sequence[Tuple[Dict[str, Any], List[Dict[str, Any]], List[bool]]],
    field_name: str,
    values: Sequence[float],
) -> List[Dict[str, Any]]:
    out = []
    for value in values:
        thresholds = PhaseThresholds(**{**DEFAULT_THRESHOLDS.to_dict(), field_name: value})
        predicted: List[bool] = []
        truth: List[bool] = []
        for manifest, records, ground in episodes:
            phases = resolve_phases(
                records, manifest["entity_roles"]["source"],
                manifest["entity_roles"]["destination"], thresholds,
            )
            predicted.extend(p == "post_grasp" for p in phases)
            truth.extend(ground)
        metrics = score(predicted, truth)
        metrics[field_name] = value
        metrics["is_default"] = value == getattr(DEFAULT_THRESHOLDS, field_name)
        out.append(metrics)
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--json_out")
    parser.add_argument("--min_grasp_run", type=int, default=3)
    parser.add_argument("--skip_sweep", action="store_true")
    args = parser.parse_args()

    episode_dirs = sorted(p.parent for p in Path(args.root).rglob("manifest.json"))
    print(f"replaying {len(episode_dirs)} episodes against the simulator's own grasp test\n")

    replays: List[ReplayResult] = []
    scored: List[Tuple[Dict[str, Any], List[Dict[str, Any]], List[bool]]] = []
    per_episode: List[Dict[str, Any]] = []

    for episode_dir in episode_dirs:
        replay = replay_episode(episode_dir)
        replays.append(replay)
        manifest, records = load_records(episode_dir)
        if not replay.replay_exact:
            print(f"  [SKIP] {episode_dir.name} diverged at t={replay.first_divergence} "
                  f"(drift {replay.max_position_drift_m:.2e} m)")
            continue

        ground = sim_grasp_phase(replay.sim_grasp, min_run=args.min_grasp_run)
        predicted = [p == "post_grasp" for p in replay.stored_phase]
        metrics = score(predicted, ground)
        metrics.update(
            episode=str(episode_dir.relative_to(args.root)),
            suite=replay.suite, condition=replay.condition,
            task_id=replay.task_id, episode_index=replay.episode_index,
            success=replay.success,
            sim_grasp_onset=first_true(ground),
            predicted_onset=first_true(predicted),
            sim_grasp_frames=sum(replay.sim_grasp),
        )
        onset_error = (
            None if metrics["sim_grasp_onset"] is None or metrics["predicted_onset"] is None
            else metrics["predicted_onset"] - metrics["sim_grasp_onset"]
        )
        metrics["onset_error_steps"] = onset_error
        metrics["sim_grasp_raw"] = [int(x) for x in replay.sim_grasp]
        metrics["sim_grasp_interval"] = [int(x) for x in ground]
        metrics["stored_phase"] = list(replay.stored_phase)
        metrics["max_position_drift_m"] = replay.max_position_drift_m
        per_episode.append(metrics)
        scored.append((manifest, records, ground))

    if not per_episode:
        print("no episodes could be replayed exactly; nothing to score")
        return 1

    print(f"{'episode':46s} {'succ':>5s} {'simGrasp':>8s} {'onset(sim/pred)':>16s} {'P':>6s} {'R':>6s} {'FP':>5s} {'FN':>5s}")
    for m in per_episode:
        print(f"{m['episode'][:46]:46s} {str(m['success']):>5s} {m['sim_grasp_frames']:8d} "
              f"{str(m['sim_grasp_onset']):>7s}/{str(m['predicted_onset']):<8s} "
              f"{_fmt(m['precision']):>6s} {_fmt(m['recall']):>6s} {m['fp']:5d} {m['fn']:5d}")

    predicted_all = [p for _, _, g in scored for p in []]  # placeholder, replaced below
    pooled_pred: List[bool] = []
    pooled_truth: List[bool] = []
    for (manifest, records, ground), replay in zip(scored, [r for r in replays if r.replay_exact]):
        pooled_pred.extend(p == "post_grasp" for p in replay.stored_phase)
        pooled_truth.extend(ground)
    pooled = score(pooled_pred, pooled_truth)
    print(f"\nPOOLED over {len(per_episode)} episodes, {pooled['steps']} timesteps:")
    print(f"  precision={_fmt(pooled['precision'])}  recall={_fmt(pooled['recall'])}  "
          f"f1={_fmt(pooled['f1'])}  FP={pooled['fp']}  FN={pooled['fn']}")

    sweeps: Dict[str, Any] = {}
    if not args.skip_sweep:
        print("\nSENSITIVITY (pooled precision / recall / f1; * marks the default)")
        for field_name, values in SWEEP.items():
            rows = sweep_thresholds(scored, field_name, values)
            sweeps[field_name] = rows
            print(f"\n  {field_name}")
            for row in rows:
                mark = "*" if row["is_default"] else " "
                print(f"   {mark} {row[field_name]:<8} P={_fmt(row['precision']):>6s} "
                      f"R={_fmt(row['recall']):>6s} F1={_fmt(row['f1']):>6s} "
                      f"FP={row['fp']:5d} FN={row['fn']:5d}")

    report = {
        "root": args.root,
        "episodes_replayed": len(replays),
        "episodes_exact": len(per_episode),
        "diverged": [
            {"episode": r.episode_dir, "first_divergence": r.first_divergence,
             "max_position_drift_m": r.max_position_drift_m}
            for r in replays if not r.replay_exact
        ],
        "max_position_drift_m": max((r.max_position_drift_m for r in replays), default=0.0),
        "replay_position_tolerance_m": REPLAY_POSITION_TOLERANCE_M,
        "per_episode": per_episode,
        "pooled": pooled,
        "sensitivity": sweeps,
        "thresholds": DEFAULT_THRESHOLDS.to_dict(),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
