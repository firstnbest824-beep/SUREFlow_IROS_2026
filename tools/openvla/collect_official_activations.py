"""Per-timestep activation collection under official LIBERO / LIBERO-PRO conditions.

Suite-independent. The suite name selects a checkpoint and a set of available
conditions; it never decides which entity moved. That is measured per episode by
diffing a vanilla reset against a perturbed reset, so a task whose perturbation
profile differs from its neighbours is labelled correctly instead of inheriting
the suite's average behaviour.

Timestep contract
-----------------
Exactly this order, enforced by ``_collect_one_timestep`` rather than by comment::

    obs_t  ->  label_t  ->  model forward / activation_t  ->  action_t  ->  env.step(action_t)

Every stored field at index ``t`` therefore describes the *same* simulator state:
the one that existed before ``action_t`` was applied. Nothing is taken from
``obs_{t+1}``, and no post-step phase timeline is reused. The record carries
``sim_state_sha`` (state observed) and ``next_sim_state_sha`` (state after the
step) so a validator can prove the chain closes rather than trust the comment.

Segmentation is analysis ground truth only -- see ``segmentation_label_provider``.
The policy receives exactly the RGB the official evaluation gives it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from activation_episode_writer import (  # noqa: E402
    REQUIRED_STAGES,
    ActivationEpisodeWriter,
    episode_dir_for,
)
from changed_entity_detector import (  # noqa: E402
    detect_changed_entities,
    extract_object_poses,
    is_clean,
    summarise,
)
from entity_role_resolver import resolve_entity_roles_from_path  # noqa: E402
from init_state_freezer import resolve_init_state  # noqa: E402
from model_input_transform import get_libero_image  # noqa: E402
from official_task_pair_resolver import (  # noqa: E402
    assert_checkpoint_matches_suite,
    resolve_official_task_pair,
    seed_everything,
    validate_seed,
)
from probe_hooks import ProbeHookManager, SINGLE_CALL_STAGES  # noqa: E402
from run_single_vanilla_rollout import (  # noqa: E402
    ACTION_DIM,
    get_libero_dummy_action,
    get_vla_action,
    invert_gripper_action,
    load_openvla,
    normalize_gripper_action,
    quat2axisangle,
)
from segmentation_label_provider import (  # noqa: E402
    DEFAULT_CAMERAS,
    POLICY_CAMERA,
    SegmentationLabelProvider,
    build_overlay,
    raw_to_policy_image,
    verify_segmentation_equivalence,
)
from task_phase_resolver import (  # noqa: E402
    DEFAULT_THRESHOLDS as PHASE_DEFAULT_THRESHOLDS,
    TaskPhaseResolver,
    compute_frame_inputs,
    phase_result_to_timeline_entry,
)

DEFAULT_MAX_STEPS = 220
DEFAULT_NUM_STEPS_WAIT = 10
DEFAULT_RESOLUTION = 256
DEFAULT_RESIZE_SIZE = 224
#: Optional 9th stage; the eight in REQUIRED_STAGES are always saved.
OPTIONAL_STAGE_LM_HEAD = "lm_head_logits"


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def make_env(bddl_path: str, resolution: int) -> Any:
    """Build the env the official evaluation builds, from an explicit BDDL path.

    ``env.seed(0)`` mirrors ``get_libero_env``. Re-seeding it with the run seed
    was measured to change the trajectory away from the validated baseline, so
    the run seed governs numpy/torch only.
    """
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution
    )
    env.seed(0)
    return env


def _poses_under_applied_init(
    bddl: str, roles: Any, resolution: int, suite: str, condition: str,
    task_id: int, task_name: str, seed: int, init_state_id: int,
) -> Tuple[Dict[str, Any], Any]:
    """Poses of the state the rollout will actually start from.

    Resetting is not enough. LIBERO samples object placements inside their
    regions on every reset, and the episode then overwrites that with a pinned
    init state -- the official ``.pruned_init`` where one ships, a locally frozen
    capture otherwise. Diffing bare resets would therefore describe a state no
    episode ever runs, and would fold per-reset placement jitter of the untouched
    objects into the perturbation's measured effect.
    """
    env = make_env(bddl, resolution)
    try:
        env.reset()
        state, record = resolve_init_state(
            env=env, bddl_path=bddl, suite=suite, condition=condition,
            task_id=task_id, task_name=task_name, seed=seed,
            init_state_id=init_state_id,
        )
        env.set_init_state(state)
        return extract_object_poses(env, roles.tracked_entities), record
    finally:
        try:
            env.close()
        except Exception:
            pass


def measure_change(
    vanilla_bddl: str,
    perturbed_bddl: Optional[str],
    roles: Any,
    resolution: int,
    suite: str,
    condition: str,
    task_id: int,
    task_name: str,
    seed: int,
    init_state_id: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Diff the two initial states the episodes actually start from."""
    vanilla_poses, vanilla_init = _poses_under_applied_init(
        vanilla_bddl, roles, resolution, suite, "vanilla", task_id, task_name,
        seed, init_state_id,
    )

    if perturbed_bddl is None:
        perturbed_poses, perturbed_init = vanilla_poses, vanilla_init
    else:
        perturbed_poses, perturbed_init = _poses_under_applied_init(
            perturbed_bddl, roles, resolution, suite, condition, task_id, task_name,
            seed, init_state_id,
        )

    report = detect_changed_entities(vanilla_poses, perturbed_poses, roles)
    poses = {
        "vanilla": {k: v.to_dict() for k, v in vanilla_poses.items()},
        "perturbed": {k: v.to_dict() for k, v in perturbed_poses.items()},
        "vanilla_init_state": vanilla_init.to_dict(),
        "perturbed_init_state": perturbed_init.to_dict(),
        # Flagged, not hidden: when the two conditions draw their init state from
        # different sources, objects the perturbation never touched can still
        # differ, and the change report above is what quantifies that.
        "init_state_sources_match": vanilla_init.source == perturbed_init.source,
    }
    return report, poses


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------
def streams_to_arrays(
    hook_manager: ProbeHookManager, saved_stages: Sequence[str]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Collect one timestep's hook output into arrays for the npz bundle.

    Multi-call stages fire once per generated action token. The prefill call
    holds the full sequence (including the 256 visual tokens) and is saved whole;
    the remaining calls are reduced to their last-token vectors, which is the
    action-generation trajectory without paying 7x the bytes for it.
    """
    arrays: Dict[str, np.ndarray] = {}
    problems: List[str] = []

    for stream in hook_manager.streams.values():
        stage = stream.functional_stage
        if stage not in saved_stages:
            continue
        if not stream.tensors:
            problems.append(f"{stage}: hook produced no tensor")
            continue
        arrays[stage] = stream.tensors[0]
        if stage not in SINGLE_CALL_STAGES and len(stream.tensors) > 1:
            rows = []
            for tensor in stream.tensors:
                array = tensor.detach().to("cpu").to(torch.float32).numpy()
                rows.append(array[0, -1, :] if array.ndim == 3 else array[-1, :])
            arrays[f"{stage}__last_token_stack"] = np.stack(rows, axis=0)

    for stage in saved_stages:
        if stage not in arrays:
            problems.append(f"{stage}: missing from this timestep")
    return arrays, problems


# -----------------------------------------------------------------------------
# Episode
# -----------------------------------------------------------------------------
@dataclass
class EpisodeContext:
    vla: Any
    processor: Any
    env: Any
    pair: Any
    roles: Any
    change_report: Any
    initial_poses: Dict[str, Any]
    provider: SegmentationLabelProvider
    init_state: np.ndarray
    init_record: Any
    saved_stages: List[str]
    args: argparse.Namespace
    dtype: torch.dtype


def collect_episode(context: EpisodeContext, episode_index: int, final_dir: Path) -> Dict[str, Any]:
    args = context.args
    env = context.env
    roles = context.roles

    hook_manager = ProbeHookManager(context.vla)
    if hook_manager.missing:
        raise RuntimeError(f"probe hooks could not be registered: {hook_manager.missing}")

    phase_resolver = TaskPhaseResolver(
        roles.source, roles.destination, thresholds=PHASE_DEFAULT_THRESHOLDS
    )
    role_map = {name: roles.role_of(name) for name in roles.tracked_entities}
    tracked = list(roles.tracked_entities)
    for entity in context.change_report.changed_entities:
        if entity.name not in tracked:
            tracked.append(entity.name)
            role_map[entity.name] = entity.role

    manifest = build_manifest(context, episode_index, tracked, role_map)
    frames: List[np.ndarray] = []
    violations: List[str] = []

    env.reset()
    obs = env.set_init_state(context.init_state)
    context.provider.refresh_bindings()

    equivalence = verify_segmentation_equivalence(
        env, height=args.resolution, width=args.resolution
    )
    manifest["segmentation_equivalence"] = equivalence
    if not equivalence["passed"]:
        raise RuntimeError(
            f"segmentation render perturbed the rollout: {json.dumps(equivalence)}"
        )

    dummy = get_libero_dummy_action()
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(dummy)

    success_flag: Optional[bool] = None
    termination = "max_steps"

    with ActivationEpisodeWriter(final_dir, manifest) as writer:
        for timestep in range(args.max_steps):
            record, obs, done, success_flag = _collect_one_timestep(
                context=context,
                writer=writer,
                hook_manager=hook_manager,
                phase_resolver=phase_resolver,
                obs=obs,
                timestep=timestep,
                tracked=tracked,
                role_map=role_map,
                frames=frames,
                violations=violations,
            )
            if done:
                termination = "success" if success_flag else "done_without_success"
                break

        writer.set_result(
            success=bool(success_flag),
            termination_reason=termination,
            violations=violations,
            video_path=writer.add_video(frames) if args.save_video else None,
        )
        steps = writer.steps_written

    return {
        "episode_index": episode_index,
        "episode_dir": str(final_dir),
        "num_timesteps": steps,
        "success": bool(success_flag),
        "termination_reason": termination,
        "violations": violations,
    }


def _collect_one_timestep(
    context: EpisodeContext,
    writer: ActivationEpisodeWriter,
    hook_manager: ProbeHookManager,
    phase_resolver: TaskPhaseResolver,
    obs: Dict[str, Any],
    timestep: int,
    tracked: List[str],
    role_map: Dict[str, str],
    frames: List[np.ndarray],
    violations: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Optional[bool]]:
    """One timestep, in the one order that keeps labels and activations aligned.

    The function body *is* the contract: obs, then labels, then forward, then
    action, and only at the very end ``env.step``. Nothing reads ``obs`` again
    after the step.
    """
    args = context.args
    env = context.env
    # Resolved fresh: env.reset() frees the previous MjSim.
    sim = context.provider.sim

    # ---- 1. obs_t -----------------------------------------------------------
    sim_state_sha = sha256_array(np.asarray(sim.get_state().flatten()))
    agentview_raw = np.asarray(obs["agentview_image"])
    eye_raw = np.asarray(obs["robot0_eye_in_hand_image"])
    agentview_policy = raw_to_policy_image(agentview_raw)
    eye_policy = raw_to_policy_image(eye_raw)

    # ---- 2. label_t (same simulator state, no stepping) ---------------------
    frame_inputs = compute_frame_inputs(env, obs, context.roles.source, context.roles.destination)
    phase_result = phase_resolver.update(timestep=timestep, **frame_inputs)
    segmentation = context.provider.labels_at_current_state(tracked)
    entity_poses = {
        name: (None if (xyz := context.provider.entity_world_xyz(name)) is None
               else [float(v) for v in xyz])
        for name in tracked
    }

    # ---- 3. activation_t ----------------------------------------------------
    proprio = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    model_obs = {"full_image": get_libero_image(obs, args.resize_size), "state": proprio}

    hook_manager.reset()
    started = time.time()
    action_model = get_vla_action(
        vla=context.vla,
        processor=context.processor,
        base_vla_name=context.pair.model_id,
        obs=model_obs,
        task_label=context.pair.instruction,
        unnorm_key=context.pair.unnorm_key,
        center_crop=args.center_crop,
        dtype=context.dtype,
    )
    latency_ms = (time.time() - started) * 1000.0

    activations, problems = streams_to_arrays(hook_manager, context.saved_stages)
    for problem in problems:
        violations.append(f"t={timestep}: {problem}")
    if problems and args.fail_fast:
        raise RuntimeError(violations[-1])

    # ---- 4. action_t --------------------------------------------------------
    if not np.isfinite(action_model).all():
        violations.append(f"t={timestep}: non-finite action {action_model}")
        if args.fail_fast:
            raise RuntimeError(violations[-1])
    action_applied = invert_gripper_action(
        normalize_gripper_action(action_model.copy(), binarize=True)
    )

    overlays = {}
    if args.save_overlays and timestep % max(1, args.overlay_every_n_steps) == 0:
        overlays["agentview"] = build_overlay(
            agentview_policy, segmentation, POLICY_CAMERA, role_map
        )
    if args.save_video:
        frames.append(agentview_policy)

    # ---- 5. env.step(action_t) ---------------------------------------------
    obs_next, _, done, info = env.step(action_applied.tolist())
    next_sim_state_sha = sha256_array(np.asarray(sim.get_state().flatten()))

    success_flag: Optional[bool] = None
    if isinstance(info, dict) and "success" in info:
        success_flag = bool(info["success"])
    else:
        checker = getattr(env, "check_success", None)
        if callable(checker):
            try:
                success_flag = bool(checker())
            except Exception:
                success_flag = None

    metrics = {
        "sim_state_sha": sim_state_sha,
        "next_sim_state_sha": next_sim_state_sha,
        "phase": phase_result.phase,
        "relevant_entity": phase_result.relevant_entity,
        "relevant_entity_role": phase_result.relevant_entity_role,
        "grasp_detected": phase_result.grasp_detected,
        "grasp_confidence": phase_result.grasp_confidence,
        "contact": phase_result.contact,
        "source_to_gripper_distance": phase_result.source_to_gripper_distance,
        "phase_timeline_entry": phase_result_to_timeline_entry(phase_result),
        "action_model": [float(v) for v in action_model],
        "action_applied": [float(v) for v in action_applied],
        "proprio_state": [float(v) for v in proprio],
        "eef_pos": [float(v) for v in obs["robot0_eef_pos"]],
        "gripper_qpos": [float(v) for v in obs["robot0_gripper_qpos"]],
        "entity_world_xyz": entity_poses,
        "segmentation": segmentation.metrics(),
        "success": success_flag,
        "done": bool(done),
        "latency_ms": latency_ms,
    }

    writer.write_step(
        timestep=timestep,
        activations=activations,
        metrics=metrics,
        agentview_rgb=agentview_policy,
        eye_in_hand_rgb=eye_policy,
        overlays=overlays,
    )
    return metrics, obs_next, bool(done), success_flag


def build_manifest(
    context: EpisodeContext,
    episode_index: int,
    tracked: List[str],
    role_map: Dict[str, str],
) -> Dict[str, Any]:
    args = context.args
    pair = context.pair
    report = context.change_report

    primary = next(
        (
            entity for entity in report.changed_entities
            if entity.name == getattr(pair, f"{pair.primary_analysis_entity}_entity", None)
        ),
        None,
    )
    measured = None if primary is None else primary.translation_norm

    return {
        "episode_index": episode_index,
        "suite": pair.suite,
        "condition": pair.perturbation_name,
        "perturbation_family": pair.perturbation_family,
        "task_id": pair.task_id,
        "task_name": pair.task_name,
        "instruction": pair.instruction,
        "seed": pair.seed,
        "checkpoint": {
            "model_id": pair.model_id,
            "revision": pair.model_revision,
            "unnorm_key": pair.unnorm_key,
        },
        "bddl_path": pair.perturbed_bddl_path or pair.vanilla_bddl_path,
        "bddl_sha256": pair.perturbed_bddl_sha256 or pair.vanilla_bddl_sha256,
        "vanilla_bddl_path": pair.vanilla_bddl_path,
        "vanilla_bddl_sha256": pair.vanilla_bddl_sha256,
        "init_state_sha256": context.init_record.sha256,
        "init_state_source": context.init_record.source,
        "init_state_path": context.init_record.path,
        "init_state_id": context.init_record.init_state_id,
        # "x0.1" is a level, not a displacement. Both are stored, never conflated.
        "requested_axis": pair.requested_axis,
        "requested_level": pair.requested_level,
        "measured_translation_m": measured,
        "entity_roles": context.roles.to_dict(),
        "change_report": report.to_dict(),
        "initial_pose_measurement": context.initial_poses,
        "change_class": report.change_class,
        "is_clean_condition": is_clean(report.change_class),
        "tracked_entities": tracked,
        "entity_role_map": role_map,
        "cameras": list(DEFAULT_CAMERAS),
        "policy_camera": POLICY_CAMERA,
        "resolution": args.resolution,
        "resize_size": args.resize_size,
        "center_crop": args.center_crop,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps,
        "phase_thresholds": PHASE_DEFAULT_THRESHOLDS.to_dict(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "segmentation_role": "analysis ground truth only; never a policy input",
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--condition", required=True, help="vanilla | swap | x0.1 | y0.1 | ...")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--num_steps_wait", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--resize_size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    parser.add_argument(
        "--save_lm_head_logits", action="store_true",
        help="also store the 32064-way logits (large); off by default",
    )
    parser.add_argument("--save_overlays", action="store_true", default=True)
    parser.add_argument("--no_overlays", dest="save_overlays", action="store_false")
    parser.add_argument("--overlay_every_n_steps", type=int, default=10)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="save_video", action="store_false")
    parser.add_argument("--fail_fast", action="store_true", default=True)
    parser.add_argument("--no_fail_fast", dest="fail_fast", action="store_false")
    parser.add_argument("--require_clean_condition", action="store_true",
                        help="abort unless the measured change is a single-role change")
    parser.add_argument("--dry_run", action="store_true",
                        help="resolve, measure the change and report; collect nothing")
    parser.add_argument("--device", default="cuda:0")
    # "eager" matches the validated baseline runners; flash_attn is not installed
    # in this environment and swapping attention kernels would change the numbers
    # the whole analysis is built on.
    parser.add_argument("--attn_implementation", default="eager")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seed = validate_seed(args.seed)
    seed_everything(seed)

    pair = resolve_official_task_pair(
        suite=args.suite, task_id=args.task_id, condition=args.condition, seed=seed
    )
    assert_checkpoint_matches_suite(args.suite, pair.model_id)

    roles = resolve_entity_roles_from_path(pair.perturbed_bddl_path or pair.vanilla_bddl_path)
    report, poses = measure_change(
        vanilla_bddl=pair.vanilla_bddl_path,
        perturbed_bddl=pair.perturbed_bddl_path,
        roles=roles,
        resolution=args.resolution,
        suite=pair.suite,
        condition=pair.perturbation_name,
        task_id=pair.task_id,
        task_name=pair.task_name,
        seed=seed,
        init_state_id=args.init_state_id,
    )

    print(f"suite/condition : {pair.suite} / {pair.perturbation_name} ({pair.perturbation_family})")
    print(f"task {pair.task_id}      : {pair.instruction}")
    print(f"checkpoint      : {pair.model_id}@{pair.model_revision[:12]}")
    print(f"bddl            : {pair.perturbed_bddl_path or pair.vanilla_bddl_path}")
    if pair.perturbation_family == "vanilla":
        print(f"change_class      : {report.change_class}  (baseline, no perturbation expected)")
    else:
        print(summarise(report))
    if pair.requested_level is not None:
        moved = next((c for c in report.changed_entities if c.name == roles.source), None)
        print(f"requested_level={pair.requested_level}  "
              f"measured_translation_m={None if moved is None else round(moved.translation_norm, 4)}")

    if args.require_clean_condition and not is_clean(report.change_class):
        print(f"[ABORT] change_class={report.change_class} is not a single-role change")
        return 2
    if args.dry_run:
        print("[dry-run] nothing collected")
        return 0

    env = None
    try:
        processor, vla = load_openvla(
            checkpoint_id=pair.model_id, revision=pair.model_revision,
            attn_implementation=args.attn_implementation, dtype=torch.bfloat16,
            device=args.device,
        )
        if vla.get_action_dim(pair.unnorm_key) != ACTION_DIM:
            raise RuntimeError("unexpected action dim for this checkpoint")

        bddl = pair.perturbed_bddl_path or pair.vanilla_bddl_path
        env = make_env(bddl, args.resolution)
        env.reset()
        init_state, init_record = resolve_init_state(
            env=env, bddl_path=bddl, suite=pair.suite, condition=pair.perturbation_name,
            task_id=pair.task_id, task_name=pair.task_name, seed=seed,
            init_state_id=args.init_state_id, resolution=args.resolution,
        )
        print(f"init state      : {init_record.source} sha={init_record.sha256[:12]} "
              f"({init_record.num_available} available)")

        saved_stages = list(REQUIRED_STAGES)
        if args.save_lm_head_logits:
            saved_stages.append(OPTIONAL_STAGE_LM_HEAD)

        context = EpisodeContext(
            vla=vla, processor=processor, env=env, pair=pair, roles=roles,
            change_report=report,
            initial_poses=poses,
            provider=SegmentationLabelProvider(
                env, height=args.resolution, width=args.resolution
            ),
            init_state=init_state, init_record=init_record,
            saved_stages=saved_stages, args=args, dtype=torch.bfloat16,
        )

        summaries = []
        for episode_index in range(args.num_episodes):
            # A deterministic policy on a pinned state replays the same episode,
            # so each episode draws its own placement: episode i gets init state
            # i. Both sources support this -- 50 shipped placements for the
            # official files, successive seeded resets for the frozen ones.
            episode_init_id = args.init_state_id + episode_index
            if episode_init_id != context.init_record.init_state_id:
                context.init_state, context.init_record = resolve_init_state(
                    env=env, bddl_path=bddl, suite=pair.suite,
                    condition=pair.perturbation_name, task_id=pair.task_id,
                    task_name=pair.task_name, seed=seed,
                    init_state_id=episode_init_id, resolution=args.resolution,
                )
            final_dir = episode_dir_for(
                args.output_root, pair.suite, pair.perturbation_name,
                pair.task_id, seed, episode_index,
            )
            print(f"\n-- episode {episode_index} (init {episode_init_id}, "
                  f"{context.init_record.source}) -> {final_dir}")
            summaries.append(collect_episode(context, episode_index, final_dir))
            last = summaries[-1]
            print(f"   timesteps={last['num_timesteps']} success={last['success']} "
                  f"termination={last['termination_reason']} "
                  f"violations={len(last['violations'])}")

        out = Path(args.output_root) / "collection_summary.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as handle:
            for entry in summaries:
                handle.write(json.dumps({**entry, "change_class": report.change_class}) + "\n")
        return 0

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
