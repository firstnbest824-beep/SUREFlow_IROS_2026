#!/usr/bin/env python3
"""Per-timestep OpenVLA activation collector (pilot scale).

Runs a real LIBERO rollout and, at every control step, saves the activations the
policy produced together with the observation, action, proprioceptive state and
task phase that belong to that same step.

Timestep semantics -- the thing most likely to be silently wrong
---------------------------------------------------------------
At control step ``k`` the loop does::

    obs_pre  = <observation the policy sees>
    action   = policy(obs_pre)          # <-- hooks fire here
    obs_post = env.step(action)

Everything stored under timestep ``k`` therefore derives from ``obs_pre``:
the activations, the proprioceptive vector handed to the model, the action, and
**the task-phase label**. ``obs_post`` is recorded separately (``*_post`` fields)
so the chain ``obs_post(k) == obs_pre(k+1)`` can be verified.

This differs from ``run_single_vanilla_rollout.py`` / ``run_failure_screening.py``,
which call the phase resolver *after* ``env.step`` and so label step ``k`` with
``obs_post(k)``. For rollout-level questions ("when did the grasp happen") that is
fine; for probe training it would shift every label one control step ahead of the
activation it is paired with. ``activation_integrity.check_alignment`` enforces the
``obs_pre`` convention here.

Scope: pilot verification only. This script does not start a large collection --
run it, read ``dashboard.html`` and ``integrity_report.json``, and only then scale up.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)

import libero  # noqa: E402
from libero.libero import benchmark  # noqa: E402

_TOOLS_OPENVLA = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_OPENVLA not in sys.path:
    sys.path.insert(0, _TOOLS_OPENVLA)

from run_single_vanilla_rollout import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_CHECKPOINT,
    DEFAULT_REVISION,
    DEFAULT_TASK_SUITE,
    DEFAULT_TASK_ID,
    DEFAULT_INIT_STATE_ID,
    DEFAULT_NUM_STEPS_WAIT,
    DEFAULT_RESOLUTION,
    DEFAULT_RESIZE_SIZE,
    DEFAULT_CENTER_CROP,
    DEFAULT_UNNORM_KEY,
    get_libero_env,
    get_libero_dummy_action,
    get_vla_action,
    invert_gripper_action,
    load_openvla,
    normalize_gripper_action,
    quat2axisangle,
)
from model_input_transform import get_libero_image  # noqa: E402
from probe_hooks import (  # noqa: E402
    FORWARD_PROBE_TARGETS,
    PRE_FORWARD_PROBE_TARGETS,
    SINGLE_CALL_STAGES,
    ProbeHookManager,
)
from spatial_task_resolver import (  # noqa: E402
    DEFAULT_OOD_SPATIAL_CONFIG,
    resolve_spatial_task_from_env,
)
from task_phase_resolver import (  # noqa: E402
    DEFAULT_THRESHOLDS as PHASE_DEFAULT_THRESHOLDS,
    TaskPhaseResolver,
    compute_frame_inputs,
    config_snapshot as phase_resolver_config_snapshot,
    phase_result_to_timeline_entry,
    save_phase_timeline,
)
from activation_integrity import (  # noqa: E402
    FAIL,
    REQUIRED_STAGES,
    STAGE_SPECS,
    build_integrity_report,
    tensor_stats,
)
from activation_storage import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGING_MAX_GB,
    DEFAULT_STAGING_MIN_FREE_GB,
    DEFAULT_STAGING_ROOT,
    DEFAULT_TRANSFER_QUEUE_SIZE,
    MODE_AUTO,
    MODE_DIRECT,
    MODE_STAGED,
    ActivationStorage,
    describe_filesystem,
    free_gb,
)
# Pilot default. Full rollouts are 220 steps; at ~28 MB/timestep that is ~6 GB,
# far more than a wiring check needs.
DEFAULT_PILOT_MAX_STEPS = 30
MIN_FREE_DISK_GB = 10.0


def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_TOOLS_OPENVLA, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def free_disk_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


# -----------------------------------------------------------------------------
# Activation persistence
# -----------------------------------------------------------------------------
def save_stage_tensors(
    stream: Any,
    stage: str,
    stage_dir: Path,
    timestep: int,
    save_dtype: np.dtype,
) -> Dict[str, Any]:
    """Persist one stage's tensors for one timestep and describe them.

    Single-call stages (vision / projector) save the whole tensor. Multi-call
    stages (LLM layers, lm_head) fire once per generated action token; the
    prefill call holds the full sequence including the 256 visual tokens, so that
    is saved in full, plus a ``[num_calls, hidden]`` last-token stack capturing
    the action-generation trajectory. Saving all seven full sequences would be
    ~7x the bytes for the same information.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    if not stream.tensors:
        return {"call_count": 0, "saved": False, "reason": "hook produced no tensor"}

    # Hooks already detach and move to CPU; make that explicit and dtype-safe.
    prefill = stream.tensors[0].detach().to("cpu").to(torch.float32).numpy()
    prefill = prefill.astype(save_dtype, copy=False)

    entry: Dict[str, Any] = {
        "call_count": len(stream.tensors),
        "hook_type": stream.hook_type,
        "module_path": stream.module_path,
        "saved": True,
    }

    primary_path = stage_dir / f"t{timestep:04d}.npy"
    np.save(primary_path, prefill)
    entry["path"] = str(primary_path)
    entry["shape"] = [int(v) for v in prefill.shape]
    entry["dtype"] = str(prefill.dtype)
    entry["stats"] = tensor_stats(prefill)

    if stage not in SINGLE_CALL_STAGES and len(stream.tensors) > 1:
        rows = []
        for tensor in stream.tensors:
            array = tensor.detach().to("cpu").to(torch.float32).numpy()
            rows.append(array[0, -1, :] if array.ndim == 3 else array[-1, :])
        stack = np.stack(rows, axis=0).astype(save_dtype, copy=False)
        stack_path = stage_dir / f"t{timestep:04d}_last_token_stack.npy"
        np.save(stack_path, stack)
        entry["last_token_stack_path"] = str(stack_path)
        entry["last_token_stack_shape"] = [int(v) for v in stack.shape]

    return entry


def verify_stage_shape(stage: str, shape: List[int]) -> Optional[str]:
    """Return an error string if ``shape`` violates the stage spec."""
    spec = STAGE_SPECS.get(stage)
    if spec is None:
        return None
    if len(shape) != spec.ndim:
        return f"{stage}: ndim {len(shape)} != {spec.ndim} (shape={shape})"
    if spec.batch is not None and shape[0] != spec.batch:
        return f"{stage}: batch {shape[0]} != {spec.batch} (shape={shape})"
    if spec.tokens is not None and shape[1] != spec.tokens:
        return f"{stage}: tokens {shape[1]} != {spec.tokens} (shape={shape})"
    if spec.hidden is not None and shape[2] != spec.hidden:
        return f"{stage}: hidden {shape[2]} != {spec.hidden} (shape={shape})"
    return None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVLA per-timestep activation pilot collector")
    parser.add_argument("--checkpoint_id", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    parser.add_argument("--task_suite", type=str, default=DEFAULT_TASK_SUITE)
    parser.add_argument("--task_id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init_state_id", type=int, default=DEFAULT_INIT_STATE_ID)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_PILOT_MAX_STEPS,
                        help=f"pilot cap on control steps (default {DEFAULT_PILOT_MAX_STEPS}; full rollouts use 220)")
    parser.add_argument("--num_steps_wait", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--resize_size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--center_crop", action="store_true", default=DEFAULT_CENTER_CROP)
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    parser.add_argument("--unnorm_key", type=str, default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="final storage root (must NOT be on '/'; default is the 4TB ext4 disk)")
    # --- storage backend -----------------------------------------------------
    parser.add_argument("--storage_mode", type=str, default=MODE_AUTO,
                        choices=[MODE_DIRECT, MODE_STAGED, MODE_AUTO],
                        help="direct: write straight to output_root. staged: write to SSD then "
                             "transfer asynchronously. auto: staged when it is safe and useful, "
                             "otherwise direct with the reason recorded.")
    parser.add_argument("--staging_root", type=str, default=DEFAULT_STAGING_ROOT,
                        help="SSD staging buffer (only used in staged mode)")
    parser.add_argument("--staging_max_gb", type=float, default=DEFAULT_STAGING_MAX_GB,
                        help="max bytes the staging buffer may hold before the collector waits")
    parser.add_argument("--staging_min_free_gb", type=float, default=DEFAULT_STAGING_MIN_FREE_GB,
                        help="free space that must remain on the staging filesystem")
    parser.add_argument("--transfer_queue_size", type=int, default=DEFAULT_TRANSFER_QUEUE_SIZE,
                        help="episodes allowed to wait for transfer before collection pauses")
    parser.add_argument("--keep_staging_on_success", action="store_true", default=False,
                        help="debugging: keep the staging copy after a verified transfer")
    parser.add_argument("--ood_config_path", type=str, default=DEFAULT_OOD_SPATIAL_CONFIG)
    parser.add_argument("--perturbation_seed", type=int, default=0)
    parser.add_argument("--save_lm_head_logits", action="store_true", default=False,
                        help="also persist lm_head_logits (~35 MB/timestep); off by default")
    parser.add_argument("--save_fp16", action="store_true", default=False,
                        help="store activations as float16 (halves size, loses precision). "
                             "The pilot defaults to float32 so precision loss cannot mask a problem.")
    parser.add_argument("--log_every_n_steps", type=int, default=5)
    parser.add_argument("--no_fail_fast", dest="fail_fast", action="store_false", default=True,
                        help="continue past shape/NaN violations instead of aborting")
    parser.add_argument("--min_free_disk_gb", type=float, default=MIN_FREE_DISK_GB)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Episode collection
# -----------------------------------------------------------------------------
def collect_episode(
    vla: Any,
    processor: Any,
    env: Any,
    task: Any,
    task_description: str,
    initial_state: np.ndarray,
    entities: Any,
    episode_dir: Path,
    episode_id: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
    save_dtype: np.dtype,
    saved_stages: List[str],
) -> Dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = episode_dir / "observations"
    act_dir = episode_dir / "activations"
    obs_dir.mkdir(exist_ok=True)
    act_dir.mkdir(exist_ok=True)

    metrics_path = episode_dir / "per_step_metrics.jsonl"
    metrics_file = open(metrics_path, "w", encoding="utf-8", buffering=1)

    metadata: Dict[str, Any] = {
        "episode_id": episode_id,
        "task_name": task.name,
        "task_instruction": task_description,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "seed": args.seed,
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        # The two fields check_alignment enforces.
        "activation_source": "obs_pre",
        "phase_source": "obs_pre",
        "activation_timing": (
            "activations are captured during policy inference on the pre-step "
            "observation, i.e. BEFORE the action is applied to the simulator"
        ),
        "saved_stages": saved_stages,
        "source_object": entities.source_object,
        "destination_object": entities.destination_object,
        "num_timesteps": 0,
        "task_success": None,
        "termination_reason": None,
        "hook_call_notes": {},
    }

    phase_resolver = TaskPhaseResolver(
        entities.source_object, entities.destination_object, thresholds=PHASE_DEFAULT_THRESHOLDS
    )
    phase_timeline: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    violations: List[str] = []
    first_frame: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None

    hook_manager = ProbeHookManager(vla)
    if hook_manager.missing:
        raise RuntimeError(f"probe hooks could not be registered: {hook_manager.missing}")

    try:
        env.reset()
        obs = env.set_init_state(initial_state)

        dummy = get_libero_dummy_action()
        env_step_index = 0
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(dummy)
            env_step_index += 1

        done = False
        success_flag: Optional[bool] = None
        termination = None

        for timestep in range(args.max_steps):
            # ---------- everything below derives from obs_pre ----------
            obs_pre = obs
            step_index_pre = env_step_index

            agentview_pre = np.asarray(obs_pre["agentview_image"])
            eye_pre = np.asarray(obs_pre["robot0_eye_in_hand_image"])
            if first_frame is None:
                first_frame = agentview_pre.copy()
            last_frame = agentview_pre

            agentview_path = obs_dir / f"t{timestep:04d}_agentview.png"
            eye_path = obs_dir / f"t{timestep:04d}_eye_in_hand.png"
            Image.fromarray(agentview_pre.astype(np.uint8)).save(agentview_path)
            Image.fromarray(eye_pre.astype(np.uint8)).save(eye_path)

            # Phase from obs_pre -- the same observation the activations come from.
            frame_inputs = compute_frame_inputs(
                env, obs_pre, entities.source_object, entities.destination_object
            )
            phase_result = phase_resolver.update(timestep=timestep, **frame_inputs)
            phase_timeline.append(phase_result_to_timeline_entry(phase_result))

            proprio = np.concatenate((
                obs_pre["robot0_eef_pos"],
                quat2axisangle(obs_pre["robot0_eef_quat"]),
                obs_pre["robot0_gripper_qpos"],
            ))
            model_obs = {
                "full_image": get_libero_image(obs_pre, args.resize_size),
                "state": proprio,
            }

            hook_manager.reset()
            inference_start = time.time()
            action_model = get_vla_action(
                vla=vla, processor=processor, base_vla_name=args.checkpoint_id,
                obs=model_obs, task_label=task_description, unnorm_key=args.unnorm_key,
                center_crop=args.center_crop, dtype=dtype,
            )
            latency_ms = (time.time() - inference_start) * 1000.0

            if not np.isfinite(action_model).all():
                violations.append(f"t={timestep}: non-finite action {action_model}")
                if args.fail_fast:
                    raise RuntimeError(violations[-1])

            # ---------- persist activations for this timestep ----------
            activations: Dict[str, Any] = {}
            for stream in hook_manager.streams.values():
                stage = stream.functional_stage
                if stage not in saved_stages:
                    continue
                entry = save_stage_tensors(
                    stream, stage, act_dir / stage, timestep, save_dtype
                )
                activations[stage] = entry
                if not entry.get("saved"):
                    violations.append(f"t={timestep}: {stage} produced no tensor")
                    if args.fail_fast:
                        raise RuntimeError(violations[-1])
                    continue
                problem = verify_stage_shape(stage, entry["shape"])
                if problem:
                    violations.append(f"t={timestep}: {problem}")
                    if args.fail_fast:
                        raise RuntimeError(problem)
                stats = entry["stats"]
                if stats["nan_count"] or stats["inf_count"]:
                    msg = (f"t={timestep}: {stage} has {stats['nan_count']} NaN / "
                           f"{stats['inf_count']} Inf")
                    violations.append(msg)
                    if args.fail_fast:
                        raise RuntimeError(msg)

            # ---------- apply the action ----------
            action_applied = invert_gripper_action(
                normalize_gripper_action(action_model.copy(), binarize=True)
            )
            obs, reward, done, info = env.step(action_applied.tolist())
            env_step_index += 1
            obs_post = obs
            step_index_post = env_step_index

            if isinstance(info, dict) and "success" in info:
                success_flag = bool(info["success"])
            else:
                checker = getattr(env, "check_success", None)
                if callable(checker):
                    try:
                        success_flag = bool(checker())
                    except Exception:
                        success_flag = None

            post_inputs = compute_frame_inputs(
                env, obs_post, entities.source_object, entities.destination_object
            )

            row = {
                "episode_id": episode_id,
                "timestep": timestep,
                "task_name": task.name,
                "seed": args.seed,
                "obs_step_index_pre": step_index_pre,
                "obs_step_index_post": step_index_post,
                "task_phase": phase_result.phase,
                "phase_source": "obs_pre",
                "relevant_entity": phase_result.relevant_entity,
                "relevant_entity_role": phase_result.relevant_entity_role,
                "grasp_detected": phase_result.grasp_detected,
                "grasp_confidence": phase_result.grasp_confidence,
                "success": success_flag,
                "done": bool(done),
                "terminal": bool(done),
                "action_model": [float(v) for v in action_model],
                "action_applied": [float(v) for v in action_applied],
                "proprio_state": [float(v) for v in proprio],
                "eef_pos_pre": [float(v) for v in obs_pre["robot0_eef_pos"]],
                "eef_pos_post": [float(v) for v in obs_post["robot0_eef_pos"]],
                "gripper_qpos_pre": [float(v) for v in obs_pre["robot0_gripper_qpos"]],
                "gripper_qpos_post": [float(v) for v in obs_post["robot0_gripper_qpos"]],
                "source_position_pre": frame_inputs.get("source_position"),
                "destination_position_pre": frame_inputs.get("destination_position"),
                "source_position_post": post_inputs.get("source_position"),
                "source_to_gripper_distance": phase_result.source_to_gripper_distance,
                "contact": phase_result.contact,
                "latency_ms": latency_ms,
                "observations": {
                    "agentview": str(agentview_path),
                    "eye_in_hand": str(eye_path),
                },
                "activations": activations,
            }
            rows.append(row)
            metrics_file.write(json.dumps(row) + "\n")

            if timestep % max(1, args.log_every_n_steps) == 0 or timestep == args.max_steps - 1:
                vision = activations.get("final_vision_dinov2", {})
                stats = vision.get("stats", {})
                print(
                    f"[Episode {episode_id}][Step {timestep + 1}/{args.max_steps}] "
                    f"phase={phase_result.phase} "
                    f"action_shape=({len(action_model)},) "
                    f"final_vision_dinov2={tuple(vision.get('shape', []))} "
                    f"norm={stats.get('l2_norm', float('nan')):.2f} "
                    f"nan={stats.get('nan_count', '?')} inf={stats.get('inf_count', '?')} "
                    f"saved={'YES' if vision.get('saved') else 'NO'}"
                )

            if success_flag:
                termination = "success"
                break
            if done:
                termination = "env_done_without_success"
                break
        else:
            termination = "pilot_max_steps_reached"

        metadata["num_timesteps"] = len(rows)
        metadata["task_success"] = bool(success_flag) if success_flag is not None else False
        metadata["termination_reason"] = termination
        metadata["violations"] = violations
        if first_frame is not None and last_frame is not None:
            metadata["first_last_frame_identical"] = bool(np.array_equal(first_frame, last_frame))
        # LLM stages legitimately fire once per generated action token.
        for stage in saved_stages:
            spec = STAGE_SPECS.get(stage)
            if spec and spec.calls_per_timestep > 1:
                metadata["hook_call_notes"][stage] = (
                    f"fires once per generated action token: 1 prefill + "
                    f"{spec.calls_per_timestep - 1} autoregressive steps"
                )
        if termination == "pilot_max_steps_reached" and args.max_steps < 220:
            metadata["count_mismatch_reason"] = None  # counts still agree; noted for clarity

    finally:
        hook_manager.remove_hooks()
        metrics_file.close()

    with open(episode_dir / "episode_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    if phase_timeline:
        save_phase_timeline(phase_timeline, str(episode_dir), thresholds=PHASE_DEFAULT_THRESHOLDS)

    return metadata


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    saved_stages = list(REQUIRED_STAGES)
    if args.save_lm_head_logits:
        saved_stages.append("lm_head_logits")
    save_dtype = np.float16 if args.save_fp16 else np.float32

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    log_section("Pre-flight")
    commit = git_commit_hash()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    run_id = (
        f"{timestamp}__{args.task_suite}_task{args.task_id}"
        f"__seed{args.seed}__rev{args.revision[:8]}__git{commit[:8]}"
    )
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    if run_dir.exists():
        print(f"[FATAL] run_id already exists, refusing to overwrite: {run_dir}")
        return 1

    storage = ActivationStorage(
        run_id=run_id,
        output_root=str(output_root),
        storage_mode=args.storage_mode,
        staging_root=args.staging_root,
        staging_max_gb=args.staging_max_gb,
        staging_min_free_gb=args.staging_min_free_gb,
        transfer_queue_size=args.transfer_queue_size,
        keep_staging_on_success=args.keep_staging_on_success,
    )
    run_dir = Path(storage.final_run_dir)

    per_step_mb = 8.65 + 4.77 * 4  # vision/projector + 4 LLM-family prefills, fp32
    if args.save_fp16:
        per_step_mb /= 2
    if args.save_lm_head_logits:
        per_step_mb += 35.0
    estimated_gb = per_step_mb * args.max_steps * args.num_episodes / 1024.0
    final_free = free_gb(str(output_root))

    print(f"Run dir            : {run_dir}")
    print(f"Task               : {task.name}")
    print(f"Instruction        : {task.language}")
    print(f"Git commit         : {commit}")
    print(f"Checkpoint         : {args.checkpoint_id} @ {args.revision}")
    print(f"Saved stages       : {saved_stages}")
    print(f"Storage dtype      : {save_dtype.__name__}")
    print(f"Estimated size     : {estimated_gb:.2f} GB ({per_step_mb:.1f} MB/timestep x {args.max_steps} steps)")
    print(f"Storage mode       : {args.storage_mode} -> {storage.resolved_mode}")
    print(f"  reason           : {storage.mode_reason}")
    print(f"  final fs         : {storage.final_fs.get('source')} {storage.final_fs.get('fstype')} "
          f"rotational={storage.final_fs.get('rotational')} free={final_free:.1f} GB")
    if storage.resolved_mode == MODE_STAGED:
        print(f"  staging          : {storage.staging_root}")
        print(f"  staging fs       : {storage.staging_fs.get('source')} {storage.staging_fs.get('fstype')} "
              f"rotational={storage.staging_fs.get('rotational')} free={free_gb(args.staging_root):.1f} GB")
        print(f"  staging budget   : max {args.staging_max_gb} GB, keep {args.staging_min_free_gb} GB free, "
              f"queue {args.transfer_queue_size}")

    if final_free - estimated_gb < args.min_free_disk_gb:
        print(f"[FATAL] final storage would drop under {args.min_free_disk_gb} GB free; aborting.")
        storage.close(timeout=5)
        return 1

    # Recover anything an interrupted run left behind before writing new data.
    recovery = storage.recover()
    if any(recovery["found"].values()):
        print(f"Recovery scan      : {recovery['found']}")
        for action in recovery["actions"]:
            print(f"  - {action}")

    run_config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": commit,
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_instruction": task.language,
        "seed": args.seed,
        "init_state_id": args.init_state_id,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "saved_stages": saved_stages,
        "save_dtype": save_dtype.__name__,
        "save_lm_head_logits": args.save_lm_head_logits,
        "estimated_size_gb": estimated_gb,
        "args": vars(args),
        "probe_targets_forward": {p: s for p, (s, _) in FORWARD_PROBE_TARGETS.items()},
        "probe_targets_forward_pre": {p: s for p, (s, _) in PRE_FORWARD_PROBE_TARGETS.items()},
        "phase_resolver_config": phase_resolver_config_snapshot(PHASE_DEFAULT_THRESHOLDS),
        "storage": storage.metadata(),
        "filesystem_free_gb_at_start": {
            "final": final_free,
            "staging": free_gb(args.staging_root) if storage.resolved_mode == MODE_STAGED else None,
            "root": free_gb("/"),
        },
        "recovery_scan": recovery,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "libero_path": list(getattr(libero, "__path__", [])),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False, default=str)

    env = None
    interrupted = False
    try:
        log_section("Loading OpenVLA")
        processor, vla = load_openvla(
            checkpoint_id=args.checkpoint_id, revision=args.revision,
            attn_implementation=args.attn_implementation, dtype=dtype, device=device,
        )
        action_dim = vla.get_action_dim(args.unnorm_key)
        if action_dim != ACTION_DIM:
            raise RuntimeError(f"expected action dim {ACTION_DIM}, got {action_dim}")

        log_section("Creating environment")
        env, task_description = get_libero_env(task, resolution=args.resolution)
        # Do NOT re-seed the env here. `get_libero_env` already calls `env.seed(0)`,
        # and that is exactly what run_single_vanilla_rollout.py and
        # run_failure_screening.py do -- re-seeding with `args.seed` made this
        # collector's trajectory diverge from the validated baseline (the episode
        # stopped reaching its grasp). `args.seed` controls numpy/torch only, as
        # in the baseline runners.
        env.reset()
        env.set_init_state(initial_states[args.init_state_id])
        entities = resolve_spatial_task_from_env(
            env=env, bddl_path=task_suite.get_task_bddl_file_path(args.task_id),
            task_suite=args.task_suite, task_name=task.name,
            ood_config_path=args.ood_config_path, perturbation_seed=args.perturbation_seed,
        )
        print(f"  source={entities.source_object} destination={entities.destination_object}")

        episode_metas = []
        for episode_id in range(args.num_episodes):
            log_section(f"Collecting episode {episode_id}")
            # begin_episode applies backpressure (staging budget / free space /
            # queue depth) BEFORE any byte of this episode is written.
            write_dir = Path(storage.begin_episode(episode_id))
            meta = collect_episode(
                vla=vla, processor=processor, env=env, task=task,
                task_description=task_description,
                initial_state=initial_states[args.init_state_id],
                entities=entities, episode_dir=write_dir, episode_id=episode_id,
                args=args, dtype=dtype, save_dtype=save_dtype, saved_stages=saved_stages,
            )
            # Seal: manifest + .partial -> .ready + enqueue async transfer.
            storage.finish_episode(episode_id, str(write_dir))
            record = storage.records[episode_id]
            meta["storage"] = {
                "mode": storage.resolved_mode,
                "raw_bytes": record.raw_bytes,
                "file_count": record.file_count,
                "write_started_at": record.write_started_at,
                "write_finished_at": record.write_finished_at,
            }
            episode_metas.append(meta)
            print(f"  timesteps={meta['num_timesteps']} success={meta['task_success']} "
                  f"termination={meta['termination_reason']}")

        if storage.resolved_mode == MODE_STAGED:
            log_section("Draining transfer queue")
            if not storage.wait_for_transfers(timeout=7200):
                raise RuntimeError(
                    "transfers did not complete; staging copies preserved: "
                    f"{storage.errors}"
                )
            print("  all episodes transferred and verified")

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPTED] flushing what has been collected so far...")
    except Exception as exc:
        print("\n[ERROR] collection failed:")
        traceback.print_exc()
        with open(run_dir / "collection_error.json", "w", encoding="utf-8") as handle:
            json.dump({"error": str(exc), "traceback": traceback.format_exc()}, handle, indent=2)
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        # Drain and shut down the transfer worker. Staging copies of anything
        # that failed to transfer are deliberately left in place.
        if not storage.close(timeout=7200):
            print(f"[WARN] transfers incomplete; staging preserved at {storage.staging_root}")
            print(f"       errors: {storage.errors}")

    log_section("Integrity checks")
    report = build_integrity_report(str(run_dir))
    with open(run_dir / "integrity_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    for episode in report["episodes"]:
        print(f"  {os.path.basename(episode['episode_dir'])}: {episode['overall']}")
        for check in episode["checks"]:
            print(f"    [{check['status']:7s}] {check['check_id']} {check['name']}: {check['detail'][:100]}")

    total_bytes = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(run_dir) for name in files
    )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "git_commit": commit,
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_name": task.name,
        "task_instruction": task.language,
        "seed": args.seed,
        "num_episodes": len(episode_metas),
        "total_timesteps": sum(m["num_timesteps"] for m in episode_metas),
        "episodes": episode_metas,
        "saved_stages": saved_stages,
        "total_size_bytes": total_bytes,
        "total_size_mb": total_bytes / 1e6,
        "interrupted": interrupted,
        "integrity_overall": report["overall"],
        "storage": storage.metadata(),
        "filesystem_free_gb_at_end": {
            "final": free_gb(str(output_root)),
            "root": free_gb("/"),
        },
    }
    with open(run_dir / "collection_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    log_section("Dashboard")
    try:
        from activation_dashboard import build_dashboard

        dashboard_path = build_dashboard(str(run_dir))
        print(f"  {dashboard_path}")
    except Exception:
        print("  dashboard generation failed:")
        traceback.print_exc()

    log_section("Result")
    print(f"Overall integrity : {report['overall']}")
    print(f"Total size        : {total_bytes / 1e6:.1f} MB")
    print(f"Run directory     : {run_dir}")
    if report["overall"] == FAIL:
        print("Large-scale collection must NOT start: integrity FAILED.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
