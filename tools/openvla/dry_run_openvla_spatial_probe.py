#!/usr/bin/env python3
"""OpenVLA spatial probe dry-run: collect 1-timestep features, labels, and actions.

This script is intentionally independent of SUREFlow. It loads OpenVLA, runs a
single LIBERO-Spatial episode for a small number of stabilization steps, captures
intermediate representations with forward / forward-pre hooks at one timestep,
and writes features, labels, actions, overlays, and metadata to a timestamped
directory.

Three things are aligned with the research plan:

1. The analysed objects come from ``spatial_task_resolver`` -- the BDDL goal
   decides ``source_object`` and ``destination_object``, and the LIBERO-PRO swap
   configuration decides ``swap_counterpart``. Nothing is hard-coded.
2. Every spatial label is stored twice: in the raw simulator frame *and* in
   OpenVLA's model-input frame (180-degree rotation, 224x224 resize, center crop
   0.9, resize back), using the shared ``model_input_transform`` chain.
3. The representation stored as ``pre_action_hidden`` is the hidden state
   entering ``lm_head`` (forward-pre hook), captured once per generation call in
   call order. ``lm_head``'s own output is kept separately as vocabulary logits.

Scope: vanilla condition, 1 task, 1 episode, 1 primary timestep (a few extra
stabilization steps are allowed). No model fine-tuning, no checkpoint download,
no destructive operations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Force headless EGL rendering and project-local LIBERO config before importing libero.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)

import libero
import mujoco
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
TOOLS_OPENVLA = TOOLS_DIR / "openvla"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(1, str(TOOLS_DIR))
if str(TOOLS_OPENVLA) not in sys.path:
    sys.path.insert(2, str(TOOLS_OPENVLA))

from run_single_vanilla_rollout import (  # type: ignore
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
    DEFAULT_GPU,
    get_libero_dummy_action,
    quat2axisangle,
    load_openvla,
    get_vla_action,
    normalize_gripper_action,
    invert_gripper_action,
)

from dry_run_libero_label_pipeline import (  # type: ignore
    CAMERA_KEY_MAP,
    find_segmentation_key,
    get_camera_metadata,
    get_robot_metadata,
    save_rgb,
    strict_json_ready,
)

from model_input_transform import (  # type: ignore
    DEFAULT_CROP_SCALE,
    MODEL_INPUT_CAMERA_KEY,
    describe_transform,
    get_libero_image,
    map_uv_raw_to_model_input,
    mask_statistics,
    mask_to_model_input,
    normalize_segmentation,
    rgb_to_model_input,
)

from probe_hooks import (  # type: ignore
    FORWARD_PROBE_TARGETS,
    PRE_FORWARD_PROBE_TARGETS,
    SINGLE_CALL_STAGES,
    ProbeHookManager,
    ProbeStream,
)

from spatial_task_resolver import (  # type: ignore
    DEFAULT_OOD_SPATIAL_CONFIG,
    get_entity_world_position,
    get_entity_world_positions,
    resolve_spatial_task_from_env,
)

from task_phase_resolver import (  # type: ignore
    DEFAULT_THRESHOLDS as PHASE_DEFAULT_THRESHOLDS,
    PHASE_UNCERTAIN,
    TaskPhaseResolver,
    compute_frame_inputs,
    config_snapshot as phase_resolver_config_snapshot,
    phase_result_to_timeline_entry,
    save_phase_timeline,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "/home/hwkim/env-audit/openvla-spatial-probe-dry-run"

# What the token axis of each saved tensor means, recorded in the manifest so a
# downstream probe never has to guess.
TOKEN_DIM_MEANING: Dict[str, str] = {
    "final_vision_dinov2": "DINOv2 patch tokens (includes prefix/CLS tokens)",
    "final_vision_siglip": "SigLIP patch tokens (256 patches, no CLS)",
    "projector_penultimate": "projected visual tokens aligned 1:1 with SigLIP patches",
    "projector_output": "projected visual tokens aligned 1:1 with SigLIP patches",
    "llm_early": "multimodal LLM sequence: BOS + projected visual tokens + text tokens",
    "llm_middle": "multimodal LLM sequence: BOS + projected visual tokens + text tokens",
    "llm_late": "multimodal LLM sequence: BOS + projected visual tokens + text tokens",
    "lm_head_logits": "vocabulary logits per sequence position",
    "pre_action_hidden": "hidden state per sequence position entering lm_head",
}

LAST_TOKEN_STACK_MEANING = (
    "row i = last-token readout of lm_head call i, in generation order "
    "(call 0 is the prompt prefill, later calls are autoregressive steps)"
)

# Roles whose spatial labels are collected. Kept separate on purpose: the
# pre-grasp phase is about the source, the post-grasp phase about the destination.
LABEL_ROLES = ("source", "destination", "swap_counterpart")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def record_gpu_memory() -> Optional[Dict[str, float]]:
    if not torch.cuda.is_available():
        return None
    mb = 1024 * 1024
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / mb),
        "reserved_mb": float(torch.cuda.memory_reserved() / mb),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / mb),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / mb),
    }


def save_array(path: Path, array: np.ndarray, readout: str, token_meaning: str) -> Dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    np.save(path, array)
    return {
        "path": str(path),
        "shape": [int(value) for value in array.shape],
        "saved_dtype": "float32",
        "readout": readout,
        "pooling": "none",
        "token_dim_meaning": token_meaning,
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "has_nan": bool(np.isnan(array).any()),
        "has_inf": bool(np.isinf(array).any()),
    }


# -----------------------------------------------------------------------------
# Segmentation env creation
# -----------------------------------------------------------------------------
def get_segmentation_env(task: Any, resolution: int = 256) -> Tuple[SegmentationRenderEnv, str]:
    """Create LIBERO SegmentationRenderEnv for the given task."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = SegmentationRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


# -----------------------------------------------------------------------------
# Label extraction (raw simulator frame + OpenVLA model-input frame)
# -----------------------------------------------------------------------------
def save_mask_overlay(
    output_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    stats: Dict[str, Any],
    caption: str,
) -> None:
    image = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_array = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    overlay_array[np.asarray(mask).astype(bool)] = (255, 0, 0, 95)
    overlay = Image.fromarray(overlay_array, mode="RGBA")

    composed = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(composed)
    draw.rectangle((0, 0, image.size[0], 24), fill=(0, 0, 0, 170))
    draw.text((4, 5), caption, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    if stats["centroid"] is not None:
        u, v = stats["centroid"]
        x0, y0, x1, y1 = stats["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0, 255), width=2)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 255, 0, 255))
    composed.convert("RGB").save(output_path)


def build_role_label(
    obs: Dict[str, Any],
    output_dir: Path,
    step: int,
    role: str,
    entity: str,
    segmentation_id: int,
    camera: str,
    mapping: Dict[str, str],
    segmentation_key: str,
    min_mask_pixels: int,
    resize_size: int,
    center_crop: bool,
    crop_scale: float,
) -> Dict[str, Any]:
    """Spatial label for one (role, camera) pair in both coordinate frames."""
    rgb = np.asarray(obs[mapping["obs_rgb"]])
    height, width = rgb.shape[:2]
    segmentation = normalize_segmentation(obs[segmentation_key])
    if segmentation.shape[:2] != (height, width):
        raise ValueError(
            f"RGB/segmentation shape mismatch for {camera}: rgb={(height, width)}, "
            f"seg={segmentation.shape[:2]}"
        )

    raw_mask = segmentation == segmentation_id
    raw_stats = mask_statistics(raw_mask, min_mask_pixels)

    prefix = f"step_{step:03d}_{role}_{entity}_{camera}"
    raw_mask_path = output_dir / f"{prefix}_mask_raw.npy"
    raw_overlay_path = output_dir / f"{prefix}_overlay_raw.png"
    np.save(raw_mask_path, raw_mask.astype(np.uint8))
    save_mask_overlay(
        raw_overlay_path,
        rgb,
        raw_mask,
        raw_stats,
        caption=(
            f"RAW {camera} {role}={entity} id={segmentation_id} "
            f"vis={raw_stats['visible']} n={raw_stats['mask_pixel_count']}"
        ),
    )

    record: Dict[str, Any] = {
        "role": role,
        "entity": entity,
        "camera": camera,
        "segmentation_key": segmentation_key,
        "segmentation_id": int(segmentation_id),
        "rgb_shape": [int(value) for value in rgb.shape],
        "segmentation_shape": [int(value) for value in segmentation.shape],
        "feeds_model_input": camera == "agentview",
        # --- raw simulator frame ---
        "visible_raw": raw_stats["visible"],
        "below_min_mask_pixels_raw": raw_stats["below_min_mask_pixels"],
        "min_mask_pixels": raw_stats["min_mask_pixels"],
        "mask_pixel_count_raw": raw_stats["mask_pixel_count"],
        "target_uv_raw": raw_stats["centroid"],
        "target_uv_raw_normalized": raw_stats["centroid_normalized"],
        "bbox_raw": raw_stats["bbox"],
        "mask_raw_path": str(raw_mask_path),
        "overlay_raw_path": str(raw_overlay_path),
        # --- model-input frame (filled in below for the agentview camera) ---
        "visible_model_input": None,
        "below_min_mask_pixels_model_input": None,
        "mask_pixel_count_model_input": None,
        "target_uv_model_input": None,
        "target_uv_model_input_normalized": None,
        "bbox_model_input": None,
        "mask_model_input_path": None,
        "overlay_model_input_path": None,
        "target_uv_model_input_analytic": None,
        "centroid_mask_vs_analytic_l2": None,
        "model_input_transform": None,
    }

    # Only the agentview camera is fed to OpenVLA, so only it has a model-input frame.
    if camera != "agentview":
        record["model_input_note"] = (
            "camera is not part of OpenVLA's input; no model-input frame exists"
        )
        return record

    model_mask = mask_to_model_input(
        raw_mask,
        resize_size=resize_size,
        center_crop=center_crop,
        crop_scale=crop_scale,
    )
    model_stats = mask_statistics(model_mask, min_mask_pixels)
    model_rgb = rgb_to_model_input(
        rgb, resize_size=resize_size, center_crop=center_crop, crop_scale=crop_scale
    )

    model_mask_path = output_dir / f"{prefix}_mask_model_input.npy"
    model_overlay_path = output_dir / f"{prefix}_overlay_model_input.png"
    np.save(model_mask_path, model_mask.astype(np.uint8))
    save_mask_overlay(
        model_overlay_path,
        model_rgb,
        model_mask,
        model_stats,
        caption=(
            f"MODEL-IN {role}={entity} id={segmentation_id} "
            f"vis={model_stats['visible']} n={model_stats['mask_pixel_count']}"
        ),
    )

    analytic_uv = None
    centroid_gap = None
    if raw_stats["centroid"] is not None:
        analytic_uv = list(
            map_uv_raw_to_model_input(
                raw_stats["centroid"][0],
                raw_stats["centroid"][1],
                raw_shape=(height, width),
                resize_size=resize_size,
                center_crop=center_crop,
                crop_scale=crop_scale,
            )
        )
        if model_stats["centroid"] is not None:
            centroid_gap = float(
                np.linalg.norm(np.array(analytic_uv) - np.array(model_stats["centroid"]))
            )

    record.update(
        {
            "visible_model_input": model_stats["visible"],
            "below_min_mask_pixels_model_input": model_stats["below_min_mask_pixels"],
            "mask_pixel_count_model_input": model_stats["mask_pixel_count"],
            "target_uv_model_input": model_stats["centroid"],
            "target_uv_model_input_normalized": model_stats["centroid_normalized"],
            "bbox_model_input": model_stats["bbox"],
            "mask_model_input_path": str(model_mask_path),
            "overlay_model_input_path": str(model_overlay_path),
            "target_uv_model_input_analytic": analytic_uv,
            "centroid_mask_vs_analytic_l2": centroid_gap,
            "model_input_transform": describe_transform(
                raw_shape=rgb.shape[:2],
                resize_size=resize_size,
                center_crop=center_crop,
                crop_scale=crop_scale,
            ),
        }
    )
    return record


# -----------------------------------------------------------------------------
# Feature persistence
# -----------------------------------------------------------------------------
def save_stream(stream: ProbeStream, output_dir: Path, save_lm_head_logits: bool = False) -> Dict[str, Any]:
    """Persist one probe stream and describe it for the feature manifest.

    ``lm_head_logits`` (vocabulary logits) is the one stream whose full prefill
    array is large (~35 MB/timestep). It is skipped by default; the hook still
    runs so ``call_count`` / NaN-Inf validation are unaffected, only the disk
    write is gated on ``save_lm_head_logits``.
    """
    stage = stream.functional_stage
    token_meaning = TOKEN_DIM_MEANING.get(stage, "unspecified")
    entry: Dict[str, Any] = {
        "module_path": stream.module_path,
        "functional_stage": stage,
        "hook_type": stream.hook_type,
        "readout_description": stream.readout_description,
        "is_action_head_input": stage == "pre_action_hidden",
        "call_count": stream.call_count,
        "non_tensor_calls": stream.non_tensor_calls,
        "calls": [record.to_dict() for record in stream.records],
        "prompt_call_indices": [
            record.call_index for record in stream.records if record.call_type == "prompt_prefill"
        ],
        "generation_call_indices": [
            record.call_index
            for record in stream.records
            if record.call_type == "autoregressive_generation"
        ],
        "captured": bool(stream.records),
        "artifacts": {},
    }
    if not stream.records:
        entry["reason"] = "hook never produced a tensor"
        return entry

    if stage == "lm_head_logits" and not save_lm_head_logits:
        entry["reason"] = (
            "lm_head_logits array saving skipped by default (~35 MB/timestep); "
            "pass --save_lm_head_logits to persist it. call_count/shape/NaN-Inf "
            "validation above is unaffected."
        )
        return entry

    safe_stage = stage.replace(".", "_")

    if stage in SINGLE_CALL_STAGES:
        array = stream.prefill_tensor()
        path = output_dir / f"feature_{safe_stage}.npy"
        entry["artifacts"]["full"] = save_array(path, array, "full_forward_output", token_meaning)
        return entry

    prefill = stream.prefill_tensor()
    if prefill is not None and prefill.ndim >= 2 and prefill.shape[1] > 1:
        path = output_dir / f"feature_{safe_stage}_prefill_full.npy"
        entry["artifacts"]["prefill_full_sequence"] = save_array(
            path, prefill, "full_sequence_of_call_0", token_meaning
        )

    stacked = stream.last_token_stack()
    if stacked is not None:
        name = (
            "pre_action_hidden_last_token.npy"
            if stage == "pre_action_hidden"
            else f"feature_{safe_stage}_last_token.npy"
        )
        entry["artifacts"]["last_token_stack"] = save_array(
            output_dir / name, stacked, "last_token_per_call", LAST_TOKEN_STACK_MEANING
        )
    return entry


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenVLA spatial probe dry-run (1 timestep, vanilla)."
    )
    parser.add_argument("--checkpoint_id", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    parser.add_argument("--task_suite", type=str, default=DEFAULT_TASK_SUITE)
    parser.add_argument("--task_id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init_state_id", type=int, default=DEFAULT_INIT_STATE_ID)
    parser.add_argument("--num_steps_wait", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--resize_size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--center_crop", action="store_true", default=DEFAULT_CENTER_CROP)
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    parser.add_argument("--crop_scale", type=float, default=DEFAULT_CROP_SCALE)
    parser.add_argument("--unnorm_key", type=str, default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_mask_pixels", type=int, default=10)
    parser.add_argument(
        "--save_lm_head_logits",
        action="store_true",
        default=False,
        help=(
            "Persist the lm_head_logits prefill array to disk (~35 MB/timestep). "
            "Off by default; vision/projector/LLM-hidden/pre_action_hidden features "
            "are always saved regardless of this flag."
        ),
    )
    parser.add_argument("--ood_config_path", type=str, default=DEFAULT_OOD_SPATIAL_CONFIG)
    parser.add_argument("--perturbation_seed", type=int, default=0)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dry_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "dry_run.log"
    log_file = open(log_path, "w", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class TeeLogger:
        def write(self, message: str) -> None:
            original_stdout.write(message)
            log_file.write(message)
            log_file.flush()

        def flush(self) -> None:
            original_stdout.flush()
            log_file.flush()

        def isatty(self) -> bool:
            return False

    tee = TeeLogger()
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "num_steps_wait": args.num_steps_wait,
        "resolution": args.resolution,
        "resize_size": args.resize_size,
        "center_crop": args.center_crop,
        "crop_scale": args.crop_scale,
        "unnorm_key": args.unnorm_key,
        "save_lm_head_logits": args.save_lm_head_logits,
        "dtype": args.dtype,
        "gpu": args.gpu,
        "output_dir": str(output_dir),
        "exception_occurred": False,
        "exception_message": None,
    }
    env = None

    try:
        log_section("Pre-flight checks")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[args.dtype]

        print(f"Output directory: {output_dir}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Device: {device}, dtype: {dtype}")

        log_section("Loading OpenVLA model")
        model_load_start = time.time()
        processor, vla = load_openvla(
            checkpoint_id=args.checkpoint_id,
            revision=args.revision,
            attn_implementation=args.attn_implementation,
            dtype=dtype,
            device=device,
        )
        metadata["model_load_time_sec"] = time.time() - model_load_start
        print(f"Model loaded in {metadata['model_load_time_sec']:.2f}s")

        action_dim = vla.get_action_dim(args.unnorm_key)
        print(f"Action dimension: {action_dim}")
        if action_dim != ACTION_DIM:
            raise RuntimeError(f"Expected action dim {ACTION_DIM}, got {action_dim}")

        log_section("Loading LIBERO benchmark and task")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        task = task_suite.get_task(args.task_id)
        initial_states = task_suite.get_task_init_states(args.task_id)
        bddl_path = task_suite.get_task_bddl_file_path(args.task_id)
        print(f"Task name: {task.name}")
        print(f"Task language: {task.language}")
        print(f"Initial states shape: {initial_states.shape}")

        metadata["task_name"] = task.name
        metadata["task_instruction"] = task.language
        metadata["bddl_path"] = bddl_path

        log_section("Creating SegmentationRenderEnv")
        env, task_description = get_segmentation_env(task, resolution=args.resolution)
        print("SegmentationRenderEnv created.")

        log_section("Resetting environment")
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state_id])
        print(f"Initial state {args.init_state_id} applied.")

        log_section("Stabilization dummy steps")
        dummy_action = get_libero_dummy_action()
        for i in range(args.num_steps_wait):
            obs, reward, done, info = env.step(dummy_action)
            if done:
                raise RuntimeError(f"Environment terminated during dummy step {i}.")
        print(f"Completed {args.num_steps_wait} dummy steps.")

        log_section("Source / destination / swap-counterpart resolution")
        entities = resolve_spatial_task_from_env(
            env=env,
            bddl_path=bddl_path,
            task_suite=args.task_suite,
            task_name=task.name,
            ood_config_path=args.ood_config_path,
            perturbation_seed=args.perturbation_seed,
        )
        for line in entities.summary_lines():
            print(f"  {line}")
        metadata["spatial_entities"] = entities.to_dict()

        log_section("Observation inspection")
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value).__name__}")

        segmentation_keys = {
            camera: find_segmentation_key(obs, mapping["seg_camera"])
            for camera, mapping in CAMERA_KEY_MAP.items()
        }
        print(f"Segmentation keys: {segmentation_keys}")

        instance_to_id = dict(getattr(env, "instance_to_id", {}))
        role_entities: Dict[str, Optional[str]] = {
            "source": entities.source_object,
            "destination": entities.destination_object,
            "swap_counterpart": entities.swap_counterpart,
        }
        segmentation_ids: Dict[str, Optional[int]] = {}
        for role, entity in role_entities.items():
            if entity is None:
                segmentation_ids[role] = None
                continue
            if entity not in instance_to_id:
                raise KeyError(
                    f"{role} object {entity!r} missing from segmentation instance map. "
                    f"Available: {sorted(instance_to_id)}"
                )
            segmentation_ids[role] = int(instance_to_id[entity])
        print(f"Role -> entity: {role_entities}")
        print(f"Role -> segmentation id: {segmentation_ids}")
        metadata["role_entities"] = role_entities
        metadata["role_segmentation_ids"] = segmentation_ids

        log_section("Task-phase / relevant-entity resolution")
        # A single-timestep dry-run has no rollout history to build streaks from,
        # so this reduces to a one-frame evidence check (see task_phase_resolver's
        # own docstring for what "sustained" evidence normally means across a
        # rollout). The full-rollout runners (run_failure_screening.py,
        # run_single_vanilla_rollout.py) are what exercise the temporal logic.
        phase_resolver = TaskPhaseResolver(
            entities.source_object, entities.destination_object, thresholds=PHASE_DEFAULT_THRESHOLDS
        )
        phase_frame_inputs = compute_frame_inputs(
            env, obs, entities.source_object, entities.destination_object
        )
        phase_result = phase_resolver.update(timestep=1, **phase_frame_inputs)
        print(
            f"  phase={phase_result.phase} relevant_entity={phase_result.relevant_entity} "
            f"role={phase_result.relevant_entity_role} grasp_confidence={phase_result.grasp_confidence:.2f}"
        )
        print(f"  reason: {phase_result.reason}")
        metadata["phase_result"] = phase_result.to_dict()
        metadata["phase_resolver_config"] = phase_resolver_config_snapshot(PHASE_DEFAULT_THRESHOLDS)

        log_section("OpenVLA inference with probes (HOOK ON)")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        img = get_libero_image(obs, args.resize_size)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        hook_manager = ProbeHookManager(vla)
        if hook_manager.missing:
            print(f"[WARN] Missing modules: {hook_manager.missing}")
        else:
            print(
                f"Registered {len(FORWARD_PROBE_TARGETS)} forward hooks and "
                f"{len(PRE_FORWARD_PROBE_TARGETS)} forward-pre hooks."
            )

        inference_start = time.time()
        action_with_hooks, model_input_image = get_vla_action(
            vla=vla,
            processor=processor,
            base_vla_name=args.checkpoint_id,
            obs=observation,
            task_label=task_description,
            unnorm_key=args.unnorm_key,
            center_crop=args.center_crop,
            dtype=dtype,
            return_model_input_image=True,
        )
        latency_with_hooks_ms = (time.time() - inference_start) * 1000.0
        print(f"Action with hooks: {action_with_hooks}")
        print(f"Latency with hooks: {latency_with_hooks_ms:.2f} ms")

        if not np.isfinite(action_with_hooks).all():
            raise RuntimeError(f"Non-finite action with hooks: {action_with_hooks}")

        log_section("Hook call accounting")
        for key, stream in hook_manager.streams.items():
            call_types = [record.call_type for record in stream.records]
            print(
                f"  {stream.functional_stage:24s} hook={stream.hook_type:11s} "
                f"calls={stream.call_count} types={call_types}"
            )
        pre_action_stream = hook_manager.stream_by_stage("pre_action_hidden")
        lm_head_stream = hook_manager.stream_by_stage("lm_head_logits")
        if pre_action_stream is None or not pre_action_stream.records:
            raise RuntimeError("pre_action_hidden was never captured.")
        metadata["lm_head_call_count"] = lm_head_stream.call_count if lm_head_stream else 0
        metadata["pre_action_hidden_call_count"] = pre_action_stream.call_count
        metadata["action_dim"] = int(action_dim)
        metadata["lm_head_calls_equal_action_dim"] = bool(
            metadata["lm_head_call_count"] == int(action_dim)
        )
        print(
            f"Measured lm_head calls: {metadata['lm_head_call_count']} "
            f"(action_dim={action_dim}, not assumed equal)"
        )

        nonfinite = hook_manager.nonfinite_stages()
        if nonfinite:
            raise RuntimeError(f"NaN/Inf detected in captured activations: {nonfinite}")
        print("No NaN/Inf in any captured activation.")

        log_section("OpenVLA inference without probes (HOOK OFF)")
        hook_manager.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_start = time.time()
        action_without_hooks = get_vla_action(
            vla=vla,
            processor=processor,
            base_vla_name=args.checkpoint_id,
            obs=observation,
            task_label=task_description,
            unnorm_key=args.unnorm_key,
            center_crop=args.center_crop,
            dtype=dtype,
        )
        latency_without_hooks_ms = (time.time() - inference_start) * 1000.0
        print(f"Action without hooks: {action_without_hooks}")
        print(f"Latency without hooks: {latency_without_hooks_ms:.2f} ms")

        if not np.isfinite(action_without_hooks).all():
            raise RuntimeError(f"Non-finite action without hooks: {action_without_hooks}")

        log_section("Action identity check")
        action_diff = float(np.max(np.abs(action_with_hooks - action_without_hooks)))
        action_identical = np.allclose(action_with_hooks, action_without_hooks)
        print(f"Max action diff: {action_diff:.6e}")
        print(f"Actions identical (allclose): {action_identical}")
        metadata["action_identity_max_diff"] = action_diff
        metadata["action_identity_pass"] = bool(action_identical)
        metadata["latency_with_hooks_ms"] = latency_with_hooks_ms
        metadata["latency_without_hooks_ms"] = latency_without_hooks_ms
        metadata["hook_overhead_ms"] = latency_with_hooks_ms - latency_without_hooks_ms

        log_section("Post-processing action")
        raw_action = action_without_hooks.copy()
        action = normalize_gripper_action(action_without_hooks.copy(), binarize=True)
        action = invert_gripper_action(action)
        print(f"Raw action: {raw_action}")
        print(f"Final action: {action}")

        log_section("Model-input coordinate verification")
        raw_agentview = np.asarray(obs[MODEL_INPUT_CAMERA_KEY])
        reconstructed = rgb_to_model_input(
            raw_agentview,
            resize_size=args.resize_size,
            center_crop=args.center_crop,
            crop_scale=args.crop_scale,
        )
        model_input_delta = float(
            np.max(np.abs(reconstructed.astype(np.int32) - model_input_image.astype(np.int32)))
        )
        print(f"Shared transform vs. actual OpenVLA input, max abs pixel diff: {model_input_delta}")
        if model_input_delta != 0.0:
            raise RuntimeError(
                "The shared model-input transform does not reproduce the image handed to "
                f"OpenVLA (max abs diff {model_input_delta}). Spatial labels would be "
                "misaligned."
            )
        metadata["model_input_transform_max_pixel_diff"] = model_input_delta
        metadata["model_input_transform_verified"] = True

        model_input_rgb_path = output_dir / "step_001_agentview_rgb_model_input.png"
        raw_rgb_path = output_dir / "step_001_agentview_rgb_raw.png"
        save_rgb(model_input_rgb_path, reconstructed)
        save_rgb(raw_rgb_path, raw_agentview)
        save_rgb(
            output_dir / "step_001_eye_in_hand_rgb_raw.png",
            np.asarray(obs[CAMERA_KEY_MAP["eye_in_hand"]["obs_rgb"]]),
        )
        print(f"Saved model-input RGB: {model_input_rgb_path}")

        log_section("Extracting spatial labels (raw and model-input frames)")
        labels: Dict[str, Dict[str, Any]] = {}
        for role in LABEL_ROLES:
            entity = role_entities.get(role)
            if entity is None:
                labels[role] = {"entity": None, "note": "role not resolved for this task"}
                continue
            labels[role] = {"entity": entity, "cameras": {}}
            for camera, mapping in CAMERA_KEY_MAP.items():
                record = build_role_label(
                    obs=obs,
                    output_dir=output_dir,
                    step=1,
                    role=role,
                    entity=entity,
                    segmentation_id=segmentation_ids[role],
                    camera=camera,
                    mapping=mapping,
                    segmentation_key=segmentation_keys[camera],
                    min_mask_pixels=args.min_mask_pixels,
                    resize_size=args.resize_size,
                    center_crop=args.center_crop,
                    crop_scale=args.crop_scale,
                )
                labels[role]["cameras"][camera] = record
                print(
                    f"  {role:17s} {camera:11s} raw uv={record['target_uv_raw']} "
                    f"n={record['mask_pixel_count_raw']} | model-input uv="
                    f"{record['target_uv_model_input']} n={record['mask_pixel_count_model_input']}"
                )

        log_section("Relevant-entity primary label (phase-gated)")
        relevant_role = phase_result.relevant_entity_role
        if phase_result.phase != PHASE_UNCERTAIN and relevant_role in labels and phase_result.relevant_entity:
            relevant_record = labels[relevant_role]["cameras"]["agentview"]
            relevant_target_label: Dict[str, Any] = {
                "relevant_target_valid": True,
                "invalid_reason": None,
                "phase": phase_result.phase,
                "relevant_target_name": phase_result.relevant_entity,
                "relevant_target_role": relevant_role,
                # Kept as aliases so consumers written against the resolver's own
                # field names keep working.
                "relevant_entity": phase_result.relevant_entity,
                "relevant_entity_role": relevant_role,
                "relevant_target_uv_raw": relevant_record["target_uv_raw"],
                "relevant_target_uv_raw_normalized": relevant_record["target_uv_raw_normalized"],
                "relevant_target_uv_model_input": relevant_record["target_uv_model_input"],
                "relevant_target_uv_model_input_normalized": relevant_record[
                    "target_uv_model_input_normalized"
                ],
                "relevant_target_world_position": get_entity_world_position(
                    env, phase_result.relevant_entity
                ),
                "relevant_target_visible": relevant_record["visible_model_input"],
                "relevant_target_mask_pixel_count": relevant_record["mask_pixel_count_model_input"],
            }
        else:
            relevant_target_label = {
                "relevant_target_valid": False,
                "invalid_reason": (
                    "phase is uncertain: relevant entity not guessed"
                    if phase_result.phase == PHASE_UNCERTAIN
                    else "relevant entity role unresolved for this task"
                ),
                "phase": phase_result.phase,
                "relevant_target_name": None,
                "relevant_target_role": "none",
                "relevant_entity": None,
                "relevant_entity_role": "none",
                "relevant_target_uv_raw": None,
                "relevant_target_uv_raw_normalized": None,
                "relevant_target_uv_model_input": None,
                "relevant_target_uv_model_input_normalized": None,
                "relevant_target_world_position": None,
                "relevant_target_visible": None,
                "relevant_target_mask_pixel_count": None,
            }
        print(f"  relevant_target_label: {relevant_target_label}")

        # Emit the same phase artifacts as the rollout runners. This dry-run has a
        # single timestep, so the "timeline" has one row -- the point is that the
        # file shape is identical to a full rollout's.
        phase_summary = save_phase_timeline(
            [phase_result_to_timeline_entry(phase_result)],
            str(output_dir),
            thresholds=PHASE_DEFAULT_THRESHOLDS,
        )
        metadata["phase_summary"] = phase_summary
        print(f"  phase artifacts written: {phase_summary['phase_summary_path']}")

        log_section("Saving features")
        feature_manifest: Dict[str, Any] = {
            "probe_targets_forward": {
                path: stage for path, (stage, _) in FORWARD_PROBE_TARGETS.items()
            },
            "probe_targets_forward_pre": {
                path: stage for path, (stage, _) in PRE_FORWARD_PROBE_TARGETS.items()
            },
            "missing_modules": hook_manager.missing,
            "streams": {},
        }
        for key, stream in hook_manager.streams.items():
            entry = save_stream(stream, output_dir, save_lm_head_logits=args.save_lm_head_logits)
            feature_manifest["streams"][key] = entry
            saved = {
                name: artifact["shape"] for name, artifact in entry["artifacts"].items()
            }
            print(f"  {stream.functional_stage:24s} calls={entry['call_count']} saved={saved}")

        nan_inf_artifacts = [
            f"{key}:{name}"
            for key, entry in feature_manifest["streams"].items()
            for name, artifact in entry["artifacts"].items()
            if artifact["has_nan"] or artifact["has_inf"]
        ]
        empty_artifacts = [
            f"{key}:{name}"
            for key, entry in feature_manifest["streams"].items()
            for name, artifact in entry["artifacts"].items()
            if 0 in artifact["shape"]
        ]
        if nan_inf_artifacts:
            raise RuntimeError(f"NaN/Inf detected in saved features: {nan_inf_artifacts}")
        if empty_artifacts:
            raise RuntimeError(f"Empty feature arrays: {empty_artifacts}")
        print("Feature validation passed: no NaN/Inf, no empty arrays.")

        log_section("Saving observation metadata")
        world_positions = get_entity_world_positions(env, entities.tracked_entities)
        observation_metadata: Dict[str, Any] = {
            "task_name": task.name,
            "task_instruction": task.language,
            "condition": "vanilla",
            "episode_id": args.init_state_id,
            "timestep": 1,
            "seed": args.seed,
            "spatial_entities": entities.to_dict(),
            "role_entities": role_entities,
            "role_segmentation_ids": segmentation_ids,
            "pre_grasp_relevant_entity": entities.pre_grasp_relevant_entity,
            "post_grasp_relevant_entity": entities.post_grasp_relevant_entity,
            "phase_result": phase_result.to_dict(),
            "phase_resolver_config": metadata["phase_resolver_config"],
            "relevant_target_label": relevant_target_label,
            "entity_world_positions": world_positions,
            "robot_metadata": get_robot_metadata(obs),
            "camera_metadata": {
                camera: get_camera_metadata(env, mapping["metadata_camera"])
                for camera, mapping in CAMERA_KEY_MAP.items()
            },
            "model_input_transform": describe_transform(
                raw_shape=raw_agentview.shape[:2],
                resize_size=args.resize_size,
                center_crop=args.center_crop,
                crop_scale=args.crop_scale,
            ),
            "model_input_transform_max_pixel_diff": model_input_delta,
            "model_input_rgb_path": str(model_input_rgb_path),
            "raw_rgb_path": str(raw_rgb_path),
            "labels": labels,
            "actions": {
                "raw": raw_action.tolist(),
                "final": action.tolist(),
                "with_hooks": action_with_hooks.tolist(),
                "without_hooks": action_without_hooks.tolist(),
                "shape": list(action.shape),
                "dtype": str(action.dtype),
            },
            "latency_ms": latency_without_hooks_ms,
            "gpu_memory_mb": record_gpu_memory(),
        }
        obs_meta_path = output_dir / "observation_metadata.json"
        with open(obs_meta_path, "w", encoding="utf-8") as f:
            json.dump(strict_json_ready(observation_metadata), f, indent=2, allow_nan=False)
        print("Saved observation_metadata.json")

        log_section("Saving feature manifest")
        manifest_path = output_dir / "feature_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(strict_json_ready(feature_manifest), f, indent=2, allow_nan=False)
        print("Saved feature_manifest.json")

        log_section("Saving dry-run summary")
        source_label = labels["source"]["cameras"]["agentview"]
        destination_label = labels["destination"]["cameras"]["agentview"]
        summary = {
            "success": True,
            "output_dir": str(output_dir),
            "checkpoint_id": args.checkpoint_id,
            "revision": args.revision,
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "task_name": task.name,
            "init_state_id": args.init_state_id,
            "timestep": 1,
            "condition": "vanilla",
            "source_object": entities.source_object,
            "destination_object": entities.destination_object,
            "swap_counterpart": entities.swap_counterpart,
            "perturbation_moved_entities": entities.perturbation_moved_entities,
            "phase": phase_result.phase,
            "relevant_entity": phase_result.relevant_entity,
            "relevant_entity_role": phase_result.relevant_entity_role,
            "grasp_confidence": phase_result.grasp_confidence,
            "relevant_target_valid": relevant_target_label["relevant_target_valid"],
            "source_uv_raw": source_label["target_uv_raw"],
            "source_uv_model_input": source_label["target_uv_model_input"],
            "destination_uv_raw": destination_label["target_uv_raw"],
            "destination_uv_model_input": destination_label["target_uv_model_input"],
            "source_overlay_model_input": source_label["overlay_model_input_path"],
            "destination_overlay_model_input": destination_label["overlay_model_input_path"],
            "model_input_transform_max_pixel_diff": model_input_delta,
            "action_identity_pass": metadata["action_identity_pass"],
            "action_identity_max_diff": metadata["action_identity_max_diff"],
            "latency_with_hooks_ms": latency_with_hooks_ms,
            "latency_without_hooks_ms": latency_without_hooks_ms,
            "hook_overhead_ms": metadata["hook_overhead_ms"],
            "lm_head_call_count": metadata["lm_head_call_count"],
            "pre_action_hidden_call_count": metadata["pre_action_hidden_call_count"],
            "action_dim": int(action_dim),
            "streams_captured": sum(
                1 for entry in feature_manifest["streams"].values() if entry["captured"]
            ),
            "streams_total": len(feature_manifest["streams"]),
            "features_missing": hook_manager.missing,
            "feature_manifest_path": str(manifest_path),
            "observation_metadata_path": str(obs_meta_path),
            "nan_inf_detected": False,
            "gpu_memory_mb": record_gpu_memory(),
        }
        summary_path = output_dir / "dry_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(strict_json_ready(summary), f, indent=2, allow_nan=False)
        print("Saved dry_run_summary.json")

        log_section("Closing environment")
        env.close()
        env = None
        print("Environment closed.")

        metadata["dry_run_summary_path"] = str(summary_path)
        metadata["dry_run_success"] = True

    except Exception as exc:
        metadata["exception_occurred"] = True
        metadata["exception_message"] = str(exc)
        metadata["exception_traceback"] = traceback.format_exc()
        print("\n[ERROR] Exception during dry-run:")
        traceback.print_exc()
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        return 1
    finally:
        metadata_path = output_dir / "dry_run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(strict_json_ready(metadata), f, indent=2, ensure_ascii=False)
        print(f"Metadata saved: {metadata_path}")
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return 0


if __name__ == "__main__":
    sys.exit(main())
