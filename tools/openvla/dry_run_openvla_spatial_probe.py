#!/usr/bin/env python3
"""OpenVLA spatial probe dry-run: collect 1-timestep features, labels, and actions.

This script is intentionally independent of SUREFlow. It loads OpenVLA, runs a
single LIBERO-Spatial episode for a small number of stabilization steps, captures
intermediate representations with forward hooks at one timestep, and writes
features, labels, actions, overlays, and metadata to a timestamped directory.

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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

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
    get_libero_image,
    get_libero_dummy_action,
    quat2axisangle,
    load_openvla,
    get_vla_action,
    normalize_gripper_action,
    invert_gripper_action,
)

from dry_run_libero_label_pipeline import (  # type: ignore
    CAMERA_KEY_MAP,
    unwrap_base_env,
    select_single_movable_source,
    target_instance_id,
    find_segmentation_key,
    normalize_segmentation,
    compute_mask_label,
    save_overlay,
    save_rgb,
    get_target_world_pos,
    get_robot_metadata,
    get_camera_metadata,
    strict_json_ready,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "/home/hwkim/env-audit/openvla-spatial-probe-dry-run"

# Module paths discovered by tools/openvla/inspect_openvla_architecture.py.
# Values are (functional_stage, readout_description).
PROBE_TARGETS: Dict[str, tuple[str, str]] = {
    "vision_backbone.featurizer.blocks.23": ("final_vision_dinov2", "final DINOv2 block output"),
    "vision_backbone.fused_featurizer.blocks.26": ("final_vision_siglip", "final SigLIP block output"),
    "projector.fc3": ("projector_penultimate", "projector fc3 output"),
    "projector": ("projector_output", "full projector output"),
    "language_model.model.layers.0": ("llm_early", "first LLM layer hidden state"),
    "language_model.model.layers.15": ("llm_middle", "middle LLM layer hidden state"),
    "language_model.model.layers.31": ("llm_late", "late LLM layer hidden state"),
    "language_model.lm_head": ("lm_head", "language model head logits"),
}


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
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / (1024 * 1024)),
        "reserved_mb": float(torch.cuda.memory_reserved() / (1024 * 1024)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }


def save_tensor(path: Path, tensor: torch.Tensor) -> Dict[str, Any]:
    """Save a tensor as float32 numpy and return shape/dtype metadata."""
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    np.save(path, arr)
    return {
        "shape": list(arr.shape),
        "saved_dtype": "float32",
        "original_dtype": str(tensor.dtype),
        "path": str(path),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
    }


# -----------------------------------------------------------------------------
# Probe hook manager
# -----------------------------------------------------------------------------
class ProbeHookManager:
    def __init__(self, vla: Any):
        self.vla = vla
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.missing: List[str] = []
        self._register_hooks()

    def _hook_fn(self, module_path: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            tensor = output
            if isinstance(output, tuple):
                tensor = output[0]
            if not isinstance(tensor, torch.Tensor):
                self.features[module_path] = None  # type: ignore[assignment]
                return
            self.features[module_path] = tensor.detach().clone()
        return hook

    def _get_module(self, path: str) -> Optional[Any]:
        parts = path.split(".")
        module: Any = self.vla
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _register_hooks(self) -> None:
        for module_path in PROBE_TARGETS:
            module = self._get_module(module_path)
            if module is None:
                self.missing.append(module_path)
                continue
            handle = module.register_forward_hook(self._hook_fn(module_path))
            self.handles.append(handle)

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# -----------------------------------------------------------------------------
# Segmentation env creation
# -----------------------------------------------------------------------------
def get_segmentation_env(task: Any, resolution: int = 256) -> tuple[SegmentationRenderEnv, str]:
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
# Label extraction
# -----------------------------------------------------------------------------
def build_label_record(
    obs: Dict[str, Any],
    output_dir: Path,
    step: int,
    camera: str,
    mapping: Dict[str, str],
    segmentation_key: str,
    target_object: str,
    target_id: int,
    min_mask_pixels: int,
) -> Dict[str, Any]:
    rgb = np.asarray(obs[mapping["obs_rgb"]])
    height, width = rgb.shape[:2]
    segmentation = obs[segmentation_key]
    label = compute_mask_label(segmentation, target_id, (height, width), min_mask_pixels)

    overlay_path = output_dir / f"step_{step:03d}_{camera}_overlay.png"
    rgb_path = output_dir / f"step_{step:03d}_{camera}_rgb.png"
    mask_path = output_dir / f"step_{step:03d}_{camera}_mask.npy"

    save_overlay(overlay_path, rgb, label, camera, target_object, target_id)
    save_rgb(rgb_path, rgb)
    np.save(mask_path, label["mask"].astype(np.uint8))

    record = {key: value for key, value in label.items() if key != "mask"}
    record.update(
        {
            "rgb_shape": list(rgb.shape),
            "segmentation_shape": list(np.asarray(segmentation).shape),
            "segmentation_key": segmentation_key,
            "overlay_path": str(overlay_path),
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path),
        }
    )
    return record


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
    parser.add_argument("--unnorm_key", type=str, default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_mask_pixels", type=int, default=10)
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
        "unnorm_key": args.unnorm_key,
        "dtype": args.dtype,
        "gpu": args.gpu,
        "output_dir": str(output_dir),
        "exception_occurred": False,
        "exception_message": None,
    }

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
        print(f"Task name: {task.name}")
        print(f"Task language: {task.language}")
        print(f"Initial states shape: {initial_states.shape}")

        metadata["task_name"] = task.name
        metadata["task_instruction"] = task.language
        metadata["bddl_path"] = task_suite.get_task_bddl_file_path(args.task_id)

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

        log_section("Observation inspection")
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value).__name__}")

        # Determine camera keys.
        camera_keys = []
        preferred = ["agentview_image", "robot0_eye_in_hand_image"]
        for key in preferred:
            if key in obs and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3 and obs[key].shape[2] == 3:
                camera_keys.append(key)
        for key in sorted(obs.keys()):
            if key not in camera_keys and key.endswith("_image") and isinstance(obs[key], np.ndarray):
                if obs[key].ndim == 3 and obs[key].shape[2] == 3:
                    camera_keys.append(key)
        print(f"Selected camera keys: {camera_keys}")

        log_section("Target object and segmentation keys")
        segmentation_keys = {
            camera: find_segmentation_key(obs, mapping["seg_camera"])
            for camera, mapping in CAMERA_KEY_MAP.items()
        }
        print(f"Segmentation keys: {segmentation_keys}")

        target_object = select_single_movable_source(env)
        target_id = target_instance_id(env, target_object)
        print(f"Target object: {target_object}, segmentation id: {target_id}")
        metadata["target_object"] = target_object
        metadata["target_segmentation_id"] = target_id

        log_section("OpenVLA inference with probes (HOOK ON)")
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
            print("All probe targets registered.")

        inference_start = time.time()
        action_with_hooks = get_vla_action(
            vla=vla,
            processor=processor,
            base_vla_name=args.checkpoint_id,
            obs=observation,
            task_label=task_description,
            unnorm_key=args.unnorm_key,
            center_crop=args.center_crop,
            dtype=dtype,
        )
        latency_with_hooks_ms = (time.time() - inference_start) * 1000.0
        print(f"Action with hooks: {action_with_hooks}")
        print(f"Latency with hooks: {latency_with_hooks_ms:.2f} ms")

        if not np.isfinite(action_with_hooks).all():
            raise RuntimeError(f"Non-finite action with hooks: {action_with_hooks}")

        log_section("OpenVLA inference without probes (HOOK OFF)")
        hook_manager.remove_hooks()
        # Clear any cached feature tensors before running again.
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

        log_section("Extracting labels from the same observation")
        labels = {}
        for camera, mapping in CAMERA_KEY_MAP.items():
            labels[camera] = build_label_record(
                obs=obs,
                output_dir=output_dir,
                step=1,
                camera=camera,
                mapping=mapping,
                segmentation_key=segmentation_keys[camera],
                target_object=target_object,
                target_id=target_id,
                min_mask_pixels=args.min_mask_pixels,
            )
            print(f"  {camera}: visible={labels[camera]['visible']}, "
                  f"count={labels[camera]['mask_pixel_count']}, "
                  f"centroid={labels[camera]['target_uv_pixel']}")

        log_section("Saving features")
        feature_manifest: Dict[str, Any] = {}
        for module_path, (functional_stage, readout_description) in PROBE_TARGETS.items():
            tensor = hook_manager.features.get(module_path)
            if tensor is None:
                feature_manifest[module_path] = {
                    "functional_stage": functional_stage,
                    "captured": False,
                    "reason": "missing_module" if module_path in hook_manager.missing else "no_output",
                }
                continue

            safe_name = module_path.replace(".", "_")
            feature_path = output_dir / f"feature_{safe_name}.npy"
            tensor_meta = save_tensor(feature_path, tensor)

            feature_manifest[module_path] = {
                "functional_stage": functional_stage,
                "readout": readout_description,
                "readout_type": "direct_forward_output",
                "pooling": "none",
                "spatial_or_token": "spatial" if tensor.ndim == 3 and tensor.shape[1] > 1 else "token",
                "captured": True,
                "timestep": 1,
                "batch_index": 0,
                "module_name": module_path,
                **tensor_meta,
            }
            print(f"  {module_path}: shape={tensor_meta['shape']}, dtype={tensor_meta['original_dtype']}")

        # Validate features.
        nan_inf_features = []
        empty_features = []
        for module_path, meta in feature_manifest.items():
            if not meta.get("captured"):
                continue
            arr = np.load(meta["path"])
            if not np.isfinite(arr).all():
                nan_inf_features.append(module_path)
            if arr.size == 0:
                empty_features.append(module_path)

        if nan_inf_features:
            raise RuntimeError(f"NaN/Inf detected in features: {nan_inf_features}")
        if empty_features:
            raise RuntimeError(f"Empty feature arrays: {empty_features}")
        print("Feature validation passed: no NaN/Inf, no empty arrays.")

        log_section("Saving observation metadata")
        observation_metadata: Dict[str, Any] = {
            "task_name": task.name,
            "task_instruction": task.language,
            "condition": "vanilla",
            "episode_id": args.init_state_id,
            "timestep": 1,
            "seed": args.seed,
            "target_object": target_object,
            "target_segmentation_id": target_id,
            "target_world_pos": get_target_world_pos(env, target_object),
            "robot_metadata": get_robot_metadata(obs),
            "camera_metadata": {
                camera: get_camera_metadata(env, mapping["metadata_camera"])
                for camera, mapping in CAMERA_KEY_MAP.items()
            },
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
        print(f"Saved observation_metadata.json")

        log_section("Saving feature manifest")
        manifest_path = output_dir / "feature_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(feature_manifest, f, indent=2)
        print(f"Saved feature_manifest.json")

        log_section("Saving dry-run summary")
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
            "action_identity_pass": metadata["action_identity_pass"],
            "action_identity_max_diff": metadata["action_identity_max_diff"],
            "latency_with_hooks_ms": latency_with_hooks_ms,
            "latency_without_hooks_ms": latency_without_hooks_ms,
            "hook_overhead_ms": metadata["hook_overhead_ms"],
            "features_captured": sum(1 for m in feature_manifest.values() if m.get("captured")),
            "features_total": len(PROBE_TARGETS),
            "features_missing": hook_manager.missing,
            "feature_manifest_path": str(manifest_path),
            "observation_metadata_path": str(obs_meta_path),
            "nan_inf_detected": False,
        }
        summary_path = output_dir / "dry_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved dry_run_summary.json")

        log_section("Closing environment")
        env.close()
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
            env.close()
        except Exception:
            pass
        return 1
    finally:
        metadata_path = output_dir / "dry_run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved: {metadata_path}")
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return 0


if __name__ == "__main__":
    sys.exit(main())
