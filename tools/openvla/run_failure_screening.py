#!/usr/bin/env python3
"""OpenVLA failure-screening runner: vanilla vs. LIBERO-PRO position perturbation.

This script performs a small smoke test that compares a vanilla OpenVLA rollout
with a position-perturbation rollout on the same LIBERO-Spatial task and initial
state. It does not run large-scale data collection.
"""

from __future__ import print_function

import argparse
import csv
import io
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)

import libero
import mujoco
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Add LIBERO-PRO path so we can import SwapPerturbator without modifying sys.path globally.
_LIBERO_PRO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO-PRO")
)
if _LIBERO_PRO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_PRO_ROOT)

# Sibling helper modules (shared with the probe dry-run).
_OPENVLA_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _OPENVLA_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_TOOLS_DIR)

from perturbation import BDDLParser, SwapPerturbator

from model_input_transform import (
    apply_center_crop,
    get_libero_image,
    pil_jpeg_encode_decode,
    resize_image,
)
from spatial_task_resolver import (
    DEFAULT_OOD_SPATIAL_CONFIG,
    SpatialTaskEntities,
    displacement,
    get_entity_world_positions,
    resolve_spatial_task,
    resolve_spatial_task_from_env,
)
from task_phase_resolver import (
    DEFAULT_THRESHOLDS as PHASE_DEFAULT_THRESHOLDS,
    TaskPhaseResolver,
    compute_frame_inputs,
    config_snapshot as phase_resolver_config_snapshot,
    per_step_phase_fields,
    phase_result_to_timeline_entry,
    save_phase_timeline,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ACTION_DIM = 7
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

DEFAULT_CHECKPOINT = "openvla/openvla-7b-finetuned-libero-spatial"
DEFAULT_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
DEFAULT_TASK_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0
DEFAULT_INIT_STATE_ID = 0
DEFAULT_NUM_STEPS_WAIT = 10
DEFAULT_MAX_STEPS = 220
DEFAULT_RESOLUTION = 256
DEFAULT_RESIZE_SIZE = 224
DEFAULT_CENTER_CROP = True
DEFAULT_UNNORM_KEY = "libero_spatial"
DEFAULT_GPU = 0
DEFAULT_OUTPUT_DIR = "/home/hwkim/env-audit/openvla-failure-screening"
DEFAULT_NUM_PERTURBATION_INITS = 10
DEFAULT_PERTURBATION_SEED = 0


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# Image preprocessing lives in model_input_transform.py (imported above) so that
# RGB frames and segmentation masks share one rotation / resize / crop chain.


# -----------------------------------------------------------------------------
# GPU memory accounting
# -----------------------------------------------------------------------------
def reset_gpu_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_report() -> Optional[Dict[str, float]]:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    mb = 1024 * 1024
    return {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / mb),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / mb),
        "final_allocated_mb": float(torch.cuda.memory_allocated() / mb),
        "final_reserved_mb": float(torch.cuda.memory_reserved() / mb),
    }


def env_success(env: Any, info: Any) -> Tuple[Optional[bool], str]:
    """Official success check: ``info["success"]`` first, then ``env.check_success()``."""
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"]), "info[\"success\"]"
    checker = getattr(env, "check_success", None)
    if callable(checker):
        try:
            return bool(checker()), "env.check_success()"
        except Exception as exc:
            return None, f"env.check_success() raised {exc}"
    return None, "unavailable"


# -----------------------------------------------------------------------------
# Action helpers
# -----------------------------------------------------------------------------
def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action[..., -1] = action[..., -1] * -1.0
    return action


# -----------------------------------------------------------------------------
# OpenVLA loading and action generation
# -----------------------------------------------------------------------------
def load_openvla(
    checkpoint_id: str,
    revision: str,
    attn_implementation: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Any, Any]:
    print(f"[*] Loading processor from {checkpoint_id} @ {revision}")
    processor = AutoProcessor.from_pretrained(
        checkpoint_id,
        revision=revision,
        trust_remote_code=True,
    )
    print(f"[*] Loading model from {checkpoint_id} @ {revision} (dtype={dtype}, attn={attn_implementation})")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_id,
        revision=revision,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    dataset_statistics_path = os.path.join(checkpoint_id, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
        print(f"[*] Loaded dataset_statistics.json; keys: {list(norm_stats.keys())}")
    else:
        print(
            "WARNING: No local dataset_statistics.json found. "
            "This is expected for base checkpoints, not fine-tuned ones."
        )
    return processor, vla


def get_vla_action(
    vla: Any,
    processor: Any,
    base_vla_name: str,
    obs: Dict[str, Any],
    task_label: str,
    unnorm_key: str,
    center_crop: bool,
    dtype: torch.dtype,
) -> np.ndarray:
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        image = apply_center_crop(image)

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"].to(vla.device)
    attention_mask = inputs["attention_mask"].to(vla.device)
    pixel_values = inputs["pixel_values"].to(vla.device, dtype=dtype)
    if input_ids[0, -1].item() != 29871:
        empty_token = torch.tensor([[29871]], dtype=input_ids.dtype, device=vla.device)
        attend_token = torch.tensor([[1]], dtype=attention_mask.dtype, device=vla.device)
        input_ids = torch.cat([input_ids, empty_token], dim=1)
        attention_mask = torch.cat([attention_mask, attend_token], dim=1)

    action = vla.predict_action(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        unnorm_key=unnorm_key,
        do_sample=False,
    )
    return action


# -----------------------------------------------------------------------------
# LIBERO environment helpers
# -----------------------------------------------------------------------------
def get_libero_env(task: Any, resolution: int = 256) -> Tuple[OffScreenRenderEnv, str]:
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_env_from_bddl(bddl_file: str, resolution: int = 256) -> OffScreenRenderEnv:
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def get_libero_dummy_action() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    qw = quat[3]
    if qw > 1.0:
        qw = 1.0
    elif qw < -1.0:
        qw = -1.0
    den = math.sqrt(1.0 - qw * qw)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(qw)) / den


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def add_text_overlay(
    frame: np.ndarray,
    task_name: str,
    step: int,
    total: int,
    action: np.ndarray,
    success: Optional[bool],
    latency_ms: float,
    camera_name: str,
    condition: str,
    phase_result: Optional[Any] = None,
    source_object: Optional[str] = None,
    destination_object: Optional[str] = None,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    color = (255, 255, 255)
    outline = (0, 0, 0)
    lines: List[str] = [
        f"task: {task_name}",
        f"condition: {condition}",
        f"step: {step + 1}/{total}",
        f"cam: {camera_name}",
        f"latency: {latency_ms:.1f}ms",
        f"action[:3]: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]",
    ]
    if success is not None:
        lines.append(f"success: {success}")
    if phase_result is not None:
        dist = phase_result.source_to_gripper_distance
        dist_str = f"{dist:.3f}" if dist is not None else "n/a"
        # Mark which of the two named entities the phase currently selects, so
        # the overlay makes the pre/post-grasp switch visually unambiguous.
        role = phase_result.relevant_entity_role
        src_mark = "*" if role == "source" else " "
        dst_mark = "*" if role == "destination" else " "
        lines.append(f"phase: {phase_result.phase}")
        lines.append(f"{src_mark}src: {source_object or 'n/a'}")
        lines.append(f"{dst_mark}dst: {destination_object or 'n/a'}")
        lines.append(f"relevant: {phase_result.relevant_entity or 'none'} ({role})")
        lines.append(f"grasp_conf={phase_result.grasp_confidence:.2f} contact={phase_result.contact}")
        lines.append(f"src->grip: {dist_str}")

    y0 = 12
    dy = 12
    for i, line in enumerate(lines):
        y = y0 + i * dy
        if y >= h:
            break
        cv2.putText(img, line, (4, y), font, scale, outline, thickness * 2, cv2.LINE_AA)
        cv2.putText(img, line, (4, y), font, scale, color, thickness, cv2.LINE_AA)
    return img


def save_initial_and_final_frames(frames: List[np.ndarray], output_dir: str) -> None:
    if not frames:
        return
    cv2.imwrite(
        os.path.join(output_dir, "initial_frame.png"),
        cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(output_dir, "final_frame.png"),
        cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
    )


def save_contact_sheet(frames: List[np.ndarray], output_dir: str) -> bool:
    if not frames:
        return False
    n = len(frames)
    indices = [0]
    if n > 1:
        indices.append(n // 5)
    if n > 2:
        indices.append(n // 3)
    if n > 3:
        indices.append(2 * n // 3)
    if n > 4:
        indices.append(4 * n // 5)
    if n > 5:
        indices.append(n - 1)
    indices = sorted(list(set(indices)))
    selected = [frames[i] for i in indices]
    n_selected = len(selected)
    cols = (n_selected + 1) // 2
    h, w = selected[0].shape[:2]
    grid = np.zeros((h * 2, w * cols, 3), dtype=np.uint8)
    for idx, frame in enumerate(selected):
        row = idx // cols
        col = idx % cols
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame
    cv2.imwrite(
        os.path.join(output_dir, "contact_sheet.png"),
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
    )
    return True


def save_video_mp4(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def save_video_gif(frames: List[np.ndarray], output_path: str, fps: float = 20.0) -> bool:
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Perturbation helpers
# -----------------------------------------------------------------------------
def load_pruned_init_zip(path: str) -> np.ndarray:
    """Load a .pruned_init file produced by generate_init_states.py (zip+pickle)."""
    with zipfile.ZipFile(path, "r") as zf:
        if "archive/data.pkl" in zf.namelist():
            data = pickle.loads(zf.read("archive/data.pkl"))
        elif "data.pkl" in zf.namelist():
            data = pickle.loads(zf.read("data.pkl"))
        else:
            raise RuntimeError(f"No data.pkl found inside {path}; contents: {zf.namelist()}")
    return np.asarray(data)


def generate_perturbation_init_states(
    bddl_dir: str,
    output_dir: str,
    num_inits: int,
    height: int,
    width: int,
    script_path: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        script_path,
        "--bddl_base_dir", bddl_dir,
        "--output_dir", output_dir,
        "--num_inits", str(num_inits),
        "--height", str(height),
        "--width", str(width),
    ]
    print(f"[PERTURB] Running init-state generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(script_path) or ".")


def apply_position_swap_perturbation(
    original_bddl_path: str,
    task_suite_name: str,
    task_name: str,
    output_bddl_path: str,
    config_path: str,
    seed: int = 0,
) -> str:
    random.seed(seed)
    with open(original_bddl_path, "r", encoding="utf-8") as f:
        content = f.read()
    parser = BDDLParser(content)
    perturbator = SwapPerturbator(parser, config_path)
    perturbed = perturbator.perturb(task_suite_name=task_suite_name, task_name=task_name)
    os.makedirs(os.path.dirname(output_bddl_path), exist_ok=True)
    with open(output_bddl_path, "w", encoding="utf-8") as f:
        f.write(perturbed)
    return perturbed


# -----------------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------------
def run_episode(
    vla: Any,
    processor: Any,
    task: Any,
    task_description: str,
    env: OffScreenRenderEnv,
    initial_state: np.ndarray,
    condition: str,
    output_dir: str,
    args: argparse.Namespace,
    entities: SpatialTaskEntities,
    enable_phase_resolver: bool = True,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "condition": condition,
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_instruction": task_description,
        "init_state_id": args.init_state_id,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps,
        "resolution": args.resolution,
        "resize_size": args.resize_size,
        "center_crop": args.center_crop,
        "unnorm_key": args.unnorm_key,
        "dtype": args.dtype,
        "gpu": args.gpu,
        "reset_success": False,
        "init_state_success": False,
        "rollout_completed": False,
        "rollout_steps": 0,
        # Success accounting: `done` alone is NOT treated as success.
        "done": False,
        "success_flag": None,
        "success_source": None,
        "task_success": False,
        "termination_reason": None,
        "max_steps_reached": False,
        "nan_inf_detected": False,
        "exception_occurred": False,
        "exception_message": None,
        "failure_type": "unclassified",
        # Which objects this episode is about (resolved from the BDDL goal).
        "source_object": entities.source_object,
        "destination_object": entities.destination_object,
        "swap_counterpart": entities.swap_counterpart,
        "pre_grasp_relevant_entity": entities.pre_grasp_relevant_entity,
        "post_grasp_relevant_entity": entities.post_grasp_relevant_entity,
        "perturbation_moved_entities": entities.perturbation_moved_entities,
        "swap_pairs": entities.swap_pairs,
        "tracked_entities": entities.tracked_entities,
        "entity_positions_initial": {},
        "entity_positions_final": {},
        "entity_displacements": {},
        "gpu_memory_mb": None,
        "phase_resolver_enabled": enable_phase_resolver,
        "phase_resolver_config": phase_resolver_config_snapshot(PHASE_DEFAULT_THRESHOLDS),
        "phase_summary": None,
    }

    try:
        log_section(f"Running {condition} episode")
        print(f"Output directory: {output_dir}")
        print(f"  source={entities.source_object} destination={entities.destination_object} "
              f"swap_counterpart={entities.swap_counterpart}")

        reset_gpu_peak_stats()

        obs = env.reset()
        metadata["reset_success"] = True

        obs = env.set_init_state(initial_state)
        metadata["init_state_success"] = True

        # World poses of the resolved entities, not of a hard-coded object name.
        metadata["entity_positions_initial"] = get_entity_world_positions(
            env, entities.tracked_entities
        )
        for entity, position in metadata["entity_positions_initial"].items():
            print(f"  initial pos {entity}: {position}")

        dummy_action = get_libero_dummy_action()
        raw_actions: List[np.ndarray] = []
        final_actions: List[np.ndarray] = []
        eef_trajectory: List[Dict[str, Any]] = []
        replay_images: List[np.ndarray] = []
        latencies_ms: List[float] = []
        per_step_records: List[Dict[str, Any]] = []
        phase_timeline: List[Dict[str, Any]] = []
        phase_resolver = (
            TaskPhaseResolver(entities.source_object, entities.destination_object, thresholds=PHASE_DEFAULT_THRESHOLDS)
            if enable_phase_resolver
            else None
        )
        phase_resolver_seeded = False

        camera_keys = []
        preferred = ["agentview_image", "robot0_eye_in_hand_image"]
        for key in preferred:
            if key in obs and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3 and obs[key].shape[2] == 3:
                camera_keys.append(key)
        for key in sorted(obs.keys()):
            if key not in camera_keys and key.endswith("_image") and isinstance(obs[key], np.ndarray):
                if obs[key].ndim == 3 and obs[key].shape[2] == 3:
                    camera_keys.append(key)
        metadata["camera_names"] = camera_keys

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map[args.dtype]

        done = False
        step_count = 0
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(dummy_action)
                t += 1
                continue

            if phase_resolver is not None and not phase_resolver_seeded:
                # Seed phase 0 from the post-stabilization, pre-action observation
                # so "initial source position" reflects the true episode start.
                seed_inputs = compute_frame_inputs(
                    env, obs, entities.source_object, entities.destination_object
                )
                seed_result = phase_resolver.update(timestep=0, **seed_inputs)
                phase_timeline.append(phase_result_to_timeline_entry(seed_result))
                phase_resolver_seeded = True

            img = get_libero_image(obs, args.resize_size)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            inference_start = time.time()
            action = get_vla_action(
                vla=vla,
                processor=processor,
                base_vla_name=args.checkpoint_id,
                obs=observation,
                task_label=task_description,
                unnorm_key=args.unnorm_key,
                center_crop=args.center_crop,
                dtype=dtype,
            )
            latency_ms = (time.time() - inference_start) * 1000.0
            latencies_ms.append(latency_ms)

            if not np.isfinite(action).all():
                metadata["nan_inf_detected"] = True
                raise RuntimeError(f"Non-finite action at step {t}: {action}")

            raw_actions.append(action.copy())
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            final_actions.append(action.copy())

            obs, reward, done, info = env.step(action.tolist())
            step_count += 1
            t += 1

            eef_trajectory.append({
                "step": t,
                "eef_pos": obs["robot0_eef_pos"].tolist(),
                "eef_quat": obs["robot0_eef_quat"].tolist(),
                "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            })

            phase_result = None
            if phase_resolver is not None:
                frame_inputs = compute_frame_inputs(
                    env, obs, entities.source_object, entities.destination_object
                )
                phase_result = phase_resolver.update(
                    timestep=step_count,
                    gripper_command=float(action[-1]) if action.size else None,
                    **frame_inputs,
                )
                phase_timeline.append(phase_result_to_timeline_entry(phase_result))

            success_flag, success_source = env_success(env, info)
            if camera_keys:
                primary_key = camera_keys[0]
                frame = np.flipud(obs[primary_key])
                labeled = add_text_overlay(
                    frame,
                    task_name=task.name,
                    step=t - args.num_steps_wait - 1,
                    total=args.max_steps,
                    action=action,
                    success=success_flag,
                    latency_ms=latency_ms,
                    camera_name=primary_key,
                    condition=condition,
                    phase_result=phase_result,
                    source_object=entities.source_object,
                    destination_object=entities.destination_object,
                )
                replay_images.append(labeled)

            per_step_records.append(dict(
                {
                    "step": t,
                    "latency_ms": latency_ms,
                    "done": bool(done),
                    "success": success_flag,
                    "success_source": success_source,
                    "raw_action": raw_actions[-1].tolist(),
                    "final_action": final_actions[-1].tolist(),
                },
                **per_step_phase_fields(phase_result),
            ))

            if (t - args.num_steps_wait) % 20 == 0 or (t - args.num_steps_wait) == 1:
                print(
                    f"  step {t - args.num_steps_wait:3d}/{args.max_steps}: "
                    f"latency={latency_ms:.1f}ms, action[:3]=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}], "
                    f"done={done}, success={success_flag}"
                )

            metadata["done"] = bool(done)
            metadata["success_flag"] = success_flag
            metadata["success_source"] = success_source

            # `done` on its own is not success: only the official success check counts.
            if success_flag:
                metadata["task_success"] = True
                metadata["termination_reason"] = "success"
                print(
                    f"Episode succeeded at step {t - args.num_steps_wait} "
                    f"(source={success_source})"
                )
                break
            if done:
                metadata["termination_reason"] = "env_done_without_success"
                print(
                    f"Environment reported done=True at step {t - args.num_steps_wait} "
                    f"but the official success check returned {success_flag}; "
                    "recording as failure."
                )
                break

        metadata["rollout_completed"] = True
        metadata["rollout_steps"] = step_count
        metadata["max_steps_reached"] = bool(step_count >= args.max_steps)
        if metadata["termination_reason"] is None:
            metadata["termination_reason"] = (
                "max_steps_reached" if metadata["max_steps_reached"] else "loop_exhausted"
            )
        if not metadata["task_success"]:
            metadata["failure_type"] = (
                "timeout_no_success" if metadata["max_steps_reached"] else "early_termination"
            )
        print(
            f"Termination: reason={metadata['termination_reason']}, done={metadata['done']}, "
            f"success_flag={metadata['success_flag']}, task_success={metadata['task_success']}, "
            f"max_steps_reached={metadata['max_steps_reached']}"
        )

        # Final world poses and displacements for every resolved entity.
        metadata["entity_positions_final"] = get_entity_world_positions(
            env, entities.tracked_entities
        )
        metadata["entity_displacements"] = {
            entity: displacement(
                metadata["entity_positions_initial"].get(entity),
                metadata["entity_positions_final"].get(entity),
            )
            for entity in entities.tracked_entities
        }
        for entity, delta in metadata["entity_displacements"].items():
            print(f"  displacement {entity}: {delta}")

        metadata["gpu_memory_mb"] = gpu_memory_report()
        if metadata["gpu_memory_mb"]:
            print(
                f"VRAM peak allocated={metadata['gpu_memory_mb']['peak_allocated_mb']:.1f} MB, "
                f"peak reserved={metadata['gpu_memory_mb']['peak_reserved_mb']:.1f} MB"
            )

        # Save actions.
        if raw_actions:
            np.save(os.path.join(output_dir, "actions_raw.npy"), np.stack(raw_actions))
        if final_actions:
            np.save(os.path.join(output_dir, "actions_final.npy"), np.stack(final_actions))

        with open(os.path.join(output_dir, "actions.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + [f"raw_a{i}" for i in range(ACTION_DIM)] + [f"final_a{i}" for i in range(ACTION_DIM)] + ["success"])
            for i, (raw, final) in enumerate(zip(raw_actions, final_actions)):
                writer.writerow([i + 1] + raw.tolist() + final.tolist() + [per_step_records[i]["success"]])

        with open(os.path.join(output_dir, "eef_trajectory.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw", "gripper_qpos"])
            for rec in eef_trajectory:
                writer.writerow([rec["step"]] + rec["eef_pos"] + rec["eef_quat"] + [rec["gripper_qpos"]])

        with open(os.path.join(output_dir, "per_step_metrics.jsonl"), "w", encoding="utf-8") as f:
            for rec in per_step_records:
                f.write(json.dumps(rec) + "\n")

        if phase_timeline:
            log_section(f"Saving phase timeline for {condition}")
            phase_summary = save_phase_timeline(phase_timeline, output_dir, thresholds=PHASE_DEFAULT_THRESHOLDS)
            metadata["phase_summary"] = phase_summary
            print(
                f"Phase counts: {phase_summary['phase_counts']}, "
                f"uncertain_fraction={phase_summary['uncertain_fraction']}, "
                f"first pre_grasp->post_grasp at timestep="
                f"{phase_summary['first_pre_grasp_to_post_grasp_timestep']}"
            )

        if latencies_ms:
            with open(os.path.join(output_dir, "inference_latency.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "latency_ms"])
                for i, lat in enumerate(latencies_ms):
                    writer.writerow([i + 1, lat])
            metadata["avg_latency_ms"] = float(np.mean(latencies_ms))
            metadata["max_latency_ms"] = float(np.max(latencies_ms))
            metadata["min_latency_ms"] = float(np.min(latencies_ms))
            print(f"Latency (ms): mean={metadata['avg_latency_ms']:.2f}, max={metadata['max_latency_ms']:.2f}")

        log_section(f"Saving visual outputs for {condition}")
        save_initial_and_final_frames(replay_images, output_dir)
        save_contact_sheet(replay_images, output_dir)
        mp4_path = os.path.join(output_dir, f"openvla_{condition}_rollout.mp4")
        mp4_ok = save_video_mp4(replay_images, mp4_path, fps=30.0)
        if mp4_ok:
            metadata["output_video_path"] = mp4_path
            print(f"MP4 saved: {mp4_path}")
        else:
            gif_path = os.path.join(output_dir, f"openvla_{condition}_rollout.gif")
            gif_ok = save_video_gif(replay_images, gif_path, fps=20.0)
            if gif_ok:
                metadata["output_gif_path"] = gif_path
                print(f"GIF saved: {gif_path}")
            else:
                print("Video saving failed.")

        env.close()

    except Exception as exc:
        metadata["exception_occurred"] = True
        metadata["exception_message"] = str(exc)
        metadata["exception_traceback"] = traceback.format_exc()
        print(f"\n[ERROR] Exception during {condition} rollout:")
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass

    if metadata["gpu_memory_mb"] is None:
        metadata["gpu_memory_mb"] = gpu_memory_report()

    metadata_path = os.path.join(output_dir, "rollout_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved: {metadata_path}")

    return metadata


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVLA vanilla vs. position-perturbation failure screening")
    parser.add_argument("--checkpoint_id", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    parser.add_argument("--task_suite", type=str, default=DEFAULT_TASK_SUITE)
    parser.add_argument("--task_id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init_state_id", type=int, default=DEFAULT_INIT_STATE_ID)
    parser.add_argument("--num_steps_wait", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
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
    parser.add_argument("--num_perturbation_inits", type=int, default=DEFAULT_NUM_PERTURBATION_INITS)
    parser.add_argument("--perturbation_seed", type=int, default=DEFAULT_PERTURBATION_SEED)
    parser.add_argument("--ood_config_path", type=str, default=DEFAULT_OOD_SPATIAL_CONFIG)
    parser.add_argument("--skip_vanilla", action="store_true", help="Skip vanilla episode")
    parser.add_argument("--skip_perturbation", action="store_true", help="Skip perturbation episode")
    parser.add_argument(
        "--disable_phase_resolver",
        action="store_true",
        help="Skip task-phase / relevant-entity resolution and logging entirely.",
    )
    parser.add_argument(
        "--action_parity_check",
        action="store_true",
        help=(
            "Instead of the normal vanilla/perturbation run, execute a short vanilla "
            "episode twice (phase resolver on vs. off, same seed/init state) and "
            "report the max abs action difference. Used to verify the phase resolver "
            "never changes policy behavior."
        ),
    )
    parser.add_argument("--action_parity_max_steps", type=int, default=15)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # Every invocation writes into its own timestamped run directory so previous
    # results are never overwritten.
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"screening_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "failure_screening.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comparison_metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "gpu": args.gpu,
        "device": str(device),
        "seed": args.seed,
        "perturbation_seed": args.perturbation_seed,
        "run_dir": run_dir,
        "vanilla": None,
        "position_perturbation": None,
    }

    try:
        log_section("Pre-flight checks")
        print(f"Run directory: {run_dir}")
        print(f"Python executable: {sys.executable}")
        print(f"libero.__path__: {libero.__path__}")
        print(f"mujoco.__version__: {mujoco.__version__}")
        print(f"robosuite.__version__: {robosuite.__version__}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"torch CUDA available: {torch.cuda.is_available()}")

        dataset_statistics_path = os.path.join(args.checkpoint_id, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r", encoding="utf-8") as f:
                norm_stats = json.load(f)
            if args.unnorm_key not in norm_stats:
                fallback = f"{args.unnorm_key}_no_noops"
                if fallback in norm_stats:
                    args.unnorm_key = fallback
                else:
                    raise RuntimeError(
                        f"unnorm_key {args.unnorm_key} not found; available keys: {list(norm_stats.keys())}"
                    )

        log_section("Loading OpenVLA model")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map[args.dtype]
        processor, vla = load_openvla(
            checkpoint_id=args.checkpoint_id,
            revision=args.revision,
            attn_implementation=args.attn_implementation,
            dtype=dtype,
            device=device,
        )
        action_dim = vla.get_action_dim(args.unnorm_key)
        print(f"Action dimension from checkpoint: {action_dim}")
        if action_dim != ACTION_DIM:
            raise RuntimeError(f"Expected action dim {ACTION_DIM}, got {action_dim}")

        log_section("Loading LIBERO benchmark and task")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        task = task_suite.get_task(args.task_id)
        vanilla_initial_states = task_suite.get_task_init_states(args.task_id)
        print(f"Task name: {task.name}")
        print(f"Task language: {task.language}")
        print(f"Vanilla initial states shape: {vanilla_initial_states.shape}")

        original_bddl_path = task_suite.get_task_bddl_file_path(args.task_id)
        print(f"Original BDDL path: {original_bddl_path}")

        log_section("Source / destination / swap-counterpart resolution")
        entities = resolve_spatial_task(
            bddl_path=original_bddl_path,
            task_suite=args.task_suite,
            task_name=task.name,
            ood_config_path=args.ood_config_path,
            perturbation_seed=args.perturbation_seed,
        )
        for line in entities.summary_lines():
            print(f"  {line}")
        comparison_metadata["spatial_entities"] = entities.to_dict()

        task_output_root = os.path.join(run_dir, task.name)
        os.makedirs(task_output_root, exist_ok=True)

        if args.action_parity_check:
            log_section("Phase resolver action-parity check (short vanilla episode, resolver on vs. off)")
            parity_args = argparse.Namespace(**vars(args))
            parity_args.max_steps = args.action_parity_max_steps
            parity_dir = os.path.join(run_dir, "phase_resolver_action_parity")
            os.makedirs(parity_dir, exist_ok=True)

            env_on, task_description_on = get_libero_env(task, resolution=args.resolution)
            entities_on = resolve_spatial_task_from_env(
                env=env_on, bddl_path=original_bddl_path, task_suite=args.task_suite,
                task_name=task.name, ood_config_path=args.ood_config_path,
                perturbation_seed=args.perturbation_seed,
            )
            run_episode(
                vla=vla, processor=processor, task=task, task_description=task_description_on,
                env=env_on, initial_state=vanilla_initial_states[args.init_state_id],
                condition="parity_resolver_on", output_dir=os.path.join(parity_dir, "resolver_on"),
                args=parity_args, entities=entities_on, enable_phase_resolver=True,
            )

            env_off, task_description_off = get_libero_env(task, resolution=args.resolution)
            entities_off = resolve_spatial_task_from_env(
                env=env_off, bddl_path=original_bddl_path, task_suite=args.task_suite,
                task_name=task.name, ood_config_path=args.ood_config_path,
                perturbation_seed=args.perturbation_seed,
            )
            run_episode(
                vla=vla, processor=processor, task=task, task_description=task_description_off,
                env=env_off, initial_state=vanilla_initial_states[args.init_state_id],
                condition="parity_resolver_off", output_dir=os.path.join(parity_dir, "resolver_off"),
                args=parity_args, entities=entities_off, enable_phase_resolver=False,
            )

            actions_on = np.load(os.path.join(parity_dir, "resolver_on", "actions_final.npy"))
            actions_off = np.load(os.path.join(parity_dir, "resolver_off", "actions_final.npy"))
            n = min(len(actions_on), len(actions_off))
            max_diff = float(np.max(np.abs(actions_on[:n] - actions_off[:n]))) if n > 0 else None
            parity_report = {
                "steps_resolver_on": int(len(actions_on)),
                "steps_resolver_off": int(len(actions_off)),
                "steps_compared": int(n),
                "identical_step_counts": bool(len(actions_on) == len(actions_off)),
                "action_max_abs_diff": max_diff,
                "pass": bool(max_diff == 0.0 and len(actions_on) == len(actions_off)) if max_diff is not None else False,
                "note": (
                    "Phase resolution reads simulator state read-only after env.step() and "
                    "never feeds back into action selection; any nonzero diff here reflects "
                    "GPU inference nondeterminism, not the resolver."
                ),
            }
            parity_path = os.path.join(parity_dir, "action_parity_report.json")
            with open(parity_path, "w", encoding="utf-8") as f:
                json.dump(parity_report, f, indent=2)
            print(f"Action parity report: {parity_report}")
            print(f"Saved: {parity_path}")
            return 0 if parity_report["pass"] else 1

        # ---------------------------------------------------------------------
        # Vanilla episode
        # ---------------------------------------------------------------------
        if not args.skip_vanilla:
            vanilla_output_dir = os.path.join(
                task_output_root, "vanilla", f"episode_{args.init_state_id}_{timestamp}"
            )
            os.makedirs(vanilla_output_dir, exist_ok=True)

            env, task_description = get_libero_env(task, resolution=args.resolution)
            vanilla_entities = resolve_spatial_task_from_env(
                env=env,
                bddl_path=original_bddl_path,
                task_suite=args.task_suite,
                task_name=task.name,
                ood_config_path=args.ood_config_path,
                perturbation_seed=args.perturbation_seed,
            )
            comparison_metadata["vanilla"] = run_episode(
                vla=vla,
                processor=processor,
                task=task,
                task_description=task_description,
                env=env,
                initial_state=vanilla_initial_states[args.init_state_id],
                condition="vanilla",
                output_dir=vanilla_output_dir,
                args=args,
                entities=vanilla_entities,
                enable_phase_resolver=not args.disable_phase_resolver,
            )
        else:
            print("[CONFIG] Skipping vanilla episode")

        # ---------------------------------------------------------------------
        # Position perturbation episode
        # ---------------------------------------------------------------------
        if not args.skip_perturbation:
            perturb_bddl_dir = os.path.join(
                run_dir, "temp", f"bddl_{task.name}"
            )
            perturb_init_dir = os.path.join(
                run_dir, "temp", f"init_{task.name}"
            )

            config_path = args.ood_config_path
            perturb_bddl_path = os.path.join(perturb_bddl_dir, f"{task.name}.bddl")

            log_section("Applying position swap perturbation")
            apply_position_swap_perturbation(
                original_bddl_path=original_bddl_path,
                task_suite_name=args.task_suite,
                task_name=task.name,
                output_bddl_path=perturb_bddl_path,
                config_path=config_path,
                seed=args.perturbation_seed,
            )

            log_section("Generating perturbation init states")
            script_path = os.path.join(
                _LIBERO_PRO_ROOT, "notebooks", "generate_init_states.py"
            )
            generate_perturbation_init_states(
                bddl_dir=perturb_bddl_dir,
                output_dir=perturb_init_dir,
                num_inits=args.num_perturbation_inits,
                height=args.resolution,
                width=args.resolution,
                script_path=script_path,
            )

            perturb_init_path = os.path.join(perturb_init_dir, f"{task.name}.pruned_init")
            print(f"Loading perturbation init states from: {perturb_init_path}")
            perturb_initial_states = load_pruned_init_zip(perturb_init_path)
            print(f"Perturbation initial states shape: {perturb_initial_states.shape}")

            perturb_output_dir = os.path.join(
                task_output_root, "position_perturbation", f"episode_{args.init_state_id}_{timestamp}"
            )
            os.makedirs(perturb_output_dir, exist_ok=True)

            perturb_env = get_libero_env_from_bddl(perturb_bddl_path, resolution=args.resolution)
            # Same resolver, now against the perturbed BDDL: the goal (and therefore
            # source/destination) is unchanged, while the init regions differ.
            perturb_entities = resolve_spatial_task_from_env(
                env=perturb_env,
                bddl_path=perturb_bddl_path,
                task_suite=args.task_suite,
                task_name=task.name,
                ood_config_path=args.ood_config_path,
                perturbation_seed=args.perturbation_seed,
            )
            perturb_entities.swap_counterpart = entities.swap_counterpart
            perturb_entities.swap_pairs = entities.swap_pairs
            perturb_entities.perturbation_moved_entities = entities.perturbation_moved_entities
            perturb_entities.swap_counterpart_partner_of = entities.swap_counterpart_partner_of
            perturb_entities.swap_note = (
                "swap fields copied from the original-vs-perturbed BDDL diff; the "
                "perturbed file is already swapped, so re-running the perturbator on "
                "it would describe a second swap"
            )
            comparison_metadata["perturbation_applied"] = {
                "bddl_path": perturb_bddl_path,
                "config_path": config_path,
                "perturbation_seed": args.perturbation_seed,
                "swap_pairs": entities.swap_pairs,
                "moved_entities": entities.perturbation_moved_entities,
                "original_init_regions": entities.init_regions,
                "perturbed_init_regions": perturb_entities.init_regions,
            }
            comparison_metadata["position_perturbation"] = run_episode(
                vla=vla,
                processor=processor,
                task=task,
                task_description=task.language,
                env=perturb_env,
                initial_state=perturb_initial_states[args.init_state_id],
                condition="position_perturbation",
                output_dir=perturb_output_dir,
                args=args,
                entities=perturb_entities,
                enable_phase_resolver=not args.disable_phase_resolver,
            )
        else:
            print("[CONFIG] Skipping perturbation episode")

        # Save comparison metadata.
        comparison_path = os.path.join(task_output_root, f"comparison_metadata_{timestamp}.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_metadata, f, indent=2, ensure_ascii=False)
        print(f"Comparison metadata saved: {comparison_path}")

    except Exception as exc:
        print("\n[ERROR] Exception during failure screening:")
        traceback.print_exc()
        comparison_metadata["exception_occurred"] = True
        comparison_metadata["exception_message"] = str(exc)
        comparison_metadata["exception_traceback"] = traceback.format_exc()
        comparison_path = os.path.join(run_dir, "comparison_metadata_error.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_metadata, f, indent=2, ensure_ascii=False)
        return 1
    finally:
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal

    return 0


if __name__ == "__main__":
    sys.exit(main())
