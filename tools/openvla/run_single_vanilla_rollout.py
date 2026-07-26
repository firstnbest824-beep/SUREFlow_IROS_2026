#!/usr/bin/env python3
"""Vanilla OpenVLA single-task single-init-state rollout on LIBERO-Spatial.

This script reimplements the official OpenVLA LIBERO evaluation path without
TensorFlow dependencies. All image preprocessing uses PIL/PyTorch equivalents.

Python 3.8 compatible syntax only.
"""

from __future__ import print_function

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Force headless EGL rendering and project-local LIBERO config before any libero import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# LIBERO_CONFIG_PATH is expected to be set externally (e.g. via Micromamba activation
# hook). If not, fall back to the project-local config directory.
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)

import libero
import mujoco
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ACTION_DIM = 7
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Default configuration matching the official OpenVLA LIBERO-Spatial evaluation.
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
DEFAULT_OUTPUT_DIR = "/home/hwkim/env-audit/openvla-vanilla-rollout"


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


# -----------------------------------------------------------------------------
# Image preprocessing (TensorFlow-free reimplementation)
# -----------------------------------------------------------------------------
def pil_jpeg_encode_decode(img: np.ndarray, quality: int = 95) -> np.ndarray:
    """Approximate tf.image.encode_jpeg / tf.io.decode_image with PIL."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.array(decoded)


def resize_image(img: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """Resize uint8 HWC image with JPEG encode/decode and Lanczos3 resize."""
    assert isinstance(resize_size, tuple) and len(resize_size) == 2
    img = pil_jpeg_encode_decode(img)
    pil_img = Image.fromarray(img)
    # PIL.resize expects (width, height); resize_size is (height, width).
    pil_img = pil_img.resize((resize_size[1], resize_size[0]), Image.LANCZOS)
    img = np.array(pil_img)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return img


def get_libero_image(obs: Dict[str, Any], resize_size: int) -> np.ndarray:
    """Extract agentview_image, rotate 180 degrees, and resize to model input size."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def apply_center_crop(image: Image.Image, crop_scale: float = 0.9, output_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Center-crop image to area crop_scale * original area, then resize back.

    Mirrors the dlimp/OpenVLA center-crop augmentation used at training time.
    """
    img_np = np.array(image).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    new_h = int(h * math.sqrt(crop_scale))
    new_w = int(w * math.sqrt(crop_scale))
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = img_np[top : top + new_h, left : left + new_w]
    cropped_uint8 = (np.clip(cropped, 0.0, 1.0) * 255).astype(np.uint8)
    pil_cropped = Image.fromarray(cropped_uint8)
    pil_cropped = pil_cropped.resize((output_size[1], output_size[0]), Image.BILINEAR)
    return pil_cropped


# -----------------------------------------------------------------------------
# Action helpers (official OpenVLA convention)
# -----------------------------------------------------------------------------
def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper action from [0, 1] to [-1, +1] and optionally binarize."""
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip sign of gripper action to align with LIBERO convention."""
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
    """Load OpenVLA processor and model, attach dataset statistics."""
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

    # Attach dataset statistics for action un-normalization.
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
    """Generate a single action from OpenVLA given a preprocessed observation."""
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

    # The OpenVLA `predict_action` wrapper appends an empty token (id 29871) to
    # `input_ids` if it is not already the last token, but it does not update the
    # `attention_mask`. With transformers 4.41.2 this creates a length mismatch in
    # the language model's causal mask. We pre-append the token to both tensors.
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
    """Create LIBERO OffScreenRenderEnv for the given task."""
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


def get_libero_dummy_action() -> List[float]:
    """Return no-op dummy action used during initial stabilization."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle."""
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
) -> np.ndarray:
    """Add small labels to an RGB uint8 frame."""
    img = frame.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    color = (255, 255, 255)
    outline = (0, 0, 0)
    lines: List[str] = [
        f"task: {task_name}",
        f"step: {step + 1}/{total}",
        f"cam: {camera_name}",
        f"latency: {latency_ms:.1f}ms",
        f"action[:3]: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]",
    ]
    if success is not None:
        lines.append(f"success: {success}")

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
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla OpenVLA rollout on LIBERO-Spatial")
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
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "rollout.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    # Reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # GPU selection.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "libero_path": list(getattr(libero, "__path__", [])),
        "mujoco_version": getattr(mujoco, "__version__", None),
        "robosuite_version": getattr(robosuite, "__version__", None),
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_name": None,
        "task_instruction": None,
        "bddl_path": None,
        "init_state_id": args.init_state_id,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps,
        "resolution": args.resolution,
        "resize_size": args.resize_size,
        "center_crop": args.center_crop,
        "unnorm_key": args.unnorm_key,
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "gpu": args.gpu,
        "device": str(device),
        "reset_success": False,
        "init_state_success": False,
        "rollout_completed": False,
        "rollout_steps": 0,
        "task_success": False,
        "nan_inf_detected": False,
        "exception_occurred": False,
        "exception_message": None,
    }

    try:
        log_section("Pre-flight checks")
        print(f"Output directory: {args.output_dir}")
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {sys.version}")
        print(f"libero.__path__: {libero.__path__}")
        print(f"mujoco.__version__: {mujoco.__version__}")
        print(f"robosuite.__version__: {robosuite.__version__}")
        print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
        print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")
        print(f"LIBERO_CONFIG_PATH={os.environ.get('LIBERO_CONFIG_PATH')}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"torch version: {torch.__version__}")
        print(f"torch CUDA available: {torch.cuda.is_available()}")
        print(f"torch CUDA version: {torch.version.cuda}")

        # Verify checkpoint contains the required unnorm key.
        dataset_statistics_path = os.path.join(args.checkpoint_id, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r", encoding="utf-8") as f:
                norm_stats = json.load(f)
            if args.unnorm_key not in norm_stats:
                fallback = f"{args.unnorm_key}_no_noops"
                if fallback in norm_stats:
                    args.unnorm_key = fallback
                    metadata["unnorm_key"] = args.unnorm_key
                else:
                    raise RuntimeError(
                        f"unnorm_key {args.unnorm_key} not found in dataset_statistics.json; "
                        f"available keys: {list(norm_stats.keys())}"
                    )

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
        print(f"Model class: {type(vla).__name__}")
        print(f"Processor class: {type(processor).__name__}")

        # Confirm action dimension.
        action_dim = vla.get_action_dim(args.unnorm_key)
        print(f"Action dimension from checkpoint: {action_dim}")
        if action_dim != ACTION_DIM:
            raise RuntimeError(f"Expected action dim {ACTION_DIM}, got {action_dim}")

        log_section("Loading LIBERO benchmark and task")
        benchmark_dict = benchmark.get_benchmark_dict()
        print(f"Available benchmarks: {sorted(benchmark_dict.keys())}")
        task_suite = benchmark_dict[args.task_suite]()
        print(f"Task suite: {args.task_suite}, n_tasks={task_suite.n_tasks}")

        task = task_suite.get_task(args.task_id)
        initial_states = task_suite.get_task_init_states(args.task_id)
        print(f"Task name: {task.name}")
        print(f"Task language: {task.language}")
        print(f"Initial states shape: {initial_states.shape}")

        metadata["task_name"] = task.name
        metadata["task_instruction"] = task.language
        metadata["bddl_path"] = task_suite.get_task_bddl_file_path(args.task_id)
        metadata["init_states_shape"] = list(initial_states.shape)

        log_section("Creating LIBERO environment")
        env, task_description = get_libero_env(task, resolution=args.resolution)
        print("Environment created.")

        log_section("Calling env.reset()")
        obs = env.reset()
        metadata["reset_success"] = True
        print("env.reset() succeeded.")

        log_section("Setting initial state")
        obs = env.set_init_state(initial_states[args.init_state_id])
        metadata["init_state_success"] = True
        print(f"env.set_init_state(initial_states[{args.init_state_id}]) succeeded.")

        log_section("Observation inspection")
        obs_summary: List[Dict[str, Any]] = []
        for key in sorted(obs.keys()):
            value = obs[key]
            info = {"key": key}
            if isinstance(value, np.ndarray):
                info["shape"] = list(value.shape)
                info["dtype"] = str(value.dtype)
            else:
                info["type"] = type(value).__name__
            obs_summary.append(info)
            print(f"  {key}: {info}")
        metadata["observation_keys"] = obs_summary
        with open(os.path.join(args.output_dir, "observations_summary.json"), "w", encoding="utf-8") as f:
            json.dump(obs_summary, f, indent=2)

        # Determine available camera keys for video.
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
        print(f"Selected camera keys: {camera_keys}")

        log_section("Starting rollout")
        dummy_action = get_libero_dummy_action()
        raw_actions: List[np.ndarray] = []
        final_actions: List[np.ndarray] = []
        eef_trajectory: List[Dict[str, Any]] = []
        replay_images: List[np.ndarray] = []
        latencies_ms: List[float] = []
        per_step_records: List[Dict[str, Any]] = []
        step_success_count = 0
        done = False
        t = 0

        # Optional GPU memory tracking.
        def record_gpu_memory() -> Optional[Dict[str, Any]]:
            if not torch.cuda.is_available():
                return None
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }

        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(dummy_action)
                t += 1
                continue

            # Preprocess image.
            img = get_libero_image(obs, args.resize_size)

            # Build observation dict for OpenVLA.
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            # Run OpenVLA inference.
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

            # Validate action.
            if not np.isfinite(action).all():
                metadata["nan_inf_detected"] = True
                raise RuntimeError(f"Non-finite action at step {t}: {action}")

            raw_actions.append(action.copy())

            # Post-process gripper.
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            final_actions.append(action.copy())

            # Execute action.
            obs, reward, done, info = env.step(action.tolist())
            step_success_count += 1
            t += 1

            # Record EEF trajectory.
            eef_trajectory.append({
                "step": t,
                "eef_pos": obs["robot0_eef_pos"].tolist(),
                "eef_quat": obs["robot0_eef_quat"].tolist(),
                "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            })

            # Record frame(s) for video.
            success_flag = info.get("success") if isinstance(info, dict) else None
            if camera_keys:
                primary_key = camera_keys[0]
                frame = np.flipud(obs[primary_key])  # match LIBERO/robosuite convention
                labeled = add_text_overlay(
                    frame,
                    task_name=task.name,
                    step=t - args.num_steps_wait - 1,
                    total=args.max_steps,
                    action=action,
                    success=success_flag,
                    latency_ms=latency_ms,
                    camera_name=primary_key,
                )
                replay_images.append(labeled)

            per_step_records.append({
                "step": t,
                "latency_ms": latency_ms,
                "success": success_flag,
                "raw_action": raw_actions[-1].tolist(),
                "final_action": final_actions[-1].tolist(),
                "gpu_memory_mb": record_gpu_memory(),
            })

            if (t - args.num_steps_wait) % 20 == 0 or (t - args.num_steps_wait) == 1:
                print(
                    f"  step {t - args.num_steps_wait:3d}/{args.max_steps}: "
                    f"latency={latency_ms:.1f}ms, action[:3]=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}], "
                    f"success={success_flag}"
                )

            if done:
                metadata["task_success"] = True
                print(f"Episode succeeded at step {t - args.num_steps_wait}")
                break

        metadata["rollout_completed"] = True
        metadata["rollout_steps"] = step_success_count
        print(f"Rollout completed: {step_success_count} steps, success={metadata['task_success']}")

        # Save actions.
        if raw_actions:
            np.save(os.path.join(args.output_dir, "actions_raw.npy"), np.stack(raw_actions))
        if final_actions:
            np.save(os.path.join(args.output_dir, "actions_final.npy"), np.stack(final_actions))

        with open(os.path.join(args.output_dir, "actions.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + [f"raw_a{i}" for i in range(ACTION_DIM)] + [f"final_a{i}" for i in range(ACTION_DIM)] + ["success"])
            for i, (raw, final) in enumerate(zip(raw_actions, final_actions)):
                writer.writerow([i + 1] + raw.tolist() + final.tolist() + [per_step_records[i]["success"]])

        with open(os.path.join(args.output_dir, "eef_trajectory.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw", "gripper_qpos"])
            for rec in eef_trajectory:
                writer.writerow([rec["step"]] + rec["eef_pos"] + rec["eef_quat"] + [rec["gripper_qpos"]])

        # Save per-step metrics.
        with open(os.path.join(args.output_dir, "per_step_metrics.jsonl"), "w", encoding="utf-8") as f:
            for rec in per_step_records:
                f.write(json.dumps(rec) + "\n")

        # Save latency summary.
        if latencies_ms:
            with open(os.path.join(args.output_dir, "inference_latency.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "latency_ms"])
                for i, lat in enumerate(latencies_ms):
                    writer.writerow([i + 1, lat])
            metadata["avg_latency_ms"] = float(np.mean(latencies_ms))
            metadata["max_latency_ms"] = float(np.max(latencies_ms))
            metadata["min_latency_ms"] = float(np.min(latencies_ms))
            print(f"Latency (ms): mean={metadata['avg_latency_ms']:.2f}, max={metadata['max_latency_ms']:.2f}")

        # Save frames and video.
        log_section("Saving visual outputs")
        save_initial_and_final_frames(replay_images, args.output_dir)
        print("Saved initial/final frames.")

        save_contact_sheet(replay_images, args.output_dir)
        print("Saved contact sheet.")

        mp4_path = os.path.join(args.output_dir, "openvla_vanilla_rollout.mp4")
        mp4_ok = save_video_mp4(replay_images, mp4_path, fps=30.0)
        if mp4_ok:
            metadata["output_video_path"] = mp4_path
            print(f"MP4 saved: {mp4_path}")
        else:
            print("MP4 saving failed; falling back to GIF.")
            gif_path = os.path.join(args.output_dir, "openvla_vanilla_rollout.gif")
            gif_ok = save_video_gif(replay_images, gif_path, fps=20.0)
            if gif_ok:
                metadata["output_gif_path"] = gif_path
                print(f"GIF saved: {gif_path}")
            else:
                print("GIF saving also failed.")

        log_section("Closing environment")
        env.close()
        print("Environment closed.")

    except Exception as exc:  # noqa: BLE001
        metadata["exception_occurred"] = True
        metadata["exception_message"] = str(exc)
        metadata["exception_traceback"] = traceback.format_exc()
        print("\n[ERROR] Exception during rollout:")
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        return 1
    finally:
        metadata_path = os.path.join(args.output_dir, "rollout_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved: {metadata_path}")
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal

    return 0


if __name__ == "__main__":
    sys.exit(main())
