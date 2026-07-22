"""Dry-run validation for SUREFlow vision-probe feature collection.

This script intentionally collects only a tiny rollout window. It validates that
RGB frames, instance segmentation masks, target 2D labels, target world position,
and camera-specific ResNet features can be read from the same pre-action
simulator state.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_PRO_ROOT = REPO_ROOT / "LIBERO-PRO"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LIBERO_PRO_ROOT.is_dir() and str(LIBERO_PRO_ROOT) not in sys.path:
    sys.path.insert(1, str(LIBERO_PRO_ROOT))


CAMERA_KEY_MAP = {
    "agentview": {
        "obs_rgb": "agentview_image",
        "model_key": "agentview_image",
        "seg_camera": "agentview",
    },
    "eye_in_hand": {
        "obs_rgb": "robot0_eye_in_hand_image",
        "model_key": "eye_in_hand_image",
        "seg_camera": "robot0_eye_in_hand",
    },
}

STAGE_SPECS = {
    "layer3": ("backbone", 6, (1, 256, 8, 8)),
    "layer4": ("backbone", 7, (1, 512, 4, 4)),
    "avgpool": ("backbone", 8, (1, 512, 1, 1)),
    "projection": ("fc_layers", None, (1, 256)),
}


@dataclass
class CheckResults:
    results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def set(self, name: str, ok: bool, detail: str | None = None) -> None:
        self.results[name] = bool(ok)
        if not ok and detail:
            self.errors.append(f"{name}: {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {"results": self.results, "errors": self.errors}


class IdentityScaler:
    def inverse_scale_output(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def scale_output(self, value: torch.Tensor) -> torch.Tensor:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run SUREFlow vision probe feature and label collection."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_suite", default="libero_object")
    parser.add_argument("--eval_suite", default=None)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--initial_state_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--output_dir", default="vision_probe_dry_run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_mask_pixels", type=int, default=10)
    parser.add_argument(
        "--allow_identity_scaler_debug",
        action="store_true",
        help="Allow IdentityScaler only for a one-step debug run when no real scaler can be restored.",
    )
    return parser.parse_args()


def repo_imports() -> dict[str, Any]:
    from configs.config import create_libero_pro_eval_config, create_libero_train_config
    from configs.factory import create_model, create_trainer

    try:
        from libero.libero import benchmark
        from libero.libero.envs import SegmentationRenderEnv
    except ModuleNotFoundError:
        from libero import benchmark
        from libero.envs import SegmentationRenderEnv

    return {
        "create_libero_train_config": create_libero_train_config,
        "create_libero_pro_eval_config": create_libero_pro_eval_config,
        "create_model": create_model,
        "create_trainer": create_trainer,
        "benchmark": benchmark,
        "SegmentationRenderEnv": SegmentationRenderEnv,
    }


def make_config(imports: dict[str, Any], train_suite: str, eval_suite: str | None, device: str):
    if eval_suite is None:
        cfg = imports["create_libero_train_config"](train_suite)
    else:
        cfg = imports["create_libero_pro_eval_config"](train_suite, eval_suite)
    cfg.device = device
    cfg.model_cfg.device = device
    cfg.model_cfg.model.device = device
    cfg.model_cfg.model.backbones.device = device
    cfg.trainer.device = device
    return cfg


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = [path / "final_model.pth", path / "model_state_dict.pth"]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_path}; expected final_model.pth or model_state_dict.pth"
            )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def checkpoint_dir_from_path(checkpoint_path: str, resolved_checkpoint_path: str | None = None) -> Path:
    path = Path(resolved_checkpoint_path or checkpoint_path)
    if path.is_file():
        return path.parent
    return path


def restore_real_scaler(
    model: torch.nn.Module,
    cfg: Any,
    imports: dict[str, Any],
    checkpoint_path: str,
    checkpoint_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir_from_path(checkpoint_path, checkpoint_info.get("path"))
    scaler_path = checkpoint_dir / "model_scaler.pkl"
    if scaler_path.is_file():
        model.load_model_scaler(str(checkpoint_dir), "model_scaler.pkl")
        return {"source": "checkpoint_model_scaler", "path": str(scaler_path)}

    try:
        trainer = imports["create_trainer"](cfg)
        model.set_scaler(trainer.scaler)
        return {"source": "create_trainer(cfg).scaler", "path": None}
    except Exception as exc:
        if args.allow_identity_scaler_debug and args.max_steps == 1:
            model.set_scaler(IdentityScaler())
            return {
                "source": "identity_debug",
                "path": None,
                "warning": (
                    "IdentityScaler is enabled only because --allow_identity_scaler_debug "
                    "was passed with --max_steps=1."
                ),
                "real_scaler_error": str(exc),
            }
        raise RuntimeError(
            "Could not restore a real action scaler. Expected model_scaler.pkl next to the "
            "checkpoint or a dataset path that lets create_trainer(cfg).scaler be built. "
            "IdentityScaler is only allowed with --allow_identity_scaler_debug --max_steps 1."
        ) from exc


def load_model(
    cfg: Any,
    imports: dict[str, Any],
    checkpoint_path: str,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    model = imports["create_model"](cfg)
    checkpoint_info = load_checkpoint(model, checkpoint_path)
    scaler_info = restore_real_scaler(model, cfg, imports, checkpoint_path, checkpoint_info, args)
    model.to(cfg.device)
    model.eval()
    return model, checkpoint_info, scaler_info


def benchmark_context(imports: dict[str, Any], benchmark_name: str, task_id: int) -> dict[str, Any]:
    benchmark_cls = imports["benchmark"].get_benchmark_dict()[benchmark_name]
    benchmark_obj = benchmark_cls()
    task_bddl_file = benchmark_obj.get_task_bddl_file_path(task_id)
    init_states = benchmark_obj.get_task_init_states(task_id)
    task = benchmark_obj.get_task(task_id)
    return {
        "benchmark": benchmark_obj,
        "task": task,
        "task_bddl_file": task_bddl_file,
        "init_states": init_states,
        "file_name": Path(task_bddl_file).stem,
    }


def make_env(imports: dict[str, Any], bddl_file: str):
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    return imports["SegmentationRenderEnv"](**env_args)


def unwrap_base_env(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def select_single_movable_source(env: Any) -> str:
    base_env = unwrap_base_env(env)
    parsed = getattr(base_env, "parsed_problem", None)
    if not parsed:
        raise RuntimeError("Could not find parsed_problem on the unwrapped LIBERO env.")

    candidates: list[str] = []
    goal_state = parsed.get("goal_state", [])
    for state in goal_state:
        if len(state) != 3:
            continue
        predicate = str(state[0]).lower()
        if predicate not in {"in", "on"}:
            continue
        source_name = state[1]
        if source_name in getattr(base_env, "objects_dict", {}):
            candidates.append(source_name)

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one movable source object from binary In/On goals, "
            f"found {len(unique_candidates)}: {unique_candidates}. goal_state={goal_state}"
        )
    return unique_candidates[0]


def format_task_embedding(value: Any, device: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32)
    value = value.float().to(device)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"Expected one task embedding with shape [1, D], got {tuple(value.shape)}")
    return value


def load_vanilla_task_embedding(benchmark_name: str, task_file_name: str, device: str) -> torch.Tensor:
    emb_path = REPO_ROOT / "SUREFlow" / "language_embeddings" / f"{benchmark_name}.pkl"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Task embedding file not found: {emb_path}")
    with emb_path.open("rb") as f:
        task_embs = pickle.load(f)
    if task_file_name not in task_embs:
        raise KeyError(
            f"Task {task_file_name!r} not found in {emb_path}. "
            f"Example keys: {list(task_embs.keys())[:5]}"
        )
    return format_task_embedding(task_embs[task_file_name], device)


def load_pro_task_embedding(cfg: Any, task_file_name: str, device: str) -> torch.Tensor:
    from libero.lifelong.utils import get_task_embs

    description = task_file_name.replace("_", " ")
    pro_cfg = SimpleNamespace()
    pro_cfg.data = SimpleNamespace()
    pro_cfg.data.max_word_len = getattr(cfg, "task_embedding_max_length", 77)
    pro_cfg.policy = SimpleNamespace()
    pro_cfg.policy.language_encoder = SimpleNamespace()
    pro_cfg.policy.language_encoder.network_kwargs = SimpleNamespace()
    pro_cfg.policy.language_encoder.network_kwargs.input_size = None
    pro_cfg.task_embedding_format = getattr(cfg, "task_embedding_format", "clip")
    pro_cfg.task_embedding_model = getattr(cfg, "task_embedding_model", "openai/clip-vit-base-patch32")
    pro_cfg.task_embedding_device = getattr(cfg, "task_embedding_device", str(device))
    pro_cfg.task_embedding_max_length = getattr(cfg, "task_embedding_max_length", 77)

    task_embs = get_task_embs(pro_cfg, [description])
    if isinstance(task_embs, dict):
        value = task_embs.get(description)
        if value is None:
            value = next(iter(task_embs.values()))
    else:
        value = task_embs[0]
    return format_task_embedding(value, device)


def load_task_embedding(
    cfg: Any,
    train_suite: str,
    eval_suite: str | None,
    task_file_name: str,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if eval_suite is None:
        embedding = load_vanilla_task_embedding(train_suite, task_file_name, device)
        return embedding, {"source": "pickle", "benchmark": train_suite}
    embedding = load_pro_task_embedding(cfg, task_file_name, device)
    return embedding, {
        "source": "runtime_get_task_embs",
        "description": task_file_name.replace("_", " "),
        "eval_suite": eval_suite,
    }


def print_obs_overview(obs: dict[str, Any]) -> list[dict[str, Any]]:
    overview = []
    print("Observation keys:")
    for key in sorted(obs.keys()):
        value = obs[key]
        shape = list(value.shape) if hasattr(value, "shape") else None
        dtype = str(getattr(value, "dtype", type(value).__name__))
        overview.append({"key": key, "shape": shape, "dtype": dtype})
        print(f"  {key}: shape={shape}, dtype={dtype}")
    return overview


def find_segmentation_key(obs: dict[str, Any], camera_name: str) -> str:
    candidates = []
    camera_tokens = [camera_name, camera_name.replace("robot0_", ""), camera_name.replace("_", "")]
    for key, value in obs.items():
        lowered = key.lower()
        if "seg" not in lowered:
            continue
        if not hasattr(value, "shape"):
            continue
        score = 0
        for token in camera_tokens:
            if token and token.lower() in lowered:
                score += 2
        if "instance" in lowered:
            score += 1
        candidates.append((score, key))
    candidates = sorted(candidates, reverse=True)
    if not candidates or candidates[0][0] <= 0:
        raise KeyError(
            f"No segmentation key found for camera {camera_name!r}. "
            f"Segmentation-like keys: {[key for key in obs if 'seg' in key.lower()]}"
        )
    return candidates[0][1]


def normalize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    arr = np.asarray(segmentation)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation image after squeeze, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def compute_mask_label(
    segmentation: np.ndarray,
    target_id: int,
    rgb_shape: tuple[int, int],
    min_mask_pixels: int,
) -> dict[str, Any]:
    seg = normalize_segmentation(segmentation)
    height, width = rgb_shape
    if seg.shape[:2] != (height, width):
        raise ValueError(f"RGB/segmentation shape mismatch: rgb={(height, width)}, seg={seg.shape[:2]}")

    mask = seg == target_id
    ys, xs = np.where(mask)
    count = int(mask.sum())
    visible = count >= min_mask_pixels
    if visible:
        u = float(xs.mean())
        v = float(ys.mean())
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        uv_pixel = [u, v]
        uv_norm = [u / float(width), v / float(height)]
    else:
        bbox = [float("nan")] * 4
        uv_pixel = [float("nan"), float("nan")]
        uv_norm = [float("nan"), float("nan")]

    return {
        "visible": visible,
        "mask_pixel_count": count,
        "target_uv_pixel": uv_pixel,
        "target_uv_normalized": uv_norm,
        "bbox": bbox,
        "mask": mask,
    }


def target_instance_id(env: Any, target_object: str) -> int:
    if not hasattr(env, "instance_to_id"):
        raise RuntimeError("Segmentation env does not expose instance_to_id.")
    if target_object not in env.instance_to_id:
        available = sorted(env.instance_to_id.keys())
        raise KeyError(
            f"Target object {target_object!r} is not in segmentation instance map. "
            f"Available instances: {available}"
        )
    return int(env.instance_to_id[target_object])


def get_target_world_pos(env: Any, target_object: str) -> np.ndarray:
    base_env = unwrap_base_env(env)
    state_obj = base_env.object_states_dict[target_object]
    return np.asarray(state_obj.get_geom_state()["pos"], dtype=np.float32)


def get_robot_metadata(obs: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos"]:
        if key in obs:
            metadata[key] = np.asarray(obs[key]).astype(np.float32).tolist()
    return metadata


def get_camera_metadata(env: Any, camera_name: str) -> dict[str, Any]:
    try:
        sim = env.sim
        cam_id = sim.model.camera_name2id(camera_name)
        return {
            "position": np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float32).tolist(),
            "xmat": np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3).tolist(),
            "fovy": float(sim.model.cam_fovy[cam_id]),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def make_obs_dict(obs: dict[str, Any], task_emb: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    agentview_rgb = (
        torch.from_numpy(obs["agentview_image"])
        .to(device)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )
    eye_rgb = (
        torch.from_numpy(obs["robot0_eye_in_hand_image"])
        .to(device)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )
    joint_state = obs["robot0_joint_pos"]
    gripper_state = obs["robot0_gripper_qpos"]
    robot_states = (
        torch.from_numpy(np.concatenate([joint_state, gripper_state], axis=-1))
        .to(device)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return {
        "agentview_image": agentview_rgb,
        "eye_in_hand_image": eye_rgb,
        "lang_emb": task_emb,
        "robot_states": robot_states,
    }


def feature_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().float()
    return {
        "shape": list(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def get_hook_module(model: torch.nn.Module, model_key: str, stage: str) -> torch.nn.Module:
    encoder = model.img_encoder.key_model_map[model_key]
    attr, index, _ = STAGE_SPECS[stage]
    module = getattr(encoder, attr)
    if index is not None:
        module = module[index]
    return module


def register_feature_hooks(
    model: torch.nn.Module,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[Any], list[str], dict[str, bool]]:
    cache: dict[str, dict[str, torch.Tensor]] = {
        camera: {} for camera in CAMERA_KEY_MAP
    }
    handles = []
    module_paths = []
    capture_state = {"enabled": False}

    def make_hook(camera: str, stage: str) -> Callable[[Any, tuple[Any, ...], torch.Tensor], None]:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not capture_state["enabled"]:
                return
            cache[camera][stage] = output.detach().cpu().clone()

        return hook

    for camera, mapping in CAMERA_KEY_MAP.items():
        for stage in STAGE_SPECS:
            module = get_hook_module(model, mapping["model_key"], stage)
            handles.append(module.register_forward_hook(make_hook(camera, stage)))
            attr, index, _ = STAGE_SPECS[stage]
            suffix = f"{attr}[{index}]" if index is not None else attr
            module_paths.append(f"img_encoder.key_model_map[{mapping['model_key']!r}].{suffix}")
    return cache, handles, module_paths, capture_state


def validate_feature_cache(cache: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, dict[str, Any]]]:
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for camera in CAMERA_KEY_MAP:
        stats[camera] = {}
        for stage, (_attr, _index, expected_shape) in STAGE_SPECS.items():
            if stage not in cache[camera]:
                raise RuntimeError(
                    f"Missing hook output for {camera}/{stage}. "
                    "Expected explicit_encoder_forward to run with capture_enabled=True. "
                    "Check image keys, model camera config, hook module path, and batch/sequence dimensions."
                )
            tensor = cache[camera][stage]
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape for {camera}/{stage}: got {tuple(tensor.shape)}, "
                    f"expected {expected_shape}. Check ResNet stage config, batch size, and observation sequence length."
                )
            stats[camera][stage] = feature_stats(tensor)
    return stats


def clear_feature_cache(cache: dict[str, dict[str, torch.Tensor]]) -> None:
    for camera in cache:
        cache[camera].clear()


def collect_explicit_encoder_features(
    model: torch.nn.Module,
    obs: dict[str, Any],
    task_emb: torch.Tensor,
    device: str,
    cache: dict[str, dict[str, torch.Tensor]],
    capture_state: dict[str, bool],
) -> dict[str, dict[str, dict[str, Any]]]:
    clear_feature_cache(cache)
    probe_obs_dict = make_obs_dict(obs, task_emb, device)
    capture_state["enabled"] = True
    try:
        with torch.no_grad():
            model._input_embeddings(probe_obs_dict)
    finally:
        capture_state["enabled"] = False
    return validate_feature_cache(cache)


def save_overlay(
    output_path: Path,
    rgb: np.ndarray,
    label: dict[str, Any],
    camera: str,
    target_object: str,
) -> None:
    image = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    mask = label["mask"]
    for y, x in zip(*np.where(mask)):
        overlay_pixels[int(x), int(y)] = (255, 0, 0, 95)
    composed = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(composed)
    text = (
        f"{camera} target={target_object} visible={label['visible']} "
        f"count={label['mask_pixel_count']} uv={label['target_uv_pixel']}"
    )
    draw.rectangle((0, 0, image.size[0], 22), fill=(0, 0, 0, 170))
    draw.text((4, 4), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    if label["visible"]:
        u, v = label["target_uv_pixel"]
        x0, y0, x1, y1 = label["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0, 255), width=2)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 255, 0, 255))
    composed.convert("RGB").save(output_path)


def sim_state_vector(env: Any) -> np.ndarray:
    return np.asarray(env.get_sim_state(), dtype=np.float64).copy()


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = CheckResults()
    imports = repo_imports()
    benchmark_name = f"{args.train_suite}_{args.eval_suite}" if args.eval_suite else args.train_suite
    cfg = make_config(imports, args.train_suite, args.eval_suite, args.device)

    model, checkpoint_info, scaler_info = load_model(cfg, imports, args.checkpoint_path, args)
    checks.set("model loaded", True)
    checks.set("scaler attached", True)

    context = benchmark_context(imports, benchmark_name, args.task_id)
    task_emb, task_embedding_info = load_task_embedding(
        cfg,
        args.train_suite,
        args.eval_suite,
        context["file_name"],
        args.device,
    )

    env = make_env(imports, context["task_bddl_file"])
    checks.set("segmentation environment created", True)

    hooks_removed = False
    cache, handles, hook_paths, capture_state = register_feature_hooks(model)
    summary: dict[str, Any] = {
        "checkpoint": checkpoint_info,
        "scaler": scaler_info,
        "task_embedding": task_embedding_info,
        "benchmark": benchmark_name,
        "task_id": args.task_id,
        "task": context["file_name"],
        "initial_state_id": args.initial_state_id,
        "max_steps": args.max_steps,
        "hook_module_paths": hook_paths,
        "steps": [],
    }

    try:
        model.reset()
        simulation_cfg = getattr(cfg, "simulation", None)
        seed = getattr(simulation_cfg, "seed", getattr(cfg, "seed", None))
        env.seed(seed)
        obs = env.reset()

        init_states = context["init_states"]
        if args.initial_state_id >= len(init_states):
            raise IndexError(
                f"initial_state_id={args.initial_state_id} out of range; "
                f"available={len(init_states)}"
            )
        obs = env.set_init_state(init_state=init_states[args.initial_state_id])

        dummy = np.zeros(7, dtype=np.float32)
        dummy[-1] = -1.0
        for _dummy_step in range(5):
            obs, _reward, done, _info = env.step(dummy)
            if done:
                raise RuntimeError("Environment terminated during the 5 stabilization dummy steps.")

        obs_overview = print_obs_overview(obs)
        summary["stabilized_observation_overview"] = obs_overview
        summary["episode_initialization"] = {
            "seed": seed,
            "init_state_id": args.initial_state_id,
            "dummy_steps": 5,
            "dummy_action": dummy.tolist(),
            "first_saved_observation": "after_dummy_steps",
        }

        target_object = select_single_movable_source(env)
        checks.set("target object selected", True)
        target_id = target_instance_id(env, target_object)
        checks.set("target instance ID mapped", True)

        available_instances = sorted(getattr(env, "instance_to_id", {}).keys())
        summary["target"] = {
            "target_object": target_object,
            "target_segmentation_id": target_id,
            "available_instance_names": available_instances,
        }

        segmentation_keys = {
            camera: find_segmentation_key(obs, mapping["seg_camera"])
            for camera, mapping in CAMERA_KEY_MAP.items()
        }
        checks.set("segmentation keys found", True)
        summary["segmentation_keys"] = segmentation_keys

        for step in range(args.max_steps):
            state_before = sim_state_vector(env)
            target_world_pos = get_target_world_pos(env, target_object)
            checks.set("target world position obtained", True)

            labels = {}
            for camera, mapping in CAMERA_KEY_MAP.items():
                rgb = np.asarray(obs[mapping["obs_rgb"]])
                height, width = rgb.shape[:2]
                seg = obs[segmentation_keys[camera]]
                label = compute_mask_label(seg, target_id, (height, width), args.min_mask_pixels)
                labels[camera] = {k: v for k, v in label.items() if k != "mask"}
                overlay_path = output_dir / f"step_{step:03d}_{camera}_overlay.png"
                save_overlay(overlay_path, rgb, label, camera, target_object)
                labels[camera]["overlay_path"] = str(overlay_path)

            checks.set("RGB/segmentation shapes matched", True)

            feature_summary = collect_explicit_encoder_features(
                model,
                obs,
                task_emb,
                args.device,
                cache,
                capture_state,
            )
            for stage in STAGE_SPECS:
                checks.set(f"{stage} hook shape valid", True)

            state_after_feature = sim_state_vector(env)
            feature_sync_ok = np.allclose(state_before, state_after_feature)
            checks.set(
                "explicit feature forward does not change sim state",
                feature_sync_ok,
                "explicit_encoder_forward changed sim state",
            )

            policy_refresh_step = int(getattr(model, "rollout_step_counter", 0)) == 0
            action_obs_dict = make_obs_dict(obs, task_emb, args.device)
            with torch.no_grad():
                action = model.predict(action_obs_dict)

            state_after_predict = sim_state_vector(env)
            sync_ok = np.allclose(state_before, state_after_predict)
            checks.set("paired timestep synchronization valid", sync_ok, "model.predict changed sim state")

            action_np = action.detach().cpu().numpy()
            step_record = {
                "timestep": step,
                "target_world_pos": target_world_pos.tolist(),
                "robot": get_robot_metadata(obs),
                "camera": {
                    "agentview": get_camera_metadata(env, "agentview"),
                    "eye_in_hand": get_camera_metadata(env, "robot0_eye_in_hand"),
                },
                "labels": labels,
                "feature_summary": feature_summary,
                "feature_source": "explicit_encoder_forward",
                "policy_refresh_step": policy_refresh_step,
                "policy_consumed_current_observation": policy_refresh_step,
                "action_shape": list(action_np.shape),
            }
            summary["steps"].append(step_record)

            obs, reward, done, _info = env.step(action_np)
            if done:
                summary["terminated_at_step"] = step
                break

        checks.set("overlay images saved", all(Path(label["overlay_path"]).is_file() for step in summary["steps"] for label in step["labels"].values()))
    finally:
        for handle in handles:
            handle.remove()
        hooks_removed = True
        env.close()
        checks.set("hooks removed", hooks_removed)

    checks.set("avgpool hook shape valid", checks.results.get("avgpool hook shape valid", False))
    checks.set("projection hook shape valid", checks.results.get("projection hook shape valid", False))
    summary["checks"] = checks.as_dict()

    summary_path = output_dir / "dry_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def print_final_report(summary: dict[str, Any]) -> None:
    print("\nDry-run summary")
    print(f"  task: {summary.get('task')}")
    print(f"  output summary: {summary.get('summary_path')}")
    print("  hook paths:")
    for path in summary.get("hook_module_paths", []):
        print(f"    - {path}")
    print("  PASS / FAIL:")
    checks = summary.get("checks", {}).get("results", {})
    for name, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'} {name}")
    errors = summary.get("checks", {}).get("errors", [])
    if errors:
        print("  Errors:")
        for error in errors:
            print(f"    - {error}")


def main() -> int:
    args = parse_args()
    try:
        summary = run_dry_run(args)
    except Exception as exc:
        print(f"Dry-run failed: {exc}", file=sys.stderr)
        return 1
    print_final_report(summary)
    failed = [
        name for name, ok in summary.get("checks", {}).get("results", {}).items()
        if not ok
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
