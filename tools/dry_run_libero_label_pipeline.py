"""Dry-run validation for LIBERO RGB/segmentation/target-label collection.

This script does not construct SUREFlow, load a checkpoint, install dependencies,
or run a policy. It validates one stabilized LIBERO simulator observation and
writes camera overlays plus a strict JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
        "seg_camera": "agentview",
        "metadata_camera": "agentview",
    },
    "eye_in_hand": {
        "obs_rgb": "robot0_eye_in_hand_image",
        "seg_camera": "robot0_eye_in_hand",
        "metadata_camera": "robot0_eye_in_hand",
    },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run LIBERO RGB/segmentation/target-label pipeline without SUREFlow."
    )
    parser.add_argument("--train_suite", default="libero_object")
    parser.add_argument("--eval_suite", default=None)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--initial_state_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dummy_steps", type=int, default=5)
    parser.add_argument("--min_mask_pixels", type=int, default=10)
    parser.add_argument("--output_dir", default="libero_label_pipeline_dry_run")
    return parser.parse_args()


def repo_imports() -> dict[str, Any]:
    try:
        from libero.libero import benchmark
        from libero.libero.envs import SegmentationRenderEnv
    except ModuleNotFoundError:
        from libero import benchmark
        from libero.envs import SegmentationRenderEnv

    return {
        "benchmark": benchmark,
        "SegmentationRenderEnv": SegmentationRenderEnv,
    }


def benchmark_context(imports: dict[str, Any], benchmark_name: str, task_id: int) -> dict[str, Any]:
    benchmark_dict = imports["benchmark"].get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        raise KeyError(
            f"Benchmark {benchmark_name!r} not found. Available benchmarks: {sorted(benchmark_dict.keys())}"
        )
    benchmark_obj = benchmark_dict[benchmark_name]()
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
    return imports["SegmentationRenderEnv"](
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )


def unwrap_base_env(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def observation_overview(obs: dict[str, Any]) -> list[dict[str, Any]]:
    overview = []
    print("Observation keys:")
    for key in sorted(obs.keys()):
        value = obs[key]
        shape = list(value.shape) if hasattr(value, "shape") else None
        dtype = str(getattr(value, "dtype", type(value).__name__))
        overview.append({"key": key, "shape": shape, "dtype": dtype})
        print(f"  {key}: shape={shape}, dtype={dtype}")
    return overview


def select_single_movable_source(env: Any) -> str:
    base_env = unwrap_base_env(env)
    parsed = getattr(base_env, "parsed_problem", None)
    if not parsed:
        raise RuntimeError("Could not find parsed_problem on the unwrapped LIBERO env.")

    candidates: list[str] = []
    goal_state = parsed.get("goal_state", [])
    objects_dict = getattr(base_env, "objects_dict", {})
    for state in goal_state:
        if len(state) != 3:
            continue
        predicate = str(state[0]).lower()
        if predicate not in {"in", "on"}:
            continue
        source_name = state[1]
        if source_name in objects_dict:
            candidates.append(source_name)

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one movable source object from binary In/On goals, "
            f"found {len(unique_candidates)}: {unique_candidates}. goal_state={goal_state}"
        )
    return unique_candidates[0]


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
        segmentation_like = [key for key in obs if "seg" in key.lower()]
        raise KeyError(
            f"No segmentation key found for camera {camera_name!r}. "
            f"Segmentation-like keys: {segmentation_like}"
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
        bbox = None
        uv_pixel = None
        uv_norm = None

    return {
        "visible": visible,
        "mask_pixel_count": count,
        "target_uv_pixel": uv_pixel,
        "target_uv_normalized": uv_norm,
        "bbox": bbox,
        "mask": mask,
    }


def save_rgb(output_path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB").save(output_path)


def save_overlay(
    output_path: Path,
    rgb: np.ndarray,
    label: dict[str, Any],
    camera: str,
    target_object: str,
    target_id: int,
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
        f"{camera} target={target_object} seg_id={target_id} visible={label['visible']} "
        f"count={label['mask_pixel_count']} centroid={label['target_uv_pixel']}"
    )
    draw.rectangle((0, 0, image.size[0], 24), fill=(0, 0, 0, 170))
    draw.text((4, 5), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    if label["visible"]:
        u, v = label["target_uv_pixel"]
        x0, y0, x1, y1 = label["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0, 255), width=2)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 255, 0, 255))
    composed.convert("RGB").save(output_path)


def get_target_world_pos(env: Any, target_object: str) -> list[float]:
    base_env = unwrap_base_env(env)
    state_obj = base_env.object_states_dict[target_object]
    return np.asarray(state_obj.get_geom_state()["pos"], dtype=np.float32).tolist()


def get_robot_metadata(obs: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos"]:
        if key in obs:
            metadata[key] = np.asarray(obs[key], dtype=np.float32).tolist()
    return metadata


def get_camera_metadata(env: Any, camera_name: str) -> dict[str, Any]:
    try:
        sim = env.sim
        cam_id = sim.model.camera_name2id(camera_name)
        return {
            "available": True,
            "position": np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float32).tolist(),
            "xmat": np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3).tolist(),
            "fovy": float(sim.model.cam_fovy[cam_id]),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def sim_state_vector(env: Any) -> np.ndarray:
    return np.asarray(env.get_sim_state(), dtype=np.float64).copy()


def strict_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): strict_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return strict_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return strict_json_ready(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def build_label_record(
    obs: dict[str, Any],
    output_dir: Path,
    camera: str,
    mapping: dict[str, str],
    segmentation_key: str,
    target_object: str,
    target_id: int,
    min_mask_pixels: int,
) -> dict[str, Any]:
    rgb = np.asarray(obs[mapping["obs_rgb"]])
    height, width = rgb.shape[:2]
    segmentation = obs[segmentation_key]
    label = compute_mask_label(segmentation, target_id, (height, width), min_mask_pixels)

    overlay_path = output_dir / f"step_000_{camera}_overlay.png"
    rgb_path = output_dir / f"step_000_{camera}_rgb.png"
    save_overlay(overlay_path, rgb, label, camera, target_object, target_id)
    save_rgb(rgb_path, rgb)

    record = {key: value for key, value in label.items() if key != "mask"}
    record.update(
        {
            "rgb_shape": list(rgb.shape),
            "segmentation_shape": list(np.asarray(segmentation).shape),
            "segmentation_key": segmentation_key,
            "overlay_path": str(overlay_path),
            "rgb_path": str(rgb_path),
        }
    )
    return record


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = CheckResults()
    env = None

    imports = repo_imports()
    benchmark_name = f"{args.train_suite}_{args.eval_suite}" if args.eval_suite else args.train_suite
    context = benchmark_context(imports, benchmark_name, args.task_id)

    try:
        env = make_env(imports, context["task_bddl_file"])
        checks.set("environment_created", True)
        env.seed(args.seed)
        env.reset()

        init_states = context["init_states"]
        if args.initial_state_id >= len(init_states):
            raise IndexError(
                f"initial_state_id={args.initial_state_id} out of range; available={len(init_states)}"
            )
        obs = env.set_init_state(init_state=init_states[args.initial_state_id])
        checks.set("initial_state_applied", True)

        dummy = np.zeros(7, dtype=np.float32)
        dummy[-1] = -1.0
        for _ in range(args.dummy_steps):
            obs, _reward, done, _info = env.step(dummy)
            if done:
                raise RuntimeError("Environment terminated during stabilization dummy steps.")

        overview = observation_overview(obs)
        segmentation_keys = {
            camera: find_segmentation_key(obs, mapping["seg_camera"])
            for camera, mapping in CAMERA_KEY_MAP.items()
        }
        checks.set("segmentation_keys_found", True)

        target_object = select_single_movable_source(env)
        checks.set("target_object_selected", True)
        target_id = target_instance_id(env, target_object)
        checks.set("target_instance_id_mapped", True)

        state_before = sim_state_vector(env)
        labels = {}
        for camera, mapping in CAMERA_KEY_MAP.items():
            labels[camera] = build_label_record(
                obs=obs,
                output_dir=output_dir,
                camera=camera,
                mapping=mapping,
                segmentation_key=segmentation_keys[camera],
                target_object=target_object,
                target_id=target_id,
                min_mask_pixels=args.min_mask_pixels,
            )
        state_after = sim_state_vector(env)

        shape_ok = all(
            labels[camera]["rgb_shape"][:2] == labels[camera]["segmentation_shape"][:2]
            for camera in CAMERA_KEY_MAP
        )
        checks.set("rgb_segmentation_shape_matched", shape_ok, "RGB and segmentation shapes differ.")
        overlay_ok = all(Path(labels[camera]["overlay_path"]).is_file() for camera in CAMERA_KEY_MAP)
        checks.set("overlay_images_saved", overlay_ok, "One or more overlay files are missing.")
        sync_ok = bool(np.allclose(state_before, state_after))
        checks.set("paired_state_synchronization_valid", sync_ok, "Label processing changed simulator state.")

        summary = {
            "scientific_scope": {
                "checkpoint_used": False,
                "model_used": False,
                "purpose": "LIBERO RGB/segmentation/target-label pipeline validation only",
            },
            "benchmark": benchmark_name,
            "task_id": args.task_id,
            "task": context["file_name"],
            "initial_state_id": args.initial_state_id,
            "seed": args.seed,
            "dummy_steps": args.dummy_steps,
            "observation_overview": overview,
            "segmentation_keys": segmentation_keys,
            "available_instance_names": sorted(getattr(env, "instance_to_id", {}).keys()),
            "target_object": target_object,
            "target_segmentation_id": target_id,
            "target_world_pos": get_target_world_pos(env, target_object),
            "robot_metadata": get_robot_metadata(obs),
            "camera_metadata": {
                camera: get_camera_metadata(env, mapping["metadata_camera"])
                for camera, mapping in CAMERA_KEY_MAP.items()
            },
            "labels": labels,
            "paired_state_synchronization_valid": sync_ok,
            "checks": checks.as_dict(),
        }
    finally:
        if env is not None:
            env.close()

    summary_path = output_dir / "dry_run_label_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(strict_json_ready(summary), f, indent=2, allow_nan=False)
    summary["summary_path"] = str(summary_path)
    return summary


def print_final_report(summary: dict[str, Any]) -> None:
    print("\nLIBERO label pipeline dry-run summary")
    print(f"  task: {summary.get('task')}")
    print(f"  output summary: {summary.get('summary_path')}")
    print("  generated files:")
    for camera, label in summary.get("labels", {}).items():
        print(f"    - {camera} overlay: {label.get('overlay_path')}")
        print(f"    - {camera} rgb: {label.get('rgb_path')}")
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
        print(f"LIBERO label pipeline dry-run failed: {exc}", file=sys.stderr)
        return 1
    print_final_report(summary)
    failed = [
        name for name, ok in summary.get("checks", {}).get("results", {}).items()
        if not ok
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
