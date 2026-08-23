#!/usr/bin/env python3
"""Run a vanilla OpenVLA baseline or verify LIBERO-PRO BDDL wiring.

This is deliberately an evaluation-only Phase 4 entry point. The only
available method is ``none`` / ``NoOverrideMethod``; it delegates every action
to the pinned base OpenVLA policy. No action-generalization intervention or
training logic belongs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_COMMON_DIR = os.path.normpath(os.path.join(_HERE, "..", "common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from methods.base import ActionGeneralizationMethod, NoOverrideMethod  # noqa: E402
from checkpoints import get_checkpoint  # noqa: E402
from experiment import ExperimentLayout, initial_metadata, prepare_experiment, write_json  # noqa: E402
from image_transform import get_libero_image  # noqa: E402
from init_state import resolve_init_state  # noqa: E402
from libero_env import configure_robosuite_logging, make_env  # noqa: E402
from openvla_model import (  # noqa: E402
    ACTION_DIM,
    get_libero_dummy_action,
    get_vla_action,
    invert_gripper_action,
    load_openvla,
    normalize_gripper_action,
    quat2axisangle,
)
from seeding import episode_seed, seed_everything  # noqa: E402
from task_resolution import ResolvedTaskCondition, resolve_task_condition  # noqa: E402


METHOD_REGISTRY: Dict[str, Type[ActionGeneralizationMethod]] = {"none": NoOverrideMethod}
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} did not parse to a dict: {type(config)}")
    return config


def resolve_method(method_name: str) -> Type[ActionGeneralizationMethod]:
    if method_name not in METHOD_REGISTRY:
        raise NotImplementedError(
            f"method {method_name!r} is not implemented for evaluation. "
            f"Registered eval methods: {sorted(METHOD_REGISTRY)}. Only 'none' exists until Phase 5."
        )
    return METHOD_REGISTRY[method_name]


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _action_statistics(actions: List[np.ndarray]) -> Dict[str, Any]:
    if not actions:
        return {"count": 0}
    stacked = np.stack(actions)
    return {
        "count": int(len(stacked)), "shape": list(stacked.shape),
        "min": stacked.min(axis=0).tolist(), "max": stacked.max(axis=0).tolist(),
        "mean": stacked.mean(axis=0).tolist(), "std": stacked.std(axis=0).tolist(),
    }


def _checkpoint_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Require an explicit baseline checkpoint to match the suite registry."""
    expected = get_checkpoint(config["suite"])
    expected_dict = {
        "model_id": expected.model_id, "revision": expected.revision, "unnorm_key": expected.unnorm_key,
    }
    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        return expected_dict
    actual = {key: checkpoint[key] for key in ("model_id", "revision", "unnorm_key")}
    if actual != expected_dict:
        raise ValueError(
            f"checkpoint for suite {config['suite']!r} must match pinned registry: "
            f"expected {expected_dict}, got {actual}"
        )
    return actual


def _prepare_run(
    config: Dict[str, Any], source_config_path: str, experiment_id: Optional[str], output_dir: Optional[str],
) -> tuple[ExperimentLayout, ResolvedTaskCondition, Dict[str, Any], torch.device, torch.dtype]:
    if config.get("method", "none") != "none":
        raise NotImplementedError("only the Phase 4 vanilla baseline (method: none) is available")
    config = dict(config)
    config["checkpoint"] = _checkpoint_from_config(config)
    gpu = int(config.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = config.get("dtype", "bfloat16")
    dtype = DTYPE_MAP[dtype_name]
    layout = prepare_experiment(config, source_config_path, experiment_id, output_dir)
    redirected_log = configure_robosuite_logging(layout.logs_dir / "robosuite.log")
    resolution = resolve_task_condition(config["suite"], int(config["task_id"]), config.get("condition", "vanilla"))
    metadata = initial_metadata(config, source_config_path, str(device), dtype_name, resolution.to_dict())
    metadata["robosuite_log_path"] = redirected_log or "/tmp/robosuite.log"
    write_json(layout.logs_dir / "bddl_resolution.json", {
        "requested_condition": resolution.requested_condition,
        "requested_bddl_path": resolution.requested_bddl_path,
        "resolved_bddl_path": resolution.resolved_bddl_path,
        "actual_env_bddl_path": None,
    })
    write_json(layout.metadata_path, metadata)
    print(f"[*] Requested condition: {resolution.requested_condition}")
    print(f"[*] Requested BDDL: {resolution.requested_bddl_path}")
    print(f"[*] Resolved BDDL: {resolution.resolved_bddl_path}")
    return layout, resolution, config, device, dtype


def run_bddl_resolution_smoke(
    config: Dict[str, Any], source_config_path: str, experiment_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a BDDL and construct/close its environment without loading OpenVLA."""
    layout, resolution, config, _, _ = _prepare_run(config, source_config_path, experiment_id, output_dir)
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    env = None
    try:
        actual_bddl = str(Path(resolution.resolved_bddl_path).resolve())
        env = make_env(actual_bddl, int(config.get("resolution", 256)))
        metadata.update({"actual_env_bddl_path": actual_bddl, "smoke_only": True, "environment_created": True})
        resolution_log = {
            "requested_condition": resolution.requested_condition,
            "requested_bddl_path": resolution.requested_bddl_path,
            "resolved_bddl_path": resolution.resolved_bddl_path,
            "actual_env_bddl_path": actual_bddl,
        }
        write_json(layout.logs_dir / "bddl_resolution.json", resolution_log)
        write_json(layout.metadata_path, metadata)
        summary = {
            "experiment_dir": str(layout.directory), "smoke_only": True,
            "environment_created": True, **resolution_log,
        }
        write_json(layout.summary_path, summary)
        print(f"[*] Actual BDDL passed to environment: {actual_bddl}")
        return summary
    finally:
        if env is not None:
            env.close()


def run_rollout(
    config: Dict[str, Any], source_config_path: str, experiment_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the no-method OpenVLA baseline and write the required artifacts."""
    layout, resolution, config, device, dtype = _prepare_run(config, source_config_path, experiment_id, output_dir)
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    checkpoint = config["checkpoint"]
    method = resolve_method(config.get("method", "none"))()
    method.setup(config)
    resolution_px = int(config.get("resolution", 256))
    resize_size = int(config.get("resize_size", 224))
    center_crop = bool(config.get("center_crop", True))
    num_steps_wait = int(config.get("num_steps_wait", 10))
    max_steps = int(config.get("max_steps", 220))

    print(f"[*] Loading OpenVLA: {checkpoint['model_id']} @ {checkpoint['revision']}")
    load_start = time.time()
    processor, vla = load_openvla(
        checkpoint_id=checkpoint["model_id"], revision=checkpoint["revision"],
        attn_implementation=config.get("attn_implementation", "eager"), dtype=dtype, device=device,
    )
    metadata["model_load_time_sec"] = time.time() - load_start
    action_dim = vla.get_action_dim(checkpoint["unnorm_key"])
    if action_dim != ACTION_DIM:
        raise RuntimeError(f"expected action dim {ACTION_DIM}, got {action_dim}")

    env = None
    raw_actions: List[np.ndarray] = []
    final_actions: List[np.ndarray] = []
    per_step_records: List[Dict[str, Any]] = []
    success_flag: Optional[bool] = None
    termination_reason = "max_steps"
    try:
        actual_bddl = str(Path(resolution.resolved_bddl_path).resolve())
        print(f"[*] Building env for task {resolution.task_id}: {resolution.task_name}")
        env = make_env(actual_bddl, resolution_px)
        metadata["actual_env_bddl_path"] = actual_bddl
        resolution_log = {
            "requested_condition": resolution.requested_condition,
            "requested_bddl_path": resolution.requested_bddl_path,
            "resolved_bddl_path": resolution.resolved_bddl_path,
            "actual_env_bddl_path": actual_bddl,
        }
        write_json(layout.logs_dir / "bddl_resolution.json", resolution_log)
        print(f"[*] Actual BDDL passed to environment: {actual_bddl}")

        reset_seed = episode_seed(
            int(config["seed"]), resolution.suite, resolution.task_id, int(config.get("init_state_id", 0))
        )
        metadata["episode_reset_seed"] = reset_seed
        np.random.seed(reset_seed)
        env.reset()
        init_state, init_record = resolve_init_state(
            env=env, bddl_path=actual_bddl, suite=resolution.suite,
            condition=resolution.requested_condition, task_id=resolution.task_id,
            task_name=resolution.task_name, seed=int(config["seed"]),
            init_state_id=int(config.get("init_state_id", 0)), resolution=resolution_px,
        )
        metadata["init_state"] = init_record.to_dict()
        obs = env.set_init_state(init_state)
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        for timestep in range(max_steps):
            img = get_libero_image(obs, resize_size)
            proprio = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            observation = {"full_image": img, "state": proprio}
            if timestep == 0:
                metadata["first_observation"] = {
                    "agentview_sha256": _array_sha256(np.asarray(obs["agentview_image"])),
                    "model_image_sha256": _array_sha256(img),
                    "sim_state_sha256": _array_sha256(
                        np.ascontiguousarray(env.sim.get_state().flatten(), dtype=np.float64)
                    ),
                    "proprio": proprio.tolist(),
                }
            step_start = time.time()
            if method.predict_action(observation, resolution.instruction) is not None:
                raise RuntimeError("Phase 4 baseline must not override the base OpenVLA action")
            action_model = np.asarray(get_vla_action(
                vla=vla, processor=processor, base_vla_name=checkpoint["model_id"], obs=observation,
                task_label=resolution.instruction, unnorm_key=checkpoint["unnorm_key"],
                center_crop=center_crop, dtype=dtype,
            ), dtype=np.float64)
            latency_ms = (time.time() - step_start) * 1000.0
            if not np.isfinite(action_model).all():
                raise RuntimeError(f"non-finite action at step {timestep}: {action_model}")
            raw_actions.append(action_model.copy())
            action_applied = invert_gripper_action(normalize_gripper_action(action_model.copy(), binarize=True))
            final_actions.append(action_applied.copy())
            obs, _, done, info = env.step(action_applied.tolist())
            if isinstance(info, dict) and "success" in info:
                success_flag = bool(info["success"])
            else:
                checker = getattr(env, "check_success", None)
                success_flag = bool(checker()) if callable(checker) else None
            per_step_records.append({
                "timestep": timestep, "action_source": "base_policy", "action_model": raw_actions[-1].tolist(),
                "action_applied": final_actions[-1].tolist(), "latency_ms": latency_ms, "success": success_flag,
            })
            if timestep == 0 or timestep % 20 == 0:
                print(f"  step {timestep:3d}/{max_steps}: latency={latency_ms:.1f}ms success={success_flag}")
            if done:
                termination_reason = "success" if success_flag else "done_without_success"
                break
    except Exception as exc:
        metadata["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if env is not None:
            env.close()
        metadata["num_timesteps"] = len(per_step_records)
        metadata["success"] = bool(success_flag)
        metadata["termination_reason"] = termination_reason
        metadata["action_statistics"] = {"raw": _action_statistics(raw_actions), "applied": _action_statistics(final_actions)}
        write_json(layout.metadata_path, metadata)

    if raw_actions:
        np.save(layout.eval_dir / "actions_raw.npy", np.stack(raw_actions))
    if final_actions:
        np.save(layout.eval_dir / "actions_final.npy", np.stack(final_actions))
    with open(layout.eval_dir / "per_step_metrics.jsonl", "w", encoding="utf-8") as handle:
        for record in per_step_records:
            handle.write(json.dumps(record) + "\n")
    with open(layout.eval_dir / "actions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step"] + [f"raw_a{i}" for i in range(ACTION_DIM)] + [f"final_a{i}" for i in range(ACTION_DIM)] + ["success"])
        for index, record in enumerate(per_step_records):
            writer.writerow([index] + record["action_model"] + record["action_applied"] + [record["success"]])
    summary = {
        "experiment_dir": str(layout.directory), "success": metadata["success"],
        "num_timesteps": metadata["num_timesteps"], "termination_reason": termination_reason,
        "action_statistics": metadata["action_statistics"], "requested_condition": resolution.requested_condition,
        "requested_bddl_path": resolution.requested_bddl_path, "resolved_bddl_path": resolution.resolved_bddl_path,
        "actual_env_bddl_path": metadata["actual_env_bddl_path"],
    }
    write_json(layout.summary_path, summary)
    print(f"[*] Rollout complete: {metadata['num_timesteps']} steps, success={metadata['success']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 OpenVLA vanilla baseline")
    parser.add_argument("--config", required=True, help="Baseline YAML config")
    parser.add_argument("--experiment_id", help="Optional experiment directory name")
    parser.add_argument("--output_dir", help="Explicit override; paths outside the required root emit a warning")
    parser.add_argument("--gpu", type=int, help="Explicit GPU override recorded in the effective config snapshot")
    parser.add_argument("--smoke_bddl_resolution", action="store_true", help="Create/close resolved env without loading OpenVLA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.gpu is not None:
        config["gpu"] = args.gpu
    if args.smoke_bddl_resolution:
        summary = run_bddl_resolution_smoke(config, args.config, args.experiment_id, args.output_dir)
    else:
        summary = run_rollout(config, args.config, args.experiment_id, args.output_dir)
    print(f"[*] Experiment directory: {summary.get('experiment_dir', 'see metadata')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
