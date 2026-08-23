#!/usr/bin/env python3
"""Single evaluation entry point for action-generalization methods.

Config-driven, same pattern as ``train.py``. Unlike ``train.py``, this script
must support running with literally no method override -- the vanilla base
OpenVLA policy through ``methods.base.NoOverrideMethod`` -- because that is
exactly what the Phase 4 baseline reproduction needs. Every future method
runs through this same rollout loop; only ``method.predict_action`` differs.

Do not import from ``tools/openvla/``; use ``tools/common/`` for shared infra.

The rollout loop below (seeded reset -> dummy-step warm-up -> per-step
obs -> action -> env.step) mirrors the official collection pipeline's reset
semantics (``tools/openvla/collect_official_activations.py``, notably its use
of ``episode_seed`` to pin LIBERO's per-reset fixture placement) so that an
episode run through this script is directly comparable to episodes already
collected by that pipeline. It is a fresh implementation built from
``tools/common/`` primitives, not an import of that file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

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
from image_transform import get_libero_image  # noqa: E402
from init_state import resolve_init_state  # noqa: E402
from libero_env import make_env  # noqa: E402
from openvla_model import (  # noqa: E402
    ACTION_DIM,
    get_libero_dummy_action,
    get_vla_action,
    invert_gripper_action,
    load_openvla,
    normalize_gripper_action,
    quat2axisangle,
)
from seeding import episode_seed  # noqa: E402

#: method name (from a config's `method:` key) -> implementation class.
#: "none" is the only one implemented in this phase: it always returns None
#: from predict_action, i.e. defers to the base OpenVLA policy. This is what
#: the Phase 4 vanilla-baseline rollout uses.
METHOD_REGISTRY: Dict[str, Type[ActionGeneralizationMethod]] = {
    "none": NoOverrideMethod,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} did not parse to a dict: {type(config)}")
    return config


def resolve_method(method_name: str) -> Type[ActionGeneralizationMethod]:
    """Look up a method class by name, raising a clear error if unimplemented."""
    if method_name not in METHOD_REGISTRY:
        raise NotImplementedError(
            f"method {method_name!r} is not implemented for evaluation. "
            f"Registered eval methods: {sorted(METHOD_REGISTRY.keys())}. "
            "Only 'none' (defer to the base OpenVLA policy) exists until "
            "Phase 5 adds the first method."
        )
    return METHOD_REGISTRY[method_name]


def seeded_reset(env: Any, seed: int) -> Any:
    """``env.reset()`` with the fixture-placement RNG pinned.

    Same mechanism as ``tools/openvla/collect_official_activations.py``'s
    ``seeded_reset`` (line 241-244): seed the global numpy RNG immediately
    before ``reset()``, since LIBERO fixture placement is drawn from it.
    """
    np.random.seed(seed)
    return env.reset()


def run_rollout(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Run one rollout episode according to ``config`` and write artifacts to ``output_dir``.

    Returns a summary dict (also written to ``rollout_metadata.json``).
    """
    os.makedirs(output_dir, exist_ok=True)

    method_name = config.get("method", "none")
    suite = config["suite"]
    task_id = int(config["task_id"])
    init_state_id = int(config.get("init_state_id", 0))
    seed = int(config.get("seed", 0))
    condition = config.get("condition", "vanilla")
    resolution = int(config.get("resolution", 256))
    resize_size = int(config.get("resize_size", 224))
    center_crop = bool(config.get("center_crop", True))
    num_steps_wait = int(config.get("num_steps_wait", 10))
    max_steps = int(config.get("max_steps", 220))
    attn_implementation = config.get("attn_implementation", "eager")
    dtype = DTYPE_MAP[config.get("dtype", "bfloat16")]
    gpu = int(config.get("gpu", 0))

    checkpoint_cfg = config.get("checkpoint")
    if checkpoint_cfg:
        model_id = checkpoint_cfg["model_id"]
        revision = checkpoint_cfg["revision"]
        unnorm_key = checkpoint_cfg["unnorm_key"]
    else:
        spec = get_checkpoint(suite)
        model_id, revision, unnorm_key = spec.model_id, spec.revision, spec.unnorm_key

    method_cls = resolve_method(method_name)
    method = method_cls()
    method.setup(config)

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method_name,
        "suite": suite,
        "task_id": task_id,
        "init_state_id": init_state_id,
        "seed": seed,
        "condition": condition,
        "checkpoint": {"model_id": model_id, "revision": revision, "unnorm_key": unnorm_key},
        "resolution": resolution,
        "resize_size": resize_size,
        "center_crop": center_crop,
        "num_steps_wait": num_steps_wait,
        "max_steps": max_steps,
        "attn_implementation": attn_implementation,
        "dtype": config.get("dtype", "bfloat16"),
        "gpu": gpu,
        "device": str(device),
    }

    from libero.libero import benchmark  # local import: keep heavy deps out of module import time

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)
    bddl_path = task_suite.get_task_bddl_file_path(task_id)
    task_name = task.name
    instruction = task.language
    metadata["task_name"] = task_name
    metadata["instruction"] = instruction
    metadata["bddl_path"] = bddl_path

    print(f"[*] Loading OpenVLA: {model_id} @ {revision}")
    load_start = time.time()
    processor, vla = load_openvla(
        checkpoint_id=model_id,
        revision=revision,
        attn_implementation=attn_implementation,
        dtype=dtype,
        device=device,
    )
    metadata["model_load_time_sec"] = time.time() - load_start
    action_dim = vla.get_action_dim(unnorm_key)
    if action_dim != ACTION_DIM:
        raise RuntimeError(f"expected action dim {ACTION_DIM}, got {action_dim}")

    print(f"[*] Building env for task {task_id}: {task_name}")
    env = make_env(bddl_path, resolution)

    reset_seed = episode_seed(seed, suite, task_id, init_state_id)
    metadata["episode_reset_seed"] = reset_seed
    seeded_reset(env, reset_seed)

    init_state, init_record = resolve_init_state(
        env=env,
        bddl_path=bddl_path,
        suite=suite,
        condition=condition,
        task_id=task_id,
        task_name=task_name,
        seed=seed,
        init_state_id=init_state_id,
        resolution=resolution,
    )
    metadata["init_state_source"] = init_record.source
    metadata["init_state_sha256"] = init_record.sha256
    obs = env.set_init_state(init_state)

    dummy_action = get_libero_dummy_action()
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(dummy_action)

    raw_actions = []
    final_actions = []
    per_step_records = []
    success_flag: Optional[bool] = None
    termination_reason = "max_steps"

    try:
        for timestep in range(max_steps):
            img = get_libero_image(obs, resize_size)
            proprio = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            observation = {"full_image": img, "state": proprio}

            step_start = time.time()
            action_override = method.predict_action(observation, instruction)
            if action_override is not None:
                action_model = np.asarray(action_override, dtype=np.float64)
                action_source = method_name
            else:
                action_model = get_vla_action(
                    vla=vla,
                    processor=processor,
                    base_vla_name=model_id,
                    obs=observation,
                    task_label=instruction,
                    unnorm_key=unnorm_key,
                    center_crop=center_crop,
                    dtype=dtype,
                )
                action_source = "base_policy"
            latency_ms = (time.time() - step_start) * 1000.0

            if not np.isfinite(action_model).all():
                raise RuntimeError(f"non-finite action at step {timestep}: {action_model}")
            raw_actions.append(np.asarray(action_model, dtype=np.float64).copy())

            action_applied = invert_gripper_action(
                normalize_gripper_action(np.asarray(action_model, dtype=np.float64).copy(), binarize=True)
            )
            final_actions.append(action_applied.copy())

            obs, _, done, info = env.step(action_applied.tolist())

            if isinstance(info, dict) and "success" in info:
                success_flag = bool(info["success"])
            else:
                checker = getattr(env, "check_success", None)
                success_flag = bool(checker()) if callable(checker) else None

            per_step_records.append({
                "timestep": timestep,
                "action_source": action_source,
                "action_model": raw_actions[-1].tolist(),
                "action_applied": final_actions[-1].tolist(),
                "latency_ms": latency_ms,
                "success": success_flag,
            })

            if timestep % 20 == 0 or timestep == 0:
                print(f"  step {timestep:3d}/{max_steps}: latency={latency_ms:.1f}ms success={success_flag}")

            if done:
                termination_reason = "success" if success_flag else "done_without_success"
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    metadata["num_timesteps"] = len(per_step_records)
    metadata["success"] = bool(success_flag)
    metadata["termination_reason"] = termination_reason

    if raw_actions:
        np.save(os.path.join(output_dir, "actions_raw.npy"), np.stack(raw_actions))
    if final_actions:
        np.save(os.path.join(output_dir, "actions_final.npy"), np.stack(final_actions))
    with open(os.path.join(output_dir, "per_step_metrics.jsonl"), "w", encoding="utf-8") as f:
        for rec in per_step_records:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(output_dir, "actions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"raw_a{i}" for i in range(ACTION_DIM)] + [f"final_a{i}" for i in range(ACTION_DIM)] + ["success"])
        for i, rec in enumerate(per_step_records):
            writer.writerow([i] + rec["action_model"] + rec["action_applied"] + [rec["success"]])
    with open(os.path.join(output_dir, "rollout_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[*] Rollout complete: {metadata['num_timesteps']} steps, success={metadata['success']}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an action-generalization method (or the base policy)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config naming the method to evaluate (see configs/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write rollout artifacts to.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print(f"[*] Loaded config: {args.config}")
    print(f"[*] Output dir: {args.output_dir}")
    run_rollout(config, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
