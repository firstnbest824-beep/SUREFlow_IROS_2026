#!/usr/bin/env python3
"""Future single-frame OpenVLA internal-grounding experiment runner.

This runner is intentionally a one-frame diagnostic: it takes the usual
stabilisation actions, invokes frozen OpenVLA exactly once, and never applies
the predicted action to the environment.  Its prediction phase has no access
to simulator segmentation.  Segmentation is read only afterwards to write
evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image

_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parent / "common"
for path in (str(_HERE), str(_COMMON)):
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoints import get_checkpoint  # noqa: E402
from experiment import initial_metadata, prepare_experiment, write_json  # noqa: E402
from image_transform import (  # noqa: E402
    describe_transform, get_libero_image, map_uv_raw_to_model_input, mask_statistics,
    mask_to_model_input, normalize_segmentation, rgb_to_model_input,
)
from init_state import resolve_init_state  # noqa: E402
from instruction_target import extract_source_phrase  # noqa: E402
from libero_env import configure_robosuite_logging  # noqa: E402
from openvla_model import (  # noqa: E402
    ACTION_DIM, get_libero_dummy_action, get_vla_action, load_openvla, quat2axisangle,
    tensor_to_numpy_for_artifact,
)
from seeding import episode_seed, seed_everything  # noqa: E402
from task_resolution import ResolvedTaskCondition, resolve_task_condition  # noqa: E402
from internal_grounding import (  # noqa: E402
    compute_cosine_grounding_map, locate_target_token_span, resolve_multimodal_layout,
)
from internal_grounding_evaluation import evaluate_grounding_prediction, save_grounding_overlay  # noqa: E402
from probe_hooks import ProbeHookManager  # noqa: E402


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("probe config must parse to a mapping")
    return config


def _assert_checkpoint(config: Dict[str, Any]) -> Dict[str, str]:
    expected = get_checkpoint(config["suite"])
    expected_dict = {"model_id": expected.model_id, "revision": expected.revision, "unnorm_key": expected.unnorm_key}
    supplied = config.get("checkpoint", expected_dict)
    actual = {key: supplied[key] for key in expected_dict}
    if actual != expected_dict:
        raise ValueError(f"checkpoint must match pinned suite registry: expected {expected_dict}, got {actual}")
    return actual


def _agentview_segmentation_key(observation: Dict[str, Any]) -> str:
    matches = [key for key in observation if "agentview" in key.lower() and "segmentation" in key.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one agentview segmentation observation, found {matches}")
    return matches[0]


def _require_prefill(hooks: ProbeHookManager, stage: str) -> np.ndarray:
    stream = hooks.stream_by_stage(stage)
    if stream is None or not stream.records:
        raise RuntimeError(f"required probe stage did not capture a prompt prefill: {stage}")
    tensor = stream.prefill_tensor()
    if tensor is None:
        raise RuntimeError(f"required probe stage has no prefill tensor: {stage}")
    return np.asarray(tensor, dtype=np.float32)


def run_single_frame_probe(
    config: Dict[str, Any], source_config_path: str, experiment_id: str | None = None,
    output_dir: str | None = None, resolution_override: Optional[ResolvedTaskCondition] = None,
) -> Dict[str, Any]:
    """Run one frozen OpenVLA prefill/generation call; never step its action."""
    checkpoint = _assert_checkpoint(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(config.get("gpu", 0)))
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPE_MAP[config.get("dtype", "bfloat16")]
    resolution = resolution_override or resolve_task_condition(
        config["suite"], int(config["task_id"]), config["condition"],
    )
    layout = prepare_experiment(config, source_config_path, experiment_id, output_dir)
    configure_robosuite_logging(layout.logs_dir / "robosuite.log")
    metadata = initial_metadata(config, source_config_path, str(device), str(config.get("dtype", "bfloat16")), resolution.to_dict())
    metadata["runner_scope"] = "single frame only; OpenVLA action is recorded and never env.step() applied"
    metadata["prediction_inputs"] = "RGB + instruction + frozen OpenVLA internal tensors only"
    metadata["evaluation_inputs"] = "simulator segmentation GT, read after prediction only"
    write_json(layout.metadata_path, metadata)

    processor, vla = load_openvla(checkpoint["model_id"], checkpoint["revision"], config.get("attn_implementation", "eager"), dtype, device)
    if vla.get_action_dim(checkpoint["unnorm_key"]) != ACTION_DIM:
        raise RuntimeError("unexpected OpenVLA action dimension")
    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=resolution.resolved_bddl_path, camera_heights=int(config["resolution"]),
        camera_widths=int(config["resolution"]),
    )
    hooks = None
    try:
        env.seed(0)
        reset_seed = episode_seed(int(config["seed"]), resolution.suite, resolution.task_id, int(config["init_state_id"]))
        np.random.seed(reset_seed)
        env.reset()
        init_state, init_record = resolve_init_state(
            env=env, bddl_path=resolution.resolved_bddl_path, suite=resolution.suite,
            condition=resolution.requested_condition, task_id=resolution.task_id, task_name=resolution.task_name,
            seed=int(config["seed"]), init_state_id=int(config["init_state_id"]), resolution=int(config["resolution"]),
        )
        obs = env.set_init_state(init_state)
        for _ in range(int(config.get("num_steps_wait", 10))):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                raise RuntimeError("environment terminated during dummy stabilisation")

        observation = {
            "full_image": get_libero_image(obs, int(config["resize_size"])),
            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
        }
        hooks = ProbeHookManager(vla)
        action, model_input_rgb, prepared = get_vla_action(
            vla=vla, processor=processor, base_vla_name=checkpoint["model_id"], obs=observation,
            task_label=resolution.instruction, unnorm_key=checkpoint["unnorm_key"],
            center_crop=bool(config["center_crop"]), dtype=dtype, return_model_input_image=True,
            return_prepared_inputs=True,
        )
        if not np.isfinite(action).all():
            raise RuntimeError("OpenVLA returned a non-finite diagnostic action")
        reconstructed = rgb_to_model_input(np.asarray(obs["agentview_image"]), int(config["resize_size"]), bool(config["center_crop"]))
        if not np.array_equal(reconstructed, model_input_rgb):
            raise RuntimeError("shared RGB transform does not equal the exact model input")

        # Prediction ends here.  No evaluation target identity or segmentation has
        # been read above this line.
        projector = _require_prefill(hooks, "projector_output")
        if projector.ndim != 3:
            raise RuntimeError(f"unexpected projector output shape {projector.shape}")
        num_visual_tokens = int(projector.shape[1])
        hidden_by_stage = {stage: _require_prefill(hooks, stage) for stage in config["readout_stages"]}
        first_hidden = next(iter(hidden_by_stage.values()))
        token_layout = resolve_multimodal_layout(prepared.input_ids, first_hidden, num_visual_tokens)
        tokenizer = getattr(processor, "tokenizer", processor)
        target_phrase = extract_source_phrase(resolution.instruction)
        target_span = locate_target_token_span(tokenizer, prepared.prompt, target_phrase, prepared.input_ids, token_layout)
        results = {
            stage: compute_cosine_grounding_map(stage, hidden, token_layout, target_span, "target_last_token", model_input_rgb.shape[:2], int(config.get("top_k", 5)))
            for stage, hidden in hidden_by_stage.items()
        }

        Image.fromarray(model_input_rgb).save(layout.eval_dir / "model_input_rgb.png")
        np.save(layout.eval_dir / "prepared_input_ids.npy", tensor_to_numpy_for_artifact(prepared.input_ids))
        np.save(layout.eval_dir / "prepared_attention_mask.npy", tensor_to_numpy_for_artifact(prepared.attention_mask))
        np.save(layout.eval_dir / "prepared_pixel_values.npy", tensor_to_numpy_for_artifact(prepared.pixel_values))
        (layout.eval_dir / "prompt.txt").write_text(prepared.prompt + "\n", encoding="utf-8")
        write_json(layout.eval_dir / "token_layout.json", token_layout.to_dict())
        write_json(layout.eval_dir / "target_token_span.json", target_span.to_dict())
        for stage, result in results.items():
            np.save(layout.eval_dir / f"grounding_{stage}.npy", result.scores.reshape(result.patch_grid_shape))
            write_json(layout.eval_dir / f"prediction_{stage}.json", result.to_dict())
        write_json(layout.eval_dir / "prediction.json", {
            "primary_readout": "target_last_token",
            "stages": {stage: result.to_dict() for stage, result in results.items()},
        })

        # EVALUATION ONLY: target identity and segmentation are accessed only after
        # every prediction artifact above has been fully created.
        evaluation_object = str(config["evaluation_target_object"])
        instance_to_id = dict(getattr(env, "instance_to_id", {}))
        if evaluation_object not in instance_to_id:
            raise RuntimeError(f"evaluation target missing from segmentation map: {evaluation_object!r}")
        segmentation = normalize_segmentation(obs[_agentview_segmentation_key(obs)])
        raw_gt_mask = segmentation == int(instance_to_id[evaluation_object])
        model_gt_mask = mask_to_model_input(
            raw_gt_mask, int(config["resize_size"]), bool(config["center_crop"]),
        )
        raw_gt_stats = mask_statistics(raw_gt_mask, int(config["min_mask_pixels"]))
        model_gt_stats = mask_statistics(model_gt_mask, int(config["min_mask_pixels"]))
        analytic_model_centroid = None
        if raw_gt_stats["centroid"] is not None:
            analytic_model_centroid = list(map_uv_raw_to_model_input(
                raw_gt_stats["centroid"][0], raw_gt_stats["centroid"][1], raw_gt_mask.shape,
                int(config["resize_size"]), bool(config["center_crop"]),
            ))
        np.save(layout.eval_dir / "gt_target_mask.npy", model_gt_mask.astype(np.uint8))
        evaluations = {}
        for stage, result in results.items():
            evaluation = evaluate_grounding_prediction(result, model_gt_mask, int(config["min_mask_pixels"]))
            evaluations[stage] = evaluation.to_dict()
            save_grounding_overlay(str(layout.eval_dir / f"overlay_{stage}.png"), model_input_rgb, result, evaluation)
        write_json(layout.eval_dir / "evaluation.json", {
            "evaluation_target_object": evaluation_object,
            "raw_mask_statistics": raw_gt_stats,
            "model_input_mask_statistics": model_gt_stats,
            "raw_centroid_mapped_analytically_to_model_input": analytic_model_centroid,
            "stages": evaluations,
        })
        metadata.update({
            "init_state": init_record.to_dict(), "diagnostic_action": np.asarray(action).tolist(),
            "action_was_applied": False, "model_input_transform_verified": True,
            "num_visual_tokens": num_visual_tokens, "prediction_completed_before_gt_evaluation": True,
            "model_input_transform": describe_transform(
                np.asarray(obs["agentview_image"]).shape[:2], int(config["resize_size"]), bool(config["center_crop"]),
            ),
        })
        write_json(layout.metadata_path, metadata)
        summary = {"experiment_dir": str(layout.directory), "action_was_applied": False, "prediction_stages": list(results), "evaluation_completed": True}
        write_json(layout.summary_path, summary)
        return summary
    finally:
        if hooks is not None:
            hooks.remove_hooks()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-frame frozen OpenVLA internal-grounding probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_id")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    summary = run_single_frame_probe(_load_config(args.config), args.config, args.experiment_id, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
