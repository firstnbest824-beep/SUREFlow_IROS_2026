#!/usr/bin/env python3
"""Repeat the unchanged single-frame internal-grounding probe and aggregate it.

Each sample delegates to ``run_single_frame_probe``: this module only supplies
different seed/init-state pairs and reads finished prediction/evaluation JSON
files afterwards.  It never calls an environment or model API itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parent / "common"
for path in (str(_HERE), str(_COMMON)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment import initial_metadata, prepare_experiment, write_json  # noqa: E402
from run_internal_grounding_probe import _load_config, run_single_frame_probe  # noqa: E402
from task_resolution import resolve_task_condition  # noqa: E402


REQUIRED_METRIC_FIELDS = (
    "gt_centroid_uv", "predicted_uv_model_input", "pixel_l2_error",
    "gt_overlapping_patch_rank", "top1_patch_hit", "top_k_patch_hit",
)


def _validate_repeat_samples(samples: Any) -> List[Dict[str, int]]:
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("repeat_samples must contain at least two seed/init_state pairs")
    normalized = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {"seed", "init_state_id"}:
            raise ValueError(f"repeat_samples[{index}] must contain exactly seed and init_state_id")
        normalized.append({"seed": int(sample["seed"]), "init_state_id": int(sample["init_state_id"])})
    if len({sample["seed"] for sample in normalized}) < 2:
        raise ValueError("repeat_samples must include multiple seeds")
    if len({sample["init_state_id"] for sample in normalized}) < 2:
        raise ValueError("repeat_samples must include multiple init_state_id values")
    if len({(sample["seed"], sample["init_state_id"]) for sample in normalized}) != len(normalized):
        raise ValueError("repeat_samples contains duplicate seed/init_state pairs")
    return normalized


def metric_rows_from_artifacts(
    sample_id: str, seed: int, init_state_id: int,
    prediction: Dict[str, Any], evaluation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Join already-completed prediction and EVAL ONLY records into CSV rows."""
    rows: List[Dict[str, Any]] = []
    for stage, predicted in prediction["stages"].items():
        metrics = evaluation["stages"].get(stage)
        if metrics is None:
            raise ValueError(f"evaluation lacks prediction stage {stage!r}")
        missing = [name for name in REQUIRED_METRIC_FIELDS if name not in metrics and name not in predicted]
        if missing:
            raise ValueError(f"missing metric fields for {stage}: {missing}")
        rows.append({
            "sample_id": sample_id,
            "seed": int(seed),
            "init_state_id": int(init_state_id),
            "layer": stage,
            "gt_centroid_u": None if metrics["gt_centroid_uv"] is None else metrics["gt_centroid_uv"][0],
            "gt_centroid_v": None if metrics["gt_centroid_uv"] is None else metrics["gt_centroid_uv"][1],
            "predicted_u": predicted["predicted_uv_model_input"][0],
            "predicted_v": predicted["predicted_uv_model_input"][1],
            "pixel_l2_error": metrics["pixel_l2_error"],
            "gt_overlapping_patch_rank": metrics["gt_overlapping_patch_rank"],
            "top1_patch_hit": metrics["top1_patch_hit"],
            "top5_patch_hit": metrics["top_k_patch_hit"],
            "valid": metrics["valid"],
        })
    return rows


def aggregate_metric_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-layer population mean/std and hit rates from valid samples."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["layer"]), []).append(row)
    summary: Dict[str, Dict[str, Any]] = {}
    for layer, layer_rows in grouped.items():
        valid = [row for row in layer_rows if row["valid"]]
        if not valid:
            summary[layer] = {"samples": len(layer_rows), "valid_samples": 0}
            continue
        pixel = np.asarray([row["pixel_l2_error"] for row in valid], dtype=np.float64)
        rank = np.asarray([row["gt_overlapping_patch_rank"] for row in valid], dtype=np.float64)
        summary[layer] = {
            "samples": len(layer_rows), "valid_samples": len(valid),
            "pixel_l2_mean": float(pixel.mean()), "pixel_l2_std": float(pixel.std(ddof=0)),
            "gt_overlapping_patch_rank_mean": float(rank.mean()),
            "gt_overlapping_patch_rank_std": float(rank.std(ddof=0)),
            "top1_success_rate": float(np.mean([row["top1_patch_hit"] for row in valid])),
            "top5_success_rate": float(np.mean([row["top5_patch_hit"] for row in valid])),
        }
    return summary


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty repeated-probe metrics table")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_repeated_probe(
    config: Dict[str, Any], source_config_path: str, experiment_id: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Run configured sample pairs through the existing single-snapshot path."""
    samples = _validate_repeat_samples(config.get("repeat_samples"))
    parent_layout = prepare_experiment(config, source_config_path, experiment_id, output_dir)
    resolution = resolve_task_condition(config["suite"], int(config["task_id"]), config["condition"])
    parent_metadata = initial_metadata(config, source_config_path, "delegated-per-sample", config.get("dtype", "bfloat16"), resolution.to_dict())
    parent_metadata.update({
        "runner_scope": "repeat of single-frame probes; each action is recorded and never applied",
        "sample_pairs": samples,
        "single_probe_reused": "run_internal_grounding_probe.run_single_frame_probe",
    })
    write_json(parent_layout.metadata_path, parent_metadata)

    rows: List[Dict[str, Any]] = []
    sample_summaries = []
    for index, sample in enumerate(samples):
        sample_id = f"sample_{index:03d}_seed{sample['seed']}_init{sample['init_state_id']:03d}"
        sample_dir = parent_layout.directory / "samples" / sample_id
        if sample_dir.exists():
            raise FileExistsError(f"refusing to overwrite sample directory: {sample_dir}")
        sample_config = dict(config)
        sample_config.pop("repeat_samples", None)
        sample_config["method"] = "internal_grounding_probe"
        sample_config.update(sample)
        sample_summary = run_single_frame_probe(
            sample_config, source_config_path, experiment_id=sample_id, output_dir=str(sample_dir),
        )
        prediction = json.loads((sample_dir / "eval" / "prediction.json").read_text(encoding="utf-8"))
        evaluation = json.loads((sample_dir / "eval" / "evaluation.json").read_text(encoding="utf-8"))
        rows.extend(metric_rows_from_artifacts(sample_id, sample["seed"], sample["init_state_id"], prediction, evaluation))
        sample_summaries.append(sample_summary)

    aggregate = aggregate_metric_rows(rows)
    _write_rows_csv(parent_layout.eval_dir / "repeat_sample_metrics.csv", rows)
    write_json(parent_layout.eval_dir / "repeat_sample_metrics.json", {"rows": rows})
    write_json(parent_layout.eval_dir / "repeat_layer_summary.json", {"layers": aggregate})
    parent_metadata.update({"completed_samples": len(samples), "layer_summary": aggregate})
    write_json(parent_layout.metadata_path, parent_metadata)
    summary = {
        "experiment_dir": str(parent_layout.directory), "sample_count": len(samples),
        "action_was_applied": False, "layer_summary": aggregate,
        "sample_experiments": [item["experiment_dir"] for item in sample_summaries],
    }
    write_json(parent_layout.summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat frozen single-frame internal-grounding probes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_id")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    summary = run_repeated_probe(_load_config(args.config), args.config, args.experiment_id, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
