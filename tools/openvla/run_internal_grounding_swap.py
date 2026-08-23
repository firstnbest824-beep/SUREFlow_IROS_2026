#!/usr/bin/env python3
"""Collect repeated one-frame OpenVLA grounding metrics on LIBERO-PRO swaps.

Each sample delegates model invocation and evaluation to the existing
``run_single_frame_probe``.  This runner only creates/reuses official swap BDDL
assets, verifies that the configured GT target actually changes init region,
and aggregates completed artifacts.  Predicted actions are never applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parent / "common"
for path in (str(_HERE), str(_COMMON)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment import initial_metadata, prepare_experiment, write_json  # noqa: E402
from libero_env import configure_robosuite_logging  # noqa: E402
from run_internal_grounding_probe import _load_config, run_single_frame_probe  # noqa: E402
from run_internal_grounding_repeat import aggregate_metric_rows, metric_rows_from_artifacts  # noqa: E402
from swap_grounding import resolve_target_moving_swap  # noqa: E402


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty swap metrics")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gt_position_range(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    unique = {}
    for row in rows:
        if row["valid"]:
            unique[row["sample_id"]] = (float(row["gt_centroid_u"]), float(row["gt_centroid_v"]))
    if len(unique) < 2:
        raise RuntimeError("swap collection produced fewer than two valid GT centroids")
    coordinates = np.asarray(list(unique.values()), dtype=np.float64)
    if np.allclose(coordinates.min(axis=0), coordinates.max(axis=0), rtol=0.0, atol=1e-6):
        raise RuntimeError("swap collection found no GT centroid variation across samples")
    return {"u": [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())], "v": [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())]}


def aggregate_existing_swap_artifacts(config: Dict[str, Any], experiment_dir: str | Path) -> Dict[str, Any]:
    """Finish parent CSV/JSON from complete sample artifacts without inference."""
    experiment_dir = Path(experiment_dir)
    rows: List[Dict[str, Any]] = []
    for index, swap_seed in enumerate(config["swap_seeds"]):
        sample_id = f"sample_{index:03d}_swap_seed{int(swap_seed)}_init{int(config['init_state_id']):03d}"
        eval_dir = experiment_dir / "samples" / sample_id / "eval"
        try:
            prediction = json.loads((eval_dir / "prediction.json").read_text(encoding="utf-8"))
            evaluation = json.loads((eval_dir / "evaluation.json").read_text(encoding="utf-8"))
            span = json.loads((eval_dir / "target_token_span.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"cannot aggregate incomplete swap sample {sample_id}") from exc
        rows.extend(metric_rows_from_artifacts(
            sample_id, int(swap_seed), int(config["init_state_id"]), [None, None, None],
            prediction, evaluation, target_phrase=span["phrase"],
        ))
    layer_summary = aggregate_metric_rows(rows)
    position_range = _gt_position_range(rows)
    _write_csv(experiment_dir / "eval" / "swap_sample_metrics.csv", rows)
    write_json(experiment_dir / "eval" / "swap_sample_metrics.json", {"rows": rows})
    write_json(experiment_dir / "eval" / "swap_layer_summary.json", {"layers": layer_summary})
    write_json(experiment_dir / "eval" / "swap_gt_position_range.json", position_range)
    metadata_path = experiment_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"completed_samples": len(config["swap_seeds"]), "gt_position_range": position_range, "layer_summary": layer_summary})
    write_json(metadata_path, metadata)
    summary = {"experiment_dir": str(experiment_dir), "sample_count": len(config["swap_seeds"]), "gt_position_range": position_range, "layer_summary": layer_summary, "action_was_applied": False}
    write_json(experiment_dir / "summary.json", summary)
    return summary


def run_swap_grounding(
    config: Dict[str, Any], source_config_path: str, experiment_id: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Run several target-moving swap BDDLs through unchanged single-frame inference."""
    swap_seeds = config.get("swap_seeds")
    if not isinstance(swap_seeds, list) or len(swap_seeds) < 2:
        raise ValueError("swap_seeds must contain at least two integer seeds")
    if len({int(seed) for seed in swap_seeds}) != len(swap_seeds):
        raise ValueError("swap_seeds must be unique")
    parent_layout = prepare_experiment(config, source_config_path, experiment_id, output_dir)
    configure_robosuite_logging(parent_layout.logs_dir / "robosuite.log")

    resolved_samples = []
    for swap_seed in swap_seeds:
        resolution, swap_provenance = resolve_target_moving_swap(config, int(swap_seed))
        resolved_samples.append((int(swap_seed), resolution, swap_provenance))
    first_resolution = resolved_samples[0][1]
    metadata = initial_metadata(config, source_config_path, "delegated-per-sample", config.get("dtype", "bfloat16"), first_resolution.to_dict())
    metadata.update({
        "runner_scope": "LIBERO-PRO swap repeats; each action is recorded and never applied",
        "prediction_inputs": "RGB + instruction + frozen OpenVLA internal tensors only",
        "evaluation_inputs": "simulator segmentation read only after prediction",
        "swap_samples": [provenance for _, _, provenance in resolved_samples],
    })
    write_json(parent_layout.metadata_path, metadata)

    rows: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    for index, (swap_seed, resolution, provenance) in enumerate(resolved_samples):
        sample_id = f"sample_{index:03d}_swap_seed{swap_seed}_init{int(config['init_state_id']):03d}"
        sample_dir = parent_layout.directory / "samples" / sample_id
        if sample_dir.exists():
            raise FileExistsError(f"refusing to overwrite sample directory: {sample_dir}")
        sample_config = dict(config)
        sample_config.update({"method": "internal_grounding_swap_probe", "condition": "swap", "seed": swap_seed})
        sample_summary = run_single_frame_probe(
            sample_config, source_config_path, experiment_id=sample_id,
            output_dir=str(sample_dir), resolution_override=resolution,
        )
        prediction = json.loads((sample_dir / "eval" / "prediction.json").read_text(encoding="utf-8"))
        evaluation = json.loads((sample_dir / "eval" / "evaluation.json").read_text(encoding="utf-8"))
        span = json.loads((sample_dir / "eval" / "target_token_span.json").read_text(encoding="utf-8"))
        target_xyz = [None, None, None]
        rows.extend(metric_rows_from_artifacts(
            sample_id, swap_seed, int(config["init_state_id"]), target_xyz,
            prediction, evaluation, target_phrase=span["phrase"],
        ))
        samples.append({"sample_id": sample_id, "summary": sample_summary, **provenance})

    layer_summary = aggregate_metric_rows(rows)
    position_range = _gt_position_range(rows)
    _write_csv(parent_layout.eval_dir / "swap_sample_metrics.csv", rows)
    write_json(parent_layout.eval_dir / "swap_sample_metrics.json", {"rows": rows})
    write_json(parent_layout.eval_dir / "swap_layer_summary.json", {"layers": layer_summary})
    write_json(parent_layout.eval_dir / "swap_gt_position_range.json", position_range)
    metadata.update({"completed_samples": len(samples), "samples": samples, "gt_position_range": position_range, "layer_summary": layer_summary})
    write_json(parent_layout.metadata_path, metadata)
    summary = {"experiment_dir": str(parent_layout.directory), "sample_count": len(samples), "gt_position_range": position_range, "layer_summary": layer_summary, "action_was_applied": False}
    write_json(parent_layout.summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeated OpenVLA internal grounding on LIBERO-PRO swap BDDLs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_id")
    parser.add_argument("--output_dir")
    parser.add_argument("--aggregate_existing", help="finish parent metrics from complete sample artifacts; never runs a model")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.aggregate_existing:
        summary = aggregate_existing_swap_artifacts(config, args.aggregate_existing)
    else:
        summary = run_swap_grounding(config, args.config, args.experiment_id, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
