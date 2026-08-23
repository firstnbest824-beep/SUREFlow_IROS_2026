"""Experiment-layout and provenance helpers for action-generalization baselines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import torch
import yaml


ACTION_GENERALIZATION_ROOT = Path("/home/user/4TB/hwkim/action_generalization")
CHECKPOINT_POLICY = {"retain": ["best", "last"], "save_every_step": False}


def _git_output(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return None


def git_provenance() -> Dict[str, Any]:
    """Capture actual repository identity at experiment start."""
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=Path(__file__).resolve().parents[2]
    ).returncode != 0
    untracked = subprocess.run(
        ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE, check=False
    ).stdout.decode("utf-8").strip()
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_working_tree": bool(dirty or untracked),
    }


def runtime_environment() -> Dict[str, Any]:
    """Return actual runtime and accelerator information without placeholders."""
    env_name = (
        os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("MAMBA_DEFAULT_ENV")
        or Path(sys.prefix).name
    )
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "environment_name": env_name,
        "environment_prefix": sys.prefix,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def default_experiment_id(config: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """Return the fixed baseline experiment-id format."""
    now = now or datetime.now()
    method = "baseline" if config.get("method", "none") == "none" else str(config["method"])
    return "{}_{}_{}_{}_seed{}".format(
        now.strftime("%Y%m%d"), method, config["suite"], config.get("condition", "vanilla"),
        int(config.get("seed", 0)),
    )


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path
    directory: Path
    config_path: Path
    metadata_path: Path
    logs_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    summary_path: Path


def prepare_experiment(
    config: Dict[str, Any], source_config_path: str | Path, experiment_id: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    root: Path = ACTION_GENERALIZATION_ROOT,
) -> ExperimentLayout:
    """Create the required layout and snapshot the exact config used for a run."""
    root = Path(root).expanduser().resolve()
    experiment_id = experiment_id or default_experiment_id(config)
    if output_dir is None:
        directory = root / experiment_id
    else:
        directory = Path(output_dir).expanduser().resolve()
        try:
            directory.relative_to(root)
        except ValueError:
            warnings.warn(
                f"output override {directory} is outside required root {root}", RuntimeWarning
            )

    logs_dir = directory / "logs"
    checkpoints_dir = directory / "checkpoints"
    eval_dir = directory / "eval"
    for path in (logs_dir, checkpoints_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_path = directory / "config.yaml"
    # Snapshot the effective configuration, including explicit runtime overrides.
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return ExperimentLayout(
        root=root,
        directory=directory,
        config_path=config_path,
        metadata_path=directory / "metadata.json",
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        eval_dir=eval_dir,
        summary_path=directory / "summary.json",
    )


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def initial_metadata(
    config: Dict[str, Any], source_config_path: str | Path, device: str, dtype: str,
    resolution: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the required metadata record from real process and resolver values."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(),
        "runtime": runtime_environment(),
        "random_seed": int(config["seed"]),
        "method": "baseline" if config.get("method", "none") == "none" else config["method"],
        "model_name": config["checkpoint"]["model_id"],
        "checkpoint_revision": config["checkpoint"]["revision"],
        "unnorm_key": config["checkpoint"]["unnorm_key"],
        "libero_suite": resolution["suite"],
        "task_id": resolution["task_id"],
        "task_name": resolution["task_name"],
        "condition": resolution["requested_condition"],
        "requested_bddl_path": resolution["requested_bddl_path"],
        "resolved_bddl_path": resolution["resolved_bddl_path"],
        "vanilla_bddl_path": resolution["vanilla_bddl_path"],
        "bddl_resolution": resolution,
        "config_path": str(Path(source_config_path).resolve()),
        "config_snapshot": config,
        "device": device,
        "dtype": dtype,
        "gpu_index": int(config.get("gpu", 0)),
        "checkpoint_policy": CHECKPOINT_POLICY,
    }
