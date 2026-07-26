# OpenVLA Spatial Diagnostics — Active Experimental Path

This directory contains the active OpenVLA-based spatial-diagnostics experiments.
It is intentionally separate from the legacy SUREFlow code (`SUREFlow/`,
`configs/`, `dataloader/`, `run.py`, and `tools/dry_run_*`).

## Scope

- Primary model: `openvla/openvla-7b-finetuned-libero-spatial`
- Secondary model: OpenVLA-OFT (future)
- Cross-architecture validation: π0.5-LIBERO (future)
- Research questions:
  1. Where does spatial information weaken between the vision encoder and action generation under LIBERO-PRO position perturbations?
  2. Does the final action fail to use spatial information even when it remains in internal representations?
  3. Can targeted repairs improve LIBERO-PRO generalization while preserving vanilla LIBERO performance?

## Scripts

- `run_single_vanilla_rollout.py` — vanilla LIBERO-Spatial rollout with OpenVLA.
- `inspect_openvla_architecture.py` — inspect module tree and tensor shapes for hook placement.
- `dry_run_openvla_spatial_probe.py` — collect observations, labels, and internal representations for probe analysis.
- `run_failure_screening.py` — compare vanilla and LIBERO-PRO position-perturbation episodes.

## Important

- Do not import or load SUREFlow modules, checkpoints, or scalers from this path.
- Do not run SUREFlow training or evaluation here.
- Keep all new results under timestamped subdirectories of `/home/hwkim/env-audit/`.
