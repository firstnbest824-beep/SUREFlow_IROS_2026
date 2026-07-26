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

## Shared modules

These three modules are the single source of truth for the probe dry-run and the
failure-screening runner; both scripts must import them rather than reimplement
their logic.

- `spatial_task_resolver.py` — derives `source_object`, `destination_object`, the
  goal predicate, the entities moved by the LIBERO-PRO swap and the
  `swap_counterpart` from the BDDL file. No object name is hard-coded. It also
  labels the research-stage semantics (`pre_grasp_relevant_entity` = source,
  `post_grasp_relevant_entity` = destination). Runnable standalone:

      python tools/openvla/spatial_task_resolver.py --bddl_path <file.bddl>

- `model_input_transform.py` — the one geometric chain (180° rotation → 224×224
  resize → center crop 0.9 → 224×224 resize) applied to RGB frames *and*
  segmentation masks, so spatial labels live in OpenVLA's input frame. Masks use
  NEAREST interpolation only. The probe asserts that this chain reproduces the
  image handed to the processor with a max abs pixel difference of 0.

- `probe_hooks.py` — call-order-preserving forward / forward-pre hooks. Every
  call is appended to a list, never overwritten. `lm_head`'s output is recorded
  as `lm_head_logits` (vocabulary logits); the hidden state entering `lm_head` is
  captured with a forward-pre hook as `pre_action_hidden` and saved as
  `pre_action_hidden_last_token.npy` with shape `[num_lm_head_calls, hidden_dim]`.
  The number of calls is measured, never assumed equal to the action dimension.

## Important

- Do not import or load SUREFlow modules, checkpoints, or scalers from this path.
- Do not run SUREFlow training or evaluation here.
- Keep all new results under timestamped subdirectories of `/home/hwkim/env-audit/`.
