# Action Generalization — Method Screening

This directory screens methods that might make OpenVLA more robust to
LIBERO-PRO spatial perturbation *faster* than understanding the failure
would (that is `tools/openvla/`'s job — spatial diagnostics). It is
intentionally separate from both the legacy SUREFlow code (`SUREFlow/`,
`configs/`, `dataloader/`, `run.py`, `tools/dry_run_*`) and the active
diagnostics line (`tools/openvla/`).

**Do not import from `tools/openvla/`; use `tools/common/` for shared
infra.** `tools/openvla/`'s modules are the diagnostics line's single source
of truth for spatial-failure analysis (task-phase resolution, entity role
resolution, perturbation/change detection, probing) and must stay untouched
by this line, the same way `tools/openvla/README.md` states diagnostics code
must not import or run legacy SUREFlow modules. `tools/common/` holds the
genuinely-reusable infra (model loading, LIBERO env construction, image
preprocessing, checkpoint bookkeeping) as physical copies, not re-exports —
see `tools/common/README.md` for that contract.

## Scope

- Candidate methods (none chosen yet, none implemented in this phase):
  action representation, action head design, flow matching, chunking, and
  eventually Mamba/SSM sequence models.
- Primary model under test: `openvla/openvla-7b-finetuned-libero-spatial`
  (same checkpoint the diagnostics line uses — see
  `tools/common/checkpoints.py`).
- Research question: does changing the action representation / head / chunk
  horizon reduce LIBERO-PRO spatial-perturbation failure faster than a
  from-scratch fix informed by the diagnostics line's causal analysis?

## Scripts

- `train.py` — single training entry point. Takes `--config <path>`, resolves
  the named method from a small registry, and delegates to it. Right now the
  registry is empty (no method has a training procedure yet); every method
  name raises a clear `NotImplementedError` until Phase 5 adds the first one.
- `eval.py` — single evaluation entry point. Same config-driven pattern.
  Supports running with **no method override** — `method: none` in a config
  resolves to `methods.base.NoOverrideMethod`, which defers to the base
  OpenVLA policy through the same rollout loop every future method will use.
  This is what `configs/baseline.yaml` uses for the vanilla-baseline
  reproduction.

      python tools/action_generalization/eval.py \
          --config tools/action_generalization/configs/baseline.yaml \
          --output_dir <run>/eval/libero_spatial/vanilla

## Shared modules

- `methods/base.py` — `ActionGeneralizationMethod`, the abstract interface a
  method implementation subclasses (`predict_action(observation, task_label,
  ...) -> action or None`, where `None` means "defer to the base policy").
  `NoOverrideMethod` is the only concrete subclass in this phase.
- `configs/baseline.yaml` — the vanilla, no-method baseline config: checkpoint
  id/revision (via `tools/common/checkpoints.py`), task suite/id, init state
  id, seed, and max steps. Matches an existing diagnostics-line reference
  episode so a run through this config is directly comparable to
  already-collected data — see the comment at the top of that file.
- `evaluation/`, `utils/` — placeholders for future evaluation-metric and
  utility code. Empty (just `__init__.py`) in this phase.

## Important

- No Phase 5 method (action-head redesign, flow matching, chunking,
  Mamba/SSM) is implemented anywhere in this directory yet. `methods/base.py`
  is an interface only.
- Do not modify `tools/openvla/`, `SUREFlow/`, `configs/`, `dataloader/`,
  `run.py`, or `tools/dry_run_*.py` from this line.
- Do not create a new conda env; use the existing
  `vla-spatial-diagnostics` env.
