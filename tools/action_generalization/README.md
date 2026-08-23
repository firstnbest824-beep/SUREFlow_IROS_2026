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

- Implemented intervention: a task-scoped, parameter-free oracle global
  approach. It moves only to a configured target object's live simulator
  waypoint, then permanently hands off to vanilla OpenVLA for all grasp and
  local manipulation. It has no training path or trainable parameters.
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
          --config tools/action_generalization/configs/baseline.yaml

  The first intervention uses the same rollout entry point:

      python tools/action_generalization/eval.py \
          --config tools/action_generalization/configs/global_approach_v1.yaml

  `global_approach_v1.yaml` is deliberately limited to `libero_object` task 0
  (`alphabet_soup_1`) under `y0.1`.  Its target coordinate comes from the live
  perturbed simulator after reset and initial-state restore, never from a
  hard-coded training coordinate. The controller commands a closed-loop OSC
  delta translation to `target + [0, 0, 0.12]`, holds orientation (zero delta)
  and opens the Panda gripper (`-1`). At 5 cm from that waypoint it latches
  one-way to OpenVLA; it never returns to geometric control, so grasp, fine
  alignment, lifting, and placement remain unchanged vanilla behavior.

  The default result root is fixed at
  `/home/user/4TB/hwkim/action_generalization/`. The default experiment id is
  `YYYYMMDD_baseline_<suite>_<condition>_seed<N>`, and each run contains
  `config.yaml`, `metadata.json`, `logs/`, `checkpoints/`, `eval/`, and
  `summary.json`. An explicit `--output_dir` outside that root is allowed only
  as an intentional override and emits a warning. An explicit `--gpu` override
  is captured in the run's effective `config.yaml` and `metadata.json`.

  The Phase 4 smoke below verifies that a non-vanilla condition selects and
  constructs the shipped LIBERO-PRO BDDL without loading a model:

      python tools/action_generalization/eval.py \
          --config tools/action_generalization/configs/libero_object_y0.1_smoke.yaml \
          --smoke_bddl_resolution

## Shared modules

- `methods/base.py` — `ActionGeneralizationMethod`, the abstract interface a
  method implementation subclasses (`predict_action(observation, task_label,
  ...) -> action or None`, where `None` means "defer to the base policy").
  `NoOverrideMethod` is the baseline implementation. `global_approach.py`
  contains only controller state, runtime-object lookup, waypoint geometry,
  bounded translation and latch logic; it does not create shared diagnostic
  dependencies or modify OpenVLA.
- `configs/baseline.yaml` — the vanilla, no-method baseline config: checkpoint
  id/revision (via `tools/common/checkpoints.py`), task suite/id, init state
  id, seed, and max steps. Matches an existing diagnostics-line reference
  episode so a run through this config is directly comparable to
  already-collected data — see the comment at the top of that file.
- `evaluation/`, `utils/` — placeholders for future evaluation-metric and
  utility code. Empty (just `__init__.py`) in this phase.

## Reproducibility and test command

- `tools/common/seeding.py` seeds Python `random`, NumPy, PyTorch, and all
  available CUDA devices before a run; the per-reset fixture seed is separately
  derived by `episode_seed`.
- Future training must retain only `best` and `last` checkpoints. The Phase 4
  baseline does not train and creates no checkpoint files.
- Run the full regression suite from the existing environment with:

      /home/hwkim/micromamba/envs/vla-spatial-diagnostics/bin/python -m pytest tests/

  `tests/conftest.py` supplies writable Numba and Matplotlib cache locations;
  no test changes diagnostic assertions or redirects application behavior.

## Important

- No learned Phase 5 method (action-head redesign, flow matching, chunking,
  Mamba/SSM, diffusion, RL) is implemented. The global approach is a
  parameter-free geometric intervention and `train.py` remains intentionally
  unable to train any method.
- Do not modify `tools/openvla/`, `SUREFlow/`, `configs/`, `dataloader/`,
  `run.py`, or `tools/dry_run_*.py` from this line.
- Do not create a new conda env; use the existing
  `vla-spatial-diagnostics` env.
