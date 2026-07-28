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

These modules are the single source of truth for the probe dry-run, the
failure-screening runner, and the single-vanilla-rollout runner; all three
scripts must import them rather than reimplement their logic.

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

  Vision/projector probe points and their verified shapes (batch 1):

  | stage | hooked module | hook | shape |
  |---|---|---|---|
  | `final_vision_dinov2` | `vision_backbone.featurizer` | forward | `[1, 256, 1024]` |
  | `final_vision_siglip` | `vision_backbone.fused_featurizer` | forward | `[1, 256, 1152]` |
  | `projector_input` | `projector` | forward_pre | `[1, 256, 2176]` |
  | `projector_output` | `projector` | forward | `[1, 256, 4096]` |

  The dry-run asserts `cat([dinov2, siglip], dim=2) == projector_input` exactly
  (`torch.equal`, `max_abs_diff == 0.0`), which is what proves the vision hooks
  sit on the tensors the model consumes.

  > **⚠️ Results collected before commit `f935f19` are not usable for probing.**
  > Earlier revisions hooked `vision_backbone.featurizer.blocks.23` and
  > `vision_backbone.fused_featurizer.blocks.26` and saved them as
  > `final_vision_dinov2` / `final_vision_siglip` with shapes `[1, 261, 1024]`
  > and `[1, 256, 1152]`. **The model never consumes those tensors.**
  > `PrismaticVisionBackbone.__init__` monkey-patches each featurizer's `forward`
  > to `get_intermediate_layers(n={len(blocks) - 2})`, so it consumes the
  > **second-to-last** block — DINOv2 block 22 of 24, SigLIP block 25 of 27 —
  > with DINOv2's 5 prefix tokens (CLS + 4 registers) stripped and no final norm.
  > The last block still executes, so the old hooks fired and produced
  > plausible-looking arrays, but their output is discarded. On the real
  > checkpoint they differ from the consumed features by rel_L2 1.61 / 1.17
  > (cos 0.59 / 0.66) — a different representation, not a small numerical drift.
  > Any `feature_final_vision_*.npy` from an older run must be re-collected.
  >
  > Those runs also contain `feature_projector_penultimate.npy` (a hook on
  > `projector.fc3`). For the fused backbone `fc3` is the final layer, so its
  > output *is* the projector's return value — the same tensor object, and the
  > file is bitwise-identical to `feature_projector_output.npy`. That hook has
  > been removed; the file is redundant, not wrong.
  `lm_head_logits`'s own prefill array (~35 MB/timestep) is skipped on disk by
  default in `dry_run_openvla_spatial_probe.py`; pass `--save_lm_head_logits` to
  persist it. All other streams (vision features, projector features, LLM
  hidden states, `pre_action_hidden`) are always saved.

- `task_phase_resolver.py` — the single source of truth for whether a timestep
  is `pre_grasp`, `post_grasp`, or `uncertain`, and therefore which resolved
  entity (`spatial_task_resolver`'s `source_object` / `destination_object`) is
  the "relevant entity" for that timestep. No object name is hard-coded and no
  image-based heuristic is used; only simulator state drives the decision
  (gripper qpos, world poses, MuJoCo contact via robosuite's
  `MujocoEnv.check_contact`). A single noisy frame never flips the phase --
  contact, proximity, gripper closure, and object displacement/comovement are
  combined into a confidence score, and a transition only fires once that
  evidence has been sustained for several consecutive timesteps (thresholds are
  code constants in `PhaseThresholds`, recorded verbatim via `config_snapshot()`).
  `uncertain` never guesses a relevant entity.

  **Comovement means *moving together*, not *a constant distance*.** Requiring
  only that the gripper-to-source distance is stable makes a closed gripper
  hovering over a resting object score as a grasp — two stationary bodies
  trivially keep a constant offset. Comovement therefore additionally requires
  that *both* bodies actually travelled at least `min_comovement_motion_m` over
  the window while their relative offset stayed within
  `comovement_relative_drift_m`.

  Used by `dry_run_openvla_spatial_probe.py`, `run_failure_screening.py`, and
  `run_single_vanilla_rollout.py`. `save_phase_timeline()` writes the shared
  artifact set for all three: `phase_timeline.csv`, `phase_timeline.jsonl`,
  `phase_summary.json`, `phase_transition_summary.json`,
  `uncertain_timesteps.json`, `relevant_entity_timeline.csv`. Per-timestep
  records get their phase columns from `per_step_phase_fields()`.
  Self-test (no simulator required):

      python tools/openvla/task_phase_resolver.py

  `run_failure_screening.py --action_parity_check` runs a short vanilla episode
  twice (phase resolver on vs. off, identical seed/init state) and reports the
  max abs action difference, to verify phase logging never changes policy
  behavior.

## Activation collection

- `collect_activation_pilot.py` — runs a real LIBERO rollout and saves, for every
  control step, the activations together with the observation, action,
  proprioceptive state and task phase that belong to that same step.

      python tools/openvla/collect_activation_pilot.py --max_steps 90

  **Timestep convention.** At control step `k` the loop is
  `obs_pre → action = policy(obs_pre) → obs_post = env.step(action)`.
  Everything stored under timestep `k` derives from `obs_pre`: the activations,
  the proprioceptive vector fed to the model, the action, *and the task-phase
  label*. `obs_post` is stored separately so `obs_post(k) == obs_pre(k+1)` can be
  checked.

  > ⚠️ `run_single_vanilla_rollout.py` and `run_failure_screening.py` call the
  > phase resolver **after** `env.step`, so they label step `k` with `obs_post(k)`.
  > For rollout-level questions ("when did the grasp happen") that is fine. For
  > probe training it would put every label one control step ahead of the
  > activation it is paired with, so the collector uses the `obs_pre` convention
  > and `activation_integrity.check_alignment` enforces it.

- `activation_integrity.py` — simulator-free verification of a collected run.
  Six check groups, each PASS/WARNING/FAIL with the numbers behind the verdict:
  **A** counts agree, **B** shapes match the spec above (including the
  `dinov2 + siglip == projector_input` channel invariant), **C** value sanity,
  **D** temporal alignment / off-by-one, **E** hooks fired the expected number of
  times and no tensor was reused, **F** the episode actually changed. Runnable on
  any past run:

      python tools/openvla/activation_integrity.py --run_dir <run>

- `activation_dashboard.py` — builds a self-contained `dashboard.html`
  (frames, activation-norm and per-step-delta plots, action plot, hook table,
  representative timesteps with 16×16 token-norm heatmaps). Everything is inlined
  as data URIs, so the file can be copied anywhere.

### Known data characteristic: LLaMA massive activations

`llm_middle` and `llm_late` put ~1.5e4 into **exactly 2 of 4096 channels**
(indices 2533 and 1415 on this checkpoint) while the median stays ~0.4–1.25 and
p99.99 is ~25–43. `llm_early` shows none, and `pre_action_hidden` (post-RMSNorm)
peaks around 76. This is normal LLaMA behaviour, not corruption — the integrity
check therefore thresholds on the **bulk** magnitude (p99.9), not the maximum.

Consequence for probing: a probe trained on raw `llm_middle` / `llm_late` will be
dominated by those two channels. Standardize per channel (or use the
post-RMSNorm `pre_action_hidden`) before drawing conclusions about spatial
decodability.

## Important

- Do not import or load SUREFlow modules, checkpoints, or scalers from this path.
- Do not run SUREFlow training or evaluation here.
- Keep all new results under timestamped subdirectories of `/home/hwkim/env-audit/`.
