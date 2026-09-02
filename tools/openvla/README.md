# OpenVLA Spatial Diagnostics — Active Experimental Path

This directory contains the active OpenVLA-based spatial-diagnostics experiments.
It is intentionally separate from the legacy SUREFlow code (`SUREFlow/`,
`configs/`, `dataloader/`, `run.py`, and `tools/dry_run_*`).

## Current research structure

The active work has two explicitly separate stages.

1. **Target-object grounding** will use Transformer relevance / attention
   attribution over the joint image-and-instruction OpenVLA path.  The retained
   generic building blocks are `probe_hooks.py` (forward tracing),
   `token_layout.py` (instruction/visual-token indices and patch mapping), and
   `grounding_evaluation.py` (segmentation **EVAL ONLY** metrics).  DINO,
   CLIP/SigLIP similarity, hidden-state cosine readout, and J_rep patch
   sensitivity are not main methods and are not active runners in this tree.
2. **Explicit spatial movement** uses vanilla-versus-perturbed trajectory
   collection, h/displacement diagnostics, grasp-phase annotations, and
   task-local J_cal analysis.  These artifacts must remain task-local and are
   not evidence for a cross-task average Jacobian.

The two stages meet only after a grounded target coordinate is available; a
grounding analysis must not silently become an action-connected J_rep method.

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

## Official LIBERO / LIBERO-PRO collection (schema v2)

`collect_official_activations.py` is the current collection path. It replaces the
pilot for anything that will be analysed, because it runs under the *official*
evaluation conditions and records which entity each perturbation actually moved.

    python tools/openvla/collect_official_activations.py \
        --suite libero_object --task_id 0 --condition x0.1 \
        --output_root outputs/<run> --num_episodes 2

### What each suite is a test of

The two suites answer different questions and must not be pooled:

| Suite | Perturbation | What generalises | Measured change class |
|---|---|---|---|
| `libero_object` | `x0.1`, `y0.1` | **source / pick** — the object to be picked moves | `clean_source_only` |
| `libero_object` | `x0.2` … `y0.5` | source moves **and a distractor is teleported out of the scene** | `source_and_distractor` |
| `libero_spatial` | `swap` | **destination / place** — but the swap partner moves too | `destination_and_distractor` |

Only `libero_object` `x0.1` / `y0.1` are single-role changes. `libero_spatial swap`
is **not** a clean destination-only condition and must never be described as one:
the destination and its swap partner both move (0.2683 m each on task 0).
Everything above `x0.1`/`y0.1` pushes a distractor to roughly x = +10 m, which is
a scene edit rather than a spatial perturbation.

`is_clean(change_class)` is the gate for causal-attribution analysis; confounded
conditions are still collected, just labelled so they can be analysed separately.

### The condition name is not the measurement

Two traps that the pipeline handles explicitly:

- **`x0.1` is a level, not a displacement.** Measured on the shipped assets,
  displacement = level × 0.7, so `x0.1` moves the source **7 cm**. The manifest
  stores `requested_level` and `measured_translation_m` as separate fields and the
  validator fails an episode that records one without the other.
- **The profile differs per task.** Static analysis over all ten `libero_object`
  tasks said `y0.4`/`y0.5` move the basket; on task 0 they do not. So the changed
  entity is measured **per episode**, by resetting the vanilla and perturbed BDDL
  and diffing every entity pose — never inferred from the suite or condition name.

### Modules

- `entity_role_resolver.py` — source / destination / distractor / fixture from the
  BDDL goal, with region → owner normalisation (`basket_1_contain_region` →
  `basket_1`, whose pose is what actually moves).
- `changed_entity_detector.py` — pose diff of two resets into one of eight
  `change_class` values. Threshold 0.05 m sits above LIBERO's per-reset placement
  jitter; anything beyond 5 m is flagged as having left the scene.
- `official_task_pair_resolver.py` — binds each suite to its checkpoint revision
  and enumerates the conditions it genuinely supports. Rejects a non-integer seed
  (the official config's `configs.get("seed", int)` defaults to the *type object*).
- `init_state_freezer.py` — uses the shipped `.pruned_init` when one exists;
  otherwise captures the state **once** under a fixed seed into
  `assets/frozen_init_states/` and refuses to regenerate it.
- `segmentation_label_provider.py` — analysis ground truth only, never a policy
  input. See below.
- `activation_episode_writer.py` / `validate_activation_episode.py` — see below.

### Segmentation labels without touching the rollout

Segmentation is obtained as an **extra read-only render at the same simulator
state** on the existing `OffScreenRenderEnv`. No `SegmentationRenderEnv`, no extra
`env.step()`, no state write. `verify_segmentation_equivalence` runs at the start
of every episode and the episode aborts if it fails; measured on both suites, RGB
before and after the segmentation renders is **bit-identical** and the full sim
state vector is unchanged.

Per entity, per camera (`agentview` + `robot0_eye_in_hand`): mask pixel count and
fraction, normalised UV of the object origin, mask centroid and bbox, `in_frame`,
and a visibility label that distinguishes `occluded` (projects inside the image
but contributes no pixels) from `out_of_view` (projects outside it).

**Three pixel frames exist and confusing them silently corrupts every label:**

    raw     = sim.render(...)      # what obs["agentview_image"] is
    upright = raw[::-1]            # what robosuite's projection returns
    policy  = raw[::-1, ::-1]      # what get_libero_image feeds OpenVLA

`policy` is a 180° rotation, so it differs from `upright` by a *horizontal* flip
as well. All `uv` fields are in the **policy** frame; `uv_pixel_upright` is kept
alongside so the transform stays auditable.

### Timestep contract

Enforced by the structure of `_collect_one_timestep`, not by a comment:

    obs_t → label_t → model forward / activation_t → action_t → env.step(action_t)

Every field at index `t` describes the same simulator state — the one before
`action_t` was applied. Each record carries `sim_state_sha` (state observed) and
`next_sim_state_sha` (state after the step), so the validator can **prove** that
`record[t].next_sim_state_sha == record[t+1].sim_state_sha` rather than trust the
declaration. A label taken from `obs_{t+1}` breaks that chain and fails check F.

### Episode layout and validation

    <root>/<suite>/<condition>/task_NN/seed_NNN/episode_NNN/
        manifest.json  per_step_metrics.jsonl  COMPLETE
        activations/step_NNNNNN.npz     images/  overlays/  rollout.mp4

Written into a sibling `.partial` directory and moved into place with a single
`os.rename` only after the manifest and `COMPLETE` marker exist, so a reader never
sees a half-written episode and an interrupted run is never silently reused.
Activations are cast to fp16 **on the storage copy only**; the model's tensors are
untouched.

    python tools/openvla/validate_activation_episode.py <root> --recursive

Checks **A** manifest/schema, **B** COMPLETE, **C** counts and contiguity,
**D** all 8 stages with stable shapes, **E** NaN/Inf and magnitude, **F** the
state-hash chain above, **G** BDDL / init-state / checkpoint hashes, **H** roles,
`change_class` and level-vs-measured separation, **I** phase ↔ relevant-entity
consistency, **J** segmentation label well-formedness.

### Cost

Measured on `libero_spatial` task 0 vanilla (81 steps, success):
**10.97 MB/timestep**, ~484 ms/timestep of inference. Per stage, compressed:
`llm_early` / `llm_middle` / `llm_late` / `pre_action_hidden` ≈ 1.83 MB each,
`projector_output` 1.62 MB, `projector_input` 0.87 MB, SigLIP 0.45 MB,
DINOv2 0.40 MB. The four LLM stages are ~68% of the bytes.

## Activation collection (pilot — superseded)

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

### Storage layout: SSD staging → 4TB ext4 final

This server has one NVMe (`/`, ~108 GB free) and two 4 TB spinning disks. The
collector produces ~28.35 MB/timestep at ~318 ms/timestep, i.e. **~89 MB/s
sustained**. Measured with the real write pattern:

| target | device | true sustained write | verdict |
|---|---|---|---|
| `/` (nvme0n1p2) | NVMe SSD | **596 MB/s** | fast, but only ~108 GB free — must not be the destination |
| `/home/user/4TB` (sdb) | **rotational** HDD, ext4 | **85.9 MB/s** | 1.9 TB free, but just under the 89 MB/s the collector needs |
| `/home/HDD` (sda2) | rotational HDD, ntfs-3g/FUSE | slower still | archive only, never the hot path |

So neither disk alone is right: the SSD has speed but no room, the HDD has room
but not quite enough speed. Hence **staged** mode — write each episode to a
bounded SSD buffer, then transfer to the HDD in the background while the next
episode is already being collected. The ~3.5 MB/s shortfall accumulates only
~10 GB of backlog over a full 250 GB run, well inside the 32 GB staging budget.

#### Modes

- `direct` — write straight to `--output_root`. Correct when the destination is
  fast enough, or when staging would not help.
- `staged` — SSD buffer + background transfer. If staging is unsafe (same
  filesystem as the destination, rotational staging disk, or not enough free
  space) it **falls back to direct and records why**.
- `auto` *(default)* — pick `staged` when it is both safe and useful.

#### Recommended command

```bash
python3 tools/openvla/collect_activation_pilot.py \
  --task_suite libero_spatial \
  --task_id 0 \
  --init_state_id 0 \
  --seed 7 \
  --num_episodes 20 \
  --max_steps 220 \
  --gpu 0 \
  --storage_mode auto \
  --output_root /home/user/4TB/hwkim/openvla_activation_collection \
  --staging_root /home/hwkim/.cache/openvla_activation_stage \
  --staging_max_gb 32 \
  --staging_min_free_gb 64
```

> ⚠️ Do **not** run this at full scale yet: the perturbation init-state
> determinism blocker is still open, so a perturbation run cannot be reproduced
> or compared. Vanilla-only collection is unaffected.

Before starting, check there is room:

```bash
df -h / /home/user/4TB
```

`/` must keep at least `--staging_min_free_gb` (64 GB) free, and the destination
needs the full run size plus margin.

#### Safety limits

- staging never exceeds `--staging_max_gb` (32 GB default)
- `/` never drops below `--staging_min_free_gb` (64 GB default)
- at most `--transfer_queue_size` (2) episodes wait for transfer, counting the
  one being copied
- when any limit is hit the collector **pauses** — it never drops activations or
  fills the disk. If it cannot resume within 30 minutes it stops with an error.
- a single transfer worker on purpose: parallel copies to one spinning disk
  multiply seeks and reduce total throughput

#### Staging states and crash recovery

| state | meaning |
|---|---|
| `<episode>.partial` | being written now, or interrupted mid-write |
| `<episode>.ready` | fully written, waiting for / undergoing transfer |
| `<final_run>/.transfer_tmp/<episode>` | copy in progress |
| `<final_run>/episodes/<episode>` | complete and verified |

An episode appears under `episodes/` only after every byte is copied and
verified against its manifest (relative path, size, file count), then moved with
an atomic rename. The staging copy is deleted only after that verification
passes; a failed transfer keeps the staging copy and records the error.

Just re-run the same command after a crash. On startup the collector:

- re-queues every `.ready` episode
- deletes stale `.transfer_tmp` copies (never trusted) and re-copies from `.ready`
- reports `.partial` directories as **orphans and never deletes them** — inspect
  them yourself
- refuses to overwrite an already-complete episode

#### Why `/home/HDD` is not the hot path

It is NTFS via ntfs-3g (FUSE): per-syscall overhead on ~3,000 files per episode,
and no POSIX ownership or permissions (everything shows as `root:root 0777`).
Use it as a cold archive only, after a run is finished and verified:

```bash
python3 tools/openvla/archive_activation_run.py \
  --run_dir /home/user/4TB/hwkim/openvla_activation_collection/<run_id> \
  --archive_root /home/HDD/hwkim/openvla_activation_archive \
  --mode tar --dry_run
```

It verifies the ext4 source first, writes one tar per episode, and **never
deletes the original**.

#### Verifying a finished run

```bash
python3 tools/openvla/activation_integrity.py --run_dir <final_run_dir>
python3 tools/openvla/activation_dashboard.py  --run_dir <final_run_dir>
```

Both operate on the final destination only. In-flight artifacts (`.partial`,
`.transfer_tmp`) are skipped, so these are safe to run while a collection is
still going.

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
