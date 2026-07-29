# Bulk activation collection — plan and commands

> **Not executed.** This file records what a bulk run would cost and what command
> would launch it. Running it needs an explicit go-ahead, and the storage
> decision below has to be settled first.

## 1. What gets collected, and why those conditions

| Suite | Condition | Family | Generalisation tested | Measured change class |
|---|---|---|---|---|
| `libero_object` | `vanilla` | — | baseline | `no_detected_change` |
| `libero_object` | `x0.1` | position_offset | **source / pick** | `clean_source_only` |
| `libero_object` | `y0.1` | position_offset | **source / pick** | `clean_source_only` |
| `libero_spatial` | `vanilla` | — | baseline | `no_detected_change` |
| `libero_spatial` | `swap` | swap | **destination / place** | `destination_and_distractor` |

`libero_spatial swap` is retained because it is the only official destination
perturbation available, but it is **confounded** — the destination and its swap
partner both move — and must be analysed separately from the clean conditions,
never pooled with them.

`x0.2`–`y0.5` are deliberately excluded: they teleport a distractor to roughly
x = +10 m, which is a scene edit rather than a spatial perturbation. Add them
only as a separate, explicitly-labelled arm.

The changed entity is re-measured for every episode, because the perturbation
profile is not constant across tasks (static analysis said `y0.4`/`y0.5` move the
basket; on task 0 they do not).

## 2. Cost

Measured on `libero_spatial` task 0 vanilla, 81 steps:

| Quantity | Value |
|---|---|
| Bytes per timestep | **10.97 MB** (activations 10.84, images 0.12, overlays 0.01) |
| Inference per timestep | ~484 ms |
| Wall clock per timestep | ~5.2 s (includes 2 segmentation renders + npz compression) |
| Episode length | 81 steps when it succeeds, 220 (the cap) when it does not |

Per-stage share of the bytes, compressed: `llm_early`, `llm_middle`, `llm_late`,
`pre_action_hidden` ≈ 1.83 MB each (**68% of the total**), `projector_output`
1.62, `projector_input` 0.87, SigLIP 0.45, DINOv2 0.40.

### Full grid

10 tasks × 5 suite-conditions × 10 episodes = **500 episodes**.

Perturbed conditions mostly fail, so they run to the 220-step cap; assume 150
steps average across the grid.

| | |
|---|---|
| Timesteps | 500 × 150 = 75,000 |
| **Storage** | 75,000 × 10.97 MB ≈ **823 GB** |
| **Wall clock, 1 GPU** | ~108 h |
| **Wall clock, 2 GPUs** | ~54 h |

`/` has ~96 GB free and is 95% full, so the full grid **cannot** land there.
`/home/HDD` has 3.0 TB free and is the only viable target.

### Levers, if 823 GB is too much

| Lever | Saving | Cost |
|---|---|---|
| Drop the three LLM full-sequence tensors, keep `pre_action_hidden` + all four last-token stacks | −45% (≈450 GB) | no per-layer sequence analysis for early/middle/late |
| Keep only the 256 visual tokens of LLM stages (drop the 35 text tokens) | −12% | loses prompt-token attention analysis |
| 5 episodes per condition instead of 10 | −50% | halves the per-condition sample |
| Every 2nd timestep | −50% | breaks the per-step state chain (validator check F) |

Recommended if a cut is needed: **5 episodes per condition** (≈411 GB, ~27 h on
2 GPUs) — it costs statistical power, which is measurable, rather than
representational coverage, which is not recoverable without re-running.

## 3. Preconditions

- [x] Gate 1 static audit
- [x] Gate 2 unit tests
- [x] Gate 6 changed-entity verification against the real simulator
- [ ] Gate 3 `libero_spatial` vanilla + swap dry-run
- [ ] Gate 4 `libero_object` vanilla + x0.1 + y0.1 dry-run
- [ ] Gate 5 smoke test
- [ ] Storage target agreed (`/home/HDD`, not `/`)
- [ ] Explicit go-ahead

## 4. Command

Do not run without the checklist above complete.

```bash
PY=/home/hwkim/micromamba/envs/vla-spatial-diagnostics/bin/python
ROOT=/home/HDD/hwkim/openvla_activations/schema_v2_$(date +%Y%m%d)
cd /home/hwkim/research/VLA-Spatial-Diagnostics/tools/openvla

# GPU 1: libero_object (source / pick generalisation)
for TASK in $(seq 0 9); do
  for COND in vanilla x0.1 y0.1; do
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $PY collect_official_activations.py \
      --suite libero_object --task_id $TASK --condition $COND \
      --output_root $ROOT --num_episodes 10 --seed 0 --max_steps 220
  done
done

# GPU 2: libero_spatial (destination / place generalisation)
for TASK in $(seq 0 9); do
  for COND in vanilla swap; do
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl $PY collect_official_activations.py \
      --suite libero_spatial --task_id $TASK --condition $COND \
      --output_root $ROOT --num_episodes 10 --seed 0 --max_steps 220
  done
done

# Validate everything before any analysis touches it
$PY validate_activation_episode.py $ROOT --recursive --json_out $ROOT/validation.json
```

GPU 0 is left alone: another user's process holds ~15 GB on it.

Each episode is sealed by an atomic rename, so an interrupted run leaves
`.partial` directories that `find_incomplete_episodes` lists and that the writer
refuses to silently reuse. Re-running the same command skips nothing — it will
fail on an existing episode directory rather than overwrite it, which is
deliberate.

## 5. Known limitation to carry into analysis

`libero_spatial swap` ships no official init states, so its episodes start from a
locally frozen capture while `vanilla` starts from the shipped `.pruned_init`.
Objects the perturbation never touched can therefore differ slightly between the
two conditions. The manifest records `initial_pose_measurement.init_state_sources_match`
and the change report quantifies every entity that differs, so the confound is
measured rather than hidden — but a vanilla-vs-swap contrast is not a
single-variable comparison and should not be presented as one.
