# Probe design

> **Status: draft, not executed.** Depends on the pilot v3 validation passing.

## The distinction the whole design turns on

Two different questions get conflated as "does the model know where the object is":

* **Spatial decodability** — is the target's position recoverable *from the
  activation*, by a probe trained to read it out? Answers "is the information
  present in this representation".
* **Action sensitivity** — does the model's *action* change when the target moves?
  Answers "does the policy use that information".

They come apart, and the interesting case is precisely when they disagree:
decodable but not acted on. A probe that reports only the first cannot tell a
representation failure from a read-out failure, and most of the value of the
collected activations is in separating them.

**One constraint from the action-schema trace: proprioception is never given to the
model** — `get_vla_action` consumes only `obs["full_image"]`. So "the model memorises
proprioception instead of looking" is not a live hypothesis for this checkpoint. The
remaining hypotheses are (a) the vision representation is weak, (b) it is present
but the action head ignores it, (c) position is inferred from the instruction text
rather than the image.

## Probe table

Stages are the eight recorded per timestep. `agentview` UV is the label frame the
policy actually sees (see `action_schema.md` §7 and the frame discussion in
`segmentation_label_provider`).

| # | Stage | Target label | Phase | Probe | Metric | Baseline | Bottleneck it isolates |
|---|---|---|---|---|---|---|---|
| P1 | `final_vision_dinov2` (256×1024) | source UV (agentview, normalised) | pre_grasp | ridge regression on the 256 patch tokens | median UV error in px; R² per axis | (a) shuffled-label, (b) mean-UV constant predictor, (c) probe on the *image* pixels directly | Is position present in DINOv2 at all |
| P2 | `final_vision_siglip` (256×1152) | same | pre_grasp | same | same | same | Same, for the second encoder — the two are concatenated, so they can differ |
| P3 | `projector_input` (256×2176) | same | pre_grasp | same | same | P1/P2 as the ceiling | Does the fusion preserve what the encoders had |
| P4 | `projector_output` (256×4096) | same | pre_grasp | same | P3 as the ceiling | P3 | Does the projection into LLM space lose position |
| P5 | `llm_early` / `llm_middle` / `llm_late` (visual-token slice) | same | pre_grasp | same, on the 256 visual token positions | same | P4 as the ceiling | Where in the LLM stack position decays, if it does |
| P6 | `pre_action_hidden` last token | same | pre_grasp | same | same | P5 as the ceiling | Is position still there at the point the action is read out |
| P7 | `pre_action_hidden` last token | **destination** UV | post_grasp | same | same | P6 | Same question for the place phase |
| P8 | all stages | **perturbed vs vanilla** condition label | pre_grasp, paired | linear classifier | AUC | shuffled-label | Does the representation distinguish the two scenes at all — a floor for P1–P6 |
| **A1** | — (behaviour) | Δaction vs Δsource position | pre_grasp, paired | regression of `action_applied[0:3]` difference on the source displacement | slope, R²; slope ≈ 0 means the action ignores the move | vanilla-vs-vanilla at different init states (the noise floor) | **Action sensitivity** |
| **A2** | `pre_action_hidden` | Δactivation vs Δsource position | pre_grasp, paired | same | same | same | Does the *representation* move when the object does, at the read-out point |
| **A3** | — (behaviour) | old-location bias (`along_axis`) | pre_grasp | see `failure_screening_plan.md` | signed along-axis error, with perpendicular reported alongside | vanilla closest-approach as scale | Is the failure directional or just degraded |

### How P-series and A-series answer different things

* **P high, A1 low** → the position is decodable but the action does not use it.
  Read-out / action-head failure. This is the shortcut hypothesis in its strong form.
* **P low at an early stage** → the representation never had it. Vision-encoder
  failure. The stage where P collapses localises it.
* **P high, A1 high, still fails** → the policy does react to the move but not
  correctly; neither a representation nor a read-out failure, and the framing
  needs to change.
* **P8 low** → nothing downstream is interpretable; the scenes are not even
  distinguishable in the representation.

A2 is the bridge: if A1 is flat while A2 is not, the signal reaches
`pre_action_hidden` and is discarded after it.

## Splits and leakage

Leakage is the failure mode that would make every probe look good. Three axes can
leak and each is handled explicitly.

| Axis | Rule |
|---|---|
| **Task** | Group-split by `task_id`. Tasks are held out entirely: train on tasks {0–5}, validate {6,7}, test {8,9}. A probe must not see any timestep from a test task |
| **Episode** | No episode ever spans two splits. Timesteps within an episode are massively autocorrelated, so a random timestep split would put near-duplicate frames on both sides and inflate every metric |
| **Condition** | Train on `vanilla` only for P1–P7, test on `vanilla` **and** perturbed separately. A probe trained on both conditions could learn "which condition is this" and reach the target through that shortcut |
| **Seed / init state** | `init_state_id` recorded per episode; splits never share an init state between train and test within a task |
| **Checkpoint** | `libero_spatial` and `libero_object` use *different* checkpoints. Never pool their activations — the representation spaces are not the same. Every probe is fit per suite |

Reported alongside every metric: the number of episodes, tasks and timesteps in
each split, so a suspiciously good number can be traced to a thin test set.

### Timestep sampling

Neighbouring timesteps are near-duplicates. Probes subsample to every `k`-th
timestep (`k` chosen so the mean UV change between retained frames exceeds the
probe's own error floor), and the subsampling is reported. Without this, "10,000
training samples" is really a few hundred.

## Label quality gating

Phase labels are a validated heuristic, not ground truth. Measured against
robosuite's own `_check_grasp` over 29 replayed episodes, the rule finalised at
commit `01fef2e` scores **precision 0.841, recall 0.928** (F1 0.882), with 2
single-frame holes inside otherwise correct intervals — down from 79 before the
release condition was required to coexist with the absence of grasp evidence.

Two things about that precision. First, it is a **lower bound**: `_check_grasp`
demands contact from *both* finger pads, so an episode carried on the finger
sides scores our correct labels as false positives — one `libero_spatial`
episode that demonstrably succeeded registered zero grasp frames. Second, the
per-condition spread matters more than the pooled figure: `libero_object`
vanilla 0.964, `libero_spatial` vanilla 0.861, `swap` 0.753. The `swap` arm's
post-grasp labels are the weakest and should carry the least weight.

Every episode records `sim_check_grasp` per timestep, so:

* P7 (post_grasp, destination target) uses only timesteps where the heuristic phase
  and `sim_check_grasp` **agree**. Disagreements are dropped, not guessed.
* The dropped fraction is reported per condition. If it is large for a condition,
  that condition's post-grasp probes are not reportable.
* `uncertain` timesteps are excluded throughout (~11% of steps).

Labels can be recomputed offline at any time: the collector stores every input
the resolver consumes, and `relabel_phases.py` writes a sidecar stamped with the
rule's commit rather than touching the original. Verified on collected episodes —
recomputing the current rule from stored inputs reproduces the collected labels
on 3428/3428 timesteps, so any future difference is attributable to the rule
change and not to the recomputation path.

## Controls that must be run, not optional

1. **Shuffled labels** — every probe, refit on permuted targets. Any probe whose
   shuffled score is not at chance has a leak.
2. **Pixel probe** — the same target read directly off the 224×224 model-input
   image. If a stage's probe does not beat this, the stage adds nothing.
3. **Constant predictor** — predict the training-set mean UV. This is the number
   that catches a probe which has learned "objects are usually here", which is the
   very shortcut under study.
4. **Vanilla-vs-vanilla noise floor** for A1/A2, from two different init states with
   the same fixture placement. Without it, any nonzero slope looks like sensitivity.

## What this design cannot settle

* ~~**One perturbation magnitude.**~~ **Resolved for the y axis.** The earlier
  claim that everything above `x0.1`/`y0.1` is contaminated came from checking the
  x axis and generalising. Measured per (task, condition, init state) over 150
  pairs, `y0.2` and `y0.3` are `clean_source_only` on 9 of 10 tasks — only
  `libero_object` task 5 is contaminated, in every init state, and it is excluded
  (see `task5_exclusion.md`). The main curve therefore has four points on one axis
  with identical task composition: **0 / ~7 / ~14 / ~21 cm** over tasks
  0,1,2,3,4,6,7,8,9. `y0.4` (4/10 clean) and `y0.5` (1/10) remain unusable, so the
  curve stops at 21 cm.
  The x axis is still single-magnitude: `x0.2` and above do teleport a distractor.
* **Correlation, not mechanism.** A probe finding position decodable does not show
  the policy could have used it. Causal claims would need intervention
  (activation patching), which is a separate experiment this collection makes
  possible but does not perform.
* **`libero_spatial swap` is confounded** (`destination_and_distractor`). Its probes
  are reportable only as a confounded arm, never pooled with the clean conditions.
* **Axes are not interchangeable, and this is not yet settled.** Over the six
  pilot episodes per axis the mean bias ratio is nearly identical (x 0.762,
  y 0.760) but the spread is not: ±0.590 against ±0.227, with one x episode biased
  in the *opposite* direction (−0.396) and three past 1.0. Treat "the bias is
  direction-invariant" as **provisional** until the axes are matched in sample
  size; the y curve brings y0.1 to 45 episodes while x0.1 stays at 6.
