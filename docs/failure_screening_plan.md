# Lightweight failure screening — plan

> **Status: draft, not executed.** Depends on the pilot v3 validation passing.
> Screening is cheap by design: it stores no activations, so it can cover the
> whole task grid at a fraction of the cost of collection.

## Why screen before collecting

Bulk collection stores ~11 MB per timestep. Spending that on episode pairs that
cannot answer the question is the main avoidable cost in this study. An episode
pair is only informative if it isolates the perturbation, and three of the four
required properties can be checked without saving a single activation.

The output is a **whitelist of (suite, task, condition, init_state_id) pairs** that
bulk collection then runs. Nothing else changes about collection.

## Selection criteria

A pair `(vanilla @ init_k, perturbed @ init_k)` is admitted only if **all** hold:

| # | Criterion | How it is decided | Why it matters |
|---|---|---|---|
| 1 | vanilla succeeds **repeatedly** | ≥ `R` of `R` repeats succeed, `R = 3` | If the baseline is unreliable the failure is not attributable to the perturbation. `libero_object` vanilla ran at 3/6 in the v1 pilot, so this filter is doing real work |
| 2 | perturbed fails **repeatedly** | 0 of `R` repeats succeed | A flaky failure is not a failure mode |
| 3 | initial state corresponds | same `init_state_id`, same `episode_reset_seed`, so identical fixture placement; `scene_body_sha256` recorded for both | Without this the contrast differs in the perturbation *and* in where the cabinet is (measured: 8–9 mm of fixture drift changed 27% of pixels and flipped an outcome) |
| 4 | the change is a **single role** | `is_clean(change_class)`, measured per episode against the applied init state, with the jitter-aware per-entity threshold | A multi-entity change cannot attribute anything |
| 5 | old-location bias is **measurable** | source moved ≥ 2× the jitter estimate for that entity, and both the moved and the original position are ≥ `2 cm` apart in the agentview projection | The bias metric needs the two hypotheses to be distinguishable |

Criterion 1 and 2 make the pair a *paired case*: the same task, the same starting
scene, differing only in the perturbation, with one side working and the other not.

Repeats are needed even though collection is now bit-reproducible: reproducibility
means the same seed gives the same episode, not that the episode is representative
of the task. `R = 3` uses three different `init_state_id` values, so it measures
across starting states rather than re-running one.

## What screening measures per episode

No activations, no images beyond a first/last frame, no overlays. Per timestep it
keeps only what the criteria and the bias metric need:

* `eef_pos`, `gripper_qpos`, `sim_check_grasp`
* `entity_world_xyz` for the tracked entities
* `phase`, `relevant_entity`
* `action_applied`
* success / done / termination reason

That is a few hundred bytes per timestep instead of ~11 MB — roughly **5 orders of
magnitude** cheaper, so the full 10-task grid is affordable.

## Old-location bias metric (defined here so screening and analysis agree)

At the timestep of closest approach to the perturbed source position:

    error       = eef_closest - source_perturbed
    axis        = unit(source_vanilla - source_perturbed)   # points at the old spot
    along_axis  = error · axis
    perpendicular = |error - along_axis · axis|

Reported per episode, aggregated per condition:

* `min |eef - source_perturbed|` — did it reach the object where it now is
* `min |eef - source_vanilla|` — did it reach where the object used to be
* `along_axis` — signed; positive means biased toward the trained location
* `perpendicular` — the part of the error the bias does not explain
* the same three for the **vanilla** episode, as the scale calibration for
  "reached it"

The v1 pilot gave along_axis ≈ +0.054 m against a 0.069 m perturbation (78%), with
11 of 12 episodes positive — but perpendicular error was 0.040–0.055 m, i.e. the
trajectory is also just worse, not purely translated. Both numbers get reported;
quoting only the first would overstate the result. **These figures came from the
unpinned-fixture pilot and must be recomputed on v3.**

## Cost

| | |
|---|---|
| Grid | 2 suites × 10 tasks × (1 baseline + 1–2 perturbed conditions) |
| Repeats | 3 init states per cell |
| Episodes | ~150 |
| Storage | < 100 MB total |
| Wall clock, 2 GPUs | ~8 h (inference-bound; the ~5 s/timestep of collection drops to the ~0.5 s of inference alone) |

## Command shape

    python tools/openvla/screen_failures.py \
        --suite libero_object --conditions vanilla,x0.1,y0.1 \
        --tasks 0-9 --init_states 0,1,2 \
        --output docs/screening/<date>/ --seed 0

Output: one JSONL row per episode plus `whitelist.json` naming the admitted pairs
and, for every rejected pair, **which criterion rejected it**. A screening run that
admits nothing is a result, not a failure — it would mean the official conditions
do not produce clean attributable pairs on this suite, which is exactly the kind of
thing that has to be known before spending 800 GB.

## What screening deliberately does not decide

It cannot tell whether the failure is a representation problem or an action
problem — that needs activations. Screening only decides *which pairs are worth
collecting activations for*.
