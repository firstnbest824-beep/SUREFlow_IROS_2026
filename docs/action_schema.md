# OpenVLA → LIBERO action schema (traced from the running code)

Everything below was read out of the code and checkpoint files actually executed
by `tools/openvla/collect_official_activations.py`, or measured on the 30 collected
episodes. Nothing is taken from the OpenVLA paper. Where a docstring in this repo
disagrees with the code, the code is reported and the docstring is flagged.

Traced at commit `3832d85`. Model files: the HF modules cache copy of
`modeling_prismatic.py` for `openvla/openvla-7b`
(`47a0ec7fc4ec123775a391911046cf33cf9ed83f`); checkpoints
`openvla-7b-finetuned-libero-spatial@962318ce` and
`...-libero-object@287d6cfd`. robosuite 1.4.0.

## 1. Shape and dtype

| Stage | Type | Shape | Notes |
|---|---|---|---|
| `input_ids` | `torch.int64` | `[1, L]` | `L` = text tokens; 29871 appended if absent |
| prefill sequence | — | `291` for libero_spatial task 0 | 256 vision patches + 35 text; varies 282–291 with instruction length |
| `generated_ids` | `torch.int64` | `[1, L+7]` | |
| action token ids | `np.int64` | `(7,)` | `generated_ids[0, -7:]` |
| bin indices | `np.int64` | `(7,)` | |
| normalised action | `np.float64` | `(7,)` | range ±0.996078431372549 |
| **`predict_action` return** | **`np.float64`** | **`(7,)`** | never batched, never a torch tensor |

`float64` is forced by `np.linspace` and by `np.array(q01/q99)` regardless of the
bfloat16 model weights.

## 2. Seven action tokens, and why seven

`max_new_tokens = self.get_action_dim(unnorm_key) = len(norm_stats[key]["action"]["q01"])`
= **7**, i.e. one token per action dimension. It is not a constant in the code.

Measured on 200 npz files spanning all 30 episodes and all 5 suite-conditions:
every `<stage>__last_token_stack` has first dimension exactly 7, with no
exceptions. Those 7 hook calls are 1 prefill + 6 autoregressive decodes producing
7 tokens; `probe_hooks.py` labels call 0 `prompt_prefill` and calls 1–6
`autoregressive_generation`. Generation never stops early — emitted ids lie in
31745–31999, far from `eos_token_id = 2`.

## 3. De-tokenisation

    bins        = np.linspace(-1, 1, 256)          # 256 edges, spacing 2/255
    bin_centers = (bins[:-1] + bins[1:]) / 2       # 255 values, ±0.996078431372549
    vocab_size  = 32064 - 64 = 32000               # deliberately overrides the parent's 32064
    discretized = 32000 - token_id
    index       = clip(discretized - 1, 0, 254)
    normalised  = bin_centers[index]

The action vocabulary is the **top 255 ids of the 32000-id Llama-2 vocabulary, in
reverse order** — a larger token id means a more negative action. Verified by
inverting the collected actions and snapping to the nearest bin centre: max error
**2.2e-16** over a whole episode.

## 4. Un-normalisation

`q01` / `q99` percentiles (not mean/std), applied as

    action = 0.5 * (normalised + 1) * (q99 - q01) + q01     where mask is True
    action = normalised                                      where mask is False

The mask is `[true, true, true, true, true, true, false]`, so **the gripper
dimension is not un-normalised** — the raw training value survives. That is why
`action_model[6]` only ever takes the two values `0.0` and `0.9960784` across all
5676 collected timesteps.

> **Docstring defect (fixed in a comment, behaviour unchanged).**
> `run_single_vanilla_rollout.py` looks for `os.path.join(checkpoint_id, "dataset_statistics.json")`,
> but `checkpoint_id` is a hub id, not a directory. The path never resolves, the
> warning branch always fires, and the statistics actually used come from
> `config.json`'s `norm_stats`. Harmless here only because the two dicts were
> verified byte-identical for both LIBERO checkpoints.

## 5. Relative, not absolute

Established from the controller, not from magnitudes: `osc_pose.json` has
`"control_delta": true`, and `set_goal_position` computes
`goal_position = current_position + delta` against `sim.data.site_xpos` re-read at
every policy step. There is **no accumulation** — the previous goal is overwritten,
so tracking error from the previous step is silently forgiven and the sum of
commanded deltas is *not* the executed trajectory.

## 6. The seven dimensions

| idx | meaning | normalised range | scale | physical meaning |
|---|---|---|---|---|
| 0–2 | Δx, Δy, Δz | [-1, 1] | **× 0.05** | metres of goal-position offset, max 5 cm/step |
| 3–5 | Δrx, Δry, Δrz | [-1, 1] | **× 0.5** | axis-angle rotation vector in radians, max 0.5 rad (28.6°) |
| 6 | gripper | — | — | **+1 = CLOSE, −1 = OPEN** (rate command) |

**`0.1` in a position dimension means 5 mm, not 0.1 m.** `action_scale =
(output_max − output_min) / (input_max − input_min)`, with zero offset because both
ranges are symmetric.

Dims 0–5 of `action_applied` are bit-identical to `action_model` (verified
`np.allclose` over all 5676 steps); only dim 6 is transformed.

### Gripper sign convention, end to end

    action_model[6]                 ∈ {0.0, 0.9960784}      (checkpoint's [0,1] convention)
    normalize_gripper_action(...)   → affine [0,1]→[-1,1], then np.sign  ⇒ {-1, +1}
    invert_gripper_action(...)      → × (-1)
    action_applied[6]               model 0.0 → +1 (CLOSE);  model ≈1.0 → −1 (OPEN)

`PandaGripper.format_action` documents `-1 => open, 1 => closed`. Both helpers
mutate in place; the collector passes `.copy()`.

## 7. Coordinate frame

**World frame** (MuJoCo global) for both translation and rotation — not the
end-effector frame.

* Translation: the delta is added to `sim.data.site_xpos[gripper0_grip_site]`, a
  world-frame position, along world axes.
* Rotation: `goal_orientation = rotation_mat_error @ current_orientation` —
  **pre**-multiplication, i.e. rotation about world axes, not tool axes.
* Verified live: `set_goal([1,0,0,0,0,0])` moved the goal from
  `[-0.20846, 0, 1.17328]` to `[-0.15846, 0, 1.17328]` — exactly +0.05 along world x.

World frame and robot **base** frame are not empirically distinguishable here:
`robot0_base` sits at world `[-0.66, 0, 0.912]` with identity rotation, so the two
share axes and a delta command reads identically under either. The code path is
world.

## 8. Rotation representation

Axis-angle rotation **vector** (scaled axis-angle / exponential coordinates: unit
axis × angle) in **radians**. Consumed by `axisangle2quat`, whose magnitude *is*
the angle. Not euler, not quaternion.

The observation uses the **same representation** (`quat2axisangle` is a
line-for-line reimplementation of robosuite's) but **different semantics**: the
observation is the *absolute* world orientation of the end-effector, while dims
3–5 are an *incremental* rotation, additionally scaled by 0.5.

> Worth carrying into the probe design: the axis-angle proprioceptive vector is
> computed and stored as `proprio_state`, but **is never given to the model** —
> `get_vla_action` consumes only `obs["full_image"]`.

## 9. Controller → joints

| Property | Value |
|---|---|
| Class | `robosuite.controllers.osc.OperationalSpaceController` (`OSC_POSE`, `control_ori=True`) |
| Config | `robosuite/controllers/config/osc_pose.json`, hardcoded by LIBERO's `ControlEnv` |
| input_min / input_max | −1 / +1 on all 6 |
| output_min / output_max | ∓[0.05, 0.05, 0.05, 0.5, 0.5, 0.5] |
| Control frequency | **20 Hz** |
| Physics timestep | **0.002 s** (500 Hz) |
| **Substeps per `env.step`** | **25** |
| Interpolation | **none** (`"interpolation": null`; `ramp_ratio: 0.2` is dead config) |
| Impedance | fixed, `kp = 150` on all axes, `damping_ratio = 1` → `kd = 24.49` |

`set_goal` runs on the first of the 25 substeps; `run_controller` recomputes
torques on all 25 against the same frozen goal.

## 10. Clipping

There is **no clipping of action values anywhere in `tools/openvla/`** (the only
`np.clip` calls in the repo are in image code). The clips that exist are outside
the repo, and the one that matters is a **bin-index** clip
(`np.clip(discretized - 1, 0, 254)`) — not a value clip. Un-normalisation does not
clip to the statistics range.

## 11. Index alignment: obs_t / activation_t / action_t / executed action

`_collect_one_timestep` reads nothing after `env.step`, and `env.step` binds a new
name (`obs_next`) so `obs` is never reassigned mid-record. `hook_manager.reset()`
runs immediately before the single `get_vla_action` call, so the harvested tensors
are from that forward pass only.

### The naive lag test fails, and that is expected

Correlating `eef_pos[t+1] − eef_pos[t]` against `action_applied` at three lags,
pooled over 30 episodes (16938 samples):

| lag | r |
|---|---|
| −1 | **0.8666** |
| 0 | 0.8591 |
| +1 | 0.7937 |

Lag 0 does **not** dominate, and it wins in only 8 of 30 episodes. This is not an
alignment bug. Two reasons, both measured:

* **Actions are highly autocorrelated** — `r(a[t], a[t−1]) = +0.8953` — so the
  three lags are badly confounded and the raw comparison cannot separate them.
* **The controller lags its goal.** With `kp = 150`, no interpolation, and 25
  substeps, a commanded 5 mm goal offset produced under 1 mm of actual motion in
  one `env.step`. Mean |commanded| is 0.00043 m against mean |achieved| 0.00255 m,
  so the displacement between `t` and `t+1` carries residual tracking from
  `a[t−1]` as well as the response to `a[t]`.

### The decisive test

Replaying each episode's recorded `action_applied` in a fresh env, reading the
observation strictly **before** `env.step` so alignment is true by construction:

| pairing | error vs recorded `eef_pos[t]` |
|---|---|
| replay pre-step eef at `t` | **0.000e+00 m** (exact on 3 spatial episodes; 2.8e-15 / 4.4e-11 on 2 object episodes) |
| either off-by-one pairing | 4.7e-3 m mean |

Eight to fifteen orders of magnitude of separation. **The alignment is correct.**

### What check F does and does not prove

The state-hash chain closes on 5646/5646 links across all 30 episodes, all 5676
hashes are distinct within their episodes, and none collide across episodes. But
it is **necessary, not sufficient**: both hashes come from the two `get_state()`
calls bracketing `env.step`, and neither touches `obs`, the hook tensors,
`eef_pos` or the labels. Deliberately shifting `action_applied` and `eef_pos` by
+1 inside a record file still closes the chain. Content alignment has to be proven
by replay, as above.

## 12. `success` and `done` are post-step

They are the **only** post-step fields in a record, read after `env.step`. LIBERO
sets `done = self._check_success()` on the stepped state, so a pre-step reading is
not even definable at index `t`.

The module docstring previously implied every field was pre-step. Records now
carry `outcome_reference: "post_step_observation"` and `post_step_fields:
["success", "done"]` explicitly.

Measured: `success != done` in 0 of 5676 records (they share one predicate); 9 of
30 episodes end `(True, True)` with `termination_reason=success`, 21 end
`(False, False)` at `max_steps`; no record before the last has either flag true.
