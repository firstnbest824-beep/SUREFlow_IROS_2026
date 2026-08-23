# tools/common/ — shared infra for non-diagnostics lines

`tools/common/` holds the pieces of the OpenVLA/LIBERO pipeline that are
genuinely reusable infrastructure (model loading, environment construction,
image preprocessing, checkpoint bookkeeping) with no diagnostics-specific
logic in them. It exists so that `tools/action_generalization/` (and any
future non-diagnostics line) does not have to import `tools/openvla/`.

## The contract: physical copy, not re-export

Every file here that has a `tools/openvla/` counterpart is a **physical copy**
of function bodies, not `from tools.openvla.X import Y`. This was a deliberate
choice, not an oversight: the diagnostics files these functions live in also
import diagnostics-only modules (`spatial_task_resolver.py`,
`task_phase_resolver.py`, `changed_entity_detector.py`, ...) at module top
level, so a plain `import` of the diagnostics file — even just to reach one
pure function in it — would drag those modules in as a side effect and create
an import-time edge from `tools/common/` back into `tools/openvla/`. That edge
is exactly what `tools/common/` must not have (verified by
`grep -rn "from task_phase_resolver\|from spatial_task_resolver\|from probe_hooks\|from changed_entity_detector\|from entity_role_resolver\|from official_task_pair_resolver\|from segmentation_label_provider\|from activation_" tools/common/`
being empty).

**Consequence: these copies do not auto-update.** If the `tools/openvla/`
original is fixed, extended, or has a bug patched, the copy here silently
keeps the old behavior until someone re-copies it by hand. There is no tooling
that detects this automatically today. Mitigation:

1. `tests/test_common_matches_diagnostics.py` runs both the `tools/common/`
   copy and the `tools/openvla/` original side by side (tiered: exact equality
   for deterministic helpers, `np.allclose` for model inference, behavioral
   equivalence for simulator rollouts — see that file's docstring for why the
   tiers differ). A future drift that changes *behavior* should surface there.
2. Periodically diff the source ranges cited per file below against the
   current `tools/openvla/` file. There is no automation for this; it is a
   manual check.

## File-by-file provenance

| `tools/common/` file | Copied from | What | Source lines (at copy time) |
|---|---|---|---|
| `openvla_model.py` | `tools/openvla/run_single_vanilla_rollout.py` | `ACTION_DIM`, `normalize_gripper_action`, `invert_gripper_action`, `load_openvla`, `get_vla_action`, `get_libero_dummy_action`, `quat2axisangle` | 55, 133-138, 141-144, 150-196, 199-251, 273-275, 278-288 |
| `libero_env.py` | `tools/openvla/collect_official_activations.py` | `make_env` | 125-138 |
| `image_transform.py` | `tools/openvla/model_input_transform.py` | full file, copied as-is | 1-261 |
| `checkpoint_verify.py` | `tools/openvla/verify_openvla_checkpoint.py` | full file, copied as-is | 1-189 |
| `init_state.py` | `tools/openvla/init_state_freezer.py` | `sha256_bytes`, `sha256_array`, `FrozenInitState`, `official_init_state_path`, `load_init_states`, `frozen_init_state_path`, `capture_init_state`, `freeze_init_state`, `resolve_init_state`, `compare_states` | 44-45, 48-49, 52-68, 74-89, 92-102, 108-115, 118-141, 144-228, 234-276, 282-294 |
| `seeding.py` | `tools/openvla/official_task_pair_resolver.py` and validated rollout runners | `validate_seed`, `seed_everything`, `episode_seed` | 93-154; CUDA seeding mirrors the rollout runners |
| `checkpoints.py` | new — no source file | `SUITE_CHECKPOINTS` registry | values transcribed from `official_task_pair_resolver.py` lines 57-68 |
| `task_resolution.py` | new — independent LIBERO-PRO path resolution | vanilla and shipped position-offset BDDL resolution | no diagnostics import |
| `experiment.py` | new — baseline artifact/provenance layout | fixed output root, metadata, `best`/`last` policy | no diagnostics import |

`libero_env.configure_robosuite_logging()` also provides a narrow runtime
fallback for Robosuite 1.4's hard-coded `/tmp/robosuite.log`: only if that
existing file cannot be opened does it redirect that one third-party log to a
writable run/test-local path. It does not change simulator or policy logic.

### Deliberately not copied: `verify_reproducibility`

`init_state_freezer.py` also defines `verify_reproducibility` (source lines
297-362). It was **not** copied. Re-reading its body (not just the
Phase-3-plan's claim that the file is "already pure") found it does a local
`from changed_entity_detector import extract_object_poses`, and
`changed_entity_detector.extract_object_poses` itself does a local
`from spatial_task_resolver import get_entity_world_position, unwrap_base_env`
— `spatial_task_resolver` is one of the four explicit diagnostics-only
"single source of truth" modules named in `tools/openvla/README.md` and
`tools/CLAUDE.md`. Copying `verify_reproducibility` verbatim would have
created exactly the import-time edge this directory exists to avoid, and
duplicating `spatial_task_resolver`'s logic to make it self-contained is
explicitly against `tools/CLAUDE.md` ("로직을 복제하지 말고 반드시 import해서
쓴다"). See the top of `tools/common/init_state.py` for the full explanation.
`compare_states` (the part of `verify_reproducibility` that was actually pure
— comparing two `env.sim.get_state()` vectors) is copied and usable directly.

## Non-goals

- `tools/common/` never imports from `tools/openvla/`, and nothing under
  `tools/action_generalization/` may import from `tools/openvla/` either —
  only from `tools/common/`.
- This directory holds infrastructure only. No spatial-diagnostics logic
  (task-phase resolution, entity role resolution, perturbation/change
  detection, probing) belongs here; that stays in `tools/openvla/` as the
  single source of truth it already is.
