"""Tests for the episode writer, the validator, and the timestep contract.

The point of these is that a plausible collector bug must make a test go red.
Each test therefore constructs a *specific* defect -- a dropped stage, a broken
state chain, a phase/entity mismatch -- and asserts the corresponding check
fails, rather than only asserting that a clean episode passes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "openvla")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from activation_episode_writer import (  # noqa: E402
    COMPLETE_MARKER,
    MANIFEST_NAME,
    METRICS_NAME,
    REQUIRED_STAGES,
    SCHEMA_VERSION,
    ActivationEpisodeWriter,
    episode_dir_for,
    find_incomplete_episodes,
    to_storage_array,
)
from validate_activation_episode import FAIL, PASS, WARN, validate_episode  # noqa: E402

SOURCE = "obj_source_1"
DESTINATION = "obj_dest_1"


def make_activations(seed: int = 0, scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    shapes = {
        "final_vision_dinov2": (1, 256, 1024),
        "final_vision_siglip": (1, 256, 1152),
        "projector_input": (1, 256, 2176),
        "projector_output": (1, 256, 4096),
        "llm_early": (1, 291, 4096),
        "llm_middle": (1, 291, 4096),
        "llm_late": (1, 291, 4096),
        "pre_action_hidden": (1, 291, 4096),
    }
    # Keep them small so the whole suite stays fast; shape fidelity is what matters.
    return {
        stage: (rng.standard_normal((shape[0], 4, 8)) * scale).astype(np.float32)
        for stage, shape in shapes.items()
    }


def make_manifest(**overrides) -> dict:
    manifest = {
        "suite": "libero_object",
        "condition": "x0.1",
        "perturbation_family": "position_offset",
        "task_id": 0,
        "task_name": "pick_up_thing",
        "seed": 0,
        "checkpoint": {"model_id": "openvla/x", "revision": "a" * 40, "unnorm_key": "libero_object"},
        "bddl_sha256": "b" * 64,
        "init_state_sha256": "c" * 64,
        "requested_level": 0.1,
        "measured_translation_m": 0.07,
        "entity_roles": {"source": SOURCE, "destination": DESTINATION},
        "change_report": {"change_class": "clean_source_only", "changed_entity_names": [SOURCE]},
    }
    manifest.update(overrides)
    return manifest


def make_metrics(timestep: int, chain: list, phase: str = "pre_grasp", **overrides) -> dict:
    relevant = {"pre_grasp": SOURCE, "post_grasp": DESTINATION, "uncertain": None}[phase]
    metrics = {
        "sim_state_sha": chain[timestep],
        "next_sim_state_sha": chain[timestep + 1],
        "phase": phase,
        "relevant_entity": relevant,
        "segmentation": {
            SOURCE: {
                "agentview": {
                    "visibility": "visible",
                    "mask_pixel_count": 120,
                    "uv": [0.4, 0.6],
                    "in_frame": True,
                }
            }
        },
    }
    metrics.update(overrides)
    return metrics


def write_episode(final_dir: Path, num_steps: int = 3, manifest=None, mutate=None) -> Path:
    chain = [f"{index:064x}" for index in range(num_steps + 1)]
    with ActivationEpisodeWriter(final_dir, manifest or make_manifest()) as writer:
        for timestep in range(num_steps):
            activations = make_activations(timestep)
            metrics = make_metrics(timestep, chain)
            if mutate is not None:
                activations, metrics = mutate(timestep, activations, metrics)
            writer.write_step(
                timestep=timestep,
                activations=activations,
                metrics=metrics,
                agentview_rgb=np.zeros((8, 8, 3), np.uint8),
            )
        writer.set_result(success=True, termination_reason="success")
    return final_dir


def status_of(report, check_name):
    return [c["status"] for c in report["checks"] if c["check"] == check_name]


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------
def test_writer_seals_atomically(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    assert (final / COMPLETE_MARKER).is_file()
    assert (final / MANIFEST_NAME).is_file()
    assert not final.with_name(final.name + ".partial").exists()
    assert len(list((final / "activations").glob("*.npz"))) == 3


def test_writer_stores_fp16_copies_only():
    source = np.array([[1.5, -2.25]], dtype=np.float32)
    stored = to_storage_array(source)
    assert stored.dtype == np.float16
    assert source.dtype == np.float32, "the caller's array must not be cast in place"


def test_writer_leaves_partial_on_failure_and_does_not_seal(tmp_path):
    final = tmp_path / "episode_000"
    with pytest.raises(ValueError):
        with ActivationEpisodeWriter(final, make_manifest()) as writer:
            writer.write_step(0, make_activations(), make_metrics(0, ["a", "b"]),
                              agentview_rgb=np.zeros((4, 4, 3), np.uint8))
            writer.write_step(5, make_activations(), make_metrics(0, ["a", "b"]))  # out of order
    assert not final.exists()
    assert final.with_name(final.name + ".partial").is_dir()
    assert find_incomplete_episodes(tmp_path) == [final.with_name(final.name + ".partial")]


def test_writer_refuses_to_reuse_a_partial(tmp_path):
    final = tmp_path / "episode_000"
    (tmp_path / "episode_000.partial").mkdir()
    with pytest.raises(FileExistsError, match="interrupted attempt"):
        ActivationEpisodeWriter(final, make_manifest()).__enter__()


def test_writer_rejects_missing_stage(tmp_path):
    activations = make_activations()
    activations.pop("llm_middle")
    with pytest.raises(KeyError, match="llm_middle"):
        with ActivationEpisodeWriter(tmp_path / "episode_000", make_manifest()) as writer:
            writer.write_step(0, activations, make_metrics(0, ["a", "b"]))


def test_episode_dir_keeps_conditions_apart():
    a = episode_dir_for("/root", "libero_object", "x0.1", 3, 0, 1)
    b = episode_dir_for("/root", "libero_object", "y0.1", 3, 0, 1)
    assert a != b and a.name == b.name == "episode_001"


# -----------------------------------------------------------------------------
# Validator: a clean episode passes
# -----------------------------------------------------------------------------
def test_clean_episode_passes(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    report = validate_episode(final)
    assert report["passed"], json.dumps(report["checks"], indent=2)
    assert status_of(report, "F_sync") == [PASS]
    assert status_of(report, "J_segmentation") == [PASS]


def test_validator_matches_expected_hashes(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    good = validate_episode(final, expected_bddl_sha="b" * 64, expected_checkpoint="a" * 40)
    assert good["passed"]
    bad = validate_episode(final, expected_bddl_sha="d" * 64)
    assert not bad["passed"]
    assert FAIL in status_of(bad, "G_hashes")


# -----------------------------------------------------------------------------
# Validator: each defect trips its own check
# -----------------------------------------------------------------------------
def test_missing_complete_marker_fails(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    (final / COMPLETE_MARKER).unlink()
    assert FAIL in status_of(validate_episode(final), "B_complete")


def test_count_mismatch_fails(tmp_path):
    final = write_episode(tmp_path / "episode_000", num_steps=3)
    sorted((final / "activations").glob("*.npz"))[-1].unlink()
    report = validate_episode(final)
    assert FAIL in status_of(report, "C_counts")


def test_non_contiguous_timesteps_fail(tmp_path):
    final = write_episode(tmp_path / "episode_000", num_steps=3)
    records = [json.loads(l) for l in (final / METRICS_NAME).read_text().splitlines() if l.strip()]
    records[1]["timestep"] = 7
    (final / METRICS_NAME).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert FAIL in status_of(validate_episode(final), "C_counts")


def test_nan_activation_fails(tmp_path):
    def mutate(timestep, activations, metrics):
        if timestep == 1:
            activations["llm_late"] = activations["llm_late"].copy()
            activations["llm_late"][0, 0, 0] = np.nan
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", mutate=mutate)
    assert FAIL in status_of(validate_episode(final), "E_finite")


def test_llama_massive_activations_do_not_fail(tmp_path):
    """~1.5e4 in a couple of channels is real, not corruption."""

    def mutate(timestep, activations, metrics):
        activations["llm_middle"] = activations["llm_middle"].copy()
        activations["llm_middle"][0, :, 0] = 1.5e4
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", mutate=mutate)
    report = validate_episode(final)
    assert FAIL not in status_of(report, "E_finite")


def test_broken_state_chain_fails(tmp_path):
    """The defining off-by-one: labels from obs_t+1 break the hash chain."""
    final = write_episode(tmp_path / "episode_000", num_steps=4)
    records = [json.loads(l) for l in (final / METRICS_NAME).read_text().splitlines() if l.strip()]
    records[2]["sim_state_sha"] = "f" * 64  # observed a state we never stepped into
    (final / METRICS_NAME).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    report = validate_episode(final)
    assert FAIL in status_of(report, "F_sync")
    assert not report["passed"]


def test_missing_chain_warns_but_does_not_pass_silently(tmp_path):
    final = write_episode(tmp_path / "episode_000", num_steps=3)
    records = [json.loads(l) for l in (final / METRICS_NAME).read_text().splitlines() if l.strip()]
    for record in records:
        record.pop("sim_state_sha", None)
        record.pop("next_sim_state_sha", None)
    (final / METRICS_NAME).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert WARN in status_of(validate_episode(final), "F_sync")


def test_wrong_contract_declaration_fails(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    records = [json.loads(l) for l in (final / METRICS_NAME).read_text().splitlines() if l.strip()]
    records[0]["label_reference"] = "post_step_observation"
    (final / METRICS_NAME).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert FAIL in status_of(validate_episode(final), "F_sync")


def test_phase_entity_mismatch_fails(tmp_path):
    """post_grasp must score against the destination, never the source."""

    def mutate(timestep, activations, metrics):
        if timestep == 2:
            metrics["phase"] = "post_grasp"
            metrics["relevant_entity"] = SOURCE
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", num_steps=3, mutate=mutate)
    assert FAIL in status_of(validate_episode(final), "I_phase")


def test_valid_phase_transition_passes(tmp_path):
    def mutate(timestep, activations, metrics):
        if timestep >= 2:
            metrics["phase"] = "post_grasp"
            metrics["relevant_entity"] = DESTINATION
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", num_steps=4, mutate=mutate)
    assert status_of(validate_episode(final), "I_phase") == [PASS]


def test_unknown_change_class_fails(tmp_path):
    manifest = make_manifest(change_report={"change_class": "mostly_fine", "changed_entity_names": []})
    final = write_episode(tmp_path / "episode_000", manifest=manifest)
    assert FAIL in status_of(validate_episode(final), "H_entities")


def test_perturbed_condition_with_no_change_fails(tmp_path):
    """A condition named x0.1 that moved nothing means the asset did not apply."""
    manifest = make_manifest(
        change_report={"change_class": "no_detected_change", "changed_entity_names": []}
    )
    final = write_episode(tmp_path / "episode_000", manifest=manifest)
    assert FAIL in status_of(validate_episode(final), "H_entities")


def test_level_without_measured_translation_fails(tmp_path):
    """"x0.1" is a level, not 0.1 m; storing only the level is a defect."""
    manifest = make_manifest()
    manifest.pop("measured_translation_m")
    final = write_episode(tmp_path / "episode_000", manifest=manifest)
    assert FAIL in status_of(validate_episode(final), "H_entities")


def test_visible_with_zero_pixels_fails(tmp_path):
    def mutate(timestep, activations, metrics):
        metrics["segmentation"][SOURCE]["agentview"]["mask_pixel_count"] = 0
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", mutate=mutate)
    assert FAIL in status_of(validate_episode(final), "J_segmentation")


def test_uv_outside_unit_range_fails(tmp_path):
    def mutate(timestep, activations, metrics):
        metrics["segmentation"][SOURCE]["agentview"]["uv"] = [1.4, 0.5]
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", mutate=mutate)
    assert FAIL in status_of(validate_episode(final), "J_segmentation")


def test_schema_version_mismatch_is_fatal(tmp_path):
    final = write_episode(tmp_path / "episode_000")
    manifest = json.loads((final / MANIFEST_NAME).read_text())
    manifest["schema_version"] = SCHEMA_VERSION - 1
    (final / MANIFEST_NAME).write_text(json.dumps(manifest))
    report = validate_episode(final)
    assert not report["passed"]
    assert len(report["checks"]) == 1, "a schema mismatch must stop the run, not be one of many"


# -----------------------------------------------------------------------------
# Phase resolver: grasp detection must not depend on object thickness
# -----------------------------------------------------------------------------
from task_phase_resolver import DEFAULT_THRESHOLDS, TaskPhaseResolver  # noqa: E402


def _drive_grasp(finger_opening: float, steps: int = 12):
    """Approach, close on an object of the given thickness, then lift.

    ``finger_opening`` is where the fingers stall: ~0.005 for a libero_spatial
    bowl, ~0.063 for a libero_object soup can. Both are firm grasps.
    """
    resolver = TaskPhaseResolver("src", "dst", thresholds=DEFAULT_THRESHOLDS)
    phases = []
    for step in range(steps):
        closing = step >= 3
        holding = step >= 5
        opening = 0.0795 if not closing else finger_opening
        lift = 0.05 * max(0, step - 5)
        phases.append(
            resolver.update(
                timestep=step,
                source_position=[0.0, 0.0, 1.0 + lift],
                destination_position=[0.4, 0.0, 1.0],
                gripper_position=[0.0, 0.0, 1.01 + lift],
                gripper_qpos=[opening / 2, -opening / 2],
                contact=holding,
            ).phase
        )
    return phases


def test_grasp_detected_for_a_thin_object():
    assert "post_grasp" in _drive_grasp(0.005)


def test_grasp_detected_for_a_thick_object():
    """Regression: an absolute qpos threshold missed every libero_object grasp.

    The soup can blocks the fingers at 0.063, so "fingers nearly touching" never
    became true and a successful episode was labelled pre_grasp end to end.
    """
    assert "post_grasp" in _drive_grasp(0.063)


def test_open_gripper_resting_on_object_is_not_a_grasp():
    """Contact plus stalled fingers is not enough if the fingers never closed."""
    assert "post_grasp" not in _drive_grasp(0.0795)


# -----------------------------------------------------------------------------
# Regressions from the adversarial audit
# -----------------------------------------------------------------------------
def _drive(steps, lift_per_step, distance, opening, contact_from=5, opening_delta=0.0):
    """Drive the resolver with an explicit geometry."""
    resolver = TaskPhaseResolver("src", "dst", thresholds=DEFAULT_THRESHOLDS)
    phases = []
    for step in range(steps):
        held = step >= contact_from
        lift = lift_per_step * max(0, step - contact_from)
        slide = 0.005 * max(0, step - contact_from)
        gap = opening if held else 0.0795
        gap += opening_delta * max(0, step - contact_from)
        phases.append(
            resolver.update(
                timestep=step,
                source_position=[slide, 0.0, 1.0 + lift],
                destination_position=[0.4, 0.0, 1.0],
                gripper_position=[slide, distance, 1.0 + lift],
                gripper_qpos=[gap / 2, -gap / 2],
                contact=held,
            ).phase
        )
    return phases


def test_slid_but_never_lifted_object_is_not_a_grasp():
    """Regression: a nudged object 8.9 cm away was labelled post_grasp for 75 steps.

    It had slid 3.1 cm (past grasp_displacement_m) but never rose above 0.4 cm,
    and the gripper was commanded open. Displacement alone must not trigger a
    transition.
    """
    phases = _drive(steps=30, lift_per_step=0.0, distance=0.089, opening=0.066)
    assert "post_grasp" not in phases


def test_genuinely_lifted_object_is_still_a_grasp():
    phases = _drive(steps=30, lift_per_step=0.01, distance=0.03, opening=0.005)
    assert "post_grasp" in phases


def test_thick_object_lifted_is_still_a_grasp():
    """The libero_object soup can: fingers stall at 0.063, object genuinely rises."""
    phases = _drive(steps=30, lift_per_step=0.01, distance=0.03, opening=0.063)
    assert "post_grasp" in phases


def test_opening_fingers_are_not_blocked_fingers():
    """A gripper whose fingers creep apart is releasing, not holding."""
    phases = _drive(steps=30, lift_per_step=0.0, distance=0.03, opening=0.066,
                    opening_delta=+2e-4)
    assert "post_grasp" not in phases


def test_validator_flags_uv_and_mask_in_different_frames(tmp_path):
    """Regression: a whole camera's projections frozen at t=0 passed validation.

    Range-checking uv cannot catch it -- robosuite clips into the image before
    normalisation -- so the check has to compare uv against its own mask.
    """
    def mutate(timestep, activations, metrics):
        label = metrics["segmentation"][SOURCE]["agentview"]
        label["uv"] = [0.9, 0.9]                 # frozen far from the mask
        label["mask_bbox"] = [0.1, 0.1, 0.2, 0.2]
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", num_steps=30, mutate=mutate)
    report = validate_episode(final)
    assert FAIL in status_of(report, "J_uv_frame")
    assert not report["passed"]


def test_validator_accepts_uv_inside_its_mask(tmp_path):
    def mutate(timestep, activations, metrics):
        label = metrics["segmentation"][SOURCE]["agentview"]
        label["uv"] = [0.15, 0.15]
        label["mask_bbox"] = [0.1, 0.1, 0.2, 0.2]
        return activations, metrics

    final = write_episode(tmp_path / "episode_000", num_steps=30, mutate=mutate)
    assert status_of(validate_episode(final), "J_uv_frame") == [PASS]


def test_jitter_raises_the_threshold_for_a_widely_placed_entity():
    """Regression: 0.05 m sits below libero_spatial's own placement spread.

    On the cabinet tasks the source bowl's placements differ by up to 0.089 m
    across the shipped init states, so a swap episode -- which draws its
    placement independently -- reported the source as moved when nothing had
    touched it.
    """
    from changed_entity_detector import ObjectPose, detect_changed_entities
    from entity_role_resolver import EntityRoles

    roles = EntityRoles(
        source="bowl_1", destination="plate_1", goal_predicate="On",
        destination_goal_argument="plate_1", destination_was_region=False,
        distractors=[], fixtures=[], movable_objects=["bowl_1", "plate_1"],
    )
    vanilla = {"bowl_1": ObjectPose("bowl_1", [0.0, 0.0, 1.0]),
               "plate_1": ObjectPose("plate_1", [0.5, 0.0, 1.0])}
    perturbed = {"bowl_1": ObjectPose("bowl_1", [0.06, 0.0, 1.0]),   # jitter, not a move
                 "plate_1": ObjectPose("plate_1", [0.5, 0.0, 1.0])}

    naive = detect_changed_entities(vanilla, perturbed, roles)
    assert naive.source_changed, "without a jitter allowance this reads as a real move"

    aware = detect_changed_entities(vanilla, perturbed, roles, entity_jitter={"bowl_1": 0.089})
    assert not aware.source_changed
    assert aware.change_class == "no_detected_change"
    assert [e["name"] for e in aware.indeterminate_entities] == ["bowl_1"]


def test_jitter_allowance_does_not_mask_a_real_perturbation():
    """libero_object's real 0.070 m offset must still register."""
    from changed_entity_detector import ObjectPose, detect_changed_entities
    from entity_role_resolver import EntityRoles

    roles = EntityRoles(
        source="soup_1", destination="basket_1", goal_predicate="In",
        destination_goal_argument="basket_1_contain_region", destination_was_region=True,
        distractors=[], fixtures=[], movable_objects=["soup_1", "basket_1"],
    )
    vanilla = {"soup_1": ObjectPose("soup_1", [0.0, 0.0, 1.0]),
               "basket_1": ObjectPose("basket_1", [0.5, 0.0, 1.0])}
    perturbed = {"soup_1": ObjectPose("soup_1", [0.070, 0.0, 1.0]),
                 "basket_1": ObjectPose("basket_1", [0.5, 0.0, 1.0])}
    report = detect_changed_entities(vanilla, perturbed, roles, entity_jitter={"soup_1": 0.0396})
    assert report.source_changed
    assert report.change_class == "clean_source_only"


# -----------------------------------------------------------------------------
# probe_control mode: it must equalise the robot and change nothing else
# -----------------------------------------------------------------------------
from probe_control_mode import (  # noqa: E402
    MODE_OFFICIAL,
    MODE_PROBE_CONTROL,
    ROBOT_QPOS_DOF,
    splice_robot_pose,
)

NQ, NV = 58, 51


def _state(robot_val: float, object_val: float, time: float = 1.5) -> np.ndarray:
    """[time, qpos(58), qvel(51)] with distinguishable robot and object blocks."""
    state = np.zeros(1 + NQ + NV, dtype=np.float64)
    state[0] = time
    state[1:1 + ROBOT_QPOS_DOF] = robot_val
    state[1 + ROBOT_QPOS_DOF:1 + NQ] = object_val
    state[1 + NQ:1 + NQ + ROBOT_QPOS_DOF] = robot_val * 10
    state[1 + NQ + ROBOT_QPOS_DOF:] = object_val * 10
    return state


def test_probe_control_takes_the_robot_from_vanilla():
    spliced = splice_robot_pose(_state(2.0, 7.0), _state(1.0, 3.0), nq=NQ)
    assert np.allclose(spliced[1:1 + ROBOT_QPOS_DOF], 1.0), "robot must come from vanilla"
    assert np.allclose(spliced[1 + NQ:1 + NQ + ROBOT_QPOS_DOF], 10.0), "robot qvel too"


def test_probe_control_leaves_the_perturbation_alone():
    """The whole point: objects keep the perturbed placement."""
    spliced = splice_robot_pose(_state(2.0, 7.0), _state(1.0, 3.0), nq=NQ)
    assert np.allclose(spliced[1 + ROBOT_QPOS_DOF:1 + NQ], 7.0), "object qpos must be perturbed"
    assert np.allclose(spliced[1 + NQ + ROBOT_QPOS_DOF:], 70.0), "object qvel must be perturbed"
    assert spliced[0] == 1.5, "simulator time comes from the perturbed state"


def test_probe_control_does_not_mutate_its_inputs():
    perturbed, vanilla = _state(2.0, 7.0), _state(1.0, 3.0)
    before_p, before_v = perturbed.copy(), vanilla.copy()
    splice_robot_pose(perturbed, vanilla, nq=NQ)
    assert np.array_equal(perturbed, before_p), "caller's perturbed state must be untouched"
    assert np.array_equal(vanilla, before_v), "caller's vanilla state must be untouched"


def test_probe_control_rejects_mismatched_states():
    with pytest.raises(ValueError, match="state shapes differ"):
        splice_robot_pose(_state(2.0, 7.0), np.zeros(10), nq=NQ)


def test_official_mode_is_the_default_and_splices_nothing():
    """Regression: the official pipeline must be unchanged by this feature."""
    import collect_official_activations as collector

    parser = collector.build_parser()
    args = parser.parse_args([
        "--suite", "libero_object", "--task_id", "0",
        "--condition", "y0.2", "--output_root", "/tmp/unused",
    ])
    assert args.evaluation_mode == MODE_OFFICIAL

    source = Path(collector.__file__).read_text(encoding="utf-8")
    # The splice is reachable only behind the explicit mode check.
    assert "splice_robot_pose(" in source
    for line in source.splitlines():
        if "init_state = splice_robot_pose(" in line:
            break
    else:
        raise AssertionError("splice call not found")
    guard = f'args.evaluation_mode == MODE_PROBE_CONTROL'
    assert guard in source, "the splice must be guarded by an explicit mode check"
