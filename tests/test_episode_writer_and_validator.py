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
