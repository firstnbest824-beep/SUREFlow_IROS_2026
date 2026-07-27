"""Unit tests for the shared OpenVLA task-phase resolver.

Covers the eight required scenarios from the research plan plus regressions for
two bugs found during integration: the hovering-reads-as-grasp comovement false
positive, and the `reason` / `phase_reason` key mismatch in the timeline writer.
"""

import json
import os
import sys

_OPENVLA_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "openvla")
if _OPENVLA_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_TOOLS_DIR)

from task_phase_resolver import (  # noqa: E402
    PHASE_POST_GRASP,
    PHASE_PRE_GRASP,
    PHASE_UNCERTAIN,
    TaskPhaseResolver,
    _run_selftest,
    phase_result_to_timeline_entry,
    save_phase_timeline,
)

SRC0 = [0.0, 0.0, 0.9]
DST = [0.3, 0.0, 0.9]
OPEN_QPOS = [0.0208, -0.0208]
CLOSED_QPOS = [0.0, 0.0]


def make_resolver():
    return TaskPhaseResolver(source_entity="source_obj", destination_entity="dest_obj")


def drive_to_post_grasp(resolver, steps=9):
    """Contact + lift + comovement sustained -- the canonical successful grasp."""
    resolver.update(0, SRC0, DST, [0.0, 0.0, 1.0], OPEN_QPOS, contact=False)
    result = None
    for i in range(1, steps):
        held = [SRC0[0], SRC0[1], SRC0[2] + 0.012 * i]
        result = resolver.update(i, held, DST, held, CLOSED_QPOS, contact=True, gripper_command=1.0)
    return result


def test_module_selftest_suite_passes():
    assert _run_selftest() is True


# --- 1. far from source, no grasp evidence -> pre_grasp ----------------------
def test_far_from_source_without_evidence_is_pre_grasp():
    result = make_resolver().update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    assert result.phase == PHASE_PRE_GRASP
    assert result.grasp_detected is False


# --- 2. gripper closed but far away -> never post_grasp ----------------------
def test_closed_gripper_far_from_source_never_becomes_post_grasp():
    resolver = make_resolver()
    phases = [
        resolver.update(i, SRC0, DST, [0.4 + 0.01 * i, 0.0, 1.1], CLOSED_QPOS, contact=False).phase
        for i in range(12)
    ]
    assert PHASE_POST_GRASP not in phases


# --- 3. one brief contact -> no immediate transition -------------------------
def test_single_timestep_contact_does_not_flip_phase():
    resolver = make_resolver()
    at_src = [0.0, 0.0, 0.92]
    resolver.update(0, SRC0, DST, at_src, OPEN_QPOS, contact=False)
    assert resolver.update(1, SRC0, DST, at_src, CLOSED_QPOS, contact=True).phase != PHASE_POST_GRASP
    assert resolver.update(2, SRC0, DST, at_src, OPEN_QPOS, contact=False).phase != PHASE_POST_GRASP


# --- 4. sustained contact + lift + comovement -> post_grasp ------------------
def test_sustained_grasp_evidence_transitions_to_post_grasp():
    result = drive_to_post_grasp(make_resolver())
    assert result.phase == PHASE_POST_GRASP
    assert result.grasp_detected is True
    assert result.grasp_confidence >= 0.6
    assert result.transition_timestep is not None


# --- 5. uncertain -> relevant_entity is null ---------------------------------
def test_uncertain_phase_never_guesses_a_relevant_entity():
    result = make_resolver().update(0, None, DST, [0.0, 0.0, 0.9], CLOSED_QPOS, contact=None)
    assert result.phase == PHASE_UNCERTAIN
    assert result.relevant_entity is None
    assert result.relevant_entity_role == "none"


# --- 6. pre_grasp -> relevant entity is source -------------------------------
def test_pre_grasp_selects_source_entity():
    result = make_resolver().update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    assert result.relevant_entity == "source_obj"
    assert result.relevant_entity_role == "source"


# --- 7. post_grasp -> relevant entity is destination -------------------------
def test_post_grasp_selects_destination_entity():
    result = drive_to_post_grasp(make_resolver())
    assert result.relevant_entity == "dest_obj"
    assert result.relevant_entity_role == "destination"


# --- 8. transient noise must not cause oscillation ---------------------------
def test_flickering_contact_does_not_oscillate_the_phase():
    resolver = make_resolver()
    drive_to_post_grasp(resolver)
    assert resolver.phase == PHASE_POST_GRASP

    base_h = SRC0[2] + 0.012 * 8
    phases = []
    for i in range(9, 25):
        held = [SRC0[0], SRC0[1], base_h + 0.001 * (i - 8)]
        phases.append(resolver.update(i, held, DST, held, CLOSED_QPOS, contact=(i % 2 == 0)).phase)
    assert set(phases) == {PHASE_POST_GRASP}


# --- regression: hovering over a resting object is not a grasp ---------------
def test_hovering_over_a_resting_object_is_not_read_as_a_grasp():
    """Both bodies stationary => the relative offset is trivially constant.

    An earlier comovement definition looked only at whether the gripper-to-source
    *distance* was stable, so a closed gripper hovering over a resting object
    scored as "moving together" and fired a premature post_grasp transition.
    Comovement now additionally requires that both bodies actually travelled.
    """
    resolver = make_resolver()
    hover_eef = [SRC0[0], SRC0[1], SRC0[2] + 0.05]
    phases = [
        resolver.update(i, SRC0, DST, hover_eef, CLOSED_QPOS, contact=True).phase
        for i in range(15)
    ]
    assert PHASE_POST_GRASP not in phases


def test_reset_clears_episode_state():
    resolver = make_resolver()
    drive_to_post_grasp(resolver)
    assert resolver.phase == PHASE_POST_GRASP

    resolver.reset()
    result = resolver.update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    assert result.phase == PHASE_PRE_GRASP
    assert result.transition_timestep is None


def test_step_remains_an_alias_for_update():
    assert TaskPhaseResolver.step is TaskPhaseResolver.update


# --- artifact writer ---------------------------------------------------------
def test_save_phase_timeline_writes_every_required_artifact(tmp_path):
    """Regression: entries must carry `phase_reason`, not the raw `reason` key.

    save_phase_timeline indexes transitions by `phase_reason`; appending a bare
    TaskPhaseResult.to_dict() previously raised KeyError mid-rollout.
    """
    resolver = make_resolver()
    timeline = [
        phase_result_to_timeline_entry(
            resolver.update(0, SRC0, DST, [0.0, 0.0, 1.0], OPEN_QPOS, contact=False)
        )
    ]
    for i in range(1, 9):
        held = [SRC0[0], SRC0[1], SRC0[2] + 0.012 * i]
        timeline.append(
            phase_result_to_timeline_entry(
                resolver.update(i, held, DST, held, CLOSED_QPOS, contact=True)
            )
        )

    summary = save_phase_timeline(timeline, str(tmp_path))

    for name in (
        "phase_timeline.csv",
        "phase_timeline.jsonl",
        "phase_transition_summary.json",
        "phase_summary.json",
        "uncertain_timesteps.json",
        "relevant_entity_timeline.csv",
    ):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"

    assert summary["total_timesteps"] == len(timeline)
    assert summary["first_pre_grasp_to_post_grasp_timestep"] is not None
    assert summary["reached_post_grasp"] is True

    # phase_summary.json must embed the exact thresholds that produced the labels.
    with open(tmp_path / "phase_summary.json", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "thresholds" in saved["phase_resolver_config"]
    assert saved["phase_resolver_config"]["image_based_heuristics_used"] is False


def test_timeline_entries_are_json_serializable():
    resolver = make_resolver()
    entry = phase_result_to_timeline_entry(
        resolver.update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    )
    json.dumps(entry)  # must not raise
    assert "phase_reason" in entry
    assert "reason" not in entry
