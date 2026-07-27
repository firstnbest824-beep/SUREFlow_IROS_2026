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
    phase_result_to_timeline_entry,
    save_phase_timeline,
    _run_selftest,
)


def test_synthetic_selftest_suite_passes():
    assert _run_selftest() is True


def test_initial_step_with_no_evidence_is_pre_grasp_with_source_relevant():
    resolver = TaskPhaseResolver(source_entity="source_obj", destination_entity="dest_obj")
    result = resolver.step(
        timestep=0,
        source_position=[0.0, 0.0, 0.9],
        destination_position=[0.3, 0.0, 0.9],
        gripper_position=[0.4, 0.0, 1.1],
        gripper_qpos=[0.0208, -0.0208],
        contact=False,
    )
    assert result.phase == PHASE_PRE_GRASP
    assert result.relevant_entity == "source_obj"
    assert result.relevant_entity_role == "source"
    assert result.grasp_detected is False


def test_missing_positions_never_guess_a_relevant_entity():
    resolver = TaskPhaseResolver(source_entity="source_obj", destination_entity="dest_obj")
    result = resolver.step(
        timestep=0,
        source_position=None,
        destination_position=[0.3, 0.0, 0.9],
        gripper_position=[0.1, 0.0, 0.9],
        gripper_qpos=[0.02, -0.02],
        contact=None,
    )
    assert result.phase == PHASE_UNCERTAIN
    assert result.relevant_entity is None
    assert result.relevant_entity_role == "none"


def test_sustained_grasp_evidence_switches_relevant_entity_to_destination():
    resolver = TaskPhaseResolver(source_entity="source_obj", destination_entity="dest_obj")
    src = [0.0, 0.0, 0.9]
    dst = [0.3, 0.0, 0.9]
    resolver.step(0, src, dst, [0.4, 0.0, 1.1], [0.0208, -0.0208], contact=False)

    last = None
    for i in range(1, 8):
        lifted = [src[0], src[1], src[2] + 0.01 * i]
        last = resolver.step(i, lifted, dst, lifted, [0.0, 0.0], contact=True)

    assert last.phase == PHASE_POST_GRASP
    assert last.relevant_entity == "dest_obj"
    assert last.relevant_entity_role == "destination"
    assert last.grasp_detected is True


def test_save_phase_timeline_round_trips_through_the_shared_entry_shape(tmp_path):
    # Regression test: phase_timeline entries must be built with
    # phase_result_to_timeline_entry (which renames `reason` -> `phase_reason`),
    # not a bare `TaskPhaseResult.to_dict()` -- save_phase_timeline's transition
    # summary indexes entries by `phase_reason` and previously raised KeyError
    # when a raw to_dict() was appended directly.
    resolver = TaskPhaseResolver(source_entity="source_obj", destination_entity="dest_obj")
    src = [0.0, 0.0, 0.9]
    dst = [0.3, 0.0, 0.9]
    timeline = []
    timeline.append(
        phase_result_to_timeline_entry(
            resolver.step(0, src, dst, [0.4, 0.0, 1.1], [0.0208, -0.0208], contact=False)
        )
    )
    for i in range(1, 8):
        lifted = [src[0], src[1], src[2] + 0.01 * i]
        timeline.append(
            phase_result_to_timeline_entry(
                resolver.step(i, lifted, dst, lifted, [0.0, 0.0], contact=True)
            )
        )

    summary = save_phase_timeline(timeline, str(tmp_path))

    assert (tmp_path / "phase_timeline.csv").exists()
    assert (tmp_path / "phase_timeline.jsonl").exists()
    assert (tmp_path / "phase_transition_summary.json").exists()
    assert summary["total_timesteps"] == len(timeline)
    assert summary["first_pre_grasp_to_post_grasp_timestep"] is not None
    assert all("reason" in t for t in summary["transitions"])
