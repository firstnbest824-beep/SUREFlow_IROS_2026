"""Gate 2 unit tests for the suite-independent official-benchmark collection path.

Covers entity role resolution, changed-entity detection, change classification,
seed validation and official task pairing. Everything that can run without the
simulator or the checkpoint runs here; the rest is covered by the dry-runs.
"""

import os
import sys

import pytest

_OPENVLA_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "openvla")
if _OPENVLA_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_TOOLS_DIR)

from changed_entity_detector import (  # noqa: E402
    DEFAULT_ROTATION_THRESHOLD_RAD,
    DEFAULT_TRANSLATION_THRESHOLD_M,
    ChangedEntity,
    ObjectPose,
    classify_change,
    detect_changed_entities,
    is_clean,
    quaternion_distance,
)
from entity_role_resolver import (  # noqa: E402
    ROLE_DESTINATION,
    ROLE_DISTRACTOR,
    ROLE_SOURCE,
    normalise_entity,
    parse_region_owners,
    resolve_entity_roles,
)
from official_task_pair_resolver import (  # noqa: E402
    FAMILY_POSITION_OFFSET,
    FAMILY_SWAP,
    SUITE_CHECKPOINTS,
    assert_checkpoint_matches_suite,
    seed_everything,
    sha256_text,
    validate_seed,
)

# --- minimal BDDL fixtures ---------------------------------------------------
SPATIAL_BDDL = """(define (problem T)
  (:language Pick the akita black bowl between the plate and the ramekin and place it on the plate)
    (:regions
      (plate_region (:target main_table) (:ranges ((0.05 0.19 0.07 0.21))))
      (ramekin_region (:target main_table) (:ranges ((-0.21 0.19 -0.19 0.21))))
    )
  (:fixtures main_table - table)
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    cookies_1 - cookies
    plate_1 - plate
  )
  (:obj_of_interest akita_black_bowl_1 plate_1)
  (:init
    (On akita_black_bowl_1 main_table_ramekin_region)
    (On akita_black_bowl_2 main_table_plate_region)
    (On cookies_1 main_table_plate_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal (And (On akita_black_bowl_1 plate_1)))
)
"""

# The destination here is a REGION, not an object -- the libero_object shape.
OBJECT_BDDL = """(define (problem T)
  (:language Pick the alphabet soup and place it in the basket)
    (:regions
      (target_object_region (:target floor) (:ranges ((-0.145 -0.265 -0.095 -0.215))))
      (other_object_region_0 (:target floor) (:ranges ((0.0 0.0 0.01 0.01))))
      (bin_region (:target floor) (:ranges ((0.1 0.1 0.2 0.2))))
      (contain_region (:target basket_1))
    )
  (:fixtures floor - floor)
  (:objects
    alphabet_soup_1 - alphabet_soup
    milk_1 - milk
    basket_1 - basket
  )
  (:obj_of_interest alphabet_soup_1 basket_1)
  (:init
    (On alphabet_soup_1 floor_target_object_region)
    (On milk_1 floor_other_object_region_0)
    (On basket_1 floor_bin_region)
  )
  (:goal (And (In alphabet_soup_1 basket_1_contain_region)))
)
"""


# --- entity role resolution --------------------------------------------------
def test_spatial_roles_object_destination():
    roles = resolve_entity_roles(SPATIAL_BDDL)
    assert roles.source == "akita_black_bowl_1"
    assert roles.destination == "plate_1"
    assert roles.destination_was_region is False
    assert roles.goal_predicate == "On"
    assert set(roles.distractors) == {"akita_black_bowl_2", "cookies_1"}


def test_object_roles_normalise_region_destination_to_owning_object():
    """`basket_1_contain_region` must resolve to `basket_1` -- the thing that moves."""
    roles = resolve_entity_roles(OBJECT_BDDL)
    assert roles.source == "alphabet_soup_1"
    assert roles.destination == "basket_1", "region destination was not normalised"
    assert roles.destination_goal_argument == "basket_1_contain_region"
    assert roles.destination_was_region is True
    assert roles.distractors == ["milk_1"]


def test_role_of_assigns_every_category():
    roles = resolve_entity_roles(OBJECT_BDDL)
    assert roles.role_of("alphabet_soup_1") == ROLE_SOURCE
    assert roles.role_of("basket_1") == ROLE_DESTINATION
    assert roles.role_of("milk_1") == ROLE_DISTRACTOR


def test_tracked_entities_lead_with_source_and_destination():
    roles = resolve_entity_roles(OBJECT_BDDL)
    assert roles.tracked_entities[:2] == ["alphabet_soup_1", "basket_1"]
    assert "milk_1" in roles.tracked_entities


def test_region_owner_parsing():
    owners = parse_region_owners(OBJECT_BDDL)
    assert owners["basket_1_contain_region"] == "basket_1"
    assert owners["floor_target_object_region"] == "floor"


def test_normalise_entity_leaves_declared_objects_alone():
    owners = parse_region_owners(OBJECT_BDDL)
    movable = ["alphabet_soup_1", "milk_1", "basket_1"]
    assert normalise_entity("basket_1", owners, movable) == "basket_1"
    assert normalise_entity("basket_1_contain_region", owners, movable) == "basket_1"


def test_unresolvable_destination_raises_instead_of_guessing():
    broken = OBJECT_BDDL.replace("(contain_region (:target basket_1))", "")
    broken = broken.replace("(In alphabet_soup_1 basket_1_contain_region)",
                            "(In alphabet_soup_1 nowhere_region)")
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        resolve_entity_roles(broken)


# --- changed entity detection ------------------------------------------------
def _roles_object():
    return resolve_entity_roles(OBJECT_BDDL)


def _poses(**kwargs):
    return {name: ObjectPose(name=name, xyz=list(xyz), quat=[0, 0, 0, 1])
            for name, xyz in kwargs.items()}


def test_detects_source_only_move_as_clean():
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.21, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.change_class == "clean_source_only"
    assert is_clean(report.change_class)
    assert report.source_changed and not report.destination_changed
    assert report.changed_entities[0].translation_norm == pytest.approx(0.21)


def test_detects_destination_only_move_as_clean():
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1.25, 0, 1), milk_1=(2, 0, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.change_class == "clean_destination_only"
    assert report.destination_changed and not report.source_changed


def test_source_plus_distractor_is_not_clean():
    """The real libero_object x0.2+ case: target moves AND a distractor is removed."""
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.21, 0, 1), basket_1=(1, 0, 1), milk_1=(12, 0, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.change_class == "source_and_distractor"
    assert not is_clean(report.change_class)
    assert report.distractors_changed == ["milk_1"]


def test_scene_exit_is_flagged_separately_from_relocation():
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.21, 0, 1), basket_1=(1, 0, 1), milk_1=(12, 0, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.entities_left_scene == ["milk_1"]
    moved = {c.name: c for c in report.changed_entities}
    assert moved["milk_1"].left_scene is True
    assert moved["alphabet_soup_1"].left_scene is False


def test_source_and_destination_both_moving_is_its_own_class():
    """The real libero_object y0.4/y0.5 case."""
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.28, 0, 1), basket_1=(1.3, 0, 1), milk_1=(2, 0, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.change_class == "source_and_destination"
    assert not is_clean(report.change_class)


def test_source_destination_and_distractor_class():
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.28, 0, 1), basket_1=(1.3, 0, 1), milk_1=(12, 0, 1))
    assert detect_changed_entities(van, per, roles).change_class == "source_destination_and_distractor"


def test_sampling_jitter_is_not_reported_as_a_change():
    """Two resets of the same BDDL differ by up to the region size; that is not a perturbation."""
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1), milk_1=(2, 0, 1))
    per = _poses(alphabet_soup_1=(0.012, 0.008, 1), basket_1=(1.01, 0, 1), milk_1=(2, 0.009, 1))
    report = detect_changed_entities(van, per, roles)
    assert report.change_class == "no_detected_change"
    assert report.changed_entities == []


def test_rotation_only_change_is_detected():
    roles = _roles_object()
    van = {"alphabet_soup_1": ObjectPose("alphabet_soup_1", [0, 0, 1], [0, 0, 0, 1]),
           "basket_1": ObjectPose("basket_1", [1, 0, 1], [0, 0, 0, 1]),
           "milk_1": ObjectPose("milk_1", [2, 0, 1], [0, 0, 0, 1])}
    per = dict(van)
    per["alphabet_soup_1"] = ObjectPose("alphabet_soup_1", [0, 0, 1], [0, 0, 0.7071, 0.7071])
    report = detect_changed_entities(van, per, roles)
    assert report.source_changed
    assert report.changed_entities[0].rotation_delta > DEFAULT_ROTATION_THRESHOLD_RAD


def test_missing_pose_is_reported_not_silently_dropped():
    roles = _roles_object()
    van = _poses(alphabet_soup_1=(0, 0, 1), basket_1=(1, 0, 1))
    per = _poses(alphabet_soup_1=(0.21, 0, 1), basket_1=(1, 0, 1))
    van["milk_1"] = ObjectPose("milk_1", None)
    per["milk_1"] = ObjectPose("milk_1", None)
    report = detect_changed_entities(van, per, roles)
    assert "milk_1" in report.unmeasurable_entities


def test_quaternion_distance_is_sign_invariant():
    q = [0.0, 0.0, 0.3827, 0.9239]
    assert quaternion_distance(q, [-v for v in q]) == pytest.approx(0.0, abs=1e-6)
    assert quaternion_distance(q, q) == pytest.approx(0.0, abs=1e-6)


def test_classify_change_is_total():
    """Every combination maps to a declared class."""
    seen = set()
    dummy = [ChangedEntity("x", "source", None, None, None, 1.0, None)]
    for src in (True, False):
        for dst in (True, False):
            for others in ([], ["d"]):
                seen.add(classify_change(src, dst, others, dummy))
    assert seen <= set(__import__("changed_entity_detector").CHANGE_CLASSES)
    assert classify_change(False, False, [], []) == "no_detected_change"


# --- seed handling -----------------------------------------------------------
def test_validate_seed_rejects_the_upstream_type_object_default():
    """LIBERO-PRO does `configs.get("seed", int)`; `int` must never be accepted."""
    with pytest.raises(TypeError):
        validate_seed(int)


@pytest.mark.parametrize("bad", ["0", None, 1.5, True, [], {}])
def test_validate_seed_rejects_non_integers(bad):
    with pytest.raises(TypeError):
        validate_seed(bad)


def test_validate_seed_accepts_integers():
    assert validate_seed(0) == 0
    assert validate_seed(42) == 42


def test_seed_everything_makes_random_draws_reproducible():
    import random

    import numpy as np

    seed_everything(7)
    a = (random.random(), float(np.random.rand()))
    seed_everything(7)
    b = (random.random(), float(np.random.rand()))
    assert a == b


# --- checkpoint / suite binding ---------------------------------------------
def test_each_supported_suite_has_a_distinct_checkpoint():
    ids = {s: c["model_id"] for s, c in SUITE_CHECKPOINTS.items()}
    assert len(set(ids.values())) == len(ids), f"checkpoints collide: {ids}"
    for suite, cfg in SUITE_CHECKPOINTS.items():
        assert suite in cfg["model_id"].replace("-", "_"), f"{suite} -> {cfg['model_id']}"
        assert len(cfg["revision"]) == 40


def test_checkpoint_suite_mismatch_is_rejected():
    assert_checkpoint_matches_suite("libero_spatial", SUITE_CHECKPOINTS["libero_spatial"]["model_id"])
    with pytest.raises(RuntimeError, match="mismatch"):
        assert_checkpoint_matches_suite("libero_spatial", SUITE_CHECKPOINTS["libero_object"]["model_id"])


def test_sha256_text_is_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


# --- official condition discovery (needs the LIBERO asset tree) --------------
libero = pytest.importorskip("libero", reason="LIBERO not importable")


def test_official_conditions_match_shipped_assets():
    from official_task_pair_resolver import available_conditions

    spatial = available_conditions("libero_spatial")
    obj = available_conditions("libero_object")

    # libero_spatial genuinely ships no position-offset assets.
    assert spatial.get("vanilla") == "vanilla"
    assert not [c for c, f in spatial.items() if f == FAMILY_POSITION_OFFSET]

    # libero_object ships x0.1..x0.5 and y0.1..y0.5.
    offsets = sorted(c for c, f in obj.items() if f == FAMILY_POSITION_OFFSET)
    assert offsets == [f"{a}0.{i}" for a in ("x", "y") for i in range(1, 6)]


def test_unknown_condition_raises_rather_than_falling_back():
    from official_task_pair_resolver import resolve_official_task_pair

    with pytest.raises(KeyError, match="not an official condition"):
        resolve_official_task_pair("libero_spatial", 0, "x0.1", seed=0)


def test_position_offset_pair_records_requested_level_without_claiming_displacement():
    from official_task_pair_resolver import resolve_official_task_pair

    pair = resolve_official_task_pair("libero_object", 0, "x0.1", seed=0)
    assert pair.perturbation_family == FAMILY_POSITION_OFFSET
    assert pair.requested_axis == "x" and pair.requested_level == 0.1
    assert pair.primary_analysis_entity == "source"
    assert pair.perturbed_bddl_sha256 and pair.vanilla_bddl_sha256
    assert pair.perturbed_bddl_sha256 != pair.vanilla_bddl_sha256
    assert any("NOT a measured displacement" in n for n in pair.notes)


def test_swap_pair_is_deterministic_across_calls(tmp_path):
    from official_task_pair_resolver import resolve_official_task_pair

    a = resolve_official_task_pair("libero_spatial", 0, "swap", seed=0, asset_dir=tmp_path)
    b = resolve_official_task_pair("libero_spatial", 0, "swap", seed=0, asset_dir=tmp_path)
    assert a.perturbed_bddl_sha256 == b.perturbed_bddl_sha256
    assert a.perturbation_family == FAMILY_SWAP
    assert a.primary_analysis_entity == "destination"


def test_suite_checkpoint_is_selected_per_suite():
    from official_task_pair_resolver import resolve_official_task_pair

    s = resolve_official_task_pair("libero_spatial", 0, "vanilla", seed=0)
    o = resolve_official_task_pair("libero_object", 0, "vanilla", seed=0)
    assert "spatial" in s.model_id and s.unnorm_key == "libero_spatial"
    assert "object" in o.model_id and o.unnorm_key == "libero_object"


def test_goal_entities_survive_the_perturbation():
    """A position perturbation must not change what the task is."""
    from official_task_pair_resolver import resolve_official_task_pair

    van = resolve_official_task_pair("libero_object", 0, "vanilla", seed=0)
    off = resolve_official_task_pair("libero_object", 0, "y0.5", seed=0)
    assert (van.source_entity, van.destination_entity) == (off.source_entity, off.destination_entity)
