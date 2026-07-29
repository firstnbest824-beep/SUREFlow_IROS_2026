"""Detect which entities an official perturbation actually moved, and classify it.

The perturbation's *name* is not evidence of what it did. ``x0.1`` does not
guarantee a 0.1 m displacement, ``swap`` does not guarantee only the destination
moved, and a suite name says nothing at all. Measured on the shipped LIBERO-PRO
assets, ``libero_object_temp_x0.2`` and above also teleport a distractor out of
the scene, and ``y0.4``/``y0.5`` additionally move the basket -- the destination.

So the pipeline resets both environments, reads every entity's initial pose, and
diffs them. The classification that downstream analysis groups on comes from that
measurement, never from the condition string.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from entity_role_resolver import (
    ROLE_DESTINATION,
    ROLE_DISTRACTOR,
    ROLE_FIXTURE,
    ROLE_SOURCE,
    ROLE_UNKNOWN,
    EntityRoles,
)

# A perturbation that moves an object by less than this is treated as "unchanged".
# LIBERO samples placements inside a small box every reset, so two resets of the
# *same* BDDL differ by up to the region size (2 cm on libero_spatial). The
# default therefore has to sit above that sampling jitter, otherwise every reset
# looks like a perturbation.
# Measured over the 50 shipped .pruned_init states: libero_object placements
# differ by at most 0.0396 m, but libero_spatial reaches 0.0890 m -- on the
# wooden-cabinet and stove tasks the source bowl sits on a region declared with
# no (:ranges ...), so it spans the whole cabinet top. A single global constant
# is therefore not safe: 0.05 m sits *below* the jitter of the very entity the
# swap condition leaves untouched, so ~12% of draws would report a source move
# that never happened. Raising it globally is not the fix either -- the real
# libero_object perturbation is only 0.070 m and would start to be masked.
#
# So this constant is the FLOOR, and `estimate_placement_jitter` raises it per
# entity from the actual placement distribution of that task.
DEFAULT_TRANSLATION_THRESHOLD_M = 0.05
DEFAULT_ROTATION_THRESHOLD_RAD = 0.20

#: Displacements between the jitter band and the threshold are not called either
#: way; they are reported as indeterminate so a downstream filter can drop them.
INDETERMINATE_BAND_FRACTION = 0.5

# Distance beyond which an entity is considered removed from the scene rather
# than relocated. LIBERO-PRO's position-offset assets push distractors to
# x ~ +10 m, which is not a spatial perturbation but a scene edit.
SCENE_EXIT_THRESHOLD_M = 5.0

CHANGE_CLASSES = (
    "no_detected_change",
    "clean_source_only",
    "clean_destination_only",
    "source_and_distractor",
    "destination_and_distractor",
    "source_and_destination",
    "source_destination_and_distractor",
    "other_multi_entity_change",
)


# -----------------------------------------------------------------------------
# Poses
# -----------------------------------------------------------------------------
@dataclass
class ObjectPose:
    name: str
    xyz: Optional[List[float]]
    quat: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def quaternion_distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    """Geodesic angle between two quaternions, in radians, sign-invariant."""
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    dot = max(-1.0, min(1.0, abs(dot)))
    return float(2.0 * math.acos(dot))


def extract_object_poses(env: Any, entities: Iterable[str]) -> Dict[str, ObjectPose]:
    """Read world pose of each entity from a live env. Missing -> xyz None."""
    from spatial_task_resolver import get_entity_world_position, unwrap_base_env

    poses: Dict[str, ObjectPose] = {}
    base_env = unwrap_base_env(env)
    states = getattr(base_env, "object_states_dict", {}) or {}
    for name in entities:
        xyz = get_entity_world_position(env, name)
        quat: Optional[List[float]] = None
        state = states.get(name)
        if state is not None:
            try:
                quat = [float(v) for v in state.get_geom_state()["quat"]]
            except Exception:
                quat = None
        poses[name] = ObjectPose(name=name, xyz=xyz, quat=quat)
    return poses


# -----------------------------------------------------------------------------
# Change detection
# -----------------------------------------------------------------------------
@dataclass
class ChangedEntity:
    name: str
    role: str
    vanilla_xyz: Optional[List[float]]
    perturbed_xyz: Optional[List[float]]
    translation_delta: Optional[List[float]]
    translation_norm: Optional[float]
    rotation_delta: Optional[float]
    left_scene: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeReport:
    changed_entities: List[ChangedEntity]
    change_class: str
    source_entity: str
    destination_entity: str
    source_changed: bool
    destination_changed: bool
    distractors_changed: List[str]
    entities_left_scene: List[str]
    translation_threshold_m: float
    rotation_threshold_rad: float
    unmeasurable_entities: List[str] = field(default_factory=list)
    #: Per-entity thresholds actually applied (jitter-aware).
    per_entity_threshold_m: Dict[str, float] = field(default_factory=dict)
    #: Entities that moved measurably but not enough to be called changed.
    indeterminate_entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["changed_entity_names"] = [c.name for c in self.changed_entities]
        return payload


def detect_changed_entities(
    vanilla_poses: Dict[str, ObjectPose],
    perturbed_poses: Dict[str, ObjectPose],
    roles: EntityRoles,
    translation_threshold: float = DEFAULT_TRANSLATION_THRESHOLD_M,
    rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD_RAD,
    scene_exit_threshold: float = SCENE_EXIT_THRESHOLD_M,
    entity_jitter: Optional[Dict[str, float]] = None,
) -> ChangeReport:
    """Diff two initial-pose maps and classify what the perturbation did.

    ``entity_jitter`` maps entity -> that entity's observed placement spread for
    this task (see ``estimate_placement_jitter``). It raises the per-entity
    threshold so an object the perturbation never touched is not called changed
    just because it was re-sampled inside a large region.
    """
    changed: List[ChangedEntity] = []
    unmeasurable: List[str] = []
    indeterminate: List[Dict[str, Any]] = []
    jitter = entity_jitter or {}
    thresholds: Dict[str, float] = {}

    for name in sorted(set(vanilla_poses) | set(perturbed_poses)):
        van = vanilla_poses.get(name)
        per = perturbed_poses.get(name)
        if van is None or per is None or van.xyz is None or per.xyz is None:
            unmeasurable.append(name)
            continue

        delta = [float(p) - float(v) for v, p in zip(van.xyz, per.xyz)]
        norm = float(math.sqrt(sum(d * d for d in delta)))
        rot = quaternion_distance(van.quat, per.quat)

        entity_threshold = max(translation_threshold, float(jitter.get(name, 0.0)))
        thresholds[name] = entity_threshold

        moved = norm >= entity_threshold
        rotated = rot is not None and rot >= rotation_threshold
        if not (moved or rotated):
            if norm >= entity_threshold * INDETERMINATE_BAND_FRACTION:
                indeterminate.append({
                    "name": name, "role": roles.role_of(name),
                    "translation_norm": norm, "threshold": entity_threshold,
                })
            continue

        changed.append(ChangedEntity(
            name=name,
            role=roles.role_of(name),
            vanilla_xyz=[float(v) for v in van.xyz],
            perturbed_xyz=[float(v) for v in per.xyz],
            translation_delta=delta,
            translation_norm=norm,
            rotation_delta=rot,
            left_scene=norm >= scene_exit_threshold,
        ))

    names = {c.name for c in changed}
    source_changed = roles.source in names
    destination_changed = roles.destination in names
    distractors_changed = sorted(
        n for n in names if roles.role_of(n) in (ROLE_DISTRACTOR, ROLE_FIXTURE, ROLE_UNKNOWN)
    )

    return ChangeReport(
        changed_entities=changed,
        change_class=classify_change(source_changed, destination_changed, distractors_changed, changed),
        source_entity=roles.source,
        destination_entity=roles.destination,
        source_changed=source_changed,
        destination_changed=destination_changed,
        distractors_changed=distractors_changed,
        entities_left_scene=sorted(c.name for c in changed if c.left_scene),
        translation_threshold_m=translation_threshold,
        rotation_threshold_rad=rotation_threshold,
        unmeasurable_entities=unmeasurable,
        per_entity_threshold_m=thresholds,
        indeterminate_entities=indeterminate,
    )


def classify_change(
    source_changed: bool,
    destination_changed: bool,
    distractors_changed: Sequence[str],
    changed: Sequence[ChangedEntity],
) -> str:
    """Bucket a measured change. `clean_*` means exactly one role moved."""
    if not changed:
        return "no_detected_change"
    others = bool(distractors_changed)

    if source_changed and destination_changed:
        return "source_destination_and_distractor" if others else "source_and_destination"
    if source_changed:
        return "source_and_distractor" if others else "clean_source_only"
    if destination_changed:
        return "destination_and_distractor" if others else "clean_destination_only"
    return "other_multi_entity_change"


def is_clean(change_class: str) -> bool:
    """Only single-role changes are usable for causal attribution."""
    return change_class in ("clean_source_only", "clean_destination_only")


# -----------------------------------------------------------------------------
# CLI helper for Gate 6
# -----------------------------------------------------------------------------
def summarise(report: ChangeReport) -> str:
    lines = [
        f"change_class      : {report.change_class}"
        f"{'  (clean)' if is_clean(report.change_class) else '  (confounded)'}",
        f"source            : {report.source_entity} changed={report.source_changed}",
        f"destination       : {report.destination_entity} changed={report.destination_changed}",
        f"distractors moved : {report.distractors_changed or '-'}",
        f"left scene        : {report.entities_left_scene or '-'}",
    ]
    for entity in report.changed_entities:
        lines.append(
            f"  {entity.name:34s} role={entity.role:11s} "
            f"|d|={entity.translation_norm:.4f} m"
            f"{' [LEFT SCENE]' if entity.left_scene else ''}"
        )
    for entry in report.indeterminate_entities:
        lines.append(
            f"  ? {entry['name']:34s} role={entry['role']:11s} "
            f"|d|={entry['translation_norm']:.4f} m below threshold "
            f"{entry['threshold']:.4f} -- indeterminate"
        )
    if report.unmeasurable_entities:
        lines.append(f"  unmeasurable: {report.unmeasurable_entities}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Placement jitter
# -----------------------------------------------------------------------------
#: Jitter is a property of (task, entity), so it is computed once per BDDL.
_JITTER_CACHE: Dict[str, Dict[str, float]] = {}


def estimate_placement_jitter(
    bddl_path: str,
    entities: Sequence[str],
    resolution: int = 256,
    num_samples: int = 6,
    make_env: Any = None,
) -> Dict[str, float]:
    """Max pairwise spread of each entity across several official placements.

    LIBERO re-samples every object inside its declared region on each reset, and
    the regions are not all small: most are a 2 cm box, but a region declared
    with no ``(:ranges ...)`` spans its whole target surface. Comparing a vanilla
    episode against a condition that drew its placement independently therefore
    shows movement for objects nobody perturbed.

    This measures that spread directly, through the same code path that reads
    poses during collection, so the number used as a threshold is the number the
    detector would actually see. Returns entity -> max pairwise displacement.
    """
    key = f"{bddl_path}|{num_samples}|{','.join(sorted(entities))}"
    if key in _JITTER_CACHE:
        return _JITTER_CACHE[key]

    from init_state_freezer import load_init_states, official_init_state_path

    official = official_init_state_path(bddl_path)
    if official is None:
        _JITTER_CACHE[key] = {}
        return {}

    states = load_init_states(official)
    if len(states) < 2:
        _JITTER_CACHE[key] = {}
        return {}

    if make_env is None:
        from libero.libero.envs import OffScreenRenderEnv

        def make_env(path):  # noqa: ANN001
            env = OffScreenRenderEnv(
                bddl_file_name=str(path), camera_heights=resolution, camera_widths=resolution
            )
            env.seed(0)
            return env

    # Evenly spaced samples so the estimate is not dominated by adjacent draws.
    step = max(1, len(states) // num_samples)
    indices = list(range(0, len(states), step))[:num_samples]

    observed: Dict[str, List[List[float]]] = {name: [] for name in entities}
    env = make_env(bddl_path)
    try:
        for index in indices:
            env.reset()
            env.set_init_state(states[index])
            for name, pose in extract_object_poses(env, entities).items():
                if pose.xyz is not None:
                    observed[name].append([float(v) for v in pose.xyz])
    finally:
        try:
            env.close()
        except Exception:
            pass

    jitter: Dict[str, float] = {}
    for name, positions in observed.items():
        if len(positions) < 2:
            continue
        jitter[name] = max(
            float(math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second))))
            for i, first in enumerate(positions)
            for second in positions[i + 1:]
        )
    _JITTER_CACHE[key] = jitter
    return jitter
