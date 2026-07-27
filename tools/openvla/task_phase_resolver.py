"""Shared pre-grasp / post-grasp phase resolver for OpenVLA spatial diagnostics.

Every rollout consumer (probe dry-run, failure-screening, and any future
large-scale collection) needs to agree on *when* a timestep is pre-grasp versus
post-grasp, because that decides which resolved entity
(``spatial_task_resolver.SpatialTaskEntities.source_object`` /
``destination_object``) is the "relevant entity" for that timestep:

* pre_grasp  -> relevant_entity = source_object
* post_grasp -> relevant_entity = destination_object
* uncertain  -> relevant_entity = None (never guessed)

This module does not hard-code any object name. It only consumes generic
simulator quantities the caller supplies (or that ``compute_frame_inputs``
reads generically through ``spatial_task_resolver`` and robosuite's
``MujocoEnv.check_contact``): world positions, gripper joint position, and a
gripper/source contact boolean.

Grasp detection intentionally does not trust a single frame. A closed gripper
touching the source object for one timestep is not enough evidence -- contact,
proximity, gripper closure, and object displacement/comovement are combined
into a confidence score, and a transition only fires once that evidence has
been sustained for ``min_consecutive_grasp_steps_for_transition`` consecutive
timesteps (see ``PhaseThresholds``). Symmetrically, a post_grasp episode that
loses contact and sees the source object fall back near its initial height is
downgraded to ``uncertain`` (not silently kept as post_grasp, and not forced
back to pre_grasp) once that evidence is itself sustained.

Image-based heuristics are intentionally not used here; only simulator state
(gripper qpos, world poses, MuJoCo contact) drives the decision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# -----------------------------------------------------------------------------
# Thresholds (code constants; recorded verbatim into episode metadata so a
# downstream consumer never has to guess what produced a given label).
# -----------------------------------------------------------------------------
@dataclass
class PhaseThresholds:
    # Evidence gates.
    grasp_distance_m: float = 0.06
    grasp_height_delta_m: float = 0.02
    grasp_displacement_m: float = 0.03
    gripper_open_qpos_sum: float = 0.03
    gripper_closed_qpos_sum: float = 0.01
    gripper_closing_qpos_delta: float = -0.0008
    release_height_drop_m: float = 0.015

    # Comovement. "Moving together" requires that BOTH bodies actually moved
    # over the window AND that their relative offset stayed rigid. Checking only
    # that the gripper-to-source *distance* is stable is not enough: a gripper
    # hovering over a resting object trivially satisfies that and would be
    # misread as a grasp.
    comovement_relative_drift_m: float = 0.015
    min_comovement_motion_m: float = 0.010

    # Temporal continuity (all in units of timesteps).
    min_consecutive_contact_steps: int = 3
    min_consecutive_comovement_steps: int = 3
    min_consecutive_grasp_steps_for_transition: int = 3
    min_consecutive_release_steps_for_transition: int = 5
    distance_history_window: int = 5

    grasp_confidence_threshold: float = 0.6

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_THRESHOLDS = PhaseThresholds()

# Evidence weights for the confidence score. Must sum to 1.0.
EVIDENCE_WEIGHTS: Dict[str, float] = {
    "contact": 0.30,
    "proximity": 0.15,
    "gripper_closed": 0.15,
    "lifted_or_displaced": 0.25,
    "comovement": 0.15,
}
assert abs(sum(EVIDENCE_WEIGHTS.values()) - 1.0) < 1e-9

PHASE_PRE_GRASP = "pre_grasp"
PHASE_POST_GRASP = "post_grasp"
PHASE_UNCERTAIN = "uncertain"


def config_snapshot(thresholds: PhaseThresholds = DEFAULT_THRESHOLDS) -> Dict[str, Any]:
    """Everything a downstream consumer needs to reproduce phase decisions."""
    return {
        "thresholds": thresholds.to_dict(),
        "evidence_weights": dict(EVIDENCE_WEIGHTS),
        "image_based_heuristics_used": False,
    }


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------
@dataclass
class TaskPhaseResult:
    timestep: int
    phase: str
    previous_phase: str
    relevant_entity: Optional[str]
    relevant_entity_role: str  # "source" | "destination" | "none"
    grasp_detected: bool
    grasp_confidence: float
    evidence: Dict[str, Any]
    transition_detected: bool
    # Timestep of the most recent phase change (this timestep when
    # ``transition_detected``, otherwise the last one seen; None before any).
    transition_timestep: Optional[int]
    reason: str

    # Convenience fields also required verbatim by the per-timestep pipeline
    # output (section 5 of the research plan).
    source_to_gripper_distance: Optional[float]
    source_height_delta: Optional[float]
    source_displacement: Optional[float]
    # source_position - gripper_position, and how much that offset moved since
    # the previous timestep. A rigidly held object keeps the offset ~constant.
    source_eef_relative_position: Optional[List[float]]
    source_eef_relative_motion: Optional[float]
    gripper_state: str
    gripper_command: Optional[float]
    contact: Optional[bool]
    source_position: Optional[List[float]]
    destination_position: Optional[List[float]]
    gripper_position: Optional[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _euclidean(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _gripper_opening(qpos: Optional[Sequence[float]]) -> Optional[float]:
    if qpos is None:
        return None
    return float(sum(abs(float(value)) for value in qpos))


# -----------------------------------------------------------------------------
# Resolver
# -----------------------------------------------------------------------------
class TaskPhaseResolver:
    """Stateful pre_grasp/post_grasp/uncertain classifier for one episode.

    One instance must be used for exactly one (source_entity, destination_entity)
    episode; state (streaks, initial source pose, history) is carried across
    ``update`` calls so the classifier can require sustained evidence instead of
    reacting to a single noisy frame. Call ``reset()`` to reuse the instance for
    a new episode.
    """

    def __init__(
        self,
        source_entity: Optional[str],
        destination_entity: Optional[str],
        thresholds: PhaseThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self.source_entity = source_entity
        self.destination_entity = destination_entity
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        """Clear all per-episode state so the instance can start a new episode."""
        self.phase: str = PHASE_PRE_GRASP
        self._initial_source_pos: Optional[List[float]] = None
        self._prev_gripper_qpos: Optional[List[float]] = None
        self._prev_relative: Optional[List[float]] = None
        self._source_history: List[List[float]] = []
        self._eef_history: List[List[float]] = []
        self._relative_history: List[List[float]] = []
        self._last_transition_timestep: Optional[int] = None

        self._contact_streak = 0
        self._grasp_evidence_streak = 0
        self._release_evidence_streak = 0
        self._comovement_streak = 0

    def update(
        self,
        timestep: int,
        source_position: Optional[Sequence[float]],
        destination_position: Optional[Sequence[float]],
        gripper_position: Optional[Sequence[float]],
        gripper_qpos: Optional[Sequence[float]],
        contact: Optional[bool],
        gripper_command: Optional[float] = None,
    ) -> TaskPhaseResult:
        t = self.thresholds
        previous_phase = self.phase

        source_position = [float(v) for v in source_position] if source_position is not None else None
        destination_position = (
            [float(v) for v in destination_position] if destination_position is not None else None
        )
        gripper_position = [float(v) for v in gripper_position] if gripper_position is not None else None
        gripper_qpos = [float(v) for v in gripper_qpos] if gripper_qpos is not None else None

        if source_position is not None and self._initial_source_pos is None:
            self._initial_source_pos = source_position

        if source_position is None or gripper_position is None:
            # No usable simulator state this frame: never guess a relevant
            # entity, but do not corrupt the streak state either.
            result = TaskPhaseResult(
                timestep=timestep,
                phase=PHASE_UNCERTAIN,
                previous_phase=previous_phase,
                relevant_entity=None,
                relevant_entity_role="none",
                grasp_detected=False,
                grasp_confidence=0.0,
                evidence={"missing_position_data": True},
                transition_detected=previous_phase != PHASE_UNCERTAIN,
                transition_timestep=(
                    timestep if previous_phase != PHASE_UNCERTAIN else self._last_transition_timestep
                ),
                reason="source or gripper world position unavailable this timestep",
                source_to_gripper_distance=None,
                source_height_delta=None,
                source_displacement=None,
                source_eef_relative_position=None,
                source_eef_relative_motion=None,
                gripper_state="unknown",
                gripper_command=gripper_command,
                contact=None,
                source_position=source_position,
                destination_position=destination_position,
                gripper_position=gripper_position,
            )
            if previous_phase != PHASE_UNCERTAIN:
                self._last_transition_timestep = timestep
            self.phase = PHASE_UNCERTAIN
            self._prev_gripper_qpos = gripper_qpos
            return result

        distance = _euclidean(source_position, gripper_position)
        height_delta = source_position[2] - self._initial_source_pos[2]
        displacement = _euclidean(source_position, self._initial_source_pos)

        opening = _gripper_opening(gripper_qpos)
        prev_opening = _gripper_opening(self._prev_gripper_qpos)
        opening_delta = (
            opening - prev_opening if opening is not None and prev_opening is not None else None
        )

        if opening is None:
            gripper_state = "unknown"
        elif opening <= t.gripper_closed_qpos_sum:
            gripper_state = "closed"
        elif opening >= t.gripper_open_qpos_sum:
            gripper_state = "opening" if (opening_delta is not None and opening_delta > 0) else "open"
        elif opening_delta is not None and opening_delta <= t.gripper_closing_qpos_delta:
            gripper_state = "closing"
        else:
            gripper_state = "transitional"

        contact_bool = bool(contact) if contact is not None else False
        self._contact_streak = self._contact_streak + 1 if contact_bool else 0

        proximity_ok = distance is not None and distance <= t.grasp_distance_m
        contact_ok = self._contact_streak >= t.min_consecutive_contact_steps
        gripper_closed_ok = gripper_state in ("closed", "closing")
        lifted_ok = height_delta >= t.grasp_height_delta_m or displacement >= t.grasp_displacement_m

        # Relative offset between the source object and the end-effector. A
        # rigidly held object keeps this vector ~constant while both bodies move.
        relative = [s - g for s, g in zip(source_position, gripper_position)]
        relative_motion = _euclidean(relative, self._prev_relative)

        self._source_history.append(source_position)
        self._eef_history.append(gripper_position)
        self._relative_history.append(relative)
        for history in (self._source_history, self._eef_history, self._relative_history):
            if len(history) > t.distance_history_window:
                history.pop(0)

        # Comovement requires all three: close enough, BOTH bodies actually
        # travelled over the window, and the relative offset stayed rigid.
        # Dropping the "actually travelled" conditions would let a gripper
        # hovering over a resting object read as a grasp -- the relative offset
        # of two stationary bodies is trivially constant.
        window = t.min_consecutive_comovement_steps
        comovement_ok = False
        source_window_motion: Optional[float] = None
        eef_window_motion: Optional[float] = None
        relative_window_drift: Optional[float] = None
        if len(self._relative_history) >= window:
            source_window_motion = _euclidean(source_position, self._source_history[-window])
            eef_window_motion = _euclidean(gripper_position, self._eef_history[-window])
            relative_window_drift = max(
                _euclidean(relative, past) for past in self._relative_history[-window:]
            )
            comovement_ok = (
                proximity_ok
                and source_window_motion >= t.min_comovement_motion_m
                and eef_window_motion >= t.min_comovement_motion_m
                and relative_window_drift <= t.comovement_relative_drift_m
            )
        self._comovement_streak = self._comovement_streak + 1 if comovement_ok else 0

        grasp_confidence = 0.0
        if contact_ok:
            grasp_confidence += EVIDENCE_WEIGHTS["contact"]
        if proximity_ok:
            grasp_confidence += EVIDENCE_WEIGHTS["proximity"]
        if gripper_closed_ok:
            grasp_confidence += EVIDENCE_WEIGHTS["gripper_closed"]
        if lifted_ok:
            grasp_confidence += EVIDENCE_WEIGHTS["lifted_or_displaced"]
        if comovement_ok:
            grasp_confidence += EVIDENCE_WEIGHTS["comovement"]

        grasp_like_now = contact_ok and gripper_closed_ok and (lifted_ok or comovement_ok)
        self._grasp_evidence_streak = self._grasp_evidence_streak + 1 if grasp_like_now else 0
        grasp_detected = (
            self._grasp_evidence_streak >= t.min_consecutive_grasp_steps_for_transition
            and grasp_confidence >= t.grasp_confidence_threshold
        )

        release_like_now = (
            previous_phase == PHASE_POST_GRASP
            and not contact_ok
            and height_delta <= t.release_height_drop_m
        )
        self._release_evidence_streak = (
            self._release_evidence_streak + 1 if release_like_now else 0
        )
        release_detected = self._release_evidence_streak >= t.min_consecutive_release_steps_for_transition

        evidence: Dict[str, Any] = {
            "distance_m": distance,
            "height_delta_m": height_delta,
            "displacement_m": displacement,
            "gripper_opening": opening,
            "gripper_opening_delta": opening_delta,
            "gripper_command": gripper_command,
            "proximity_ok": proximity_ok,
            "contact_ok_sustained": contact_ok,
            "contact_streak": self._contact_streak,
            "gripper_closed_ok": gripper_closed_ok,
            "lifted_ok": lifted_ok,
            "comovement_ok": comovement_ok,
            "comovement_streak": self._comovement_streak,
            "source_window_motion_m": source_window_motion,
            "eef_window_motion_m": eef_window_motion,
            "relative_window_drift_m": relative_window_drift,
            "relative_motion_m": relative_motion,
            "grasp_evidence_streak": self._grasp_evidence_streak,
            "release_evidence_streak": self._release_evidence_streak,
            "grasp_like_now": grasp_like_now,
            "release_like_now": release_like_now,
        }

        reason: str
        if previous_phase in (PHASE_PRE_GRASP, PHASE_UNCERTAIN):
            if grasp_detected:
                new_phase = PHASE_POST_GRASP
                reason = (
                    f"sustained grasp evidence for {self._grasp_evidence_streak} steps "
                    f"(confidence={grasp_confidence:.2f} >= {t.grasp_confidence_threshold})"
                )
            elif contact_ok and not (lifted_ok or comovement_ok):
                new_phase = PHASE_UNCERTAIN
                reason = "sustained contact with source but no displacement/comovement evidence yet"
            elif proximity_ok and gripper_closed_ok and not contact_ok:
                new_phase = PHASE_UNCERTAIN
                reason = "gripper closed near source object without confirmed contact"
            else:
                new_phase = PHASE_PRE_GRASP
                reason = "no sustained grasp evidence: approaching or idle before contact"
        else:  # previous_phase == PHASE_POST_GRASP
            if release_detected:
                new_phase = PHASE_UNCERTAIN
                reason = (
                    f"contact lost and source height returned near initial for "
                    f"{self._release_evidence_streak} steps: grasp likely released"
                )
            else:
                new_phase = PHASE_POST_GRASP
                reason = "grasp evidence still consistent with holding the source object"

        role = {PHASE_PRE_GRASP: "source", PHASE_POST_GRASP: "destination"}.get(new_phase, "none")
        relevant_entity = {
            "source": self.source_entity,
            "destination": self.destination_entity,
        }.get(role)

        transition_detected = new_phase != previous_phase
        if transition_detected:
            self._last_transition_timestep = timestep

        result = TaskPhaseResult(
            timestep=timestep,
            phase=new_phase,
            previous_phase=previous_phase,
            relevant_entity=relevant_entity,
            relevant_entity_role=role if relevant_entity is not None else "none",
            grasp_detected=grasp_detected,
            grasp_confidence=grasp_confidence,
            evidence=evidence,
            transition_detected=transition_detected,
            transition_timestep=self._last_transition_timestep,
            reason=reason,
            source_to_gripper_distance=distance,
            source_height_delta=height_delta,
            source_displacement=displacement,
            source_eef_relative_position=relative,
            source_eef_relative_motion=relative_motion,
            gripper_state=gripper_state,
            gripper_command=gripper_command,
            contact=contact_bool if contact is not None else None,
            source_position=source_position,
            destination_position=destination_position,
            gripper_position=gripper_position,
        )

        self.phase = new_phase
        self._prev_gripper_qpos = gripper_qpos
        self._prev_relative = relative
        return result

    # Backwards-compatible alias: earlier callers on this branch used `step`.
    step = update


# -----------------------------------------------------------------------------
# Simulator-facing convenience: pull the generic quantities TaskPhaseResolver
# needs out of a live LIBERO/robosuite env, without hard-coding any object name.
# -----------------------------------------------------------------------------
def get_contact(env: Any, entity: Optional[str]) -> Optional[bool]:
    """MuJoCo contact between ``entity``'s geoms and the robot gripper's geoms.

    Returns ``None`` (never ``False``) when the query itself is not possible
    (missing object/robot registration), so callers can distinguish "no
    contact" from "couldn't check contact".
    """
    if entity is None:
        return None
    from spatial_task_resolver import unwrap_base_env  # local import: avoid cycle at module load

    base_env = unwrap_base_env(env)
    objects_dict = getattr(base_env, "objects_dict", None)
    robots = getattr(base_env, "robots", None)
    if not objects_dict or entity not in objects_dict or not robots:
        return None
    try:
        gripper = robots[0].gripper
        return bool(base_env.check_contact(objects_dict[entity], gripper))
    except Exception:
        return None


def compute_frame_inputs(
    env: Any,
    obs: Dict[str, Any],
    source_entity: Optional[str],
    destination_entity: Optional[str],
) -> Dict[str, Any]:
    """Generic per-timestep inputs for ``TaskPhaseResolver.step``.

    Positions come from ``spatial_task_resolver.get_entity_world_position``
    (the same source used to log ``entity_positions_initial`` / ``_final``),
    gripper pose/qpos come from the observation dict every LIBERO env already
    returns, and contact comes from robosuite's own ``MujocoEnv.check_contact``.
    Nothing here is specific to any particular object name.
    """
    from spatial_task_resolver import get_entity_world_position

    source_position = get_entity_world_position(env, source_entity) if source_entity else None
    destination_position = (
        get_entity_world_position(env, destination_entity) if destination_entity else None
    )
    gripper_position = obs.get("robot0_eef_pos")
    gripper_position = [float(v) for v in gripper_position] if gripper_position is not None else None
    gripper_qpos = obs.get("robot0_gripper_qpos")
    gripper_qpos = [float(v) for v in gripper_qpos] if gripper_qpos is not None else None

    return {
        "source_position": source_position,
        "destination_position": destination_position,
        "gripper_position": gripper_position,
        "gripper_qpos": gripper_qpos,
        "contact": get_contact(env, source_entity),
    }


# -----------------------------------------------------------------------------
# Shared per-episode artifact writer (phase_timeline.csv / .jsonl / transition
# summary). Every rollout script uses this so the file shape never diverges.
# -----------------------------------------------------------------------------
PHASE_TIMELINE_CSV_FIELDS = [
    "timestep",
    "phase",
    "previous_phase",
    "transition_detected",
    "transition_timestep",
    "relevant_entity",
    "relevant_entity_role",
    "grasp_detected",
    "grasp_confidence",
    "source_to_gripper_distance",
    "source_height_delta",
    "source_displacement",
    "source_eef_relative_motion",
    "gripper_state",
    "gripper_command",
    "contact",
    "phase_reason",
]

RELEVANT_ENTITY_TIMELINE_CSV_FIELDS = [
    "timestep",
    "phase",
    "relevant_entity",
    "relevant_entity_role",
    "relevant_target_valid",
    "grasp_confidence",
]


def phase_result_to_timeline_entry(result: "TaskPhaseResult") -> Dict[str, Any]:
    """``TaskPhaseResult.to_dict()`` with ``reason`` renamed to ``phase_reason``.

    The resolver's own output field is ``reason`` (section 1 of the research
    plan); the shared per-timestep pipeline record calls the same value
    ``phase_reason`` (section 3). This is the one place that mapping happens,
    so ``phase_timeline.csv`` / ``.jsonl`` and ``save_phase_timeline`` never
    have to guess which key is present.
    """
    entry = result.to_dict()
    entry["phase_reason"] = entry.pop("reason")
    return entry


def phase_record_to_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {key: entry.get(key) for key in PHASE_TIMELINE_CSV_FIELDS}


# Fields every runner mixes into its own per-timestep record (per_step_metrics
# .jsonl). Defined once so the three rollout scripts cannot drift apart.
PER_STEP_PHASE_FIELDS = [
    "phase",
    "relevant_entity",
    "relevant_entity_role",
    "grasp_detected",
    "grasp_confidence",
    "transition_detected",
    "transition_timestep",
    "source_to_gripper_distance",
    "source_height_delta",
    "source_displacement",
    "source_eef_relative_position",
    "source_eef_relative_motion",
    "gripper_state",
    "gripper_command",
    "contact",
    "source_position",
    "destination_position",
    "gripper_position",
]


def per_step_phase_fields(result: Optional["TaskPhaseResult"]) -> Dict[str, Any]:
    """Phase columns for one per-timestep record; all ``None`` when disabled."""
    if result is None:
        fields: Dict[str, Any] = {key: None for key in PER_STEP_PHASE_FIELDS}
        fields["phase_reason"] = None
        fields["phase_evidence"] = None
        return fields
    entry = result.to_dict()
    fields = {key: entry.get(key) for key in PER_STEP_PHASE_FIELDS}
    fields["phase_reason"] = entry.get("reason")
    fields["phase_evidence"] = entry.get("evidence")
    return fields


def save_phase_timeline(
    phase_timeline: List[Dict[str, Any]],
    output_dir: str,
    thresholds: PhaseThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Write every per-episode phase artifact and return the summary dict.

    Files written into ``output_dir``:
      * ``phase_timeline.csv``           -- one row per timestep, flat columns
      * ``phase_timeline.jsonl``         -- one full record per timestep
      * ``phase_transition_summary.json``-- transitions only
      * ``phase_summary.json``           -- counts, fractions, config snapshot
      * ``uncertain_timesteps.json``     -- uncertain timesteps and their reasons
      * ``relevant_entity_timeline.csv`` -- timestep -> selected relevant entity
    """
    import csv
    import json
    import os

    csv_path = os.path.join(output_dir, "phase_timeline.csv")
    jsonl_path = os.path.join(output_dir, "phase_timeline.jsonl")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE_TIMELINE_CSV_FIELDS)
        writer.writeheader()
        for entry in phase_timeline:
            writer.writerow(phase_record_to_row(entry))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in phase_timeline:
            f.write(json.dumps(entry) + "\n")

    relevant_csv_path = os.path.join(output_dir, "relevant_entity_timeline.csv")
    with open(relevant_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RELEVANT_ENTITY_TIMELINE_CSV_FIELDS)
        writer.writeheader()
        for entry in phase_timeline:
            writer.writerow(
                {
                    "timestep": entry.get("timestep"),
                    "phase": entry.get("phase"),
                    "relevant_entity": entry.get("relevant_entity"),
                    "relevant_entity_role": entry.get("relevant_entity_role"),
                    "relevant_target_valid": entry.get("relevant_entity") is not None,
                    "grasp_confidence": entry.get("grasp_confidence"),
                }
            )

    transitions = [
        {
            "timestep": entry["timestep"],
            "from_phase": entry["previous_phase"],
            "to_phase": entry["phase"],
            "reason": entry["phase_reason"],
            "grasp_confidence": entry.get("grasp_confidence"),
        }
        for entry in phase_timeline
        if entry["transition_detected"]
    ]
    uncertain_entries = [
        {
            "timestep": entry["timestep"],
            "reason": entry["phase_reason"],
            "grasp_confidence": entry.get("grasp_confidence"),
            "source_to_gripper_distance": entry.get("source_to_gripper_distance"),
            "contact": entry.get("contact"),
        }
        for entry in phase_timeline
        if entry["phase"] == PHASE_UNCERTAIN
    ]
    uncertain_timesteps = [entry["timestep"] for entry in uncertain_entries]
    phase_counts: Dict[str, int] = {}
    for entry in phase_timeline:
        phase_counts[entry["phase"]] = phase_counts.get(entry["phase"], 0) + 1

    pre_to_post = next(
        (
            t["timestep"]
            for t in transitions
            if t["to_phase"] == PHASE_POST_GRASP and t["from_phase"] != PHASE_POST_GRASP
        ),
        None,
    )

    total = len(phase_timeline)
    summary = {
        "total_timesteps": total,
        "phase_counts": phase_counts,
        "phase_fractions": {k: v / total for k, v in phase_counts.items()} if total else {},
        "uncertain_timesteps": uncertain_timesteps,
        "uncertain_count": len(uncertain_timesteps),
        "uncertain_fraction": (len(uncertain_timesteps) / total if total else None),
        "transitions": transitions,
        "transition_count": len(transitions),
        "first_pre_grasp_to_post_grasp_timestep": pre_to_post,
        "reached_post_grasp": PHASE_POST_GRASP in phase_counts,
        "final_phase": phase_timeline[-1]["phase"] if phase_timeline else None,
        "phase_timeline_csv_path": csv_path,
        "phase_timeline_jsonl_path": jsonl_path,
        "relevant_entity_timeline_csv_path": relevant_csv_path,
    }

    transition_summary_path = os.path.join(output_dir, "phase_transition_summary.json")
    with open(transition_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transitions": transitions,
                "transition_count": len(transitions),
                "first_pre_grasp_to_post_grasp_timestep": pre_to_post,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    uncertain_path = os.path.join(output_dir, "uncertain_timesteps.json")
    with open(uncertain_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "uncertain_count": len(uncertain_entries),
                "uncertain_fraction": summary["uncertain_fraction"],
                "uncertain_timesteps": uncertain_entries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_path = os.path.join(output_dir, "phase_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(summary, phase_resolver_config=config_snapshot(thresholds)),
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary["phase_transition_summary_path"] = transition_summary_path
    summary["uncertain_timesteps_path"] = uncertain_path
    summary["phase_summary_path"] = summary_path
    return summary


# -----------------------------------------------------------------------------
# Self-test: synthetic trajectories, no simulator required.
# -----------------------------------------------------------------------------
def _run_selftest() -> bool:
    """Synthetic trajectories covering the required phase scenarios.

    No simulator involved: positions/qpos/contact are handed in directly so the
    temporal logic can be exercised deterministically.
    """
    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            ok = False
        print(f"  [{status}] {name}")

    SRC0 = [0.0, 0.0, 0.9]
    DST = [0.3, 0.0, 0.9]
    OPEN_QPOS = [0.0208, -0.0208]
    CLOSED_QPOS = [0.0, 0.0]

    def new_resolver():
        return TaskPhaseResolver(source_entity="src", destination_entity="dst")

    # -- 1. Far from source, no grasp evidence -> pre_grasp, entity = source ---
    r1 = new_resolver()
    result = r1.update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    check("1. far from source with no evidence -> pre_grasp", result.phase == PHASE_PRE_GRASP)
    check("1. pre_grasp relevant_entity is source", result.relevant_entity == "src")
    check("1. pre_grasp role is 'source'", result.relevant_entity_role == "source")

    # -- 2. Gripper closed but far from source -> must NOT become post_grasp ---
    r2 = new_resolver()
    phases2 = []
    for i in range(12):
        # Gripper closed and moving, but 40cm away and never touching the source.
        eef = [0.4 + 0.01 * i, 0.0, 1.1]
        phases2.append(r2.update(i, SRC0, DST, eef, CLOSED_QPOS, contact=False).phase)
    check("2. closed gripper far from source never reaches post_grasp", PHASE_POST_GRASP not in phases2)

    # -- 3. One brief contact near source -> must NOT flip immediately ---------
    r3 = new_resolver()
    at_src = [0.0, 0.0, 0.92]
    r3.update(0, SRC0, DST, at_src, OPEN_QPOS, contact=False)
    result = r3.update(1, SRC0, DST, at_src, CLOSED_QPOS, contact=True)
    check("3. single-timestep contact does not flip to post_grasp", result.phase != PHASE_POST_GRASP)
    result = r3.update(2, SRC0, DST, at_src, OPEN_QPOS, contact=False)
    check("3. contact that immediately disappears leaves phase non-post_grasp", result.phase != PHASE_POST_GRASP)

    # -- 4. Contact + lift + comovement sustained -> post_grasp ----------------
    r4 = new_resolver()
    r4.update(0, SRC0, DST, [0.0, 0.0, 1.0], OPEN_QPOS, contact=False)
    result = None
    for i in range(1, 9):
        held = [SRC0[0], SRC0[1], SRC0[2] + 0.012 * i]  # object rises with the eef
        result = r4.update(i, held, DST, held, CLOSED_QPOS, contact=True, gripper_command=1.0)
    check("4. sustained contact+lift+comovement -> post_grasp", result.phase == PHASE_POST_GRASP)
    check("4. grasp_detected is True", result.grasp_detected is True)
    check("4. transition_timestep is recorded", result.transition_timestep is not None)

    # -- 5. uncertain -> relevant_entity is null ------------------------------
    r5 = new_resolver()
    result = r5.update(0, None, DST, [0.0, 0.0, 0.9], CLOSED_QPOS, contact=None)
    check("5. uncertain phase yields relevant_entity None", result.phase == PHASE_UNCERTAIN and result.relevant_entity is None)
    check("5. uncertain role is 'none'", result.relevant_entity_role == "none")

    # -- 6. pre_grasp -> relevant_entity is source (covered in 1, asserted again)
    check("6. pre_grasp maps to source entity", r1.update(1, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False).relevant_entity == "src")

    # -- 7. post_grasp -> relevant_entity is destination -----------------------
    check("7. post_grasp maps to destination entity", result is not None and r4.phase == PHASE_POST_GRASP)
    held_now = [SRC0[0], SRC0[1], SRC0[2] + 0.12]
    result = r4.update(20, held_now, DST, held_now, CLOSED_QPOS, contact=True)
    check("7. post_grasp relevant_entity is destination", result.relevant_entity == "dst")
    check("7. post_grasp role is 'destination'", result.relevant_entity_role == "destination")

    # -- 8. Transient sensor noise must not cause phase oscillation -----------
    r8 = new_resolver()
    r8.update(0, SRC0, DST, [0.0, 0.0, 1.0], OPEN_QPOS, contact=False)
    for i in range(1, 9):
        held = [SRC0[0], SRC0[1], SRC0[2] + 0.012 * i]
        r8.update(i, held, DST, held, CLOSED_QPOS, contact=True)
    assert r8.phase == PHASE_POST_GRASP, "scenario 8 precondition: must be holding"
    noisy_phases = []
    base_h = SRC0[2] + 0.012 * 8
    for i in range(9, 25):
        held = [SRC0[0], SRC0[1], base_h + 0.001 * (i - 8)]
        # Contact sensor drops out on alternating frames (classic MuJoCo flicker).
        flaky_contact = (i % 2 == 0)
        noisy_phases.append(r8.update(i, held, DST, held, CLOSED_QPOS, contact=flaky_contact).phase)
    check("8. flickering contact does not knock the phase out of post_grasp", set(noisy_phases) == {PHASE_POST_GRASP})

    # -- 9. Regression: hovering over a resting object is NOT a grasp ----------
    # Both bodies stationary => relative offset trivially constant. Only the
    # motion requirement in comovement_ok keeps this out of post_grasp.
    r9 = new_resolver()
    hover_phases = []
    hover_eef = [SRC0[0], SRC0[1], SRC0[2] + 0.05]
    for i in range(15):
        hover_phases.append(
            r9.update(i, SRC0, DST, hover_eef, CLOSED_QPOS, contact=True).phase
        )
    check("9. closed gripper hovering on a resting object never reads as post_grasp", PHASE_POST_GRASP not in hover_phases)

    # -- 10. reset() clears episode state -------------------------------------
    r10 = new_resolver()
    for i in range(9):
        held = [SRC0[0], SRC0[1], SRC0[2] + 0.012 * i]
        r10.update(i, held, DST, held, CLOSED_QPOS, contact=True)
    r10.reset()
    result = r10.update(0, SRC0, DST, [0.4, 0.0, 1.1], OPEN_QPOS, contact=False)
    check("10. reset() returns the resolver to pre_grasp with clean streaks", result.phase == PHASE_PRE_GRASP and result.transition_timestep is None)

    return ok


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="task_phase_resolver self-test (no simulator required).")
    parser.parse_args()

    print("Running task_phase_resolver synthetic self-test...")
    ok = _run_selftest()
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
