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
    comovement_distance_span_m: float = 0.015
    release_height_drop_m: float = 0.015

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
    reason: str

    # Convenience fields also required verbatim by the per-timestep pipeline
    # output (section 3 of the research plan).
    source_to_gripper_distance: Optional[float]
    source_height_delta: Optional[float]
    source_displacement: Optional[float]
    gripper_state: str
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
    ``step`` calls so the classifier can require sustained evidence instead of
    reacting to a single noisy frame.
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

        self.phase: str = PHASE_PRE_GRASP
        self._initial_source_pos: Optional[List[float]] = None
        self._prev_gripper_qpos: Optional[List[float]] = None
        self._distance_history: List[float] = []

        self._contact_streak = 0
        self._grasp_evidence_streak = 0
        self._release_evidence_streak = 0
        self._comovement_streak = 0

    def step(
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
                reason="source or gripper world position unavailable this timestep",
                source_to_gripper_distance=None,
                source_height_delta=None,
                source_displacement=None,
                gripper_state="unknown",
                contact=None,
                source_position=source_position,
                destination_position=destination_position,
                gripper_position=gripper_position,
            )
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

        self._distance_history.append(distance)
        if len(self._distance_history) > t.distance_history_window:
            self._distance_history.pop(0)
        comovement_ok = False
        if len(self._distance_history) >= t.min_consecutive_comovement_steps:
            recent = self._distance_history[-t.min_consecutive_comovement_steps :]
            comovement_ok = proximity_ok and (max(recent) - min(recent)) <= t.comovement_distance_span_m
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

        result = TaskPhaseResult(
            timestep=timestep,
            phase=new_phase,
            previous_phase=previous_phase,
            relevant_entity=relevant_entity,
            relevant_entity_role=role if relevant_entity is not None else "none",
            grasp_detected=grasp_detected,
            grasp_confidence=grasp_confidence,
            evidence=evidence,
            transition_detected=new_phase != previous_phase,
            reason=reason,
            source_to_gripper_distance=distance,
            source_height_delta=height_delta,
            source_displacement=displacement,
            gripper_state=gripper_state,
            contact=contact_bool if contact is not None else None,
            source_position=source_position,
            destination_position=destination_position,
            gripper_position=gripper_position,
        )

        self.phase = new_phase
        self._prev_gripper_qpos = gripper_qpos
        return result


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
    "relevant_entity",
    "relevant_entity_role",
    "grasp_detected",
    "grasp_confidence",
    "source_to_gripper_distance",
    "source_height_delta",
    "source_displacement",
    "gripper_state",
    "contact",
    "phase_reason",
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


def save_phase_timeline(phase_timeline: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """Write phase_timeline.csv / .jsonl and return a transition/uncertainty summary."""
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

    transitions = [
        {
            "timestep": entry["timestep"],
            "from_phase": entry["previous_phase"],
            "to_phase": entry["phase"],
            "reason": entry["phase_reason"],
        }
        for entry in phase_timeline
        if entry["transition_detected"]
    ]
    uncertain_timesteps = [entry["timestep"] for entry in phase_timeline if entry["phase"] == PHASE_UNCERTAIN]
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

    summary = {
        "total_timesteps": len(phase_timeline),
        "phase_counts": phase_counts,
        "uncertain_timesteps": uncertain_timesteps,
        "uncertain_fraction": (
            len(uncertain_timesteps) / len(phase_timeline) if phase_timeline else None
        ),
        "transitions": transitions,
        "first_pre_grasp_to_post_grasp_timestep": pre_to_post,
        "phase_timeline_csv_path": csv_path,
        "phase_timeline_jsonl_path": jsonl_path,
    }
    summary_path = os.path.join(output_dir, "phase_transition_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["phase_transition_summary_path"] = summary_path
    return summary


# -----------------------------------------------------------------------------
# Self-test: synthetic trajectories, no simulator required.
# -----------------------------------------------------------------------------
def _run_selftest() -> bool:
    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            ok = False
        print(f"  [{status}] {name}")

    # --- Scenario 1: approach, sustained grasp, then post_grasp holds. ---
    resolver = TaskPhaseResolver(source_entity="src", destination_entity="dst")
    src0 = [0.0, 0.0, 0.9]
    dst = [0.3, 0.0, 0.9]
    gripper_far = [0.4, 0.0, 1.1]
    r = resolver.step(0, src0, dst, gripper_far, [0.0208, -0.0208], contact=False)
    check("t0 starts pre_grasp", r.phase == PHASE_PRE_GRASP and r.relevant_entity == "src")

    # Approach: gripper closes in on source, not yet touching.
    for i in range(1, 4):
        gripper_near = [0.02 * i, 0.0, 0.95]
        r = resolver.step(i, src0, dst, gripper_near, [0.0208, -0.0208], contact=False)
    check("still pre_grasp while approaching without contact", r.phase == PHASE_PRE_GRASP)

    # Contact + closing gripper + lift sustained for several steps -> post_grasp.
    lifted = list(src0)
    for i in range(4, 10):
        lifted = [src0[0], src0[1], src0[2] + 0.01 * (i - 3)]
        gripper_at_src = lifted
        r = resolver.step(
            i, lifted, dst, gripper_at_src, [0.0, 0.0], contact=True, gripper_command=1.0
        )
    check("transitions to post_grasp after sustained grasp evidence", r.phase == PHASE_POST_GRASP)
    check("relevant_entity is destination in post_grasp", r.relevant_entity == "dst")
    check("grasp_detected true", r.grasp_detected is True)

    # Holding steady near destination: stays post_grasp.
    for i in range(10, 13):
        r = resolver.step(i, lifted, dst, lifted, [0.0, 0.0], contact=True)
    check("remains post_grasp while still held", r.phase == PHASE_POST_GRASP)

    # Drop: contact lost, height falls back near initial. Per spec this may
    # resolve to "uncertain" or self-heal back to "pre_grasp" once the gripper
    # is clearly away from the (now stationary) source -- either is acceptable,
    # but it must NOT keep reporting post_grasp/destination once contact and
    # lift evidence are gone.
    dropped = [src0[0], src0[1], src0[2] + 0.001]
    drop_phases = []
    for i in range(13, 20):
        r = resolver.step(i, dropped, dst, [dropped[0], dropped[1], dropped[2] + 0.1], [0.02, -0.02], contact=False)
        drop_phases.append(r.phase)
    check(
        "post_grasp does not persist once contact and lift evidence disappear",
        drop_phases[-1] != PHASE_POST_GRASP,
    )
    check("phase passes through uncertain during the drop (never forced straight to a guess)", PHASE_UNCERTAIN in drop_phases)
    check("final relevant_entity is never destination once released", r.relevant_entity != "dst")

    # --- Scenario 2: single-frame contact must NOT flip the phase. ---
    resolver2 = TaskPhaseResolver(source_entity="src", destination_entity="dst")
    r = resolver2.step(0, [0, 0, 0.9], dst, [0, 0, 0.9], [0.0, 0.0], contact=True)
    check("single-frame contact alone stays pre_grasp/uncertain, never post_grasp", r.phase != PHASE_POST_GRASP)

    # --- Scenario 3: missing position data never guesses a relevant entity. ---
    resolver3 = TaskPhaseResolver(source_entity="src", destination_entity="dst")
    r = resolver3.step(0, None, dst, [0, 0, 0.9], [0.0, 0.0], contact=None)
    check("missing source position -> uncertain with no relevant entity", r.phase == PHASE_UNCERTAIN and r.relevant_entity is None)

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
