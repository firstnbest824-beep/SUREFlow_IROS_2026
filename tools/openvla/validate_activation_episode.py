"""Integrity validation for one collected episode (schema v2).

Every check is falsifiable and names the field it read. A validator that says
"looks fine" without being able to fail is worthless, so each check below is
written so that a plausible bug in the collector would trip it:

===  ==========================================================================
 A   manifest present, parses, schema_version == 2
 B   COMPLETE marker present, no leftover .partial
 C   counts agree: manifest.num_timesteps == JSONL lines == activation files,
     and timesteps are contiguous 0..N-1
 D   every required stage present in every .npz, with consistent shapes
 E   no NaN / Inf; flags implausible magnitudes on the p99.9 (not the max,
     because LLaMA genuinely carries a few ~1e4 massive-activation channels)
 F   timestep synchronisation: the declared contract is
     obs_t -> label_t -> activation_t -> action_t -> env.step(action_t), and the
     recorded state-hash chain must actually close:
     record[t].next_sim_state_sha == record[t+1].sim_state_sha
     NECESSARY BUT NOT SUFFICIENT. Both hashes come from the two get_state()
     calls bracketing env.step; neither touches obs, the hook tensors, eef_pos or
     the labels. Deliberately shifting action_applied and eef_pos by +1 inside a
     record file leaves the chain 100% closed. What F does prove is that
     consecutive records are temporally contiguous and that no env.step was
     skipped, duplicated or interleaved. Proving the *content* alignment needs a
     replay -- see validate_phase_against_simulator.py, which reproduces the
     recorded eef_pos exactly (0.0 m) from a pre-step read while either
     off-by-one pairing lands 4.7e-3 m away.
 G   BDDL / init-state / checkpoint hashes recorded, and equal to expectations
 H   source / destination / changed_entities / change_class recorded, and the
     change_class is one the detector can actually emit
 I   phase labels valid and consistent with the auto-assigned relevant entity
 J   segmentation labels well-formed: uv inside [0, 1] when in_frame, visibility
     drawn from the enum, mask counts non-negative
===  ==========================================================================

Exit code is 0 only when no check FAILs. WARN does not fail the run but is
always printed, because a silent warning is how a confound reaches a paper.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from activation_episode_writer import (  # noqa: E402
    ACTION_REFERENCE,
    ACTIVATION_REFERENCE,
    COMPLETE_MARKER,
    LABEL_REFERENCE,
    MANIFEST_NAME,
    METRICS_NAME,
    REQUIRED_STAGES,
    SCHEMA_VERSION,
)
from changed_entity_detector import CHANGE_CLASSES  # noqa: E402

VALID_PHASES = ("pre_grasp", "post_grasp", "uncertain")
VALID_VISIBILITY = ("visible", "occluded", "out_of_view", "unknown")

#: A visible entity's projected origin must land near its own mask. Tolerance is
#: generous because a partly-occluded object's visible fragment is legitimately
#: offset from its centre -- this is a frame check, not a precision check.
UV_MASK_TOLERANCE = 0.05
#: Fraction of visible labels allowed to miss before the frame is judged wrong.
UV_MASK_MAX_MISS_RATE = 0.25

#: Phase -> the entity the probe must be scored against at that phase.
PHASE_TO_ROLE = {"pre_grasp": "source", "post_grasp": "destination", "uncertain": None}

#: p99.9 above this is implausible even allowing for LLaMA massive activations.
BULK_EXTREME_ABS_VALUE = 1e3
#: Any single element above this is a hard failure regardless of distribution.
ABSOLUTE_EXTREME_VALUE = 1e6

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    check: str
    status: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "status": self.status, "message": self.message, **({"detail": self.detail} if self.detail else {})}


class EpisodeValidator:
    def __init__(
        self,
        episode_dir: str | os.PathLike,
        expected_bddl_sha: Optional[str] = None,
        expected_init_state_sha: Optional[str] = None,
        expected_checkpoint: Optional[str] = None,
        activation_sample_stride: int = 1,
    ) -> None:
        self.dir = Path(episode_dir)
        self.expected_bddl_sha = expected_bddl_sha
        self.expected_init_state_sha = expected_init_state_sha
        self.expected_checkpoint = expected_checkpoint
        self.stride = max(1, int(activation_sample_stride))
        self.results: List[CheckResult] = []
        self.manifest: Dict[str, Any] = {}
        self.records: List[Dict[str, Any]] = []

    # -- helpers --------------------------------------------------------------
    def _add(self, check: str, status: str, message: str, **detail: Any) -> None:
        self.results.append(CheckResult(check, status, message, detail))

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == WARN]

    # -- checks ---------------------------------------------------------------
    def check_a_manifest(self) -> bool:
        path = self.dir / MANIFEST_NAME
        if not path.is_file():
            self._add("A_manifest", FAIL, f"missing {MANIFEST_NAME}")
            return False
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._add("A_manifest", FAIL, f"manifest does not parse: {exc}")
            return False
        version = self.manifest.get("schema_version")
        if version != SCHEMA_VERSION:
            self._add(
                "A_manifest", FAIL,
                f"schema_version {version!r} != {SCHEMA_VERSION}; refusing to mix schemas",
            )
            return False
        self._add("A_manifest", PASS, f"schema v{version}")
        return True

    def check_b_complete(self) -> None:
        if (self.dir / COMPLETE_MARKER).is_file():
            self._add("B_complete", PASS, "COMPLETE marker present")
        else:
            self._add("B_complete", FAIL, "COMPLETE marker missing: episode was interrupted")
        stray = self.dir.with_name(self.dir.name + ".partial")
        if stray.exists():
            self._add("B_complete", WARN, f"leftover partial directory: {stray}")

    def check_c_counts(self) -> None:
        metrics_path = self.dir / METRICS_NAME
        if not metrics_path.is_file():
            self._add("C_counts", FAIL, f"missing {METRICS_NAME}")
            return
        self.records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        activation_files = sorted((self.dir / "activations").glob("step_*.npz"))
        declared = self.manifest.get("num_timesteps")

        if not (len(self.records) == len(activation_files) == declared):
            self._add(
                "C_counts", FAIL,
                "count mismatch between manifest, metrics and activation files",
                manifest_num_timesteps=declared,
                metrics_lines=len(self.records),
                activation_files=len(activation_files),
            )
            return

        timesteps = [r.get("timestep") for r in self.records]
        if timesteps != list(range(len(timesteps))):
            self._add("C_counts", FAIL, "timesteps are not contiguous 0..N-1", first_bad=_first_gap(timesteps))
            return
        self._add("C_counts", PASS, f"{len(self.records)} timesteps, contiguous, all artefacts present")

    def check_d_e_activations(self) -> None:
        activation_files = sorted((self.dir / "activations").glob("step_*.npz"))
        if not activation_files:
            self._add("D_stages", FAIL, "no activation files")
            return

        shapes: Dict[str, Tuple[int, ...]] = {}
        missing_stage_steps: Dict[str, List[int]] = {}
        nan_steps: List[str] = []
        bulk_extreme: List[str] = []
        hard_extreme: List[str] = []
        inconsistent: List[str] = []

        for index, path in enumerate(activation_files):
            if index % self.stride:
                continue
            with np.load(path) as bundle:
                present = set(bundle.files)
                for stage in REQUIRED_STAGES:
                    if stage not in present:
                        missing_stage_steps.setdefault(stage, []).append(index)
                        continue
                    array = bundle[stage]
                    if not np.isfinite(array).all():
                        nan_steps.append(f"{path.name}:{stage}")
                    else:
                        finite = array.astype(np.float32)
                        if np.abs(finite).max() > ABSOLUTE_EXTREME_VALUE:
                            hard_extreme.append(f"{path.name}:{stage}")
                        elif float(np.percentile(np.abs(finite), 99.9)) > BULK_EXTREME_ABS_VALUE:
                            bulk_extreme.append(f"{path.name}:{stage}")
                    key = array.shape[:1] + array.shape[2:]
                    if stage in shapes and shapes[stage] != key:
                        inconsistent.append(f"{path.name}:{stage} {array.shape} vs {shapes[stage]}")
                    shapes.setdefault(stage, key)

        if missing_stage_steps:
            self._add("D_stages", FAIL, "required stages missing", missing=missing_stage_steps)
        elif inconsistent:
            self._add("D_stages", FAIL, "stage shape changed mid-episode", examples=inconsistent[:5])
        else:
            self._add(
                "D_stages", PASS,
                f"all {len(REQUIRED_STAGES)} stages present with stable shapes",
                stages=sorted(shapes),
            )

        if nan_steps:
            self._add("E_finite", FAIL, "NaN or Inf in stored activations", examples=nan_steps[:8], count=len(nan_steps))
        elif hard_extreme:
            self._add("E_finite", FAIL, f"values above {ABSOLUTE_EXTREME_VALUE:g}", examples=hard_extreme[:8])
        elif bulk_extreme:
            self._add(
                "E_finite", WARN,
                f"p99.9 above {BULK_EXTREME_ABS_VALUE:g} (expected only for LLaMA massive-activation channels)",
                examples=bulk_extreme[:8],
            )
        else:
            self._add("E_finite", PASS, "all activations finite and within plausible magnitude")

    def check_f_timestep_sync(self) -> None:
        if not self.records:
            self._add("F_sync", FAIL, "no records to check")
            return

        bad_contract = [
            r["timestep"] for r in self.records
            if r.get("label_reference") != LABEL_REFERENCE
            or r.get("activation_reference") != ACTIVATION_REFERENCE
            or r.get("action_reference") != ACTION_REFERENCE
        ]
        if bad_contract:
            self._add(
                "F_sync", FAIL,
                "records do not declare the obs_t -> label_t -> activation_t -> action_t contract",
                timesteps=bad_contract[:10],
            )
            return

        # The chain check: the state we said we would step into must be the state
        # the next record says it observed. This catches an off-by-one that the
        # declared contract alone cannot.
        broken: List[Dict[str, Any]] = []
        chained = 0
        for current, following in zip(self.records, self.records[1:]):
            expected = current.get("next_sim_state_sha")
            observed = following.get("sim_state_sha")
            if expected is None or observed is None:
                continue
            chained += 1
            if expected != observed:
                broken.append({
                    "t": current["timestep"],
                    "expected_next": expected[:12],
                    "observed": observed[:12],
                })

        if broken:
            self._add("F_sync", FAIL, "simulator state chain is broken: labels and activations are misaligned", breaks=broken[:5], num_breaks=len(broken))
        elif chained == 0:
            self._add("F_sync", WARN, "no sim_state_sha chain recorded; only the declared contract was checked")
        else:
            self._add("F_sync", PASS, f"state chain closes across {chained} consecutive timesteps")

    def check_g_hashes(self) -> None:
        pairs = [
            ("bddl_sha256", self.expected_bddl_sha, "BDDL"),
            ("init_state_sha256", self.expected_init_state_sha, "init state"),
        ]
        for key, expected, label in pairs:
            actual = self.manifest.get(key)
            if not actual:
                self._add("G_hashes", FAIL, f"{label} hash not recorded ({key})")
            elif expected and actual != expected:
                self._add("G_hashes", FAIL, f"{label} hash mismatch", expected=expected, actual=actual)
            else:
                self._add("G_hashes", PASS, f"{label} hash recorded{' and matches' if expected else ''}")

        checkpoint = self.manifest.get("checkpoint") or {}
        revision = checkpoint.get("revision")
        if not revision:
            self._add("G_hashes", FAIL, "checkpoint revision not recorded")
        elif self.expected_checkpoint and revision != self.expected_checkpoint:
            self._add("G_hashes", FAIL, "checkpoint revision mismatch", expected=self.expected_checkpoint, actual=revision)
        else:
            self._add("G_hashes", PASS, f"checkpoint pinned at {revision[:12]}")

    def check_h_entities(self) -> None:
        roles = self.manifest.get("entity_roles") or {}
        change = self.manifest.get("change_report") or {}
        problems: List[str] = []

        if not roles.get("source"):
            problems.append("entity_roles.source missing")
        if not roles.get("destination"):
            problems.append("entity_roles.destination missing")

        change_class = change.get("change_class")
        if change_class is None:
            problems.append("change_report.change_class missing")
        elif change_class not in CHANGE_CLASSES:
            problems.append(f"change_class {change_class!r} is not one the detector emits")
        if "changed_entity_names" not in change and "changed_entities" not in change:
            problems.append("change_report has no changed-entity list")

        condition = self.manifest.get("condition")
        if condition and condition != "vanilla" and change_class == "no_detected_change":
            problems.append(
                f"condition {condition!r} but no entity moved: the perturbation did not apply"
            )

        if problems:
            self._add("H_entities", FAIL, "entity/change metadata incomplete", problems=problems)
        else:
            self._add(
                "H_entities", PASS,
                f"source={roles['source']} destination={roles['destination']} change_class={change_class}",
            )

        # Level vs measured displacement must be stored separately: "x0.1" is not 0.1 m.
        if self.manifest.get("requested_level") is not None and self.manifest.get(
            "measured_translation_m"
        ) is None:
            self._add("H_entities", FAIL, "requested_level recorded without measured_translation_m")

    def check_i_phase(self) -> None:
        if not self.records:
            return
        bad_phase: List[int] = []
        bad_entity: List[Dict[str, Any]] = []
        roles = self.manifest.get("entity_roles") or {}

        for record in self.records:
            phase = record.get("phase")
            if phase not in VALID_PHASES:
                bad_phase.append(record.get("timestep"))
                continue
            expected_role = PHASE_TO_ROLE[phase]
            relevant = record.get("relevant_entity")
            expected_entity = roles.get(expected_role) if expected_role else None
            if relevant != expected_entity:
                bad_entity.append({"t": record.get("timestep"), "phase": phase, "relevant_entity": relevant, "expected": expected_entity})

        if bad_phase:
            self._add("I_phase", FAIL, "invalid phase labels", timesteps=bad_phase[:10])
        elif bad_entity:
            self._add("I_phase", FAIL, "relevant_entity does not follow from phase", examples=bad_entity[:5], count=len(bad_entity))
        else:
            counts = {p: sum(1 for r in self.records if r.get("phase") == p) for p in VALID_PHASES}
            self._add("I_phase", PASS, f"phase labels valid and role-consistent {counts}")

    def check_j_segmentation(self) -> None:
        if not self.records:
            return
        problems: List[Dict[str, Any]] = []
        labelled = 0
        checked: Dict[str, int] = {}
        missed: Dict[str, int] = {}

        for record in self.records:
            segmentation = record.get("segmentation") or {}
            for entity, per_camera in segmentation.items():
                for camera, label in per_camera.items():
                    labelled += 1
                    visibility = label.get("visibility")
                    if visibility not in VALID_VISIBILITY:
                        problems.append({"t": record["timestep"], "entity": entity, "camera": camera, "why": f"visibility {visibility!r}"})
                    count = label.get("mask_pixel_count")
                    if count is None or count < 0:
                        problems.append({"t": record["timestep"], "entity": entity, "camera": camera, "why": f"mask_pixel_count {count!r}"})
                    uv = label.get("uv")
                    if label.get("in_frame") and uv is not None:
                        if not all(-0.001 <= v <= 1.001 for v in uv):
                            problems.append({"t": record["timestep"], "entity": entity, "camera": camera, "why": f"uv {uv} outside [0,1] while in_frame"})
                    if visibility == "visible" and (count or 0) <= 0:
                        problems.append({"t": record["timestep"], "entity": entity, "camera": camera, "why": "visible with zero mask pixels"})
                    # The one invariant that actually tests the frame convention:
                    # a visible entity's projected origin and its own mask must
                    # live in the same coordinate system. Range-checking uv alone
                    # cannot fail -- robosuite clips into the image before we
                    # normalise -- which is how a whole camera's projections
                    # being frozen at t=0 passed validation.
                    bbox = label.get("mask_bbox")
                    if visibility == "visible" and uv is not None and bbox:
                        checked[camera] = checked.get(camera, 0) + 1
                        u_min, v_min, u_max, v_max = bbox
                        if not (u_min - UV_MASK_TOLERANCE <= uv[0] <= u_max + UV_MASK_TOLERANCE
                                and v_min - UV_MASK_TOLERANCE <= uv[1] <= v_max + UV_MASK_TOLERANCE):
                            missed[camera] = missed.get(camera, 0) + 1

        if not labelled:
            self._add("J_segmentation", WARN, "no segmentation labels recorded")
        elif problems:
            self._add("J_segmentation", FAIL, "malformed segmentation labels", examples=problems[:5], count=len(problems))
        else:
            self._add("J_segmentation", PASS, f"{labelled} entity-camera labels well-formed")

        bad_frames = {
            camera: {"checked": total, "missed": missed.get(camera, 0),
                     "miss_rate": round(missed.get(camera, 0) / total, 3)}
            for camera, total in checked.items()
            if total >= 20 and missed.get(camera, 0) / total > UV_MASK_MAX_MISS_RATE
        }
        if bad_frames:
            self._add(
                "J_uv_frame", FAIL,
                "projected uv and its own mask do not share a frame",
                cameras=bad_frames, tolerance=UV_MASK_TOLERANCE,
            )
        elif checked:
            rates = {c: round(missed.get(c, 0) / t, 3) for c, t in checked.items()}
            self._add("J_uv_frame", PASS, f"uv lands inside its own mask {rates}")

    # -- driver ---------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        if self.check_a_manifest():
            self.check_b_complete()
            self.check_c_counts()
            self.check_d_e_activations()
            self.check_f_timestep_sync()
            self.check_g_hashes()
            self.check_h_entities()
            self.check_i_phase()
            self.check_j_segmentation()

        report = {
            "episode_dir": str(self.dir),
            "passed": not self.failed,
            "num_failed": len(self.failed),
            "num_warned": len(self.warned),
            "checks": [r.to_dict() for r in self.results],
        }
        return report


def _first_gap(timesteps: Sequence[Any]) -> Optional[int]:
    for index, value in enumerate(timesteps):
        if value != index:
            return index
    return None


def validate_episode(episode_dir: str | os.PathLike, **kwargs: Any) -> Dict[str, Any]:
    return EpisodeValidator(episode_dir, **kwargs).run()


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"episode: {report['episode_dir']}"]
    for check in report["checks"]:
        marker = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}[check["status"]]
        lines.append(f"  [{marker}] {check['check']:18s} {check['message']}")
        if check["status"] != PASS and check.get("detail"):
            lines.append(f"         {json.dumps(check['detail'], ensure_ascii=False)[:400]}")
    verdict = "PASS" if report["passed"] else "FAIL"
    lines.append(f"  => {verdict} ({report['num_failed']} failed, {report['num_warned']} warned)")
    return "\n".join(lines)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate collected activation episodes.")
    parser.add_argument("episode_dir", nargs="+")
    parser.add_argument("--recursive", action="store_true", help="treat paths as roots and validate every episode under them")
    parser.add_argument("--expected_bddl_sha")
    parser.add_argument("--expected_init_state_sha")
    parser.add_argument("--expected_checkpoint")
    parser.add_argument("--stride", type=int, default=1, help="check every Nth activation file")
    parser.add_argument("--json_out")
    args = parser.parse_args()

    targets: List[Path] = []
    for raw in args.episode_dir:
        path = Path(raw)
        if args.recursive:
            targets.extend(sorted(p.parent for p in path.rglob(MANIFEST_NAME)))
        else:
            targets.append(path)

    reports = []
    for target in targets:
        report = validate_episode(
            target,
            expected_bddl_sha=args.expected_bddl_sha,
            expected_init_state_sha=args.expected_init_state_sha,
            expected_checkpoint=args.expected_checkpoint,
            activation_sample_stride=args.stride,
        )
        reports.append(report)
        print(format_report(report))
        try:
            (target / "validation_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    failed = [r for r in reports if not r["passed"]]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} episodes passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
