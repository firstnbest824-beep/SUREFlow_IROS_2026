"""Recompute phase labels for already-collected episodes, without touching them.

The phase resolver is a heuristic that has been corrected three times, and it will
probably be corrected again. Re-running a 10-hour GPU collection every time the
labelling rule changes would be absurd, so the collector stores every input the
resolver consumes -- entity world positions, end-effector position, gripper qpos,
gripper/source contact, and ``supported_by_other`` -- and this recomputes the
labels from them offline.

Two properties make that sound:

* it is a pure function of stored data; no simulator, no GPU, no policy
* it writes a **sidecar**, never the original. ``per_step_metrics.jsonl`` and
  ``manifest.json`` are opened read-only, and the recomputed labels land in
  ``phase_labels_<commit>.jsonl`` beside them, stamped with the git commit of the
  rule that produced them. Two rule versions can therefore coexist and be
  compared, and nothing that was already validated is invalidated.

Episodes collected before ``supported_by_other`` was recorded cannot be relabelled
this way -- the current release rule needs it. Those are reported as skipped
rather than silently relabelled with a missing input, and require the replay path
in ``validate_phase_against_simulator.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from task_phase_resolver import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    TaskPhaseResolver,
    phase_result_to_timeline_entry,
)


def rule_commit() -> str:
    """The commit of the rule doing the relabelling, recorded in every output."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def stored_support(record: Dict[str, Any]) -> Optional[bool]:
    """``supported_by_other`` as the collector recorded it, or None if absent."""
    if "supported_by_other" in record:
        return record["supported_by_other"]
    evidence = (record.get("phase_timeline_entry") or {}).get("evidence") or {}
    return evidence.get("supported_by_other")


def relabel_episode(episode_dir: Path, commit: str, dry_run: bool = False) -> Dict[str, Any]:
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (episode_dir / "per_step_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roles = manifest.get("entity_roles") or {}
    source, destination = roles.get("source"), roles.get("destination")

    if not records or source is None:
        return {"episode": str(episode_dir), "status": "skipped", "reason": "no records or no source"}

    missing_support = sum(1 for r in records if stored_support(r) is None)
    if missing_support == len(records):
        return {
            "episode": str(episode_dir), "status": "skipped",
            "reason": "supported_by_other not recorded; use the replay path instead",
        }

    resolver = TaskPhaseResolver(source, destination, thresholds=DEFAULT_THRESHOLDS)
    out_rows: List[Dict[str, Any]] = []
    for record in records:
        result = resolver.update(
            timestep=record["timestep"],
            source_position=record["entity_world_xyz"][source],
            destination_position=record["entity_world_xyz"][destination],
            gripper_position=record["eef_pos"],
            gripper_qpos=record["gripper_qpos"],
            contact=record["contact"],
            supported_by_other=stored_support(record),
        )
        out_rows.append({
            "timestep": record["timestep"],
            "phase": result.phase,
            "relevant_entity": result.relevant_entity,
            "relevant_entity_role": result.relevant_entity_role,
            "grasp_detected": result.grasp_detected,
            "grasp_confidence": result.grasp_confidence,
            "phase_timeline_entry": phase_result_to_timeline_entry(result),
            # Kept alongside so agreement can be filtered on without a second pass.
            "sim_check_grasp": record.get("sim_check_grasp"),
            "phase_original": record.get("phase"),
        })

    changed = sum(1 for r, o in zip(records, out_rows) if r.get("phase") != o["phase"])
    summary = {
        "episode": str(episode_dir),
        "status": "relabelled",
        "rule_commit": commit,
        "timesteps": len(out_rows),
        "changed": changed,
        "phases_before": dict(Counter(r.get("phase") for r in records)),
        "phases_after": dict(Counter(o["phase"] for o in out_rows)),
        "missing_support_steps": missing_support,
        "sidecar": str(episode_dir / f"phase_labels_{commit}.jsonl"),
    }

    if not dry_run:
        header = {
            "_rule_commit": commit,
            "_source": "recomputed offline from per_step_metrics.jsonl",
            "_thresholds": DEFAULT_THRESHOLDS.to_dict(),
            "_episode": str(episode_dir),
            "_timesteps": len(out_rows),
        }
        target = episode_dir / f"phase_labels_{commit}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, target)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would change without writing sidecars")
    parser.add_argument("--json_out")
    args = parser.parse_args()

    commit = rule_commit()
    print(f"relabelling with rule commit {commit}"
          f"{' (dry run)' if args.dry_run else ''}\n")

    results: List[Dict[str, Any]] = []
    for root in args.roots:
        for manifest_path in sorted(Path(root).rglob("manifest.json")):
            results.append(relabel_episode(manifest_path.parent, commit, args.dry_run))

    relabelled = [r for r in results if r["status"] == "relabelled"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"{'episode':56s} {'steps':>6s} {'changed':>8s}  before -> after")
    for r in relabelled:
        name = r["episode"].split("/")[-5:]
        before = "/".join(str(r["phases_before"].get(p, 0)) for p in ("pre_grasp", "post_grasp", "uncertain"))
        after = "/".join(str(r["phases_after"].get(p, 0)) for p in ("pre_grasp", "post_grasp", "uncertain"))
        print(f"{'/'.join(name)[:56]:56s} {r['timesteps']:6d} {r['changed']:8d}  {before} -> {after}")

    total_steps = sum(r["timesteps"] for r in relabelled)
    total_changed = sum(r["changed"] for r in relabelled)
    print(f"\n{len(relabelled)} episodes relabelled, {len(skipped)} skipped")
    if total_steps:
        print(f"{total_changed}/{total_steps} timesteps changed label "
              f"({total_changed / total_steps:.1%})")
    for r in skipped:
        print(f"  SKIP {r['episode']}: {r['reason']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"rule_commit": commit, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
