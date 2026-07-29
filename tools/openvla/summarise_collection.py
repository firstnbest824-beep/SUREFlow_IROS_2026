"""Cross-episode summary of a collection run.

Reads only manifests and per-step metrics -- never the activation tensors -- so it
runs in seconds over a run that takes hours to collect. Groups by
(suite, condition) and reports the things that decide whether a run is usable:

* success rate, and whether it separates baseline from perturbed
* the *measured* change class, and whether every episode in a condition agrees
* requested level vs measured displacement, kept apart
* phase coverage -- a condition with no post_grasp timesteps yields no
  destination-phase probe targets, which is a finding, not a bug
* segmentation label coverage and visibility breakdown
* per-episode integrity: contiguous timesteps and a closed sim-state chain

Nothing here interprets results. It reports what is in the files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_episode(episode_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = episode_dir / "manifest.json"
    metrics_path = episode_dir / "per_step_metrics.jsonl"
    if not manifest_path.is_file() or not metrics_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None

    phases = Counter(r.get("phase") for r in rows)
    chain_ok = all(
        rows[i].get("next_sim_state_sha") == rows[i + 1].get("sim_state_sha")
        for i in range(len(rows) - 1)
    )
    contiguous = [r.get("timestep") for r in rows] == list(range(len(rows)))

    source = (manifest.get("entity_roles") or {}).get("source")
    destination = (manifest.get("entity_roles") or {}).get("destination")
    visibility = Counter()
    for row in rows:
        for entity, per_camera in (row.get("segmentation") or {}).items():
            label = per_camera.get("agentview")
            if label and entity in (source, destination):
                visibility[f"{'src' if entity == source else 'dst'}:{label.get('visibility')}"] += 1

    return {
        "dir": str(episode_dir),
        "suite": manifest.get("suite"),
        "condition": manifest.get("condition"),
        "task_id": manifest.get("task_id"),
        "episode_index": manifest.get("episode_index"),
        "steps": len(rows),
        "success": bool(manifest.get("success")),
        "termination": manifest.get("termination_reason"),
        "change_class": manifest.get("change_class"),
        "is_clean": manifest.get("is_clean_condition"),
        "requested_level": manifest.get("requested_level"),
        "measured_translation_m": manifest.get("measured_translation_m"),
        "init_state_source": manifest.get("init_state_source"),
        "init_state_sha": manifest.get("init_state_sha256"),
        "segmentation_equivalence": (manifest.get("segmentation_equivalence") or {}).get("passed"),
        "phases": dict(phases),
        "chain_ok": chain_ok,
        "contiguous": contiguous,
        "violations": len(manifest.get("violations") or []),
        "visibility": dict(visibility),
        "source": source,
        "destination": destination,
    }


def summarise(root: Path) -> Dict[str, Any]:
    episodes = [
        e for e in (load_episode(p.parent) for p in sorted(root.rglob("manifest.json")))
        if e is not None
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[(episode["suite"], episode["condition"])].append(episode)

    summary = []
    for (suite, condition), items in sorted(groups.items()):
        classes = Counter(e["change_class"] for e in items)
        measured = [e["measured_translation_m"] for e in items if e["measured_translation_m"] is not None]
        phases = Counter()
        for episode in items:
            phases.update(episode["phases"])
        total_steps = sum(e["steps"] for e in items)
        summary.append({
            "suite": suite,
            "condition": condition,
            "episodes": len(items),
            "tasks": sorted({e["task_id"] for e in items}),
            "success_rate": sum(e["success"] for e in items) / len(items),
            "successes": sum(e["success"] for e in items),
            "mean_steps": statistics.mean(e["steps"] for e in items),
            "change_classes": dict(classes),
            "change_class_consistent": len(classes) == 1,
            "is_clean": all(e["is_clean"] for e in items) if len(classes) == 1 else False,
            "requested_levels": sorted({e["requested_level"] for e in items if e["requested_level"] is not None}),
            "measured_translation_m": {
                "min": min(measured), "max": max(measured),
                "mean": statistics.mean(measured),
            } if measured else None,
            "init_state_sources": sorted({e["init_state_source"] for e in items}),
            "distinct_init_states": len({e["init_state_sha"] for e in items}),
            "phase_steps": dict(phases),
            "post_grasp_fraction": phases.get("post_grasp", 0) / max(1, total_steps),
            "segmentation_equivalence_all_passed": all(
                e["segmentation_equivalence"] is True for e in items
            ),
            "all_chains_closed": all(e["chain_ok"] for e in items),
            "all_contiguous": all(e["contiguous"] for e in items),
            "total_violations": sum(e["violations"] for e in items),
        })

    return {"root": str(root), "num_episodes": len(episodes), "groups": summary, "episodes": episodes}


def format_summary(report: Dict[str, Any]) -> str:
    lines = [f"{report['root']}  ({report['num_episodes']} episodes)", ""]
    header = (
        f"{'suite':15s} {'cond':8s} {'eps':>3s} {'succ':>6s} {'steps':>6s} "
        f"{'change_class':26s} {'clean':>5s} {'lvl':>4s} {'measured_m':>10s} "
        f"{'pre/post/unc':>18s} {'inits':>5s} {'ok'}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for group in report["groups"]:
        phases = group["phase_steps"]
        measured = group["measured_translation_m"]
        measured_text = "-" if not measured else (
            f"{measured['mean']:.4f}" if abs(measured["max"] - measured["min"]) < 1e-6
            else f"{measured['min']:.3f}-{measured['max']:.3f}"
        )
        classes = group["change_classes"]
        class_text = next(iter(classes)) if group["change_class_consistent"] else f"MIXED{list(classes)}"
        integrity = "OK" if (
            group["all_chains_closed"] and group["all_contiguous"]
            and group["segmentation_equivalence_all_passed"] and group["total_violations"] == 0
        ) else "CHECK"
        lines.append(
            f"{group['suite']:15s} {group['condition']:8s} {group['episodes']:3d} "
            f"{group['successes']:2d}/{group['episodes']:<3d} {group['mean_steps']:6.0f} "
            f"{class_text:26s} {('yes' if group['is_clean'] else 'no'):>5s} "
            f"{str(group['requested_levels'][0] if group['requested_levels'] else '-'):>4s} "
            f"{measured_text:>10s} "
            f"{phases.get('pre_grasp', 0):5d}/{phases.get('post_grasp', 0):5d}/{phases.get('uncertain', 0):5d} "
            f"{group['distinct_init_states']:5d} {integrity}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise a collection run.")
    parser.add_argument("root")
    parser.add_argument("--json_out")
    args = parser.parse_args()

    report = summarise(Path(args.root))
    print(format_summary(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
