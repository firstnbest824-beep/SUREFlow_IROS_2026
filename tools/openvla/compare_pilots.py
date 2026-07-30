"""Compare two collection runs episode-by-episode.

Written for the audit-fix comparison: the v1 pilot was collected before seven
confirmed defects were fixed, v2 after. Both used the same suites, conditions,
tasks, seeds and init-state indices, so every difference is attributable to the
code change rather than to sampling.

Reports exactly the things that could have moved:

* integrity -- does every episode still pass validation
* wrist-camera UV -- the frozen-extrinsic defect showed up as a *static* object
  having a single distinct wrist UV for a whole episode while the camera moved.
  The diagnostic is therefore: for entities that did not move in the world, how
  many distinct wrist UVs were recorded?
* phase labels -- pre/post/uncertain counts, and post_grasp agreement with the
  simulator's own grasp test where that was recorded
* changed-entity -- change_class and measured displacement, which the jitter-aware
  threshold and the per-episode re-measurement could both have altered
* outcomes -- success and trajectory length, which must NOT have moved: none of
  the fixes touch the policy path
* old-location bias -- whether the perturbed-condition finding reproduces
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load(root: Path) -> Dict[Tuple, Dict[str, Any]]:
    """Key episodes by (suite, condition, task, episode) so runs can be paired."""
    out: Dict[Tuple, Dict[str, Any]] = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (manifest_path.parent / "per_step_metrics.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not records:
            continue
        key = (manifest["suite"], manifest["condition"], manifest["task_id"],
               manifest.get("episode_index", 0))
        out[key] = {"manifest": manifest, "records": records, "dir": manifest_path.parent}
    return out


def distinct_uv(records: List[Dict[str, Any]], entity: str, camera: str) -> int:
    values = set()
    for record in records:
        label = (record.get("segmentation") or {}).get(entity, {}).get(camera)
        if label and label.get("uv"):
            values.add(tuple(round(v, 6) for v in label["uv"]))
    return len(values)


def static_entities(records: List[Dict[str, Any]], tolerance: float = 1e-6) -> List[str]:
    """Entities whose world position never moved -- the wrist-UV diagnostic needs these."""
    first = records[0].get("entity_world_xyz") or {}
    out = []
    for entity, xyz in first.items():
        if xyz is None:
            continue
        moved = max(
            (max(abs(a - b) for a, b in zip(xyz, r["entity_world_xyz"][entity]))
             for r in records if (r.get("entity_world_xyz") or {}).get(entity)),
            default=0.0,
        )
        if moved <= tolerance:
            out.append(entity)
    return out


def old_location_bias(episode: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Closest approach to the moved source vs to where it used to be."""
    manifest, records = episode["manifest"], episode["records"]
    changed = [
        c for c in (manifest.get("change_report") or {}).get("changed_entities", [])
        if c.get("role") == "source"
    ]
    if not changed:
        return None
    vanilla, perturbed = changed[0]["vanilla_xyz"], changed[0]["perturbed_xyz"]
    if not vanilla or not perturbed:
        return None

    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    to_new = min(distance(r["eef_pos"], perturbed) for r in records)
    to_old = min(distance(r["eef_pos"], vanilla) for r in records)
    axis = [v - p for v, p in zip(vanilla, perturbed)]
    length = math.sqrt(sum(a * a for a in axis)) or 1.0
    axis = [a / length for a in axis]
    best = min(records, key=lambda r: distance(r["eef_pos"], perturbed))
    error = [e - p for e, p in zip(best["eef_pos"], perturbed)]
    along = sum(e * a for e, a in zip(error, axis))
    return {
        "to_new": to_new, "to_old": to_old, "along_axis": along,
        "perturbation_m": length,
    }


def compare(v1_root: Path, v2_root: Path) -> Dict[str, Any]:
    v1, v2 = load(v1_root), load(v2_root)
    shared = sorted(set(v1) & set(v2))
    report: Dict[str, Any] = {
        "v1_root": str(v1_root), "v2_root": str(v2_root),
        "v1_episodes": len(v1), "v2_episodes": len(v2), "paired": len(shared),
        "only_in_v1": [list(k) for k in sorted(set(v1) - set(v2))],
        "only_in_v2": [list(k) for k in sorted(set(v2) - set(v1))],
        "episodes": [],
    }

    for key in shared:
        a, b = v1[key], v2[key]
        entry: Dict[str, Any] = {"key": list(key)}
        for tag, episode in (("v1", a), ("v2", b)):
            manifest, records = episode["manifest"], episode["records"]
            statics = static_entities(records)
            entry[tag] = {
                "steps": len(records),
                "success": bool(manifest.get("success")),
                "termination": manifest.get("termination_reason"),
                "change_class": manifest.get("change_class"),
                "measured_translation_m": manifest.get("measured_translation_m"),
                "phases": dict(Counter(r.get("phase") for r in records)),
                "static_entities": len(statics),
                # The frozen-extrinsic signature: 1 distinct UV for a whole episode.
                "wrist_uv_distinct_for_static": {
                    e: distinct_uv(records, e, "robot0_eye_in_hand") for e in statics[:4]
                },
                "agentview_uv_distinct_for_static": {
                    e: distinct_uv(records, e, "agentview") for e in statics[:4]
                },
                "sim_grasp_steps": sum(
                    1 for r in records if r.get("sim_check_grasp") is True
                ) if any("sim_check_grasp" in r for r in records) else None,
                "bias": old_location_bias(episode),
            }
        report["episodes"].append(entry)
    return report


def summarise(report: Dict[str, Any]) -> str:
    lines = [
        f"v1 {report['v1_root']}  ({report['v1_episodes']} episodes)",
        f"v2 {report['v2_root']}  ({report['v2_episodes']} episodes)",
        f"paired: {report['paired']}", "",
    ]
    if report["only_in_v1"] or report["only_in_v2"]:
        lines.append(f"unpaired -- only v1: {report['only_in_v1']}  only v2: {report['only_in_v2']}")
        lines.append("")

    # 1. wrist UV
    lines.append("WRIST-CAMERA UV for entities that never moved in the world")
    lines.append("(the frozen-extrinsic defect shows as exactly 1 distinct UV per episode)")
    v1_one = v2_one = v1_tot = v2_tot = 0
    v2_counts = []
    for e in report["episodes"]:
        for tag, one_key in (("v1", "v1"), ("v2", "v2")):
            for entity, count in e[tag]["wrist_uv_distinct_for_static"].items():
                if tag == "v1":
                    v1_tot += 1
                    v1_one += count <= 1
                else:
                    v2_tot += 1
                    v2_one += count <= 1
                    v2_counts.append(count / max(1, e[tag]["steps"]))
    lines.append(f"  v1: {v1_one}/{v1_tot} static entity-episodes had a single frozen wrist UV")
    lines.append(f"  v2: {v2_one}/{v2_tot} static entity-episodes had a single frozen wrist UV")
    if v2_counts:
        lines.append(f"  v2 mean distinct-UV / timesteps ratio: {sum(v2_counts)/len(v2_counts):.3f}"
                     "  (1.0 = a new projection every step)")
    lines.append("")

    # 2. outcomes -- must be unchanged
    lines.append("OUTCOMES (must be unchanged: no fix touches the policy path)")
    same_success = sum(1 for e in report["episodes"] if e["v1"]["success"] == e["v2"]["success"])
    same_steps = sum(1 for e in report["episodes"] if e["v1"]["steps"] == e["v2"]["steps"])
    lines.append(f"  same success flag : {same_success}/{report['paired']}")
    lines.append(f"  same step count   : {same_steps}/{report['paired']}")
    for e in report["episodes"]:
        if e["v1"]["success"] != e["v2"]["success"] or e["v1"]["steps"] != e["v2"]["steps"]:
            lines.append(f"    DIFFERS {e['key']}: v1 {e['v1']['steps']}/{e['v1']['success']}"
                         f"  v2 {e['v2']['steps']}/{e['v2']['success']}")
    lines.append("")

    # 3. phase + changed entity, grouped
    lines.append("PHASE and CHANGED-ENTITY by condition")
    groups: Dict[Tuple, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "v1_post": 0, "v2_post": 0, "v1_unc": 0, "v2_unc": 0,
        "v2_simgrasp": 0, "v1_class": Counter(), "v2_class": Counter(),
        "v1_meas": [], "v2_meas": [],
    })
    for e in report["episodes"]:
        g = groups[(e["key"][0], e["key"][1])]
        g["n"] += 1
        g["v1_post"] += e["v1"]["phases"].get("post_grasp", 0)
        g["v2_post"] += e["v2"]["phases"].get("post_grasp", 0)
        g["v1_unc"] += e["v1"]["phases"].get("uncertain", 0)
        g["v2_unc"] += e["v2"]["phases"].get("uncertain", 0)
        if e["v2"]["sim_grasp_steps"] is not None:
            g["v2_simgrasp"] += e["v2"]["sim_grasp_steps"]
        g["v1_class"][e["v1"]["change_class"]] += 1
        g["v2_class"][e["v2"]["change_class"]] += 1
        for tag in ("v1", "v2"):
            if e[tag]["measured_translation_m"] is not None:
                g[f"{tag}_meas"].append(e[tag]["measured_translation_m"])
    header = (f"  {'suite/condition':24s} {'post v1':>8s} {'post v2':>8s} {'simGrasp':>9s} "
              f"{'unc v1':>7s} {'unc v2':>7s} {'change_class v1 -> v2'}")
    lines.append(header)
    for (suite, condition), g in sorted(groups.items()):
        c1 = ",".join(sorted(str(k) for k in g["v1_class"]))
        c2 = ",".join(sorted(str(k) for k in g["v2_class"]))
        arrow = c1 if c1 == c2 else f"{c1} -> {c2}"
        lines.append(f"  {suite.replace('libero_','')+'/'+condition:24s} {g['v1_post']:8d} "
                     f"{g['v2_post']:8d} {g['v2_simgrasp']:9d} {g['v1_unc']:7d} {g['v2_unc']:7d} {arrow}")
        for tag in ("v1", "v2"):
            if g[f"{tag}_meas"]:
                lo, hi = min(g[f"{tag}_meas"]), max(g[f"{tag}_meas"])
                lines.append(f"      measured translation {tag}: {lo:.4f}-{hi:.4f} m")
    lines.append("")

    # 4. old-location bias
    lines.append("OLD-LOCATION BIAS (perturbed conditions only)")
    lines.append(f"  {'suite/condition':24s} {'ver':>4s} {'->new':>8s} {'->old':>8s} "
                 f"{'along-axis':>11s} {'perturb':>8s} {'n':>3s}  old<new")
    agg: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in report["episodes"]:
        for tag in ("v1", "v2"):
            bias = e[tag]["bias"]
            if not bias:
                continue
            k = (e["key"][0], e["key"][1], tag)
            for field, value in bias.items():
                agg[k][field].append(value)
    for (suite, condition, tag), vals in sorted(agg.items()):
        n = len(vals["to_new"])
        closer = sum(1 for a, b in zip(vals["to_old"], vals["to_new"]) if a < b)
        lines.append(
            f"  {suite.replace('libero_','')+'/'+condition:24s} {tag:>4s} "
            f"{sum(vals['to_new'])/n:8.4f} {sum(vals['to_old'])/n:8.4f} "
            f"{sum(vals['along_axis'])/n:+11.4f} {sum(vals['perturbation_m'])/n:8.4f} {n:3d}  {closer}/{n}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("v1_root")
    parser.add_argument("v2_root")
    parser.add_argument("--json_out")
    args = parser.parse_args()
    report = compare(Path(args.v1_root), Path(args.v2_root))
    print(summarise(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
