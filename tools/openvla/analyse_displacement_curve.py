"""Displacement-response curves: does the policy follow the object, or its memory?

Groups collected episodes by perturbation magnitude and reports, per magnitude,
the three curves that separate the two hypotheses this study is built around:

* **success rate** -- coarse, and by itself uninformative (a 7 cm shift failing is
  unsurprising). It is here as a sanity check that the perturbation bites.
* **error to the object's actual position** -- how badly the reach missed.
* **old-location bias** -- whether the miss points at where the object used to be.

A shortcut policy is expected to show bias immediately at the smallest magnitude
and to keep it as the magnitude grows. A policy that simply runs out of
generalisation range is expected to degrade smoothly with little directional
structure.

Definitions
-----------
All quantities are evaluated at ``t*``, the timestep of the end-effector's closest
approach to the perturbed source position, using positions recorded at that
timestep::

    e          = eef(t*) - p_perturbed                      (error vector, metres)
    a          = (p_vanilla - p_perturbed) / |p_vanilla - p_perturbed|
                                                            (unit vector pointing
                                                             at the old location)
    d_new      = min_t |eef(t) - p_perturbed|               (reached the object?)
    d_old      = min_t |eef(t) - p_vanilla|                 (reached the old spot?)
    along      = e . a                                      (signed; > 0 means the
                                                             miss points at the old
                                                             location)
    perp       = |e - (e . a) a|                            (the part of the miss
                                                             the bias does not
                                                             explain)
    bias_ratio = along / |p_vanilla - p_perturbed|

``bias_ratio`` is the fraction of the applied displacement that the end-effector
failed to follow, measured along the displacement axis:

* ``1.0`` -- the reach ended exactly where the object used to be; the policy moved
  as if no perturbation had happened.
* ``0.0`` -- the reach ended at the object's true position along that axis; the
  policy fully tracked the change.
* ``< 0`` -- the reach overshot past the object, away from the old location.

It is reported next to ``perp`` at all times. A large ``bias_ratio`` with a
comparably large ``perp`` means the trajectory is both biased *and* degraded, and
only the first of those is evidence for the shortcut account.

Axes are never pooled: ``x0.1`` and ``y0.1`` are the same nominal level but
different directions, and mixing them averages away the directional structure the
bias metric exists to detect.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def episode_metrics(manifest: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = (manifest.get("entity_roles") or {}).get("source")
    perturbed = manifest.get("source_position_perturbed")
    vanilla = manifest.get("source_position_vanilla")

    # Fall back to the change report for episodes collected before the manifest
    # carried explicit positions.
    if perturbed is None or vanilla is None:
        for entry in (manifest.get("change_report") or {}).get("changed_entities", []):
            if entry.get("role") == "source":
                vanilla = vanilla or entry.get("vanilla_xyz")
                perturbed = perturbed or entry.get("perturbed_xyz")

    row: Dict[str, Any] = {
        "episode": manifest.get("episode_index"),
        "task_id": manifest.get("task_id"),
        "suite": manifest.get("suite"),
        "condition": manifest.get("condition"),
        "axis": manifest.get("requested_axis"),
        "level": manifest.get("requested_level"),
        "comparison_group": manifest.get("comparison_group"),
        "success": bool(manifest.get("success")),
        "steps": len(records),
        "change_class": manifest.get("change_class"),
        "clean": manifest.get("is_clean_condition"),
        "measured_translation_m": manifest.get("measured_translation_m"),
    }

    if perturbed is None:
        # A baseline episode: the object never moved, so there is no old location.
        actual = None
        for record in records:
            xyz = (record.get("entity_world_xyz") or {}).get(source)
            if xyz:
                actual = xyz
                break
        if actual is not None:
            row["d_new"] = min(norm(sub(r["eef_pos"], actual)) for r in records)
        row["displacement_m"] = 0.0
        return row

    displacement = norm(sub(vanilla, perturbed)) if vanilla else 0.0
    row["displacement_m"] = displacement
    row["d_new"] = min(norm(sub(r["eef_pos"], perturbed)) for r in records)
    if vanilla:
        row["d_old"] = min(norm(sub(r["eef_pos"], vanilla)) for r in records)

    if displacement > 1e-9 and vanilla:
        axis = [c / displacement for c in sub(vanilla, perturbed)]
        best = min(records, key=lambda r: norm(sub(r["eef_pos"], perturbed)))
        error = sub(best["eef_pos"], perturbed)
        along = dot(error, axis)
        row["t_star"] = best["timestep"]
        row["along_axis_m"] = along
        row["perpendicular_m"] = math.sqrt(max(norm(error) ** 2 - along ** 2, 0.0))
        row["bias_ratio"] = along / displacement
    return row


def load_run(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = manifest_path.parent / "per_step_metrics.jsonl"
        if not metrics_path.is_file():
            continue
        records = [
            json.loads(line) for line in
            metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not records:
            continue
        row = episode_metrics(manifest, records)
        row["dir"] = str(manifest_path.parent)
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], group_axis: bool = True) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Baselines belong to every axis's curve as its zero point, so they are
        # keyed with axis None and merged in by the caller.
        key = (row["suite"], row["axis"] if group_axis else None, row["condition"])
        groups[key].append(row)

    out = []
    for (suite, axis, condition), items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        def mean(field):
            vals = [i[field] for i in items if i.get(field) is not None]
            return None if not vals else stats.mean(vals)

        biases = [i["bias_ratio"] for i in items if i.get("bias_ratio") is not None]
        closer = sum(
            1 for i in items
            if i.get("d_old") is not None and i.get("d_new") is not None
            and i["d_old"] < i["d_new"]
        )
        out.append({
            "suite": suite, "axis": axis, "condition": condition,
            "episodes": len(items),
            "displacement_m": mean("displacement_m"),
            "success_rate": sum(i["success"] for i in items) / len(items),
            "d_new_m": mean("d_new"),
            "d_old_m": mean("d_old"),
            "along_axis_m": mean("along_axis_m"),
            "perpendicular_m": mean("perpendicular_m"),
            "bias_ratio": None if not biases else stats.mean(biases),
            "bias_ratio_sd": None if len(biases) < 2 else stats.stdev(biases),
            "old_closer_than_new": f"{closer}/{len(items)}" if closer or items else None,
            "clean": all(bool(i.get("clean")) for i in items) if items else None,
            "tasks": sorted({i["task_id"] for i in items}),
        })
    return out


def fmt(value: Optional[float], width: int = 8, digits: int = 4) -> str:
    return "—".rjust(width) if value is None else f"{value:{width}.{digits}f}"


def render(curves: List[Dict[str, Any]]) -> str:
    lines = [
        f"{'suite':14s} {'cond':7s} {'ax':3s} {'n':>3s} {'shift(m)':>9s} {'succ':>6s} "
        f"{'→object':>8s} {'→old':>8s} {'along':>8s} {'perp':>8s} {'bias':>7s} {'old<new':>8s}",
        "-" * 108,
    ]
    for c in curves:
        suite = c["suite"].replace("libero_", "")
        bias = "—".rjust(7) if c["bias_ratio"] is None else f"{c['bias_ratio']:7.3f}"
        lines.append(
            f"{suite:14s} {c['condition']:7s} {(c['axis'] or '-'):3s} {c['episodes']:3d} "
            f"{fmt(c['displacement_m'],9,4)} {c['success_rate']:6.2f} "
            f"{fmt(c['d_new_m'])} {fmt(c['d_old_m'])} "
            f"{fmt(c['along_axis_m'])} {fmt(c['perpendicular_m'])} {bias} "
            f"{(c['old_closer_than_new'] or '—'):>8s}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="+", help="collection run directories")
    parser.add_argument("--json_out")
    parser.add_argument("--pool_axes", action="store_true",
                        help="pool x and y (off by default: it averages away direction)")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for root in args.roots:
        rows.extend(load_run(Path(root)))
    if not rows:
        print("no episodes found")
        return 1

    curves = aggregate(rows, group_axis=not args.pool_axes)
    print(render(curves))

    print("\nbias_ratio = (error at closest approach, projected on the displacement axis)"
          " / (applied displacement)")
    print("  1.0 = ended where the object used to be   0.0 = fully tracked the object")
    print("  reported beside `perp`; a large bias with a large perp means biased AND degraded")

    groups = defaultdict(list)
    for row in rows:
        if row.get("comparison_group"):
            groups[row["comparison_group"]].append(row["condition"])
    complete = sum(1 for v in groups.values() if len(set(v)) > 1)
    print(f"\nmatched comparison groups: {complete} of {len(groups)} have more than one condition")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"curves": curves, "episodes": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
