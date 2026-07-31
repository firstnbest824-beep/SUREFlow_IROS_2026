"""Measure, per task, exactly what each official perturbation condition changes.

Written because a single task's answer does not generalise. On ``libero_object``
task 0, ``y0.2`` and ``y0.3`` move only the source, while ``x0.2`` and above also
teleport a distractor ~10 m out of the scene -- and an earlier static analysis
disagreed with the simulator about which tasks ``y0.4`` affects. So every
(task, condition) pair is measured individually and reported individually.

No policy is run and nothing is rendered: each pair builds the vanilla and the
perturbed environment, applies the same pinned initial state to both, reads every
tracked entity's world position, and diffs them.

Outputs, side by side, into one directory:

* ``condition_verification.csv``  -- one row per (task, condition)
* ``condition_verification.md``   -- the same table, readable
* ``condition_verification.json`` -- full detail including per-entity deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: Anything beyond this is a scene edit, not a spatial perturbation.
LEFT_SCENE_M = 5.0


def verify_pair(
    suite: str, task_id: int, condition: str, seed: int, init_state_id: int, resolution: int
) -> Dict[str, Any]:
    from changed_entity_detector import is_clean
    from collect_official_activations import measure_change
    from entity_role_resolver import resolve_entity_roles_from_path
    from official_task_pair_resolver import resolve_official_task_pair

    row: Dict[str, Any] = {
        "suite": suite, "task_id": task_id, "condition": condition,
        "seed": seed, "init_state_id": init_state_id,
    }
    try:
        pair = resolve_official_task_pair(
            suite=suite, task_id=task_id, condition=condition, seed=seed
        )
        roles = resolve_entity_roles_from_path(
            pair.perturbed_bddl_path or pair.vanilla_bddl_path
        )
        report, poses = measure_change(
            vanilla_bddl=pair.vanilla_bddl_path,
            perturbed_bddl=pair.perturbed_bddl_path,
            roles=roles,
            resolution=resolution,
            suite=suite,
            condition=pair.perturbation_name,
            task_id=task_id,
            task_name=pair.task_name,
            seed=seed,
            init_state_id=init_state_id,
        )
    except Exception as exc:  # a task that cannot even be built is a result
        row.update(
            init_state_ok=False, error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-800:], change_class=None, clean=None,
        )
        return row

    changed = {c.name: c for c in report.changed_entities}
    source = report.source_entity
    destination = report.destination_entity
    source_change = changed.get(source)

    others = [
        {
            "name": c.name, "role": c.role,
            "displacement_m": round(c.translation_norm, 4),
            "left_scene": bool(c.left_scene or (c.translation_norm or 0) >= LEFT_SCENE_M),
        }
        for c in report.changed_entities if c.name != source
    ]

    vanilla_pose = poses["vanilla"].get(source, {}).get("xyz")
    perturbed_pose = poses["perturbed"].get(source, {}).get("xyz")

    row.update(
        init_state_ok=True,
        error=None,
        task_name=pair.task_name,
        instruction=pair.instruction,
        source=source,
        destination=destination,
        requested_axis=pair.requested_axis,
        requested_level=pair.requested_level,
        source_displacement_m=(
            None if source_change is None else round(source_change.translation_norm, 4)
        ),
        source_moved=source in changed,
        destination_moved=destination in changed,
        destination_displacement_m=(
            None if destination not in changed
            else round(changed[destination].translation_norm, 4)
        ),
        other_moved_objects=[o["name"] for o in others],
        other_moved_detail=others,
        left_scene=[o["name"] for o in others if o["left_scene"]],
        change_class=report.change_class,
        clean=bool(is_clean(report.change_class)),
        source_position_vanilla=vanilla_pose,
        source_position_perturbed=perturbed_pose,
        destination_position=poses["perturbed"].get(destination, {}).get("xyz"),
        init_state_sources_match=poses.get("init_state_sources_match"),
        per_entity_threshold_m=report.per_entity_threshold_m,
        indeterminate_entities=[e["name"] for e in report.indeterminate_entities],
    )
    return row


CSV_FIELDS = [
    "suite", "task_id", "condition", "requested_axis", "requested_level",
    "source", "source_displacement_m", "destination_moved",
    "destination_displacement_m", "other_moved_objects", "left_scene",
    "change_class", "clean", "init_state_ok", "error",
]


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    head = ("| task | condition | source displacement | other moved objects | "
            "left scene | change class | clean |")
    sep = "|---|---|---:|---|---|---|:--:|"
    lines = [head, sep]
    for r in rows:
        if not r.get("init_state_ok"):
            lines.append(
                f"| {r['task_id']} | `{r['condition']}` | — | — | — | "
                f"**BUILD FAILED** | — |"
            )
            continue
        others = ", ".join(
            f"`{o['name']}` ({o['displacement_m']} m)" for o in r["other_moved_detail"]
        ) or "없음"
        left = ", ".join(f"`{n}`" for n in r["left_scene"]) or "—"
        disp = "—" if r["source_displacement_m"] is None else f"{r['source_displacement_m']:.3f} m"
        mark = "✅" if r["clean"] else "❌"
        lines.append(
            f"| {r['task_id']} | `{r['condition']}` | {disp} | {others} | {left} | "
            f"`{r['change_class']}` | {mark} |"
        )
    return "\n".join(lines)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        entry = by_condition.setdefault(
            r["condition"],
            {"tasks": 0, "clean_tasks": [], "dirty_tasks": [], "failed_tasks": [],
             "displacements": []},
        )
        entry["tasks"] += 1
        if not r.get("init_state_ok"):
            entry["failed_tasks"].append(r["task_id"])
        elif r["clean"]:
            entry["clean_tasks"].append(r["task_id"])
            if r["source_displacement_m"] is not None:
                entry["displacements"].append(r["source_displacement_m"])
        else:
            entry["dirty_tasks"].append(r["task_id"])
    for entry in by_condition.values():
        d = entry.pop("displacements")
        entry["clean_count"] = len(entry["clean_tasks"])
        entry["all_clean"] = (
            len(entry["clean_tasks"]) == entry["tasks"] and not entry["failed_tasks"]
        )
        entry["source_displacement_m"] = (
            None if not d else {"min": min(d), "max": max(d), "mean": round(sum(d) / len(d), 4)}
        )
    return by_condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--conditions", default="y0.1,y0.2,y0.3")
    parser.add_argument("--tasks", default="0-9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if "-" in args.tasks:
        lo, hi = args.tasks.split("-")
        tasks = list(range(int(lo), int(hi) + 1))
    else:
        tasks = [int(t) for t in args.tasks.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for task_id in tasks:
        for condition in conditions:
            row = verify_pair(
                args.suite, task_id, condition, args.seed, args.init_state_id, args.resolution
            )
            rows.append(row)
            status = (
                "BUILD FAILED" if not row.get("init_state_ok")
                else f"{row['change_class']:26s} src={row['source_displacement_m']}"
            )
            print(f"  task {task_id:2d} {condition:6s} {status}", flush=True)

    with open(out / "condition_verification.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "other_moved_objects": ";".join(row.get("other_moved_objects") or []),
                "left_scene": ";".join(row.get("left_scene") or []),
            })

    summary = summarise(rows)
    (out / "condition_verification.md").write_text(
        f"# 조건 오염 검증 — {args.suite}\n\n"
        f"seed {args.seed}, init state {args.init_state_id}. "
        "정책은 실행하지 않고 vanilla / perturbed 환경에 같은 시작 상태를 적용해 좌표만 비교했습니다.\n\n"
        + to_markdown(rows)
        + "\n\n## 조건별 요약\n\n"
        + "\n".join(
            f"- `{c}` — clean {e['clean_count']}/{e['tasks']} task"
            + (f", 오염된 task: {e['dirty_tasks']}" if e["dirty_tasks"] else "")
            + (f", 빌드 실패: {e['failed_tasks']}" if e["failed_tasks"] else "")
            + (f", 이동량 {e['source_displacement_m']['min']:.3f}–{e['source_displacement_m']['max']:.3f} m"
               if e["source_displacement_m"] else "")
            for c, e in sorted(summary.items())
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "condition_verification.json").write_text(
        json.dumps({"rows": rows, "summary": summary,
                    "args": vars(args)}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + to_markdown(rows))
    print()
    for condition, entry in sorted(summary.items()):
        print(f"{condition}: clean {entry['clean_count']}/{entry['tasks']}"
              f"  dirty={entry['dirty_tasks']}  failed={entry['failed_tasks']}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
