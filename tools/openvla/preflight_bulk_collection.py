"""Ten checks that must all pass before a bulk collection is allowed to start.

The point is that the run plan is validated as data, not as intention. Each check
either passes with the number it verified or fails with the number that was wrong;
none of them can pass vacuously.

The plan being checked is fixed by a decision taken *before* any of these episodes
were collected: libero_object task 5 is excluded because its y0.2 and y0.3 assets
move a second object (``chocolate_pudding_1``, ~10 m) in addition to the source,
so it does not meet the single-source-displacement control the curve requires.
That exclusion is recorded in docs/task5_exclusion.md and is enforced here rather
than left to the operator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: Fixed before collection, not chosen after seeing results.
INCLUDED_TASKS = [0, 1, 2, 3, 4, 6, 7, 8, 9]
EXCLUDED_TASKS = [5]
CONDITIONS = ["vanilla", "y0.1", "y0.2", "y0.3"]
EPISODES_PER_CELL = 5
INIT_STATE_IDS = list(range(EPISODES_PER_CELL))

#: Measured on the smoke test: 4 episodes occupied 8.0 GB.
GB_PER_EPISODE = 2.0
SAFETY_FACTOR = 1.25

PASS, FAIL = "PASS", "FAIL"


def check(results: List[Dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    results.append({"check": name, "status": PASS if ok else FAIL, "detail": detail})
    return ok


def load_verification(verification_dir: Path) -> Dict[Tuple[int, str, int], Dict[str, Any]]:
    """Every (task, condition, init_state) pair that has been measured."""
    out: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    candidates = [verification_dir / "condition_verification.json"]
    candidates += sorted(verification_dir.glob("init_*/condition_verification.json"))
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        init = payload.get("args", {}).get("init_state_id", 0)
        for row in payload["rows"]:
            out[(row["task_id"], row["condition"], init)] = row
    return out


def build_plan(seed: int) -> List[Dict[str, Any]]:
    """One collector invocation per (task, condition); each yields EPISODES_PER_CELL."""
    plan = []
    for index, task_id in enumerate(INCLUDED_TASKS):
        for condition_index, condition in enumerate(CONDITIONS):
            plan.append({
                "task_id": task_id,
                "condition": condition,
                "seed": seed,
                "episodes": EPISODES_PER_CELL,
                "init_state_ids": list(INIT_STATE_IDS),
                # Round-robin over the flat invocation list keeps the two GPUs
                # balanced across conditions as well as tasks, which matters
                # because perturbed episodes run to the step cap and baselines
                # terminate on success.
                "gpu": 1 if (index * len(CONDITIONS) + condition_index) % 2 == 0 else 2,
            })
    return plan


def run_checks(
    output_root: Path, verification_dir: Path, seed: int,
    existing_roots: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    plan = build_plan(seed)

    # 1 --------------------------------------------------------------------
    tasks = sorted({cell["task_id"] for cell in plan})
    check(results, "1_included_tasks", tasks == INCLUDED_TASKS,
          f"tasks in plan = {tasks}, expected {INCLUDED_TASKS}")

    # 2 --------------------------------------------------------------------
    per_condition = {
        condition: sorted(c["task_id"] for c in plan if c["condition"] == condition)
        for condition in CONDITIONS
    }
    identical = len({tuple(v) for v in per_condition.values()}) == 1
    check(results, "2_conditions_share_tasks", identical,
          "; ".join(f"{c}={v}" for c, v in per_condition.items()))

    # 3 --------------------------------------------------------------------
    excluded_present = sorted(set(tasks) & set(EXCLUDED_TASKS))
    check(results, "3_task5_absent", not excluded_present,
          f"excluded tasks appearing in the plan: {excluded_present or 'none'}")

    # 4 + 5 ----------------------------------------------------------------
    verified = load_verification(verification_dir)
    missing: List[str] = []
    dirty: List[str] = []
    contaminated: List[str] = []
    for cell in plan:
        if cell["condition"] == "vanilla":
            continue
        for init in cell["init_state_ids"]:
            key = (cell["task_id"], cell["condition"], init)
            row = verified.get(key)
            if row is None:
                missing.append(f"task{cell['task_id']}/{cell['condition']}/init{init}")
                continue
            if row.get("change_class") != "clean_source_only":
                dirty.append(f"task{cell['task_id']}/{cell['condition']}/init{init}"
                             f"={row.get('change_class')}")
            if row.get("other_moved_objects"):
                contaminated.append(
                    f"task{cell['task_id']}/{cell['condition']}/init{init}"
                    f"={row['other_moved_objects']}")

    expected_pairs = len(INCLUDED_TASKS) * (len(CONDITIONS) - 1) * len(INIT_STATE_IDS)
    check(results, "4_clean_source_only",
          not missing and not dirty,
          f"{expected_pairs - len(missing) - len(dirty)}/{expected_pairs} verified "
          f"clean_source_only" + (f"; missing={missing[:6]}" if missing else "")
          + (f"; not clean={dirty[:6]}" if dirty else ""))
    check(results, "5_no_other_object_moved", not contaminated,
          f"pairs with other moved objects: {contaminated[:6] or 'none'}")

    # 6 --------------------------------------------------------------------
    total = sum(cell["episodes"] for cell in plan)
    expected = len(INCLUDED_TASKS) * len(CONDITIONS) * EPISODES_PER_CELL
    check(results, "6_episode_count", total == expected == 180,
          f"{total} episodes planned (expected {expected})")

    # 7 --------------------------------------------------------------------
    clashes = [
        str(root) for root in existing_roots
        if root.exists() and (root == output_root or root in output_root.parents
                              or output_root in root.parents)
    ]
    check(results, "7_output_path_separate",
          not output_root.exists() and not clashes,
          f"output_root={output_root} exists={output_root.exists()}"
          + (f"; overlaps {clashes}" if clashes else "; no overlap with existing runs"))

    # 8 --------------------------------------------------------------------
    need_gb = total * GB_PER_EPISODE * SAFETY_FACTOR
    target = output_root
    while not target.exists() and target != target.parent:
        target = target.parent
    free_gb = shutil.disk_usage(target).free / 1e9
    check(results, "8_disk_space", free_gb > need_gb,
          f"need ~{need_gb:.0f} GB (at {GB_PER_EPISODE} GB/episode x{SAFETY_FACTOR}), "
          f"free {free_gb:.0f} GB on {target}")

    # 9 --------------------------------------------------------------------
    gpu_cells: Dict[int, List[str]] = {}
    for cell in plan:
        gpu_cells.setdefault(cell["gpu"], []).append(f"t{cell['task_id']}/{cell['condition']}")
    overlap = set(gpu_cells.get(1, [])) & set(gpu_cells.get(2, []))
    balanced = abs(len(gpu_cells.get(1, [])) - len(gpu_cells.get(2, []))) <= 1
    check(results, "9_gpu_partition_disjoint", not overlap and balanced,
          f"gpu1={len(gpu_cells.get(1, []))} cells, gpu2={len(gpu_cells.get(2, []))} cells, "
          f"overlap={sorted(overlap) or 'none'}")

    # 10 -------------------------------------------------------------------
    collector = Path(_HERE) / "collect_official_activations.py"
    source = collector.read_text(encoding="utf-8")
    resumable = "--skip_existing" in source and 'COMPLETE").is_file()' in source
    check(results, "10_resumable", resumable,
          "collector honours --skip_existing by testing the COMPLETE marker"
          if resumable else "collector cannot skip already-finished episodes")

    return results, plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--verification_dir",
                        default=str(Path(_HERE).parents[1] / "docs" / "condition_verification"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--existing_roots", nargs="*", default=[])
    parser.add_argument("--plan_out")
    args = parser.parse_args()

    results, plan = run_checks(
        Path(args.output_root), Path(args.verification_dir), args.seed,
        [Path(p) for p in args.existing_roots],
    )

    width = max(len(r["check"]) for r in results)
    for r in results:
        print(f"[{r['status']}] {r['check']:{width}s}  {r['detail']}")

    failed = [r for r in results if r["status"] == FAIL]
    print()
    if failed:
        print(f"PREFLIGHT FAILED: {len(failed)} of {len(results)} checks")
        for r in failed:
            print(f"  - {r['check']}: {r['detail']}")
        return 1

    print(f"PREFLIGHT PASSED: {len(results)}/{len(results)} checks")
    print(f"  {len(plan)} collector invocations, "
          f"{sum(c['episodes'] for c in plan)} episodes, "
          f"tasks {INCLUDED_TASKS}, conditions {CONDITIONS}")
    if args.plan_out:
        Path(args.plan_out).write_text(
            json.dumps({"plan": plan, "checks": results,
                        "included_tasks": INCLUDED_TASKS,
                        "excluded_tasks": EXCLUDED_TASKS,
                        "conditions": CONDITIONS,
                        "episodes_per_cell": EPISODES_PER_CELL,
                        "init_state_ids": INIT_STATE_IDS,
                        "output_root": args.output_root,
                        "seed": args.seed}, indent=2),
            encoding="utf-8")
        print(f"  plan written to {args.plan_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
