"""LIBERO-PRO swap-specific resolution for internal-grounding collection.

The common task resolver intentionally has no dynamic-BDDL side effects.  This
module is the narrow orchestration layer that invokes the repository's existing
official LIBERO-PRO swap generator and converts its frozen asset into the
path-only resolution consumed by the unchanged single-frame grounding probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from official_task_pair_resolver import resolve_official_task_pair
from spatial_task_resolver import diff_init_regions, parse_init_regions
from task_resolution import ResolvedTaskCondition


class SwapGroundingConfigurationError(RuntimeError):
    """The selected swap does not move the object evaluated by the probe."""


def resolve_target_moving_swap(
    config: Dict[str, Any], swap_seed: int,
) -> Tuple[ResolvedTaskCondition, Dict[str, Any]]:
    """Generate/reuse one official swap BDDL and require that its GT target moved."""
    pair = resolve_official_task_pair(
        suite=str(config["suite"]), task_id=int(config["task_id"]), condition="swap",
        seed=int(swap_seed),
        asset_dir=Path(__file__).resolve().parents[2] / "assets" / "libero_pro_generated" / str(config["suite"]),
    )
    if not pair.perturbed_bddl_path or not pair.perturbed_bddl_sha256:
        raise SwapGroundingConfigurationError("official swap resolver returned no perturbed BDDL")
    vanilla_regions = parse_init_regions(Path(pair.vanilla_bddl_path).read_text(encoding="utf-8"))
    swapped_regions = parse_init_regions(Path(pair.perturbed_bddl_path).read_text(encoding="utf-8"))
    moved_entities, swap_pairs = diff_init_regions(vanilla_regions, swapped_regions)
    target_object = str(config["evaluation_target_object"])
    if target_object not in moved_entities:
        raise SwapGroundingConfigurationError(
            f"swap seed {swap_seed} does not move evaluation target {target_object!r}; "
            f"moved entities: {moved_entities}"
        )
    resolution = ResolvedTaskCondition(
        suite=pair.suite, task_id=pair.task_id, task_name=pair.task_name,
        instruction=pair.instruction, requested_condition="swap", perturbation_family="swap",
        requested_bddl_path=pair.perturbed_bddl_path, vanilla_bddl_path=pair.vanilla_bddl_path,
        resolved_bddl_path=pair.perturbed_bddl_path,
        vanilla_bddl_sha256=pair.vanilla_bddl_sha256,
        resolved_bddl_sha256=pair.perturbed_bddl_sha256,
    )
    return resolution, {
        "swap_seed": int(swap_seed), "target_object": target_object,
        "moved_entities": moved_entities, "swap_pairs": swap_pairs,
        "swap_bddl_path": pair.perturbed_bddl_path,
        "swap_bddl_sha256": pair.perturbed_bddl_sha256,
        "swap_notes": pair.notes,
    }
