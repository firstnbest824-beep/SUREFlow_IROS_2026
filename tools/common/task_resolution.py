"""Resolve baseline LIBERO and shipped LIBERO-PRO BDDL files without diagnostics.

This module intentionally contains only path selection.  It does not import
``tools/openvla`` or any diagnostics resolver.  ``vanilla`` resolves through
the official LIBERO benchmark; shipped position-offset conditions such as
``y0.1`` resolve to ``<suite>_temp_y0.1/<task-bddl-name>`` under LIBERO-PRO.
Dynamic ``swap`` generation is deliberately unsupported here because it is not
a path-only operation and would create an experiment asset as a side effect.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_OFFSET_CONDITION_RE = re.compile(r"^(?P<axis>[xy])(?P<level>[0-9]+(?:\.[0-9]+)?)$")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a BDDL file without loading it all at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedTaskCondition:
    """The concrete BDDL selected for one baseline rollout or smoke test."""

    suite: str
    task_id: int
    task_name: str
    instruction: str
    requested_condition: str
    perturbation_family: str
    requested_bddl_path: str
    vanilla_bddl_path: str
    resolved_bddl_path: str
    vanilla_bddl_sha256: str
    resolved_bddl_sha256: str
    requested_axis: Optional[str] = None
    requested_level: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_task_condition(suite: str, task_id: int, condition: str) -> ResolvedTaskCondition:
    """Resolve ``vanilla`` or a shipped LIBERO-PRO position-offset condition.

    Unknown conditions are rejected rather than silently running the vanilla
    scene.  This is the safety property that the old action-generalization
    evaluation path lacked.
    """
    from libero.libero import benchmark, get_libero_path

    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_id)
    vanilla = Path(suite_obj.get_task_bddl_file_path(task_id)).resolve()
    if not vanilla.is_file():
        raise FileNotFoundError(f"official vanilla BDDL is missing: {vanilla}")

    requested_condition = str(condition)
    family = "vanilla"
    axis: Optional[str] = None
    level: Optional[float] = None
    requested = vanilla
    resolved = vanilla

    if requested_condition != "vanilla":
        match = _OFFSET_CONDITION_RE.fullmatch(requested_condition)
        if match is None:
            raise ValueError(
                f"unsupported LIBERO-PRO condition {requested_condition!r}; "
                "use 'vanilla' or a shipped position offset such as 'y0.1'"
            )
        bddl_root = Path(get_libero_path("bddl_files")).resolve()
        directory = bddl_root / f"{suite}_temp_{requested_condition}"
        requested = directory / vanilla.name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"LIBERO-PRO condition directory does not exist: {directory}"
            )
        if not requested.is_file():
            raise FileNotFoundError(
                f"LIBERO-PRO condition {requested_condition!r} has no BDDL for "
                f"task {task.name!r}: {requested}"
            )
        resolved = requested.resolve()
        family = "position_offset"
        axis = match.group("axis")
        level = float(match.group("level"))

    return ResolvedTaskCondition(
        suite=suite,
        task_id=int(task_id),
        task_name=task.name,
        instruction=task.language,
        requested_condition=requested_condition,
        perturbation_family=family,
        requested_bddl_path=str(requested),
        vanilla_bddl_path=str(vanilla),
        resolved_bddl_path=str(resolved),
        vanilla_bddl_sha256=sha256_file(vanilla),
        resolved_bddl_sha256=sha256_file(resolved),
        requested_axis=axis,
        requested_level=level,
    )
