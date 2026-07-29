"""Pair an official LIBERO task with its official LIBERO-PRO perturbed counterpart.

Two perturbation families are supported, and they are shipped very differently.

``position_offset`` (LIBERO-Object)
    LIBERO-PRO ships pre-generated BDDL directories
    ``bddl_files/libero_object_temp_{x,y}0.{1..5}``. Each shifts the
    ``target_object_region`` sampling box -- i.e. it moves the object the
    instruction names. Instructions in this suite are name-based ("Pick the
    alphabet soup"), so the referent stays valid at any offset. Nothing is
    generated at run time; the perturbed BDDL is read straight off disk.

``swap`` (LIBERO-Spatial and others)
    Nothing is shipped. ``SwapPerturbator`` rewrites the ``(:init ...)`` block at
    run time from ``libero_ood/ood_spatial_relation.yaml``. It has no ``seed``
    parameter and draws from the global ``random`` module, so this resolver seeds
    the RNG itself and writes the generated text to a content-addressed asset
    file. The pairing is therefore reproducible and hashable, which the
    "official files with a recorded sha256" requirement needs.

Both families produce the same ``OfficialTaskPair``. Nothing downstream needs to
know which family it came from except to record it.

The official README drives position offsets by ``cp -r libero_object_temp_x0.3/*
libero_object_temp/`` before launching. That mutates a shared directory and makes
concurrent conditions clobber each other, so it is not used here: the resolver
hands out explicit per-condition BDDL paths instead, and the original
directories are never written to.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from entity_role_resolver import EntityRoles, resolve_entity_roles  # noqa: E402

FAMILY_SWAP = "swap"
FAMILY_POSITION_OFFSET = "position_offset"
FAMILY_VANILLA = "vanilla"

# Suite -> the OpenVLA checkpoint fine-tuned on it. Running a suite against the
# wrong checkpoint produces plausible-looking but meaningless rollouts, so this
# mapping is enforced rather than advisory.
SUITE_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "libero_spatial": {
        "model_id": "openvla/openvla-7b-finetuned-libero-spatial",
        "revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "unnorm_key": "libero_spatial",
    },
    "libero_object": {
        "model_id": "openvla/openvla-7b-finetuned-libero-object",
        "revision": "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
        "unnorm_key": "libero_object",
    },
}

# Which role each experiment is designed to interrogate. Recorded per episode as
# `primary_analysis_entity`; it selects the probe label, not what gets saved.
FAMILY_PRIMARY_ENTITY = {
    FAMILY_POSITION_OFFSET: "source",
    FAMILY_SWAP: "destination",
    FAMILY_VANILLA: "source",
}

_OFFSET_DIR_RE = re.compile(r"^(?P<suite>.+)_temp_(?P<axis>[xy])(?P<level>[0-9.]+)$")


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_seed(seed: Any) -> int:
    """Reject the upstream ``configs.get("seed", int)`` failure mode.

    LIBERO-PRO's ``perturbation.py`` defaults the seed to the *type object*
    ``int`` when the key is absent from ``evaluation_config.yaml``. That is then
    passed to ``random.seed(int)``, which does not raise -- it hash-seeds from an
    address, so it differs per process under ASLR and silently destroys
    reproducibility. A seed must be a real integer here.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"seed must be an int, got {seed!r} ({type(seed).__name__}). "
            "Upstream LIBERO-PRO defaults this to the type object `int`, which "
            "silently produces a per-process random seed."
        )
    return int(seed)


def seed_everything(seed: int) -> int:
    """Seed every RNG the perturbation and placement samplers actually use."""
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


# -----------------------------------------------------------------------------
# Pair
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class OfficialTaskPair:
    suite: str
    task_id: int
    task_name: str
    instruction: str

    vanilla_bddl_path: str
    perturbed_bddl_path: Optional[str]
    vanilla_bddl_sha256: str
    perturbed_bddl_sha256: Optional[str]

    perturbation_family: str
    perturbation_name: str
    requested_axis: Optional[str]
    requested_level: Optional[float]

    source_entity: str
    destination_entity: str
    primary_analysis_entity: str

    model_id: str
    model_revision: str
    unnorm_key: str

    seed: int
    roles: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def bddl_root() -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files"))


def discover_position_offset_conditions(suite: str) -> Dict[str, Tuple[str, float, Path]]:
    """Find shipped ``<suite>_temp_{axis}{level}`` directories for ``suite``.

    Returns ``{"x0.1": ("x", 0.1, path), ...}``. Empty when the suite has none --
    ``libero_spatial`` genuinely ships no position-offset assets.
    """
    root = bddl_root()
    found: Dict[str, Tuple[str, float, Path]] = {}
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _OFFSET_DIR_RE.match(entry.name)
        if not match or match.group("suite") != suite:
            continue
        axis = match.group("axis")
        try:
            level = float(match.group("level"))
        except ValueError:
            continue
        found[f"{axis}{match.group('level')}"] = (axis, level, entry)
    return found


def available_conditions(suite: str) -> Dict[str, str]:
    """Condition name -> family, for every official condition this suite supports."""
    conditions = {"vanilla": FAMILY_VANILLA}
    for name in discover_position_offset_conditions(suite):
        conditions[name] = FAMILY_POSITION_OFFSET
    if _swap_config_has_suite(suite):
        conditions["swap"] = FAMILY_SWAP
    return conditions


def _swap_config_path() -> Path:
    return Path(_HERE).parents[1] / "LIBERO-PRO" / "libero_ood" / "ood_spatial_relation.yaml"


def _swap_config_has_suite(suite: str) -> bool:
    path = _swap_config_path()
    if not path.is_file():
        return False
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return bool(config.get(suite))


# -----------------------------------------------------------------------------
# Swap asset generation
# -----------------------------------------------------------------------------
def generate_swap_bddl(
    vanilla_bddl_path: str,
    suite: str,
    task_name: str,
    seed: int,
    asset_dir: str | os.PathLike,
    config_path: Optional[str] = None,
) -> Tuple[str, str, List[str]]:
    """Deterministically generate the official swap BDDL and freeze it as an asset.

    Returns ``(path, sha256, notes)``. Regenerating with the same seed reproduces
    the same text, so an existing asset is reused rather than rewritten.
    """
    notes: List[str] = []
    config = Path(config_path) if config_path else _swap_config_path()
    if not config.is_file():
        raise FileNotFoundError(f"official swap config not found: {config}")

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / f"{task_name}__swap__seed{seed}.bddl"

    libero_pro_root = str(Path(_HERE).parents[1] / "LIBERO-PRO")
    if libero_pro_root not in sys.path:
        sys.path.insert(0, libero_pro_root)
    from perturbation import BDDLParser, SwapPerturbator  # type: ignore

    original = Path(vanilla_bddl_path).read_text(encoding="utf-8")
    # SwapPerturbator.perturb() takes no seed and uses the global `random`.
    seed_everything(seed)
    perturbed = SwapPerturbator(BDDLParser(original), str(config)).perturb(
        task_suite_name=suite, task_name=task_name
    )

    if perturbed.strip() == original.strip():
        notes.append(
            "swap produced no change for this task -- the official config lists no "
            "usable candidate for any object of interest"
        )

    digest = sha256_text(perturbed)
    if target.is_file():
        existing = sha256_text(target.read_text(encoding="utf-8"))
        if existing != digest:
            raise RuntimeError(
                f"existing swap asset {target} has sha256 {existing} but regeneration "
                f"with seed {seed} produced {digest}. Refusing to overwrite a frozen asset."
            )
        notes.append("reused frozen swap asset")
    else:
        target.write_text(perturbed, encoding="utf-8")
        notes.append("generated and froze swap asset")
    return str(target), digest, notes


# -----------------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------------
def resolve_official_task_pair(
    suite: str,
    task_id: int,
    condition: str,
    seed: int = 0,
    asset_dir: str | os.PathLike | None = None,
) -> OfficialTaskPair:
    """Resolve one (suite, task, condition) into a fully-specified pair.

    ``condition`` is ``"vanilla"``, ``"swap"``, or a position-offset name such as
    ``"x0.1"``. Unknown conditions raise rather than silently falling back.
    """
    from libero.libero import benchmark as libero_benchmark

    seed = validate_seed(seed)
    if suite not in SUITE_CHECKPOINTS:
        raise KeyError(
            f"no checkpoint mapping for suite {suite!r}; known: {sorted(SUITE_CHECKPOINTS)}"
        )

    suite_obj = libero_benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_id)
    vanilla_bddl = suite_obj.get_task_bddl_file_path(task_id)
    vanilla_text = Path(vanilla_bddl).read_text(encoding="utf-8")
    roles = resolve_entity_roles(vanilla_text)

    conditions = available_conditions(suite)
    if condition not in conditions:
        raise KeyError(
            f"condition {condition!r} is not an official condition for {suite}. "
            f"Available: {sorted(conditions)}"
        )
    family = conditions[condition]

    notes: List[str] = []
    perturbed_path: Optional[str] = None
    perturbed_sha: Optional[str] = None
    axis: Optional[str] = None
    level: Optional[float] = None

    if family == FAMILY_POSITION_OFFSET:
        axis, level, directory = discover_position_offset_conditions(suite)[condition]
        candidate = directory / Path(vanilla_bddl).name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"official position-offset BDDL missing for task {task.name!r}: {candidate}"
            )
        perturbed_path = str(candidate)
        perturbed_sha = sha256_file(candidate)
        notes.append(
            "requested_level is the official condition name, NOT a measured "
            "displacement; measure the actual delta from initial poses"
        )
    elif family == FAMILY_SWAP:
        target_dir = Path(asset_dir) if asset_dir else (
            Path(_HERE).parents[1] / "assets" / "libero_pro_generated" / suite
        )
        perturbed_path, perturbed_sha, swap_notes = generate_swap_bddl(
            vanilla_bddl_path=vanilla_bddl, suite=suite, task_name=task.name,
            seed=seed, asset_dir=target_dir,
        )
        notes.extend(swap_notes)

    # Roles are read from the *vanilla* BDDL: a perturbation must not change the
    # task's goal, and if it did we would want that to surface as a mismatch.
    if perturbed_path:
        perturbed_roles = resolve_entity_roles(Path(perturbed_path).read_text(encoding="utf-8"))
        if (perturbed_roles.source, perturbed_roles.destination) != (roles.source, roles.destination):
            raise RuntimeError(
                f"perturbation changed the goal entities: vanilla "
                f"({roles.source}, {roles.destination}) vs perturbed "
                f"({perturbed_roles.source}, {perturbed_roles.destination}). "
                "That is not a position perturbation."
            )

    checkpoint = SUITE_CHECKPOINTS[suite]
    return OfficialTaskPair(
        suite=suite,
        task_id=task_id,
        task_name=task.name,
        instruction=task.language,
        vanilla_bddl_path=str(vanilla_bddl),
        perturbed_bddl_path=perturbed_path,
        vanilla_bddl_sha256=sha256_file(vanilla_bddl),
        perturbed_bddl_sha256=perturbed_sha,
        perturbation_family=family,
        perturbation_name=condition,
        requested_axis=axis,
        requested_level=level,
        source_entity=roles.source,
        destination_entity=roles.destination,
        primary_analysis_entity=FAMILY_PRIMARY_ENTITY[family],
        model_id=checkpoint["model_id"],
        model_revision=checkpoint["revision"],
        unnorm_key=checkpoint["unnorm_key"],
        seed=seed,
        roles=roles.to_dict(),
        notes=notes,
    )


def assert_checkpoint_matches_suite(suite: str, model_id: str) -> None:
    """Refuse to roll out a suite against another suite's fine-tune."""
    expected = SUITE_CHECKPOINTS.get(suite, {}).get("model_id")
    if expected is None:
        raise KeyError(f"unknown suite {suite!r}")
    if model_id != expected:
        raise RuntimeError(
            f"checkpoint/suite mismatch: suite {suite!r} requires {expected!r}, got {model_id!r}"
        )


def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Resolve official LIBERO/LIBERO-PRO task pairs.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--condition", default=None,
                        help="omit to list every official condition for the suite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.condition is None:
        print(json.dumps(available_conditions(args.suite), indent=2))
        return 0
    pair = resolve_official_task_pair(args.suite, args.task_id, args.condition, seed=args.seed)
    print(json.dumps(pair.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
