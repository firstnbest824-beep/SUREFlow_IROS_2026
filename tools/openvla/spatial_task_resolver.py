"""Shared source/destination/swap-counterpart resolver for LIBERO spatial tasks.

Both the OpenVLA probe dry-run and the failure-screening runner need to agree on
*which* object is being analysed. Previously they disagreed: the probe picked the
single movable source of the BDDL goal (``akita_black_bowl_1``) while the
failure-screening runner hard-coded ``akita_black_bowl_2`` (which is not the task
object at all -- it is the object that the LIBERO-PRO position perturbation swaps
with the destination plate).

This module is the single source of truth. Nothing here hard-codes an object
name; everything is derived from the BDDL file:

* ``source_object``      -- first argument of the binary ``On``/``In`` goal
                            predicate, restricted to movable ``(:objects ...)``.
* ``destination_object`` -- second argument of that same goal predicate.
* ``goal_predicate``     -- ``On`` or ``In``.
* ``swap_pairs`` / ``perturbation_moved_entities`` / ``swap_counterpart``
                         -- obtained by running the LIBERO-PRO ``SwapPerturbator``
                            deterministically in memory and diffing the ``(:init ...)``
                            regions of the original and perturbed BDDL text.

Research-stage semantics are attached as well:

* ``pre_grasp_relevant_entity``  == ``source_object``
* ``post_grasp_relevant_entity`` == ``destination_object``

Automatic task-phase detection is intentionally *not* implemented here; the two
roles are simply kept separate so that a later phase classifier can use them.
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LIBERO_PRO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO-PRO")
)

BINARY_GOAL_PREDICATES = {"on", "in"}

DEFAULT_OOD_SPATIAL_CONFIG = os.path.join(
    _LIBERO_PRO_ROOT, "libero_ood", "ood_spatial_relation.yaml"
)


# -----------------------------------------------------------------------------
# Minimal BDDL parsing (balanced-paren block extraction + predicate regex)
# -----------------------------------------------------------------------------
def _balanced_block(text: str, header: str) -> Optional[str]:
    """Return the ``(header ...)`` s-expression including its closing paren."""
    start = text.find(header)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_typed_names(text: str, header: str) -> List[str]:
    """Parse ``(:objects a b - type\n c - type)`` style blocks into a name list."""
    block = _balanced_block(text, header)
    if block is None:
        return []
    inner = block[len(header) : -1]
    names: List[str] = []
    for line in inner.splitlines():
        entry = line.strip()
        if not entry:
            continue
        if " - " in entry:
            entry = entry.split(" - ", 1)[0]
        names.extend(entry.split())
    return names


def _parse_plain_names(text: str, header: str) -> List[str]:
    block = _balanced_block(text, header)
    if block is None:
        return []
    inner = block[len(header) : -1]
    return [token for token in inner.split() if token not in {"(", ")"}]


_BINARY_PREDICATE_RE = re.compile(r"\(\s*(\w+)\s+([\w\-]+)\s+([\w\-]+)\s*\)")


def _parse_binary_predicates(block: Optional[str]) -> List[Tuple[str, str, str]]:
    if not block:
        return []
    found: List[Tuple[str, str, str]] = []
    for predicate, first, second in _BINARY_PREDICATE_RE.findall(block):
        found.append((predicate, first, second))
    return found


def parse_init_regions(text: str) -> Dict[str, str]:
    """Map every entity in ``(:init ...)`` to the region it is placed on/in."""
    block = _balanced_block(text, "(:init")
    regions: Dict[str, str] = {}
    for predicate, entity, region in _parse_binary_predicates(block):
        if predicate.lower() in BINARY_GOAL_PREDICATES:
            regions[entity] = region
    return regions


def parse_goal_relations(text: str) -> List[Tuple[str, str, str]]:
    """Return every binary ``On``/``In`` relation inside ``(:goal ...)``."""
    block = _balanced_block(text, "(:goal")
    return [
        (predicate, first, second)
        for predicate, first, second in _parse_binary_predicates(block)
        if predicate.lower() in BINARY_GOAL_PREDICATES
    ]


# -----------------------------------------------------------------------------
# Resolver result
# -----------------------------------------------------------------------------
@dataclass
class SpatialTaskEntities:
    bddl_path: Optional[str]
    task_suite: Optional[str]
    task_name: Optional[str]

    source_object: str
    destination_object: str
    goal_predicate: str
    goal_relations: List[List[str]]

    movable_objects: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    obj_of_interest: List[str] = field(default_factory=list)
    init_regions: Dict[str, str] = field(default_factory=dict)

    # Research-stage semantics (phase classification itself is out of scope here).
    pre_grasp_relevant_entity: str = ""
    post_grasp_relevant_entity: str = ""

    # LIBERO-PRO position (swap) perturbation.
    swap_config_path: Optional[str] = None
    swap_perturbation_seed: Optional[int] = None
    swap_config_candidates: Dict[str, List[str]] = field(default_factory=dict)
    swap_pairs: List[List[str]] = field(default_factory=list)
    perturbation_moved_entities: List[str] = field(default_factory=list)
    perturbed_init_regions: Dict[str, str] = field(default_factory=dict)
    swap_counterpart: Optional[str] = None
    swap_counterpart_partner_of: Optional[str] = None
    swap_resolved: bool = False
    swap_note: Optional[str] = None

    @property
    def tracked_entities(self) -> List[str]:
        """Entities whose world pose must be logged for every rollout."""
        ordered = [self.source_object, self.destination_object]
        if self.swap_counterpart and self.swap_counterpart not in ordered:
            ordered.append(self.swap_counterpart)
        for entity in self.perturbation_moved_entities:
            if entity not in ordered:
                ordered.append(entity)
        return ordered

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tracked_entities"] = self.tracked_entities
        return payload

    def summary_lines(self) -> List[str]:
        return [
            f"source_object      = {self.source_object}",
            f"destination_object = {self.destination_object}",
            f"goal_predicate     = {self.goal_predicate}",
            f"swap_counterpart   = {self.swap_counterpart}",
            f"swap_pairs         = {self.swap_pairs}",
            f"moved_entities     = {self.perturbation_moved_entities}",
            f"pre_grasp_entity   = {self.pre_grasp_relevant_entity}",
            f"post_grasp_entity  = {self.post_grasp_relevant_entity}",
        ]


# -----------------------------------------------------------------------------
# Goal resolution
# -----------------------------------------------------------------------------
def resolve_source_destination(text: str) -> Tuple[str, str, str, List[Tuple[str, str, str]]]:
    """Pick the single movable ``(On|In) source destination`` goal relation."""
    movable = set(_parse_typed_names(text, "(:objects"))
    relations = parse_goal_relations(text)
    if not relations:
        raise RuntimeError("No binary On/In relation found in the BDDL (:goal ...) block.")

    movable_relations = [rel for rel in relations if rel[1] in movable]
    if len(movable_relations) != 1:
        raise RuntimeError(
            "Expected exactly one binary In/On goal whose first argument is a movable "
            f"object, found {len(movable_relations)}: {movable_relations}. "
            f"All goal relations: {relations}"
        )

    predicate, source, destination = movable_relations[0]
    return source, destination, predicate, relations


# -----------------------------------------------------------------------------
# Swap perturbation resolution
# -----------------------------------------------------------------------------
def _load_swap_perturbator():
    if _LIBERO_PRO_ROOT not in sys.path:
        sys.path.insert(0, _LIBERO_PRO_ROOT)
    from perturbation import BDDLParser, SwapPerturbator  # type: ignore

    return BDDLParser, SwapPerturbator


def apply_swap_perturbation(
    bddl_text: str,
    task_suite: str,
    task_name: str,
    config_path: str,
    seed: int,
) -> str:
    """Run the LIBERO-PRO SwapPerturbator deterministically and return new BDDL text."""
    BDDLParser, SwapPerturbator = _load_swap_perturbator()
    random.seed(seed)
    parser = BDDLParser(bddl_text)
    perturbator = SwapPerturbator(parser, config_path)
    return perturbator.perturb(task_suite_name=task_suite, task_name=task_name)


def diff_init_regions(
    original: Dict[str, str], perturbed: Dict[str, str]
) -> Tuple[List[str], List[List[str]]]:
    """Return (moved entities, swap pairs) between two ``(:init ...)`` region maps."""
    moved = sorted(
        entity
        for entity in original
        if entity in perturbed and original[entity] != perturbed[entity]
    )
    pairs: List[List[str]] = []
    consumed: set = set()
    for entity in moved:
        if entity in consumed:
            continue
        for other in moved:
            if other == entity or other in consumed:
                continue
            if perturbed[entity] == original[other] and perturbed[other] == original[entity]:
                pairs.append(sorted([entity, other]))
                consumed.add(entity)
                consumed.add(other)
                break
    return moved, pairs


def _read_swap_config_candidates(
    config_path: str, task_suite: str, task_name: str
) -> Dict[str, List[str]]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a LIBERO dependency
        return {}
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    task_cfg = (config.get(task_suite) or {}).get(task_name)
    if isinstance(task_cfg, dict):
        return {key: list(value) for key, value in task_cfg.items() if isinstance(value, list)}
    if isinstance(task_cfg, list):
        return {"__any__": list(task_cfg)}
    return {}


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------
def resolve_spatial_task(
    bddl_path: str,
    task_suite: Optional[str] = None,
    task_name: Optional[str] = None,
    ood_config_path: Optional[str] = DEFAULT_OOD_SPATIAL_CONFIG,
    perturbation_seed: int = 0,
    resolve_swap: bool = True,
) -> SpatialTaskEntities:
    """Resolve source/destination/swap-counterpart for one LIBERO BDDL task."""
    with open(bddl_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    source, destination, predicate, relations = resolve_source_destination(text)
    entities = SpatialTaskEntities(
        bddl_path=str(bddl_path),
        task_suite=task_suite,
        task_name=task_name,
        source_object=source,
        destination_object=destination,
        goal_predicate=predicate,
        goal_relations=[list(rel) for rel in relations],
        movable_objects=_parse_typed_names(text, "(:objects"),
        fixtures=_parse_typed_names(text, "(:fixtures"),
        obj_of_interest=_parse_plain_names(text, "(:obj_of_interest"),
        init_regions=parse_init_regions(text),
        pre_grasp_relevant_entity=source,
        post_grasp_relevant_entity=destination,
    )

    if not resolve_swap:
        entities.swap_note = "swap resolution disabled by caller"
        return entities

    if not (task_suite and task_name and ood_config_path):
        entities.swap_note = "task_suite/task_name/ood_config_path not provided"
        return entities

    entities.swap_config_path = ood_config_path
    entities.swap_perturbation_seed = perturbation_seed
    entities.swap_config_candidates = _read_swap_config_candidates(
        ood_config_path, task_suite, task_name
    )

    try:
        perturbed_text = apply_swap_perturbation(
            bddl_text=text,
            task_suite=task_suite,
            task_name=task_name,
            config_path=ood_config_path,
            seed=perturbation_seed,
        )
    except Exception as exc:  # keep the resolver usable without LIBERO-PRO deps
        entities.swap_note = f"swap perturbation unavailable: {exc}"
        return entities

    perturbed_regions = parse_init_regions(perturbed_text)
    moved, pairs = diff_init_regions(entities.init_regions, perturbed_regions)

    entities.perturbed_init_regions = perturbed_regions
    entities.perturbation_moved_entities = moved
    entities.swap_pairs = pairs
    entities.swap_resolved = True

    counterpart, partner_of = _select_swap_counterpart(pairs, source, destination)
    entities.swap_counterpart = counterpart
    entities.swap_counterpart_partner_of = partner_of
    if counterpart is None:
        entities.swap_note = (
            "no swap pair involves the source or destination object; "
            f"moved entities were {moved}"
        )
    return entities


def _select_swap_counterpart(
    pairs: List[List[str]], source: str, destination: str
) -> Tuple[Optional[str], Optional[str]]:
    """The entity that the position perturbation exchanges with the task object."""
    for role_name, role_entity in (("destination_object", destination), ("source_object", source)):
        for pair in pairs:
            if role_entity in pair:
                other = pair[0] if pair[1] == role_entity else pair[1]
                return other, role_name
    return None, None


def resolve_spatial_task_from_env(
    env: Any,
    bddl_path: str,
    task_suite: Optional[str] = None,
    task_name: Optional[str] = None,
    ood_config_path: Optional[str] = DEFAULT_OOD_SPATIAL_CONFIG,
    perturbation_seed: int = 0,
) -> SpatialTaskEntities:
    """Resolve from the BDDL file and cross-check against the live environment.

    The BDDL text is authoritative; the environment check only guards against a
    task/env mismatch (e.g. a perturbed BDDL paired with a vanilla env).
    """
    entities = resolve_spatial_task(
        bddl_path=bddl_path,
        task_suite=task_suite,
        task_name=task_name,
        ood_config_path=ood_config_path,
        perturbation_seed=perturbation_seed,
    )

    base_env = unwrap_base_env(env)
    parsed = getattr(base_env, "parsed_problem", None)
    if parsed:
        env_goals = [
            (str(state[0]), str(state[1]), str(state[2]))
            for state in parsed.get("goal_state", [])
            if len(state) == 3 and str(state[0]).lower() in BINARY_GOAL_PREDICATES
        ]
        env_pair = {(src, dst) for _, src, dst in env_goals}
        if env_pair and (entities.source_object, entities.destination_object) not in env_pair:
            raise RuntimeError(
                "Resolver/environment mismatch: BDDL goal is "
                f"({entities.source_object}, {entities.destination_object}) but the live "
                f"environment reports {sorted(env_pair)}."
            )
    return entities


# -----------------------------------------------------------------------------
# Environment helpers shared by both runners
# -----------------------------------------------------------------------------
def unwrap_base_env(env: Any) -> Any:
    current = env
    seen: set = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def get_entity_world_position(env: Any, entity: str) -> Optional[List[float]]:
    """World position of a BDDL entity, or ``None`` if it cannot be resolved."""
    base_env = unwrap_base_env(env)
    states = getattr(base_env, "object_states_dict", {})
    if entity in states:
        try:
            pos = states[entity].get_geom_state()["pos"]
            return [float(value) for value in pos]
        except Exception:
            pass
    try:
        sim = env.sim
        body_names = [name for name in sim.model.body_names if name.startswith(entity)]
        if body_names:
            body_id = sim.model.body_name2id(body_names[0])
            return [float(value) for value in sim.data.body_xpos[body_id]]
    except Exception:
        pass
    return None


def get_entity_world_positions(env: Any, entities: List[str]) -> Dict[str, Optional[List[float]]]:
    return {entity: get_entity_world_position(env, entity) for entity in entities}


def displacement(initial: Optional[List[float]], final: Optional[List[float]]) -> Optional[Dict[str, Any]]:
    if initial is None or final is None:
        return None
    delta = [float(f) - float(i) for i, f in zip(initial, final)]
    norm = float(sum(value * value for value in delta) ** 0.5)
    return {"delta": delta, "norm": norm}


# -----------------------------------------------------------------------------
# CLI (self-check without loading any model)
# -----------------------------------------------------------------------------
def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Resolve LIBERO source/destination/swap entities.")
    parser.add_argument("--bddl_path", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--ood_config_path", type=str, default=DEFAULT_OOD_SPATIAL_CONFIG)
    parser.add_argument("--perturbation_seed", type=int, default=0)
    args = parser.parse_args()

    task_name = args.task_name or os.path.splitext(os.path.basename(args.bddl_path))[0]
    entities = resolve_spatial_task(
        bddl_path=args.bddl_path,
        task_suite=args.task_suite,
        task_name=task_name,
        ood_config_path=args.ood_config_path,
        perturbation_seed=args.perturbation_seed,
    )
    print(json.dumps(entities.to_dict(), indent=2))
    for line in entities.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
