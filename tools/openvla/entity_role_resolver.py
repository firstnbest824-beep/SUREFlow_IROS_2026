"""Suite-independent source / destination / distractor role assignment.

Nothing here is keyed on a suite name or an object name. Roles come from the
BDDL itself:

* ``source``      -- first argument of the single binary ``On``/``In`` goal
                     predicate whose first argument is a movable object.
* ``destination`` -- second argument of that predicate, **normalised from a
                     region back to the object that owns it**.
* ``distractor``  -- any other movable object placed in ``(:init ...)``.
* ``fixture``     -- anything declared under ``(:fixtures ...)``.

The region normalisation matters. ``libero_spatial`` goals read
``(On akita_black_bowl_1 plate_1)`` -- the destination is already an object.
``libero_object`` goals read ``(In alphabet_soup_1 basket_1_contain_region)``
-- the destination is a *region*, and the object that actually moved during a
perturbation is ``basket_1``. Comparing poses of ``basket_1_contain_region``
would silently find nothing, so the destination is resolved to ``basket_1``.

Region names in LIBERO are ``{owner}_{region}`` where ``owner`` is the
``(:target ...)`` of the region declaration, so the mapping is recovered by
parsing ``(:regions ...)`` rather than by string heuristics.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spatial_task_resolver import (
    _balanced_block,
    _parse_typed_names,
    parse_init_regions,
    resolve_source_destination,
)

ROLE_SOURCE = "source"
ROLE_DESTINATION = "destination"
ROLE_DISTRACTOR = "distractor"
ROLE_FIXTURE = "fixture"
ROLE_UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Region -> owning object
# -----------------------------------------------------------------------------
_REGION_HEAD_RE = re.compile(r"\(\s*([A-Za-z_][\w]*)\s*\(\s*:target\s+([\w]+)\s*\)")


def parse_region_owners(bddl_text: str) -> Dict[str, str]:
    """Map fully-qualified region name -> the entity that owns it.

    ``(contain_region (:target basket_1) ...)`` inside ``(:regions ...)`` yields
    ``{"basket_1_contain_region": "basket_1"}``. Regions targeting a fixture
    (e.g. ``main_table``) are included too, so callers can tell "this is a spot
    on the table" from "this is a spot on a movable object".
    """
    block = _balanced_block(bddl_text, "(:regions")
    if not block:
        return {}
    owners: Dict[str, str] = {}
    for region_name, target in _REGION_HEAD_RE.findall(block):
        owners[f"{target}_{region_name}"] = target
        # LIBERO also refers to regions by their bare name in some places.
        owners.setdefault(region_name, target)
    return owners


def normalise_entity(name: str, region_owners: Dict[str, str], movable: List[str]) -> str:
    """Resolve a goal argument to the entity whose pose actually moves.

    Returns ``name`` unchanged when it is already a declared object.
    """
    if name in movable:
        return name
    owner = region_owners.get(name)
    if owner:
        return owner
    # `<object>_<something>_region` without a matching declaration: fall back to
    # the longest declared object that prefixes the name.
    candidates = [obj for obj in movable if name.startswith(obj + "_")]
    if candidates:
        return max(candidates, key=len)
    return name


# -----------------------------------------------------------------------------
# Roles
# -----------------------------------------------------------------------------
@dataclass
class EntityRoles:
    source: str
    destination: str
    goal_predicate: str
    #: Destination exactly as written in the goal, before region normalisation.
    destination_goal_argument: str
    destination_was_region: bool
    distractors: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    movable_objects: List[str] = field(default_factory=list)
    obj_of_interest: List[str] = field(default_factory=list)
    region_owners: Dict[str, str] = field(default_factory=dict)

    def role_of(self, entity: str) -> str:
        if entity == self.source:
            return ROLE_SOURCE
        if entity == self.destination:
            return ROLE_DESTINATION
        if entity in self.distractors:
            return ROLE_DISTRACTOR
        if entity in self.fixtures:
            return ROLE_FIXTURE
        return ROLE_UNKNOWN

    @property
    def tracked_entities(self) -> List[str]:
        """Every entity whose pose is worth recording, source/destination first."""
        ordered = [self.source, self.destination]
        for entity in self.distractors + self.fixtures:
            if entity not in ordered:
                ordered.append(entity)
        return ordered

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tracked_entities"] = self.tracked_entities
        return payload


def resolve_entity_roles(bddl_text: str) -> EntityRoles:
    """Derive every role from one BDDL. Raises if the goal is not interpretable."""
    source, destination_arg, predicate, _ = resolve_source_destination(bddl_text)

    movable = _parse_typed_names(bddl_text, "(:objects")
    fixtures = _parse_typed_names(bddl_text, "(:fixtures")
    region_owners = parse_region_owners(bddl_text)

    destination = normalise_entity(destination_arg, region_owners, movable)
    if destination not in movable and destination not in fixtures:
        raise RuntimeError(
            f"goal destination {destination_arg!r} normalised to {destination!r}, "
            f"which is neither a declared object {movable} nor a fixture {fixtures}. "
            "Refusing to guess."
        )

    placed = list(parse_init_regions(bddl_text).keys())
    distractors = [
        entity for entity in movable
        if entity not in (source, destination) and entity in placed
    ]

    return EntityRoles(
        source=source,
        destination=destination,
        goal_predicate=predicate,
        destination_goal_argument=destination_arg,
        destination_was_region=destination != destination_arg,
        distractors=distractors,
        fixtures=fixtures,
        movable_objects=movable,
        obj_of_interest=_parse_plain_obj_of_interest(bddl_text),
        region_owners=region_owners,
    )


def _parse_plain_obj_of_interest(bddl_text: str) -> List[str]:
    block = _balanced_block(bddl_text, "(:obj_of_interest")
    if not block:
        return []
    inner = block[len("(:obj_of_interest") : -1]
    return [tok for tok in inner.split() if tok not in {"(", ")"}]


def resolve_entity_roles_from_path(bddl_path: str) -> EntityRoles:
    with open(bddl_path, "r", encoding="utf-8") as handle:
        return resolve_entity_roles(handle.read())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Resolve entity roles from a BDDL file.")
    parser.add_argument("--bddl_path", required=True)
    args = parser.parse_args()
    roles = resolve_entity_roles_from_path(args.bddl_path)
    print(json.dumps(roles.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
