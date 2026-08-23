#!/usr/bin/env python3
"""Single training entry point for action-generalization methods.

Skeleton only. This does not implement any method's training loop -- Phase 5
adds the first one (action-head redesign, flow matching, chunking,
Mamba/SSM, ...). What this script does right now:

1. Parse ``--config <path>``.
2. Load that YAML config and read its ``method:`` key.
3. Resolve the method by name from ``METHOD_REGISTRY`` below.
4. If the name is not in the registry (true for every name today, since no
   trainable method exists yet), raise a clear ``NotImplementedError`` instead
   of silently doing nothing.

Do not import from ``tools/openvla/``; use ``tools/common/`` for shared infra.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Type

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from methods.base import ActionGeneralizationMethod  # noqa: E402

#: method name (from a config's `method:` key) -> implementation class.
#: Empty today: no method has a training procedure yet. `methods.base
#: .NoOverrideMethod` is deliberately excluded here (it has no learnable
#: state, so "training" it is not a meaningful operation) and is only
#: registered in `eval.py` for the Phase 4 vanilla-baseline rollout.
METHOD_REGISTRY: Dict[str, Type[ActionGeneralizationMethod]] = {}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} did not parse to a dict: {type(config)}")
    return config


def resolve_method(method_name: str) -> Type[ActionGeneralizationMethod]:
    """Look up a method class by name, raising a clear error if unimplemented."""
    if method_name not in METHOD_REGISTRY:
        raise NotImplementedError(
            f"method {method_name!r} is not implemented for training. "
            f"Registered training methods: {sorted(METHOD_REGISTRY.keys()) or '(none yet)'}. "
            "This is expected in the skeleton phase -- Phase 5 adds the first "
            "trainable method."
        )
    return METHOD_REGISTRY[method_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an action-generalization method")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config naming the method to train (see configs/).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    method_name = config.get("method")
    if method_name is None:
        raise ValueError(f"config at {args.config} is missing a top-level 'method' key")

    print(f"[*] Loaded config: {args.config}")
    print(f"[*] Requested method: {method_name}")

    method_cls = resolve_method(method_name)  # raises NotImplementedError today
    method = method_cls()
    method.setup(config)
    raise NotImplementedError(
        "train.py has no training loop yet -- Phase 5 adds it alongside the "
        "first implemented method."
    )


if __name__ == "__main__":
    sys.exit(main())
