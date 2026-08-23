"""Deterministic seeding for action-generalization evaluation.

``episode_seed`` is a physical copy (not a re-export) from
``tools/openvla/official_task_pair_resolver.py`` line 111. That file imports
``entity_role_resolver`` at module top level for unrelated reasons (building
``OfficialTaskPair``s); ``episode_seed`` itself never touches it -- it is pure
``hashlib`` arithmetic on four plain values, verified by re-reading the
function body before copying.

The mechanism (and the exact formula) is unchanged. The docstring below keeps
the original's technical explanation of *why* fixture placement needs a
deterministic seed, but generalises the framing: the source docstring framed
this in terms of the diagnostics line's vanilla/perturbed paired-comparison
study specifically; action-generalization work needs the same deterministic
reset behaviour for its own (unrelated) reason -- reproducible evaluation
episodes across method runs.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def validate_seed(seed: Any) -> int:
    """Require an integer seed rather than accepting Python's ``int`` type object.

    This is the same guard used by the diagnostics task-pair resolver.  In
    particular, LIBERO-PRO's upstream default can be the ``int`` type itself;
    passing that to ``random.seed`` silently makes a process-specific seed.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r} ({type(seed).__name__})")
    return int(seed)


def seed_everything(seed: int) -> int:
    """Seed the Python, NumPy, PyTorch, and CUDA RNGs used by the baseline.

    The diagnostics collector seeds Python/NumPy/PyTorch through its
    ``seed_everything`` helper.  The validated rollout runners additionally
    seed all CUDA devices.  Keeping both steps here makes the action-
    generalization baseline use the same reproducibility policy without an
    import edge into ``tools/openvla``.
    """
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Keeping this helper importable is useful for pure resolver tests.
        pass
    return seed


def episode_seed(base_seed: int, suite: str, task_id: int, init_state_id: int) -> int:
    """A deterministic seed for one episode's environment reset.

    Needed because LIBERO re-samples FIXTURE placements on every ``reset()``, and
    those live in ``sim.model.body_pos`` / ``body_quat`` -- not in ``qpos`` -- so
    they are neither captured by ``sim.get_state()`` nor restored by
    ``set_init_state``. Measured on libero_spatial task 0: successive resets of the
    same env move fixtures by up to 1.5 cm and rotate them by up to 0.011 in
    quaternion distance. Two episodes with a byte-identical ``init_state_sha256``
    differed in 27% of the pixels the policy sees, and the trajectory diverged
    (105 steps and success, versus 177 steps).

    Seeding the global RNG immediately before ``reset()`` makes the placement a
    pure function of these four values, so it no longer depends on how many other
    environments happened to be constructed earlier in the process.

    This is deliberately a function of (base_seed, suite, task_id, init_state_id)
    only -- not of anything downstream of the reset, such as a perturbation
    condition or a method name -- so that two runs sharing those four values
    always draw the same fixture placement and stay directly comparable,
    whatever else differs between them.

    Note this pins something the official evaluation leaves free: official LIBERO
    also re-samples fixtures per reset, so there is no "official" placement to
    match. Pinning it is a deliberate, recorded deviation that buys
    reproducibility and paired comparison.
    """
    payload = f"{base_seed}|{suite}|{task_id}|{init_state_id}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)
