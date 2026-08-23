"""Abstract interface a future action-generalization method implements.

This interface deliberately stays small so a non-learning intervention can
share the Phase 4 rollout path with the vanilla baseline.  Implementations
must not import from ``tools/openvla``; they receive an environment and the
current raw observation from the orchestration layer when that is necessary.

A method implementation lives under ``tools/action_generalization/methods/``
and subclasses ``ActionGeneralizationMethod``. ``predict_action`` is the only
required hook. Returning ``None`` (the base-class default) means "defer to
the base OpenVLA policy" -- that is exactly what the Phase 4 baseline run
does: it uses this same interface with no override, so
``tools/common/openvla_model.get_vla_action`` is the effective policy.

Do not import from ``tools/openvla/`` here or in any subclass; use
``tools/common/`` for shared infra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class ActionGeneralizationMethod(ABC):
    """Interface every action-generalization method implements.

    A method wraps (or replaces part of) the base OpenVLA policy. It is
    responsible for its own setup (loading extra weights, building an action
    head, etc.) in ``__init__`` / ``setup``, and for producing an action given
    an observation in ``predict_action``.
    """

    #: Registry name this method is selected by from a config's ``method:`` key.
    name: str = "base"

    def setup(self, config: Dict[str, Any]) -> None:
        """Optional one-time setup (load weights, build modules, ...).

        Default is a no-op, which is correct for a method that only wraps the
        base policy and needs no extra state.
        """
        return None

    def begin_episode(self, **kwargs: Any) -> None:
        """Receive the live environment after reset / initial-state restore."""
        return None

    def episode_summary(self) -> Dict[str, Any]:
        """Return JSON-safe method-specific episode facts for metadata."""
        return {}

    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        task_label: str,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        """Return an action for ``observation``, or ``None`` to defer to the base policy.

        ``observation`` follows the same convention as
        ``tools/common/openvla_model.get_vla_action``'s ``obs`` argument: a
        dict with at least ``"full_image"`` (uint8 HxWx3) and ``"state"``
        (proprioceptive vector). Returning ``None`` means "this method has no
        override for this step, run the base OpenVLA policy instead" -- the
        mechanism the Phase 4 vanilla baseline uses so it goes through this
        same interface without a separate no-method code path.  An override is
        an *already normalized, directly executable* LIBERO action.  Thus the
        eval loop only applies OpenVLA's gripper conversion when this method
        returns ``None``.
        """
        raise NotImplementedError


class NoOverrideMethod(ActionGeneralizationMethod):
    """The vanilla baseline: always defers to the base OpenVLA policy.

    Used by ``configs/baseline.yaml``. This is the only concrete method
    implemented in this phase; every other method name is deliberately
    unimplemented (see ``train.py`` / ``eval.py``'s registry).
    """

    name = "none"

    def predict_action(
        self,
        observation: Dict[str, Any],
        task_label: str,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        return None
