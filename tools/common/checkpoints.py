"""Registry of OpenVLA checkpoint ids and pinned revisions, by LIBERO suite.

New file -- no diagnostics-line source to copy from. The two entries below are
transcribed from values confirmed elsewhere in the repo:

- ``openvla-7b-finetuned-libero-spatial`` @ ``962318cec55ac10993ff0f5f43eda9a270b4c873``
  -- ``tools/openvla/official_task_pair_resolver.py`` line 60 (``SUITE_CHECKPOINTS``),
  also ``tools/openvla/run_single_vanilla_rollout.py`` line 63 (``DEFAULT_REVISION``).
- ``openvla-7b-finetuned-libero-object`` @ ``287d6cfdf12d07b1449505f66d9bf3550257e9b3``
  -- ``tools/openvla/official_task_pair_resolver.py`` line 65 (``SUITE_CHECKPOINTS``).

Pinning the revision matters: the un-normalisation statistics used to decode
actions live in the checkpoint's ``config.json`` at that specific commit. See
``tools/common/openvla_model.py``'s ``load_openvla`` docstring/comment for the
caveat about where norm stats actually come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CheckpointSpec:
    model_id: str
    revision: str
    unnorm_key: str


#: suite name -> checkpoint fine-tuned on that suite.
SUITE_CHECKPOINTS: Dict[str, CheckpointSpec] = {
    "libero_spatial": CheckpointSpec(
        model_id="openvla/openvla-7b-finetuned-libero-spatial",
        revision="962318cec55ac10993ff0f5f43eda9a270b4c873",
        unnorm_key="libero_spatial",
    ),
    "libero_object": CheckpointSpec(
        model_id="openvla/openvla-7b-finetuned-libero-object",
        revision="287d6cfdf12d07b1449505f66d9bf3550257e9b3",
        unnorm_key="libero_object",
    ),
}


def get_checkpoint(suite: str) -> CheckpointSpec:
    """Look up the checkpoint spec for a suite; raises KeyError with the valid options."""
    try:
        return SUITE_CHECKPOINTS[suite]
    except KeyError as exc:
        raise KeyError(
            f"no checkpoint registered for suite {suite!r}; known suites: "
            f"{sorted(SUITE_CHECKPOINTS.keys())}"
        ) from exc
