"""CPU-only structural contracts retained for future Transformer relevance."""

import os
import sys

import numpy as np

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "openvla")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from grounding_evaluation import evaluate_patch_scores, patch_mask_overlap  # noqa: E402
from token_layout import patch_index_to_uv, resolve_multimodal_layout  # noqa: E402


def test_multimodal_layout_and_patch_centers_are_model_structure_not_a_readout():
    layout = resolve_multimodal_layout([[1, 2, 3]], np.zeros((1, 259, 4)), 256)
    assert (layout.visual_start, layout.visual_end) == (1, 257)
    assert layout.llm_position_for_input_token(1) == 257
    assert patch_index_to_uv(0, 256) == (7.0, 7.0)


def test_segmentation_is_only_used_after_patch_scores_exist():
    mask = np.zeros((224, 224), dtype=bool)
    mask[:14, :14] = True
    scores = np.zeros(256)
    scores[0] = 1.0
    evaluation = evaluate_patch_scores(scores, mask)
    assert patch_mask_overlap(mask, 256)[0]
    assert evaluation.valid and evaluation.top1_patch_hit and evaluation.top_k_patch_hit
