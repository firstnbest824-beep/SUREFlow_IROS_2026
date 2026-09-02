"""CPU-only structural contracts retained for future Transformer relevance."""

import os
import sys

import numpy as np
import pytest

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "openvla")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from geometry_projection import pixel_depth_to_world, robust_depth_at_pixel, world_to_pixel  # noqa: E402
from grounding_evaluation import evaluate_patch_scores, patch_mask_overlap  # noqa: E402
from token_layout import (  # noqa: E402
    TokenLayoutError,
    locate_target_token_span,
    patch_index_to_uv,
    resolve_multimodal_layout,
)


class _TokenizerWithoutOffsets:
    def __init__(self):
        self.vocab = {"<bos>": 1, "pick": 2, "up": 3, "apple": 4}

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if return_offsets_mapping:
            raise NotImplementedError
        tokens = text.strip().lower().split()
        ids = [self.vocab[token] for token in tokens]
        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids
        return {"input_ids": np.asarray([ids], dtype=np.int64)}

    def convert_ids_to_tokens(self, ids):
        reverse = {value: key for key, value in self.vocab.items()}
        return [reverse[value] for value in ids]


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
    assert evaluation.gt_mask_pixels == 196
    assert evaluation.pixel_l2_error == pytest.approx(np.sqrt(0.5))
    assert evaluation.predicted_to_gt_patch_distance == pytest.approx(0.0)


def test_evaluation_marks_tiny_masks_unavailable_without_using_prediction_method_details():
    scores = np.zeros(256)
    mask = np.zeros((224, 224), dtype=bool)
    mask[0, 0] = True
    evaluation = evaluate_patch_scores(scores, mask, min_mask_pixels=2)
    assert not evaluation.valid
    assert evaluation.invalid_reason == "GT target mask absent or below min_mask_pixels"
    assert evaluation.gt_mask_pixels == 1


def test_token_id_fallback_requires_exactly_one_target_span():
    tokenizer = _TokenizerWithoutOffsets()
    prompt = "pick up apple"
    input_ids = tokenizer(prompt)["input_ids"]
    layout = resolve_multimodal_layout(input_ids, np.zeros((1, 260, 4)), 256)
    span = locate_target_token_span(tokenizer, prompt, "apple", input_ids, layout)
    assert span.resolution_method == "token_id_subsequence"
    assert span.input_token_indices == (3,)
    duplicate = np.asarray([[1, 2, 3, 4, 4]], dtype=np.int64)
    duplicate_layout = resolve_multimodal_layout(duplicate, np.zeros((1, 261, 4)), 256)
    with pytest.raises(TokenLayoutError, match="exactly one"):
        locate_target_token_span(tokenizer, "pick up apple", "apple", duplicate, duplicate_layout)


def test_rgbd_geometry_round_trip_and_robust_depth_are_method_neutral():
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4)
    world = pixel_depth_to_world((60, 45), 2.0, intrinsic, camera_to_world)
    assert world == pytest.approx([0.2, 0.1, 2.0])
    assert world_to_pixel(world, intrinsic, camera_to_world) == (60, 45)
    depth = np.array([[1.0, np.nan, 3.0], [0.0, 2.0, 4.0]])
    assert robust_depth_at_pixel(depth, (1, 0), radius=1) == pytest.approx(2.5)
