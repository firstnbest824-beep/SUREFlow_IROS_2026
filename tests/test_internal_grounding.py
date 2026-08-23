"""CPU-only contracts for the frozen OpenVLA internal-grounding probe."""

import inspect
import os
import re
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_ROOT, "tools", "common"), os.path.join(_ROOT, "tools", "openvla"), os.path.join(_ROOT, "tools", "action_generalization")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from instruction_target import InstructionTargetError, extract_source_phrase  # noqa: E402
from internal_grounding import (  # noqa: E402
    InternalGroundingError, compute_cosine_grounding_map, locate_target_token_span,
    patch_index_to_uv, resolve_multimodal_layout,
)
from internal_grounding_evaluation import evaluate_grounding_prediction  # noqa: E402
from openvla_model import (  # noqa: E402
    build_openvla_prompt, get_vla_action, prepare_openvla_inputs, tensor_to_numpy_for_artifact,
)
from perception.target_localizer import extract_source_phrase as detector_extract_source_phrase  # noqa: E402
from run_internal_grounding_repeat import (  # noqa: E402
    aggregate_metric_rows, metric_rows_from_artifacts,
)
from target_pose_selection import (  # noqa: E402
    TargetPoseCandidate, TargetPoseSelectionError, select_distinct_target_poses,
)
from run_internal_grounding_swap import _gt_position_range  # noqa: E402


class _Tokenizer:
    def __init__(self, offsets=True):
        self.offsets = offsets
        self.vocab = {"<bos>": 1}
        self.reverse = {1: "<bos>"}

    def _id(self, token):
        if token not in self.vocab:
            index = len(self.vocab) + 10
            self.vocab[token] = index
            self.reverse[index] = token
        return self.vocab[token]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        if return_offsets_mapping and not self.offsets:
            raise NotImplementedError("offsets unavailable")
        matches = list(re.finditer(r"[A-Za-z]+|[^\w\s]", text))
        ids = [self._id(match.group(0).lower()) for match in matches]
        positions = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            ids, positions = [1] + ids, [(0, 0)] + positions
        result = {"input_ids": np.asarray([ids], dtype=np.int64)}
        if return_offsets_mapping:
            result["offset_mapping"] = np.asarray([positions], dtype=np.int64)
        return result

    def convert_ids_to_tokens(self, ids):
        return [self.reverse[int(value)] for value in ids]


def _prompt_and_layout(tokenizer=None):
    tokenizer = tokenizer or _Tokenizer()
    prompt = "In: What action should the robot take to pick up the alphabet soup and place it in the basket?\nOut:"
    ids = tokenizer(prompt)["input_ids"]
    hidden = np.zeros((1, ids.shape[1] + 256, 4), dtype=np.float32)
    return tokenizer, prompt, ids, resolve_multimodal_layout(ids, hidden, 256), hidden


def test_source_phrase_and_grounding_dino_shared_parser():
    instruction = "pick up the alphabet soup and place it in the basket"
    assert extract_source_phrase(instruction) == "alphabet soup"
    assert detector_extract_source_phrase(instruction) == "alphabet soup"
    with pytest.raises(InstructionTargetError):
        extract_source_phrase("move the alphabet soup")


def test_multimodal_layout_maps_bos_visual_and_text_without_hardcoding():
    tokenizer, _, ids, layout, _ = _prompt_and_layout()
    assert layout.visual_start == 1
    assert layout.visual_end == 257
    assert layout.llm_position_for_input_token(0) == 0
    assert layout.llm_position_for_input_token(1) == 257
    assert layout.llm_position_for_input_token(ids.shape[1] - 1) == ids.shape[1] - 1 + 256
    with pytest.raises(InternalGroundingError, match="does not match"):
        resolve_multimodal_layout(ids, np.zeros((1, ids.shape[1] + 255, 4)), 256)


def test_target_span_uses_unique_offsets_and_llm_mapping():
    tokenizer, prompt, ids, layout, _ = _prompt_and_layout()
    span = locate_target_token_span(tokenizer, prompt, "alphabet soup", ids, layout)
    assert span.resolution_method == "offset_mapping"
    assert span.decoded_tokens == ("alphabet", "soup")
    assert span.llm_token_positions == tuple(index + 256 for index in span.input_token_indices)


def test_target_span_fallback_and_ambiguous_input_fail_closed():
    tokenizer, prompt, ids, layout, _ = _prompt_and_layout(_Tokenizer(offsets=False))
    span = locate_target_token_span(tokenizer, prompt, "alphabet soup", ids, layout)
    assert span.resolution_method == "token_id_subsequence"
    duplicate_ids = np.concatenate((ids, ids[:, list(span.input_token_indices)]), axis=1)
    duplicate_hidden = np.zeros((1, duplicate_ids.shape[1] + 256, 4), dtype=np.float32)
    duplicate_layout = resolve_multimodal_layout(duplicate_ids, duplicate_hidden, 256)
    with pytest.raises(InternalGroundingError, match="exactly one"):
        locate_target_token_span(tokenizer, prompt, "alphabet soup", duplicate_ids, duplicate_layout)


def test_cosine_map_finds_known_patch_and_patch_centers():
    tokenizer, prompt, ids, layout, hidden = _prompt_and_layout()
    span = locate_target_token_span(tokenizer, prompt, "alphabet soup", ids, layout)
    hidden[0, layout.visual_start:layout.visual_end] = [0, 1, 0, 0]
    target_position = span.llm_token_positions[-1]
    hidden[0, target_position] = [1, 0, 0, 0]
    hidden[0, layout.visual_start + 37] = [5, 0, 0, 0]
    result = compute_cosine_grounding_map("llm_late", hidden, layout, span)
    assert result.predicted_patch_index == 37
    assert result.patch_grid_shape == (16, 16)
    assert result.scores.shape == (256,)
    assert patch_index_to_uv(0, 256) == (7.0, 7.0)
    assert patch_index_to_uv(255, 256) == (217.0, 217.0)


def test_evaluation_metrics_are_post_prediction_and_patch_aligned():
    tokenizer, prompt, ids, layout, hidden = _prompt_and_layout()
    span = locate_target_token_span(tokenizer, prompt, "alphabet soup", ids, layout)
    hidden[0, layout.visual_start:layout.visual_end] = [0, 1, 0, 0]
    hidden[0, span.llm_token_positions[-1]] = [1, 0, 0, 0]
    hidden[0, layout.visual_start + 0] = [2, 0, 0, 0]
    result = compute_cosine_grounding_map("llm_early", hidden, layout, span)
    mask = np.zeros((224, 224), dtype=bool)
    mask[:14, :14] = True
    metrics = evaluate_grounding_prediction(result, mask, min_mask_pixels=10)
    assert metrics.valid and metrics.top1_patch_hit and metrics.top_k_patch_hit
    assert metrics.gt_overlapping_patch_rank == 1
    assert metrics.gt_centroid_patch_index == 0
    assert metrics.gt_overlapping_patch_indices == (0,)
    assert metrics.predicted_to_gt_patch_distance == pytest.approx(0.0)
    # GT pixel centers are integer-indexed (mean 6.5), whereas the patch
    # representative is the continuous cell center (7.0).
    assert metrics.pixel_l2_error == pytest.approx(np.sqrt(0.5))


def test_prediction_api_has_no_environment_or_ground_truth_parameters():
    parameters = inspect.signature(compute_cosine_grounding_map).parameters
    forbidden = {"env", "segmentation", "mask", "gt_mask", "target_object", "depth"}
    assert not forbidden.intersection(parameters)
    source = inspect.getsource(compute_cosine_grounding_map)
    assert "segmentation" not in source and "gt_mask" not in source


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, image):
        self.calls.append((prompt, np.asarray(image)))
        return {
            "input_ids": torch.tensor([[42, 43]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2)),
        }


class _FakeVLA:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return np.arange(7, dtype=np.float32)


def test_openvla_prompt_preparation_refactor_preserves_action_path():
    vla, processor = _FakeVLA(), _FakeProcessor()
    obs = {"full_image": np.full((4, 4, 3), 127, dtype=np.uint8)}
    expected_prompt = build_openvla_prompt("some-openvla", "Pick up the alphabet soup")
    prepared = prepare_openvla_inputs(vla, processor, "some-openvla", obs, "Pick up the alphabet soup", False, torch.float32)
    assert prepared.prompt == expected_prompt
    assert prepared.input_ids.tolist() == [[42, 43, 29871]]
    assert prepared.attention_mask.tolist() == [[1, 1, 1]]
    action, image, returned = get_vla_action(
        vla, processor, "some-openvla", obs, "Pick up the alphabet soup", "libero_object", False,
        torch.float32, return_model_input_image=True, return_prepared_inputs=True,
    )
    assert np.array_equal(action, np.arange(7, dtype=np.float32))
    assert np.array_equal(image, obs["full_image"])
    assert returned.prompt == expected_prompt
    assert vla.kwargs["input_ids"].tolist() == [[42, 43, 29871]]


def test_tensor_artifact_conversion_handles_bfloat16_and_preserves_integer_dtype():
    floating = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    serialized_floating = tensor_to_numpy_for_artifact(floating)
    assert serialized_floating.dtype == np.float32
    assert np.allclose(serialized_floating, [1.25, -2.5])

    ids = torch.tensor([[1, 29871]], dtype=torch.int64)
    serialized_ids = tensor_to_numpy_for_artifact(ids)
    assert serialized_ids.dtype == np.int64
    assert np.array_equal(serialized_ids, [[1, 29871]])


def test_repeat_metrics_preserve_each_sample_and_aggregate_per_layer():
    prediction = {"stages": {"llm_late": {
        "predicted_patch_index": 0, "predicted_uv_model_input": [91.0, 35.0],
    }}}
    evaluation_a = {"stages": {"llm_late": {
        "gt_centroid_uv": [90.0, 34.0], "pixel_l2_error": 1.0,
        "gt_overlapping_patch_rank": 1, "top1_patch_hit": True,
        "top_k_patch_hit": True, "valid": True,
    }}}
    evaluation_b = {"stages": {"llm_late": {
        "gt_centroid_uv": [100.0, 50.0], "pixel_l2_error": 3.0,
        "gt_overlapping_patch_rank": 5, "top1_patch_hit": False,
        "top_k_patch_hit": True, "valid": True,
    }}}
    rows = metric_rows_from_artifacts("sample_a", 0, 0, [0.1, 0.2, 0.3], prediction, evaluation_a)
    rows += metric_rows_from_artifacts("sample_b", 1, 1, [0.2, 0.2, 0.3], prediction, evaluation_b)
    summary = aggregate_metric_rows(rows)["llm_late"]
    assert [(row["seed"], row["init_state_id"]) for row in rows] == [(0, 0), (1, 1)]
    assert summary["pixel_l2_mean"] == pytest.approx(2.0)
    assert summary["pixel_l2_std"] == pytest.approx(1.0)
    assert summary["gt_overlapping_patch_rank_mean"] == pytest.approx(3.0)
    assert summary["top1_success_rate"] == pytest.approx(0.5)
    assert summary["top5_success_rate"] == pytest.approx(1.0)
    assert rows[0]["target_x"] == pytest.approx(0.1)
    assert rows[0]["grounding_argmax_visual_token_index"] == 0


def test_target_pose_selector_keeps_only_distinct_target_xyz_and_fails_without_variation():
    candidates = [
        TargetPoseCandidate(0, (0.1, 0.2, 0.3)),
        TargetPoseCandidate(1, (0.1 + 1e-8, 0.2, 0.3)),
        TargetPoseCandidate(2, (0.2, 0.2, 0.3)),
    ]
    selected = select_distinct_target_poses(candidates, tolerance_m=1e-6, max_samples=2)
    assert [candidate.init_state_id for candidate in selected] == [0, 2]
    with pytest.raises(TargetPoseSelectionError, match="variation is absent"):
        select_distinct_target_poses(candidates[:2], tolerance_m=1e-6, max_samples=2)


def test_swap_gt_position_range_requires_multiple_distinct_valid_centroids():
    rows = [
        {"sample_id": "swap_0", "valid": True, "gt_centroid_u": 10.0, "gt_centroid_v": 20.0},
        {"sample_id": "swap_0", "valid": True, "gt_centroid_u": 10.0, "gt_centroid_v": 20.0},
        {"sample_id": "swap_1", "valid": True, "gt_centroid_u": 30.0, "gt_centroid_v": 40.0},
    ]
    assert _gt_position_range(rows) == {"u": [10.0, 30.0], "v": [20.0, 40.0]}
    with pytest.raises(RuntimeError, match="no GT centroid variation"):
        _gt_position_range([rows[0], {**rows[0], "sample_id": "swap_same"}])
