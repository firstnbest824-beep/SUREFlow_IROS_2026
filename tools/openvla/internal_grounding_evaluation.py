"""Evaluation-only segmentation metrics for internal OpenVLA grounding maps.

This module is intentionally separate from ``internal_grounding``.  Its mask
arguments are never accepted by prediction functions, making simulator GT a
post-prediction evaluation concern rather than a possible prediction input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from internal_grounding import GroundingMapResult


@dataclass(frozen=True)
class GroundingEvaluation:
    valid: bool
    invalid_reason: Optional[str]
    gt_mask_pixels: int
    gt_centroid_uv: Optional[Tuple[float, float]]
    gt_bbox_xyxy: Optional[Tuple[float, float, float, float]]
    pixel_l2_error: Optional[float]
    top1_patch_hit: Optional[bool]
    top_k_patch_hit: Optional[bool]
    gt_overlapping_patch_rank: Optional[int]
    gt_centroid_patch_index: Optional[int]
    gt_overlapping_patch_indices: Optional[Tuple[int, ...]]
    predicted_to_gt_patch_distance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "gt_mask_pixels": self.gt_mask_pixels,
            "gt_centroid_uv": None if self.gt_centroid_uv is None else list(self.gt_centroid_uv),
            "gt_bbox_xyxy": None if self.gt_bbox_xyxy is None else list(self.gt_bbox_xyxy),
            "pixel_l2_error": self.pixel_l2_error,
            "top1_patch_hit": self.top1_patch_hit,
            "top_k_patch_hit": self.top_k_patch_hit,
            "gt_overlapping_patch_rank": self.gt_overlapping_patch_rank,
            "gt_centroid_patch_index": self.gt_centroid_patch_index,
            "gt_overlapping_patch_indices": (
                None if self.gt_overlapping_patch_indices is None
                else list(self.gt_overlapping_patch_indices)
            ),
            "predicted_to_gt_patch_distance": self.predicted_to_gt_patch_distance,
            "ground_truth_note": "EVAL ONLY: simulator segmentation is not provided to prediction code",
        }


def patch_mask_overlap(mask: np.ndarray, num_visual_tokens: int) -> np.ndarray:
    """Return which visual patch cells overlap a model-input boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D GT mask, got {mask.shape}")
    side = int(round(np.sqrt(num_visual_tokens)))
    if side * side != int(num_visual_tokens):
        raise ValueError(f"visual token count {num_visual_tokens} is not square")
    height, width = mask.shape
    hits = np.zeros(num_visual_tokens, dtype=bool)
    for row in range(side):
        y0, y1 = int(np.floor(row * height / side)), int(np.floor((row + 1) * height / side))
        for col in range(side):
            x0, x1 = int(np.floor(col * width / side)), int(np.floor((col + 1) * width / side))
            hits[row * side + col] = bool(mask[y0:y1, x0:x1].any())
    return hits


def evaluate_grounding_prediction(
    prediction: GroundingMapResult, gt_mask: np.ndarray, min_mask_pixels: int,
) -> GroundingEvaluation:
    """Compare a completed prediction against GT in the same model-input frame."""
    mask = np.asarray(gt_mask, dtype=bool)
    count = int(mask.sum())
    if count < int(min_mask_pixels):
        return GroundingEvaluation(
            False, "GT target mask absent or below min_mask_pixels", count,
            None, None, None, None, None, None, None, None, None,
        )
    ys, xs = np.where(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    hits = patch_mask_overlap(mask, prediction.num_visual_tokens)
    overlap_indices = np.flatnonzero(hits)
    ranked = np.argsort(-prediction.scores, kind="stable")
    overlap_ranks = np.where(hits[ranked])[0]
    best_rank = None if overlap_ranks.size == 0 else int(overlap_ranks[0] + 1)
    predicted_uv = np.asarray(prediction.predicted_uv_model_input, dtype=np.float64)
    rows, cols = prediction.patch_grid_shape
    centroid_col = min(cols - 1, int(np.floor(centroid[0] * cols / mask.shape[1])))
    centroid_row = min(rows - 1, int(np.floor(centroid[1] * rows / mask.shape[0])))
    centroid_patch = centroid_row * cols + centroid_col
    predicted_row, predicted_col = divmod(prediction.predicted_patch_index, cols)
    overlap_rc = np.asarray([divmod(int(index), cols) for index in overlap_indices], dtype=np.float64)
    patch_distance = float(np.linalg.norm(overlap_rc - [predicted_row, predicted_col], axis=1).min())
    return GroundingEvaluation(
        True, None, count, centroid, bbox, float(np.linalg.norm(predicted_uv - np.asarray(centroid))),
        bool(hits[prediction.predicted_patch_index]),
        bool(hits[list(prediction.top_k_patch_indices)].any()), best_rank,
        int(centroid_patch), tuple(int(index) for index in overlap_indices), patch_distance,
    )


def save_grounding_overlay(
    output_path: str, model_input_rgb: np.ndarray, prediction: GroundingMapResult,
    evaluation: GroundingEvaluation,
) -> None:
    """Write an overlay whose GT annotations are visibly labelled EVAL ONLY."""
    image = Image.fromarray(np.asarray(model_input_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    height, width = image.height, image.width
    rows, cols = prediction.patch_grid_shape
    cell_w, cell_h = width / cols, height / rows
    row, col = prediction.predicted_patch_row, prediction.predicted_patch_col
    draw.rectangle((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h), outline=(255, 0, 0), width=2)
    u, v = prediction.predicted_uv_model_input
    draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 0, 0))
    if evaluation.valid and evaluation.gt_centroid_uv is not None and evaluation.gt_bbox_xyxy is not None:
        x0, y0, x1, y1 = evaluation.gt_bbox_xyxy
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)
        gu, gv = evaluation.gt_centroid_uv
        draw.ellipse((gu - 3, gv - 3, gu + 3, gv + 3), fill=(0, 255, 0))
    draw.rectangle((0, 0, width, 15), fill=(0, 0, 0))
    draw.text((2, 2), "red=prediction; green=EVAL ONLY segmentation GT", fill=(255, 255, 255), font=ImageFont.load_default())
    image.save(output_path)
