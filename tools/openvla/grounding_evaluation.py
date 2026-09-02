"""Generic, evaluation-only patch-grounding metrics and visualization.

Prediction scores are supplied by a caller (for example, a Transformer
relevance method). Simulator segmentation is used only after those scores have
already been produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from token_layout import patch_index_to_uv


@dataclass(frozen=True)
class PatchGroundingEvaluation:
    valid: bool
    invalid_reason: Optional[str]
    predicted_patch_index: Optional[int]
    gt_centroid_patch_index: Optional[int]
    top1_patch_hit: Optional[bool]
    top_k_patch_hit: Optional[bool]
    gt_overlapping_patch_rank: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def patch_mask_overlap(mask: np.ndarray, num_visual_tokens: int) -> np.ndarray:
    """Return visual patch cells that overlap a boolean GT mask."""
    mask = np.asarray(mask, dtype=bool)
    side = int(round(np.sqrt(num_visual_tokens)))
    if mask.ndim != 2 or side * side != int(num_visual_tokens):
        raise ValueError("expected a 2-D mask and a square visual-token grid")
    height, width = mask.shape
    hits = np.zeros(num_visual_tokens, dtype=bool)
    for row in range(side):
        y0, y1 = int(np.floor(row * height / side)), int(np.floor((row + 1) * height / side))
        for col in range(side):
            x0, x1 = int(np.floor(col * width / side)), int(np.floor((col + 1) * width / side))
            hits[row * side + col] = bool(mask[y0:y1, x0:x1].any())
    return hits


def evaluate_patch_scores(scores: Sequence[float], gt_mask: np.ndarray, top_k: int = 5) -> PatchGroundingEvaluation:
    """Evaluate already-computed patch scores against segmentation GT."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be one finite value per visual token")
    hits = patch_mask_overlap(gt_mask, len(values))
    if not hits.any():
        return PatchGroundingEvaluation(False, "GT target mask is absent", None, None, None, None, None)
    ranked = np.argsort(-values, kind="stable")
    predicted = int(ranked[0])
    overlap_ranks = np.flatnonzero(hits[ranked])
    height, width = np.asarray(gt_mask).shape
    ys, xs = np.where(np.asarray(gt_mask, dtype=bool))
    side = int(round(np.sqrt(len(values))))
    centroid = min(side - 1, int(np.floor(float(ys.mean()) * side / height))) * side + min(side - 1, int(np.floor(float(xs.mean()) * side / width)))
    return PatchGroundingEvaluation(
        True, None, predicted, int(centroid), bool(hits[predicted]),
        bool(hits[ranked[:max(1, int(top_k))]].any()), int(overlap_ranks[0] + 1),
    )


def save_patch_overlay(
    output_path: str, model_input_rgb: np.ndarray, predicted_patch_index: int,
    num_visual_tokens: int, gt_mask: Optional[np.ndarray] = None,
) -> None:
    """Draw a predicted patch and optional **EVAL ONLY** segmentation boundary."""
    image = Image.fromarray(np.asarray(model_input_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    side = int(round(np.sqrt(num_visual_tokens)))
    row, col = divmod(int(predicted_patch_index), side)
    cell_w, cell_h = image.width / side, image.height / side
    draw.rectangle((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h), outline=(255, 0, 0), width=2)
    if gt_mask is not None and np.asarray(gt_mask, dtype=bool).any():
        mask = Image.fromarray(np.asarray(gt_mask, dtype=np.uint8) * 255).resize(image.size)
        draw.bitmap((0, 0), mask, fill=(0, 255, 0))
    u, v = patch_index_to_uv(predicted_patch_index, num_visual_tokens, (image.height, image.width))
    draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 0, 0))
    draw.text((2, 2), "red=prediction; green=EVAL ONLY segmentation GT", fill=(255, 255, 255), font=ImageFont.load_default())
    image.save(output_path)
