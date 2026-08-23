"""Frozen language-conditioned 2-D grounding; no simulator-state access."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


class VisionLocalizationError(RuntimeError):
    """Raised when the perception front-end cannot safely choose a target."""


@dataclass(frozen=True)
class LocalizationResult:
    target_phrase: str
    bbox_xyxy: Tuple[float, float, float, float]
    target_pixel_uv: Tuple[int, int]
    confidence: float
    model_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_source_phrase(instruction: str) -> str:
    """Extract the picked source noun phrase, not the placement destination.

    The initial POC supports LIBERO's ``pick up X and place it in Y`` wording.
    It intentionally fails closed for unrelated instructions rather than
    guessing a destination or a simulator entity name.
    """
    match = re.match(
        r"^\s*(?:pick up|pick|grab)\s+(?:the\s+)?(.+?)\s+and\s+(?:place|put)\s+it\s+(?:in|on|into|onto)\s+.+\s*$",
        instruction,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise VisionLocalizationError(
            "unsupported instruction for source extraction; expected 'pick up X and place it in/on Y': "
            f"{instruction!r}"
        )
    return match.group(1).strip()


class GroundingDinoTargetLocalizer:
    """External pretrained Grounding DINO front-end, used without fine-tuning."""

    def __init__(
        self, model_id: str, confidence_threshold: float, text_threshold: float,
        device: str = "cuda", cache_dir: str | None = None,
    ) -> None:
        import torch
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.model_id = model_id
        self.confidence_threshold = float(confidence_threshold)
        self.text_threshold = float(text_threshold)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.processor = GroundingDinoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True
        ).to(self.device).eval()

    @staticmethod
    def select_highest_confidence(
        target_phrase: str, boxes: np.ndarray, scores: np.ndarray, model_id: str,
    ) -> LocalizationResult:
        if len(boxes) == 0:
            raise VisionLocalizationError(f"no grounding detection for target phrase {target_phrase!r}")
        index = int(np.argmax(scores))
        x0, y0, x1, y1 = [float(value) for value in boxes[index]]
        if not (x1 > x0 and y1 > y0):
            raise VisionLocalizationError(f"invalid detector box {(x0, y0, x1, y1)!r}")
        return LocalizationResult(
            target_phrase=target_phrase,
            bbox_xyxy=(x0, y0, x1, y1),
            target_pixel_uv=(int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))),
            confidence=float(scores[index]),
            model_id=model_id,
        )

    def localize(self, rgb: np.ndarray, instruction: str) -> LocalizationResult:
        """Ground the source phrase in RGB; does not accept depth or environment data."""
        import torch

        target_phrase = extract_source_phrase(instruction)
        inputs = self.processor(images=np.asarray(rgb), text=target_phrase + ".", return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        height, width = np.asarray(rgb).shape[:2]
        processed = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], box_threshold=self.confidence_threshold,
            text_threshold=self.text_threshold, target_sizes=[(height, width)],
        )[0]
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        result = self.select_highest_confidence(target_phrase, boxes, scores, self.model_id)
        u = min(max(result.target_pixel_uv[0], 0), width - 1)
        v = min(max(result.target_pixel_uv[1], 0), height - 1)
        return LocalizationResult(result.target_phrase, result.bbox_xyxy, (u, v), result.confidence, result.model_id)
