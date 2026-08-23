"""Frozen perception front-ends and RGB-D geometry for action-generalization."""

from .target_localizer import GroundingDinoTargetLocalizer, LocalizationResult, VisionLocalizationError

__all__ = ["GroundingDinoTargetLocalizer", "LocalizationResult", "VisionLocalizationError"]
