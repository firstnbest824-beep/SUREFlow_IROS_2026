"""Shared geometry between the LIBERO simulator image space and OpenVLA's input.

OpenVLA never sees the raw simulator frame. The rollout code applies

    1. 180-degree rotation of the ``agentview`` frame  (``img[::-1, ::-1]``)
    2. resize to 224x224
    3. center crop with ``crop_scale = 0.9``
    4. resize back to 224x224

before handing the image to the processor. Segmentation masks and target
centroids, however, used to be computed in the *raw* 256x256 simulator frame, so
features and spatial labels lived in two different coordinate systems.

This module is the single implementation of that chain. ``rgb_to_model_input``
and ``mask_to_model_input`` share the exact same rotation, resize target and crop
box; the only difference is the interpolation filter (LANCZOS/BILINEAR for RGB,
NEAREST for masks, so mask ids are never blended).

``get_libero_image`` and ``apply_center_crop`` are re-exported here so the rollout
scripts and the label pipeline provably use one code path.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

DEFAULT_RESIZE_SIZE = 224
DEFAULT_CROP_SCALE = 0.9
MODEL_INPUT_CAMERA_KEY = "agentview_image"


# -----------------------------------------------------------------------------
# Primitive steps
# -----------------------------------------------------------------------------
def rotate_180(array: np.ndarray) -> np.ndarray:
    """The rotation the rollout code applies via ``img[::-1, ::-1]``."""
    return np.ascontiguousarray(np.asarray(array)[::-1, ::-1])


def pil_jpeg_encode_decode(img: np.ndarray, quality: int = 95) -> np.ndarray:
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.array(decoded)


def resize_image(img: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """RGB resize used by the rollout code (JPEG round-trip + LANCZOS)."""
    assert isinstance(resize_size, tuple) and len(resize_size) == 2
    img = pil_jpeg_encode_decode(img)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((resize_size[1], resize_size[0]), Image.LANCZOS)
    img = np.array(pil_img)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return img


def center_crop_box(
    height: int, width: int, crop_scale: float = DEFAULT_CROP_SCALE
) -> Tuple[int, int, int, int]:
    """Return ``(top, left, crop_height, crop_width)`` -- identical for RGB and masks."""
    new_h = int(height * math.sqrt(crop_scale))
    new_w = int(width * math.sqrt(crop_scale))
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    return top, left, new_h, new_w


def apply_center_crop(
    image: Image.Image,
    crop_scale: float = DEFAULT_CROP_SCALE,
    output_size: Tuple[int, int] = (DEFAULT_RESIZE_SIZE, DEFAULT_RESIZE_SIZE),
) -> Image.Image:
    """RGB center crop exactly as performed right before ``processor(prompt, image)``."""
    img_np = np.array(image).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    top, left, new_h, new_w = center_crop_box(h, w, crop_scale)
    cropped = img_np[top : top + new_h, left : left + new_w]
    cropped_uint8 = (np.clip(cropped, 0.0, 1.0) * 255).astype(np.uint8)
    pil_cropped = Image.fromarray(cropped_uint8)
    pil_cropped = pil_cropped.resize((output_size[1], output_size[0]), Image.BILINEAR)
    return pil_cropped


def get_libero_image(
    obs: Dict[str, Any],
    resize_size: int,
    camera_key: str = MODEL_INPUT_CAMERA_KEY,
) -> np.ndarray:
    """Steps 1-2 of the chain: rotate the agentview frame and resize it."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[camera_key]
    img = img[::-1, ::-1]
    img = resize_image(img, resize_size)
    return img


# -----------------------------------------------------------------------------
# Full chain: raw simulator frame -> OpenVLA model input
# -----------------------------------------------------------------------------
def rgb_to_model_input(
    rgb: np.ndarray,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    center_crop: bool = True,
    crop_scale: float = DEFAULT_CROP_SCALE,
) -> np.ndarray:
    """Raw simulator RGB -> the exact uint8 array OpenVLA's processor receives."""
    rotated = rotate_180(rgb)
    resized = resize_image(rotated, (resize_size, resize_size))
    if not center_crop:
        return resized
    cropped = apply_center_crop(
        Image.fromarray(resized).convert("RGB"),
        crop_scale=crop_scale,
        output_size=(resize_size, resize_size),
    )
    return np.array(cropped)


def mask_to_model_input(
    mask: np.ndarray,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    center_crop: bool = True,
    crop_scale: float = DEFAULT_CROP_SCALE,
) -> np.ndarray:
    """Raw simulator boolean mask -> OpenVLA model-input frame, NEAREST only."""
    rotated = rotate_180(np.asarray(mask).astype(bool)).astype(np.uint8)
    pil_mask = Image.fromarray(rotated, mode="L")
    pil_mask = pil_mask.resize((resize_size, resize_size), Image.NEAREST)
    if center_crop:
        top, left, new_h, new_w = center_crop_box(resize_size, resize_size, crop_scale)
        pil_mask = pil_mask.crop((left, top, left + new_w, top + new_h))
        pil_mask = pil_mask.resize((resize_size, resize_size), Image.NEAREST)
    return np.asarray(pil_mask).astype(bool)


def map_uv_raw_to_model_input(
    u: float,
    v: float,
    raw_shape: Tuple[int, int],
    resize_size: int = DEFAULT_RESIZE_SIZE,
    center_crop: bool = True,
    crop_scale: float = DEFAULT_CROP_SCALE,
) -> Tuple[float, float]:
    """Analytic point mapping, used only to cross-check the mask-derived centroid."""
    height, width = raw_shape
    u_rot = (width - 1) - float(u)
    v_rot = (height - 1) - float(v)

    scale_x = resize_size / float(width)
    scale_y = resize_size / float(height)
    u_res = (u_rot + 0.5) * scale_x - 0.5
    v_res = (v_rot + 0.5) * scale_y - 0.5
    if not center_crop:
        return u_res, v_res

    top, left, new_h, new_w = center_crop_box(resize_size, resize_size, crop_scale)
    u_crop = u_res - left
    v_crop = v_res - top
    u_out = (u_crop + 0.5) * (resize_size / float(new_w)) - 0.5
    v_out = (v_crop + 0.5) * (resize_size / float(new_h)) - 0.5
    return u_out, v_out


def describe_transform(
    raw_shape: Sequence[int],
    resize_size: int = DEFAULT_RESIZE_SIZE,
    center_crop: bool = True,
    crop_scale: float = DEFAULT_CROP_SCALE,
) -> Dict[str, Any]:
    """Machine-readable description of the chain, stored alongside every label."""
    top, left, new_h, new_w = center_crop_box(resize_size, resize_size, crop_scale)
    steps: List[Dict[str, Any]] = [
        {"step": "rotate_180", "rgb_filter": "exact", "mask_filter": "exact"},
        {
            "step": "resize",
            "output_size": [resize_size, resize_size],
            "rgb_filter": "jpeg_q95+LANCZOS",
            "mask_filter": "NEAREST",
        },
    ]
    if center_crop:
        steps.append(
            {
                "step": "center_crop",
                "crop_scale": crop_scale,
                "box_top_left_h_w": [top, left, new_h, new_w],
                "rgb_filter": "exact",
                "mask_filter": "exact",
            }
        )
        steps.append(
            {
                "step": "resize",
                "output_size": [resize_size, resize_size],
                "rgb_filter": "BILINEAR",
                "mask_filter": "NEAREST",
            }
        )
    return {
        "raw_shape": [int(value) for value in raw_shape],
        "model_input_shape": [resize_size, resize_size],
        "center_crop": bool(center_crop),
        "crop_scale": crop_scale,
        "camera_fed_to_model": MODEL_INPUT_CAMERA_KEY,
        "steps": steps,
    }


# -----------------------------------------------------------------------------
# Label computation in either coordinate frame
# -----------------------------------------------------------------------------
def mask_statistics(mask: np.ndarray, min_mask_pixels: int) -> Dict[str, Any]:
    """Pixel count, visibility, centroid, normalized centroid and bbox for a mask.

    Masks below ``min_mask_pixels`` are *not* dropped: the record is still
    returned with ``visible=False`` and ``below_min_mask_pixels=True`` so the
    caller can log the occlusion instead of silently discarding the sample.
    """
    mask = np.asarray(mask).astype(bool)
    height, width = mask.shape[:2]
    ys, xs = np.where(mask)
    count = int(mask.sum())
    visible = count >= int(min_mask_pixels)
    if count > 0:
        u = float(xs.mean())
        v = float(ys.mean())
        centroid = [u, v]
        centroid_norm = [u / float(width), v / float(height)]
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    else:
        centroid = None
        centroid_norm = None
        bbox = None
    return {
        "shape": [height, width],
        "mask_pixel_count": count,
        "visible": bool(visible),
        "below_min_mask_pixels": bool(count < int(min_mask_pixels)),
        "min_mask_pixels": int(min_mask_pixels),
        "centroid": centroid,
        "centroid_normalized": centroid_norm,
        "bbox": bbox,
    }


def normalize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(segmentation))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation image after squeeze, got {array.shape}")
    return array.astype(np.int64, copy=False)
