"""RGB-D pixel projection using robosuite's official camera calibration APIs."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def pixel_depth_to_world(
    pixel_uv: Tuple[int, int], depth_m: float, intrinsic: np.ndarray, camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project a raw-image pixel and metric optical depth into world XYZ."""
    u, v = pixel_uv
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError(f"depth must be finite and positive, got {depth_m!r}")
    point_camera = np.array([
        (float(u) - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
        (float(v) - intrinsic[1, 2]) * depth_m / intrinsic[1, 1],
        depth_m,
        1.0,
    ])
    return (camera_to_world @ point_camera)[:3]


def world_to_pixel(world_xyz: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray) -> Tuple[int, int]:
    """Project a world point for evaluation-only debug overlays."""
    camera = np.linalg.inv(camera_to_world) @ np.append(np.asarray(world_xyz, dtype=np.float64), 1.0)
    if camera[2] <= 0:
        raise ValueError("world point is behind the camera")
    return (
        int(round(intrinsic[0, 0] * camera[0] / camera[2] + intrinsic[0, 2])),
        int(round(intrinsic[1, 1] * camera[1] / camera[2] + intrinsic[1, 2])),
    )


def robust_depth_at_pixel(depth_map: np.ndarray, pixel_uv: Tuple[int, int], radius: int = 2) -> float:
    """Use a small center patch median, not a segmentation-derived object mask."""
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    u, v = pixel_uv
    y0, y1 = max(v - radius, 0), min(v + radius + 1, depth.shape[0])
    x0, x1 = max(u - radius, 0), min(u + radius + 1, depth.shape[1])
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError(f"no valid depth near pixel {pixel_uv!r}")
    return float(np.median(values))


def project_observation_pixel_to_world(sim, camera_name: str, image: np.ndarray, depth_map: np.ndarray, pixel_uv: Tuple[int, int]) -> np.ndarray:
    """Convert LIBERO's normalized depth observation to world XYZ officially."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    height, width = np.asarray(image).shape[:2]
    intrinsic = get_camera_intrinsic_matrix(sim, camera_name, height, width)
    camera_to_world = get_camera_extrinsic_matrix(sim, camera_name)
    metric_depth = get_real_depth_map(sim, np.asarray(depth_map).squeeze())
    return pixel_depth_to_world(pixel_uv, robust_depth_at_pixel(metric_depth, pixel_uv), intrinsic, camera_to_world)
