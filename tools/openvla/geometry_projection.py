"""Method-neutral RGB-D pixel ↔ world-coordinate geometry utilities.

These helpers use camera calibration only.  They do not inspect segmentation,
choose a target object, or issue robot actions, so a grounding method can use
them after it has independently selected an image location.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np


def pixel_depth_to_world(
    pixel_uv: Tuple[int, int], depth_m: float, intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project a raw-image pixel and metric optical depth into world XYZ."""
    u, v = pixel_uv
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("expected 3x3 intrinsic and 4x4 camera_to_world matrices")
    if not np.isfinite(depth_m) or depth_m <= 0 or intrinsic[0, 0] == 0 or intrinsic[1, 1] == 0:
        raise ValueError("depth must be finite/positive and focal lengths must be non-zero")
    point_camera = np.array([
        (float(u) - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
        (float(v) - intrinsic[1, 2]) * depth_m / intrinsic[1, 1],
        depth_m,
        1.0,
    ])
    world = camera_to_world @ point_camera
    if not np.isfinite(world).all() or world[3] == 0:
        raise ValueError("camera projection produced an invalid homogeneous world point")
    return world[:3] / world[3]


def world_to_pixel(
    world_xyz: Sequence[float], intrinsic: np.ndarray, camera_to_world: np.ndarray,
) -> Tuple[int, int]:
    """Project world XYZ into the raw camera image for evaluation/debug only."""
    xyz = np.asarray(world_xyz, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if xyz.shape != (3,) or intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("expected XYZ, 3x3 intrinsic, and 4x4 camera_to_world")
    camera = np.linalg.inv(camera_to_world) @ np.append(xyz, 1.0)
    if not np.isfinite(camera).all() or camera[2] <= 0:
        raise ValueError("world point is behind the camera or invalid")
    return (
        int(round(intrinsic[0, 0] * camera[0] / camera[2] + intrinsic[0, 2])),
        int(round(intrinsic[1, 1] * camera[1] / camera[2] + intrinsic[1, 2])),
    )


def robust_depth_at_pixel(depth_map: np.ndarray, pixel_uv: Tuple[int, int], radius: int = 2) -> float:
    """Return a local median of valid positive metric-depth values."""
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or radius < 0:
        raise ValueError("depth_map must be 2-D and radius non-negative")
    u, v = pixel_uv
    y0, y1 = max(v - radius, 0), min(v + radius + 1, depth.shape[0])
    x0, x1 = max(u - radius, 0), min(u + radius + 1, depth.shape[1])
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("no valid depth near pixel {!r}".format(pixel_uv))
    return float(np.median(values))


def project_observation_pixel_to_world(
    sim: Any, camera_name: str, image: np.ndarray, depth_map: np.ndarray,
    pixel_uv: Tuple[int, int],
) -> np.ndarray:
    """Use robosuite calibration APIs to convert a LIBERO RGB-D pixel to XYZ."""
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
