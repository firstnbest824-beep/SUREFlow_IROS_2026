"""Ground-truth spatial labels (mask / UV / visibility) for arbitrary entities.

Scope: **analysis labels only.** Nothing here is an input to OpenVLA. The policy
still receives exactly the RGB that ``get_libero_image`` produces from the
unmodified ``OffScreenRenderEnv`` observation.

Why no ``SegmentationRenderEnv``
--------------------------------
Switching env class would change the object the rollout runs on, so it has to be
justified rather than assumed. Measured on this stack (robosuite 1.4.0,
libero_spatial task 0, 256x256):

* ``sim.render(..., segmentation=True)`` on the *existing* ``OffScreenRenderEnv``
  returns a valid ``(H, W, 2)`` int32 buffer.
* An agentview RGB render taken before and after two segmentation renders is
  **bit-identical** (max abs diff 0), and ``sim.data.qpos`` is unchanged.

So segmentation is obtained as an extra read-only render at the *same* simulator
state. No ``env.step()``, no state write, no env class change, and therefore
nothing for a rollout-equivalence test to catch. ``verify_segmentation_equivalence``
re-runs that check on demand.

Frame conventions (the part that silently corrupts labels if you get it wrong)
------------------------------------------------------------------------------
Three different pixel frames are in play::

    raw      = sim.render(...)                  # what obs["agentview_image"] is
    upright  = raw[::-1]                        # robosuite camera_utils convention
    policy   = raw[::-1, ::-1]                  # what get_libero_image feeds OpenVLA

``policy`` is a 180 degree rotation, not a vertical flip, so it differs from
``upright`` by a *horizontal* flip as well::

    row_policy = row_upright
    col_policy = (W - 1) - col_upright

Verified empirically: ``project_points_from_world_to_camera`` returns ``upright``
coordinates (raising a world point by +0.30 m moved its row from 142 to 53, i.e.
up the image). Every label below is reported in **both** frames, with the
normalised ``uv`` field in the policy frame, because that is the frame in which
"the object is on the left" means what the analysis needs it to mean.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: MuJoCo mjtObj value for a geom; segmentation channel 0 carries the object type.
MJ_OBJ_GEOM = 5

DEFAULT_CAMERAS: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")

#: Camera whose frame the policy actually consumes.
POLICY_CAMERA = "agentview"

VISIBILITY_VISIBLE = "visible"
VISIBILITY_OCCLUDED = "occluded"
VISIBILITY_OUT_OF_VIEW = "out_of_view"
VISIBILITY_UNKNOWN = "unknown"

#: A mask this small is treated as noise rather than a sighting.
MIN_VISIBLE_PIXELS = 1


# -----------------------------------------------------------------------------
# Frame conversion
# -----------------------------------------------------------------------------
def upright_to_policy(row: float, col: float, height: int, width: int) -> Tuple[float, float]:
    """robosuite projection frame -> the frame OpenVLA sees."""
    return float(row), float((width - 1) - col)


def project_world_point(
    xyz: Sequence[float], world_to_pixel: np.ndarray
) -> Tuple[float, float, float]:
    """Project one world point, WITHOUT clipping, returning (row, col, depth).

    ``robosuite.project_points_from_world_to_camera`` clips row/col into the
    image and rounds to int, and discards the homogeneous depth. That makes any
    downstream ``0 <= row < height`` test a tautology: a point ten metres off the
    left edge lands on column 0, and a point *behind* the camera lands on the
    principal point -- dead centre. Both then read as "in frame". Keeping the
    unclipped value and the depth is what lets out-of-view be told from occluded.

    Coordinates are in robosuite's upright frame, matching what the clipped
    helper returns for in-frame points.
    """
    homogeneous = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0])
    projected = np.asarray(world_to_pixel) @ homogeneous
    depth = float(projected[2])
    if abs(depth) < 1e-12:
        return float("nan"), float("nan"), depth
    # robosuite swaps axes: row comes from the second component, col from the first.
    return float(projected[1] / depth), float(projected[0] / depth), depth


def raw_to_policy_image(image: np.ndarray) -> np.ndarray:
    """Rotate a raw render into the policy frame (same op as get_libero_image)."""
    return np.asarray(image)[::-1, ::-1]


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
@dataclass
class EntityCameraLabel:
    entity: str
    camera: str
    mask_pixel_count: int
    mask_fraction: float
    #: Projected object origin, policy frame, pixels.
    uv_pixel: Optional[List[float]]
    #: Projected object origin, policy frame, normalised to [0, 1] (u=col, v=row).
    uv: Optional[List[float]]
    #: Same point in robosuite's upright frame, kept so the transform is auditable.
    uv_pixel_upright: Optional[List[float]]
    #: Centroid of the visible mask, policy frame, normalised. None when unseen.
    mask_centroid: Optional[List[float]]
    #: (u_min, v_min, u_max, v_max), policy frame, normalised. None when unseen.
    mask_bbox: Optional[List[float]]
    in_frame: bool
    visibility: str
    depth_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepSegmentationLabels:
    """All entity labels at one simulator state, plus the packed label maps."""

    entities: Dict[str, Dict[str, EntityCameraLabel]] = field(default_factory=dict)
    label_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    legend: Dict[str, int] = field(default_factory=dict)
    cameras: List[str] = field(default_factory=list)

    def metrics(self) -> Dict[str, Any]:
        """JSON-serialisable per-entity labels (label maps excluded)."""
        return {
            entity: {camera: label.to_dict() for camera, label in per_camera.items()}
            for entity, per_camera in self.entities.items()
        }

    def label_of(self, entity: str, camera: str = POLICY_CAMERA) -> Optional[EntityCameraLabel]:
        return self.entities.get(entity, {}).get(camera)


# -----------------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------------
class SegmentationLabelProvider:
    """Reads masks and projections for named entities at the current sim state.

    The provider never advances or mutates the simulator. It is safe to call at
    any point between ``env.reset()`` and ``env.step()``.
    """

    def __init__(
        self,
        env: Any,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        height: int = 256,
        width: int = 256,
        min_visible_pixels: int = MIN_VISIBLE_PIXELS,
    ) -> None:
        self.env = env
        self.cameras = list(cameras)
        self.height = int(height)
        self.width = int(width)
        self.min_visible_pixels = int(min_visible_pixels)
        self._geom_to_body: Dict[int, str] = {}
        self.refresh_bindings()

    @property
    def sim(self) -> Any:
        """Resolved on every access.

        ``env.reset()`` rebuilds the MuJoCo model and frees the previous
        ``MjSim``; a cached handle turns into an object whose ``.model`` has been
        deleted. Looking it up each time costs an attribute walk and removes a
        whole class of stale-handle bugs.
        """
        return _unwrap_sim(self.env)

    # -- setup ----------------------------------------------------------------
    def refresh_bindings(self) -> None:
        """Rebuild geom->body and camera matrices. Call after every env.reset()."""
        model = self.sim.model
        self._geom_to_body = {}
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            name = model.body_id2name(body_id)
            if name:
                self._geom_to_body[geom_id] = name

    def _transform(self, camera: str) -> np.ndarray:
        """World -> pixel matrix at the CURRENT simulator state. Never cached.

        ``get_camera_transform_matrix`` reads ``sim.data.cam_xpos`` /
        ``sim.data.cam_xmat``, which are live state. ``robot0_eye_in_hand`` is
        mounted on the wrist, so the matrix changes every timestep. Caching it
        per episode froze every wrist projection at the t=0 camera pose while the
        wrist *mask* kept tracking the real camera -- the two stopped sharing a
        frame, and a static object's wrist uv stayed bit-identical for a whole
        episode while its mask swept 246 px across a 256 px image.
        """
        from robosuite.utils import camera_utils

        return camera_utils.get_camera_transform_matrix(
            sim=self.sim, camera_name=camera, camera_height=self.height, camera_width=self.width
        )

    def geom_ids_for(self, entity: str) -> List[int]:
        """Geoms belonging to an entity, matched on body-name boundary.

        LIBERO names object bodies ``<entity>_main`` (plus ``<entity>_base`` and
        friends), so ``akita_black_bowl_1`` must not swallow
        ``akita_black_bowl_12``. Matching requires an exact hit or an ``_``
        boundary.
        """
        prefix = entity + "_"
        return sorted(
            geom_id
            for geom_id, body in self._geom_to_body.items()
            if body == entity or body.startswith(prefix)
        )

    def entity_world_xyz(self, entity: str) -> Optional[np.ndarray]:
        model, data = self.sim.model, self.sim.data
        for candidate in (f"{entity}_main", entity, f"{entity}_base"):
            try:
                body_id = model.body_name2id(candidate)
            except Exception:
                continue
            return np.array(data.body_xpos[body_id], dtype=float)
        return None

    # -- rendering ------------------------------------------------------------
    def render_segmentation(self, camera: str) -> np.ndarray:
        """Read-only extra render at the current state. Returns (H, W) geom ids.

        Background and non-geom pixels become -1.
        """
        raw = self.sim.render(
            camera_name=camera, height=self.height, width=self.width, segmentation=True
        )
        geom_ids = np.asarray(raw[..., 1]).astype(np.int32)
        types = np.asarray(raw[..., 0]).astype(np.int32)
        geom_ids = np.where(types == MJ_OBJ_GEOM, geom_ids, -1)
        # Into the policy frame so masks and UVs share one coordinate system.
        return raw_to_policy_image(geom_ids)

    # -- main entry point -----------------------------------------------------
    def labels_at_current_state(
        self, entities: Iterable[str], build_label_maps: bool = True
    ) -> StepSegmentationLabels:
        entities = list(dict.fromkeys(entities))
        legend = {name: index + 1 for index, name in enumerate(entities)}
        result = StepSegmentationLabels(legend=legend, cameras=list(self.cameras))

        entity_geoms = {name: set(self.geom_ids_for(name)) for name in entities}
        world_xyz = {name: self.entity_world_xyz(name) for name in entities}

        for camera in self.cameras:
            seg = self.render_segmentation(camera)
            label_map = np.zeros(seg.shape, dtype=np.uint8) if build_label_maps else None

            for name in entities:
                gids = entity_geoms[name]
                mask = np.isin(seg, list(gids)) if gids else np.zeros(seg.shape, dtype=bool)
                count = int(mask.sum())
                if label_map is not None and count:
                    label_map[mask] = legend[name]
                result.entities.setdefault(name, {})[camera] = self._build_label(
                    name, camera, mask, count, world_xyz[name]
                )

            if label_map is not None:
                result.label_maps[camera] = label_map

        return result

    def _build_label(
        self,
        entity: str,
        camera: str,
        mask: np.ndarray,
        count: int,
        xyz: Optional[np.ndarray],
    ) -> EntityCameraLabel:
        height, width = mask.shape
        uv_pixel = uv_norm = uv_upright = None
        in_frame = False

        depth = None
        if xyz is not None:
            row_up, col_up, depth = project_world_point(xyz, self._transform(camera))
            uv_upright = [row_up, col_up]
            row, col = upright_to_policy(row_up, col_up, height, width)
            uv_pixel = [row, col]
            uv_norm = [col / max(width - 1, 1), row / max(height - 1, 1)]
            in_frame = depth > 0 and 0 <= row <= height - 1 and 0 <= col <= width - 1

        centroid = bbox = None
        if count >= self.min_visible_pixels:
            rows, cols = np.nonzero(mask)
            centroid = [
                float(cols.mean()) / max(width - 1, 1),
                float(rows.mean()) / max(height - 1, 1),
            ]
            bbox = [
                float(cols.min()) / max(width - 1, 1),
                float(rows.min()) / max(height - 1, 1),
                float(cols.max()) / max(width - 1, 1),
                float(rows.max()) / max(height - 1, 1),
            ]

        if xyz is None:
            visibility = VISIBILITY_UNKNOWN
        elif count >= self.min_visible_pixels:
            visibility = VISIBILITY_VISIBLE
        elif not in_frame:
            # Projects outside the image, or behind the camera: not visible, and
            # not occluded either -- the camera is simply not pointed at it.
            visibility = VISIBILITY_OUT_OF_VIEW
        else:
            # Projects inside the image but contributes no pixels: something is
            # in front of it.
            visibility = VISIBILITY_OCCLUDED

        return EntityCameraLabel(
            entity=entity,
            camera=camera,
            mask_pixel_count=count,
            mask_fraction=float(count) / float(height * width),
            uv_pixel=uv_pixel,
            uv=uv_norm,
            uv_pixel_upright=uv_upright,
            mask_centroid=centroid,
            mask_bbox=bbox,
            in_frame=in_frame,
            visibility=visibility,
            depth_m=depth,
        )


# -----------------------------------------------------------------------------
# Overlay
# -----------------------------------------------------------------------------
#: Stable colours so a reviewer can read overlays without a legend lookup.
ROLE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "source": (255, 64, 64),
    "destination": (64, 160, 255),
    "distractor": (255, 200, 32),
    "fixture": (150, 150, 150),
    "unknown": (200, 32, 255),
}


def build_overlay(
    policy_frame_rgb: np.ndarray,
    labels: StepSegmentationLabels,
    camera: str,
    roles: Optional[Dict[str, str]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint each entity's mask and mark its projected origin with a cross."""
    image = np.asarray(policy_frame_rgb).astype(np.float32).copy()
    height, width = image.shape[:2]
    label_map = labels.label_maps.get(camera)
    roles = roles or {}

    if label_map is not None and label_map.shape[:2] == (height, width):
        for entity, index in labels.legend.items():
            mask = label_map == index
            if not mask.any():
                continue
            colour = np.array(ROLE_COLOURS.get(roles.get(entity, "unknown"), ROLE_COLOURS["unknown"]))
            image[mask] = (1.0 - alpha) * image[mask] + alpha * colour

    for entity, per_camera in labels.entities.items():
        label = per_camera.get(camera)
        if label is None or label.uv_pixel is None or not label.in_frame:
            continue
        colour = np.array(ROLE_COLOURS.get(roles.get(entity, "unknown"), ROLE_COLOURS["unknown"]))
        row, col = int(round(label.uv_pixel[0])), int(round(label.uv_pixel[1]))
        for delta in range(-4, 5):
            for r, c in ((row + delta, col), (row, col + delta)):
                if 0 <= r < height and 0 <= c < width:
                    image[r, c] = colour

    return image.clip(0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Equivalence check
# -----------------------------------------------------------------------------
def verify_segmentation_equivalence(
    env: Any, cameras: Sequence[str] = DEFAULT_CAMERAS, height: int = 256, width: int = 256
) -> Dict[str, Any]:
    """Prove the extra segmentation render changes neither pixels nor physics.

    Renders RGB, then segmentation on every camera, then RGB again, and compares
    both the images and the full simulator state vector.
    """
    sim = _unwrap_sim(env)
    before_rgb = {
        cam: np.array(sim.render(camera_name=cam, height=height, width=width), copy=True)
        for cam in cameras
    }
    before_state = np.array(sim.get_state().flatten(), copy=True)

    for cam in cameras:
        sim.render(camera_name=cam, height=height, width=width, segmentation=True)

    after_rgb = {
        cam: np.array(sim.render(camera_name=cam, height=height, width=width), copy=True)
        for cam in cameras
    }
    after_state = np.array(sim.get_state().flatten(), copy=True)

    per_camera = {
        cam: {
            "identical": bool(np.array_equal(before_rgb[cam], after_rgb[cam])),
            "max_abs_diff": int(
                np.abs(before_rgb[cam].astype(np.int32) - after_rgb[cam].astype(np.int32)).max()
            ),
        }
        for cam in cameras
    }
    state_identical = bool(np.array_equal(before_state, after_state))
    return {
        "rgb": per_camera,
        "sim_state_identical": state_identical,
        "passed": state_identical and all(v["identical"] for v in per_camera.values()),
    }


def _unwrap_sim(env: Any) -> Any:
    for candidate in (env, getattr(env, "env", None), getattr(getattr(env, "env", None), "env", None)):
        sim = getattr(candidate, "sim", None)
        if sim is not None:
            return sim
    raise AttributeError("could not locate a MuJoCo sim on the supplied env")


# -----------------------------------------------------------------------------
# CLI: self-test against a real task
# -----------------------------------------------------------------------------
def _main() -> int:
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from entity_role_resolver import resolve_entity_roles_from_path
    from libero.libero import benchmark, get_libero_path
    from run_single_vanilla_rollout import get_libero_env

    parser = argparse.ArgumentParser(description="Self-test the segmentation label provider.")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task_id)
    env, description = get_libero_env(task, resolution=args.resolution)
    env.reset()

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    roles = resolve_entity_roles_from_path(bddl)

    equivalence = verify_segmentation_equivalence(
        env, height=args.resolution, width=args.resolution
    )
    provider = SegmentationLabelProvider(env, height=args.resolution, width=args.resolution)
    labels = provider.labels_at_current_state(roles.tracked_entities)

    print(f"task        : {description}")
    print(f"equivalence : {'PASS' if equivalence['passed'] else 'FAIL'}  {json.dumps(equivalence)}")
    for entity in roles.tracked_entities:
        label = labels.label_of(entity)
        if label is None:
            continue
        uv = "  -  " if label.uv is None else f"({label.uv[0]:.3f}, {label.uv[1]:.3f})"
        print(
            f"  {entity:36s} role={roles.role_of(entity):11s} uv={uv} "
            f"px={label.mask_pixel_count:6d} {label.visibility}"
        )
    return 0 if equivalence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
