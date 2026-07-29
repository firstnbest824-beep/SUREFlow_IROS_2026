"""Crash-safe writer for one activation-collection episode.

Layout produced (schema v2):

    <root>/<suite>/<condition>/task_NN/seed_NNN/episode_NNN/
        manifest.json
        per_step_metrics.jsonl
        activations/step_000000.npz
        images/agentview/step_000000.png
        images/eye_in_hand/step_000000.png
        overlays/step_000000_agentview.png
        rollout.mp4
        validation_report.json
        COMPLETE

Everything is written into a sibling ``.partial`` directory and moved into place
with a single ``os.rename`` only after the manifest is written and the
``COMPLETE`` marker exists. A reader therefore never observes a half-written
episode, and an interrupted run leaves a ``.partial`` directory that is easy to
find and is never silently reused.

Activations are stored as one compressed ``.npz`` per timestep holding all eight
stages. Only the *saved copy* is cast: the hook output is detached, moved to CPU
and converted to fp16 for storage, and the tensor the model computed with is
never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2
COMPLETE_MARKER = "COMPLETE"
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "per_step_metrics.jsonl"
VALIDATION_NAME = "validation_report.json"

REQUIRED_STAGES: Sequence[str] = (
    "final_vision_dinov2",
    "final_vision_siglip",
    "projector_input",
    "projector_output",
    "llm_early",
    "llm_middle",
    "llm_late",
    "pre_action_hidden",
)

#: What every stored record asserts about where its numbers came from.
LABEL_REFERENCE = "pre_step_observation"
ACTIVATION_REFERENCE = "pre_step_observation"
ACTION_REFERENCE = "computed_from_same_observation"
#: `success` and `done` are the ONLY post-step fields in a record. LIBERO
#: evaluates its goal predicate after the step (bddl_base_domain.py sets
#: `done = self._check_success()` on the stepped state), so a pre-step reading
#: would be impossible: at index t the episode has not yet been acted on.
#: Recorded explicitly so nothing has to infer it from the other three.
OUTCOME_REFERENCE = "post_step_observation"
POST_STEP_FIELDS = ("success", "done")


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_storage_array(tensor: Any, dtype: np.dtype = np.float16) -> np.ndarray:
    """Detach -> CPU -> numpy -> storage dtype, on a copy only.

    The model keeps computing in its own dtype; this never writes back into the
    graph or mutates the hook's tensor.
    """
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().to("cpu")
            # float16 has no bfloat16 cast path on some builds; go via float32.
            array = tensor.to(torch.float32).numpy()
            return array.astype(dtype, copy=False)
    except ImportError:
        pass
    return np.asarray(tensor).astype(dtype, copy=False)


@dataclass
class StepRecord:
    """One timestep. Every field derives from the same pre-step observation."""

    timestep: int
    payload: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        record = dict(self.payload)
        record["timestep"] = self.timestep
        record.setdefault("observation_timestep", self.timestep)
        record.setdefault("action_timestep", self.timestep)
        record["label_reference"] = LABEL_REFERENCE
        record["activation_reference"] = ACTIVATION_REFERENCE
        record["action_reference"] = ACTION_REFERENCE
        record["outcome_reference"] = OUTCOME_REFERENCE
        record["post_step_fields"] = list(POST_STEP_FIELDS)
        return record


class ActivationEpisodeWriter:
    """Writes one episode atomically.

    Usage::

        with ActivationEpisodeWriter(final_dir, manifest) as writer:
            for t in range(...):
                writer.write_step(t, activations=..., metrics=..., images=...)
            writer.set_result(success=True, termination_reason="success")

    Leaving the ``with`` block without an exception seals and renames. An
    exception leaves the ``.partial`` directory behind untouched.
    """

    def __init__(
        self,
        final_dir: str | os.PathLike,
        manifest: Dict[str, Any],
        save_dtype: np.dtype = np.float16,
        required_stages: Sequence[str] = REQUIRED_STAGES,
        overwrite: bool = False,
    ) -> None:
        self.final_dir = Path(final_dir)
        self.partial_dir = self.final_dir.with_name(self.final_dir.name + ".partial")
        self.manifest = dict(manifest)
        self.save_dtype = save_dtype
        self.required_stages = tuple(required_stages)
        self.overwrite = overwrite

        self._steps_written = 0
        self._stage_shapes: Dict[str, List[int]] = {}
        self._metrics_handle = None
        self._sealed = False
        self._result: Dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ActivationEpisodeWriter":
        if self.final_dir.exists() and not self.overwrite:
            raise FileExistsError(
                f"episode already exists, refusing to overwrite: {self.final_dir}"
            )
        if self.partial_dir.exists():
            raise FileExistsError(
                f"a previous interrupted attempt is still present, refusing to reuse it: "
                f"{self.partial_dir}. Inspect or move it aside first."
            )
        for sub in ("activations", "images/agentview", "images/eye_in_hand", "overlays"):
            (self.partial_dir / sub).mkdir(parents=True, exist_ok=True)
        self._metrics_handle = open(
            self.partial_dir / METRICS_NAME, "w", encoding="utf-8", buffering=1
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._metrics_handle is not None:
            self._metrics_handle.close()
            self._metrics_handle = None
        if exc_type is None and not self._sealed:
            self.seal()
        return False

    # -- writing --------------------------------------------------------------
    def write_step(
        self,
        timestep: int,
        activations: Dict[str, Any],
        metrics: Dict[str, Any],
        agentview_rgb: Optional[np.ndarray] = None,
        eye_in_hand_rgb: Optional[np.ndarray] = None,
        overlays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Persist one timestep and return the activation descriptor."""
        if self._sealed:
            raise RuntimeError("writer already sealed")
        if timestep != self._steps_written:
            raise ValueError(
                f"timestep {timestep} written out of order; expected {self._steps_written}. "
                "Records must be contiguous from 0 so activation and label indices cannot drift."
            )

        missing = [s for s in self.required_stages if s not in activations]
        if missing:
            raise KeyError(f"timestep {timestep} missing required stages: {missing}")

        arrays: Dict[str, np.ndarray] = {}
        descriptor: Dict[str, Any] = {}
        for stage, tensor in activations.items():
            array = to_storage_array(tensor, self.save_dtype)
            arrays[stage] = array
            shape = [int(v) for v in array.shape]
            descriptor[stage] = {
                "shape": shape,
                "dtype": str(array.dtype),
                "has_nan": bool(np.isnan(array).any()),
                "has_inf": bool(np.isinf(array).any()),
            }
            known = self._stage_shapes.setdefault(stage, shape)
            if stage in self.required_stages and shape[:1] + shape[2:] != known[:1] + known[2:]:
                # Sequence length may legitimately vary; batch/hidden may not.
                raise ValueError(
                    f"stage {stage} changed shape at timestep {timestep}: {shape} vs {known}"
                )

        activation_path = self.partial_dir / "activations" / f"step_{timestep:06d}.npz"
        np.savez_compressed(activation_path, **arrays)

        images: Dict[str, str] = {}
        if agentview_rgb is not None:
            images["agentview"] = self._save_png(
                self.partial_dir / "images/agentview" / f"step_{timestep:06d}.png", agentview_rgb
            )
        if eye_in_hand_rgb is not None:
            images["eye_in_hand"] = self._save_png(
                self.partial_dir / "images/eye_in_hand" / f"step_{timestep:06d}.png",
                eye_in_hand_rgb,
            )
        for name, image in (overlays or {}).items():
            self._save_png(
                self.partial_dir / "overlays" / f"step_{timestep:06d}_{name}.png", image
            )

        record = StepRecord(timestep, dict(metrics, activations=descriptor, images=images))
        assert self._metrics_handle is not None
        self._metrics_handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self._steps_written += 1
        return descriptor

    @staticmethod
    def _save_png(path: Path, array: np.ndarray) -> str:
        from PIL import Image

        Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB").save(path)
        return str(path)

    def set_result(self, **fields: Any) -> None:
        """Record episode outcome (success, termination_reason, ...)."""
        self._result.update(fields)

    def add_video(self, frames: Sequence[np.ndarray], fps: float = 30.0) -> Optional[str]:
        if not frames:
            return None
        try:
            import cv2

            path = self.partial_dir / "rollout.mp4"
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                return None
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(np.asarray(frame).astype(np.uint8), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return str(path)
        except Exception:
            return None

    # -- sealing --------------------------------------------------------------
    def seal(self) -> Path:
        """Write the manifest, drop COMPLETE, then rename into place atomically."""
        if self._sealed:
            return self.final_dir
        if self._metrics_handle is not None:
            self._metrics_handle.close()
            self._metrics_handle = None

        activation_files = sorted((self.partial_dir / "activations").glob("step_*.npz"))
        if len(activation_files) != self._steps_written:
            raise RuntimeError(
                f"activation file count {len(activation_files)} != steps written "
                f"{self._steps_written}"
            )
        metrics_lines = sum(
            1 for line in open(self.partial_dir / METRICS_NAME, encoding="utf-8") if line.strip()
        )
        if metrics_lines != self._steps_written:
            raise RuntimeError(
                f"per_step_metrics line count {metrics_lines} != steps written {self._steps_written}"
            )

        manifest = dict(self.manifest)
        manifest.update(
            schema_version=SCHEMA_VERSION,
            num_timesteps=self._steps_written,
            activation_stages=list(self.required_stages),
            activation_dtype=str(np.dtype(self.save_dtype)),
            activation_stage_shapes=self._stage_shapes,
            label_reference=LABEL_REFERENCE,
            activation_reference=ACTIVATION_REFERENCE,
            action_reference=ACTION_REFERENCE,
            **self._result,
        )
        with open(self.partial_dir / MANIFEST_NAME, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)

        (self.partial_dir / COMPLETE_MARKER).write_text("", encoding="utf-8")
        self._fsync_dir(self.partial_dir)

        if self.final_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"episode appeared while writing: {self.final_dir}")
            shutil.rmtree(self.final_dir)
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(self.partial_dir, self.final_dir)
        self._fsync_dir(self.final_dir.parent)
        self._sealed = True
        return self.final_dir

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    # -- helpers --------------------------------------------------------------
    @property
    def steps_written(self) -> int:
        return self._steps_written

    @property
    def sealed(self) -> bool:
        return self._sealed


def episode_dir_for(
    root: str | os.PathLike,
    suite: str,
    condition: str,
    task_id: int,
    seed: int,
    episode_index: int,
) -> Path:
    """Layout that keeps suite / condition / checkpoint data from mixing."""
    return (
        Path(root) / suite / condition / f"task_{task_id:02d}"
        / f"seed_{seed:03d}" / f"episode_{episode_index:03d}"
    )


def find_incomplete_episodes(root: str | os.PathLike) -> List[Path]:
    """Directories left behind by an interrupted run, for quarantine or retry."""
    root = Path(root)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for path in root.rglob("episode_*"):
        if not path.is_dir():
            continue
        if path.name.endswith(".partial"):
            out.append(path)
        elif not (path / COMPLETE_MARKER).exists():
            out.append(path)
    return sorted(out)
