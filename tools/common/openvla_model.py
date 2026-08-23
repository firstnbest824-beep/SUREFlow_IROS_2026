"""OpenVLA model loading and action-generation helpers, extracted for reuse.

Physical copy (not a re-export) of the following functions from
``tools/openvla/run_single_vanilla_rollout.py``, verified line-by-line before
copying to confirm none of them reference that file's diagnostics-only
top-level imports (``spatial_task_resolver``, ``task_phase_resolver``):

- ``ACTION_DIM``                  -- source line 55
- ``normalize_gripper_action``    -- source lines 133-138
- ``invert_gripper_action``       -- source lines 141-144
- ``load_openvla``                -- source lines 150-196
- ``get_vla_action``              -- source lines 199-251
- ``get_libero_dummy_action``     -- source lines 273-275
- ``quat2axisangle``              -- source lines 278-288

The only adjustment made to the bodies themselves is the import of
``apply_center_crop``, which now comes from ``tools/common/image_transform.py``
(the same content as ``tools/openvla/model_input_transform.py``) instead of the
sibling-module import the original file used. See ``tools/common/README.md``
for the drift-tracking contract: this file does not auto-update if the
original changes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from image_transform import apply_center_crop  # noqa: E402

ACTION_DIM = 7

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass(frozen=True)
class PreparedOpenVLAInputs:
    """The exact prompt and tensors handed to ``vla.predict_action``.

    Keeping this as a shared preparation result prevents an analysis script
    from reconstructing a prompt or processor input with subtly different
    special-token handling than the action path.
    """

    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    model_input_image: np.ndarray


def build_openvla_prompt(base_vla_name: str, task_label: str) -> str:
    """Build the exact task prompt used by the action-generation helper."""
    if "openvla-v01" in base_vla_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def prepare_openvla_inputs(
    vla: Any,
    processor: Any,
    base_vla_name: str,
    obs: Dict[str, Any],
    task_label: str,
    center_crop: bool,
    dtype: torch.dtype,
) -> PreparedOpenVLAInputs:
    """Prepare the exact image, prompt, and tensors for ``predict_action``."""
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        image = apply_center_crop(image)
    prompt = build_openvla_prompt(base_vla_name, task_label)
    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"].to(vla.device)
    attention_mask = inputs["attention_mask"].to(vla.device)
    pixel_values = inputs["pixel_values"].to(vla.device, dtype=dtype)

    # The OpenVLA wrapper appends this empty token when absent. Append it here
    # together with its attention entry so the causal-mask sequence is valid.
    if input_ids[0, -1].item() != 29871:
        empty_token = torch.tensor([[29871]], dtype=input_ids.dtype, device=vla.device)
        attend_token = torch.tensor([[1]], dtype=attention_mask.dtype, device=vla.device)
        input_ids = torch.cat([input_ids, empty_token], dim=1)
        attention_mask = torch.cat([attention_mask, attend_token], dim=1)
    return PreparedOpenVLAInputs(
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        model_input_image=np.array(image),
    )


# -----------------------------------------------------------------------------
# Action helpers (official OpenVLA convention)
# -----------------------------------------------------------------------------
def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper action from [0, 1] to [-1, +1] and optionally binarize."""
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip sign of gripper action to align with LIBERO convention."""
    action[..., -1] = action[..., -1] * -1.0
    return action


# -----------------------------------------------------------------------------
# OpenVLA loading and action generation
# -----------------------------------------------------------------------------
def load_openvla(
    checkpoint_id: str,
    revision: str,
    attn_implementation: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Any, Any]:
    """Load OpenVLA processor and model, attach dataset statistics."""
    print(f"[*] Loading processor from {checkpoint_id} @ {revision}")
    processor = AutoProcessor.from_pretrained(
        checkpoint_id,
        revision=revision,
        trust_remote_code=True,
    )

    print(f"[*] Loading model from {checkpoint_id} @ {revision} (dtype={dtype}, attn={attn_implementation})")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_id,
        revision=revision,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # Attach dataset statistics for action un-normalization.
    # NOTE: `checkpoint_id` is a HF hub id ("openvla/openvla-7b-finetuned-..."),
    # not a local directory, so this path never resolves and the warning branch
    # below always fires. That is currently harmless -- config.json's `norm_stats`
    # and the hub's dataset_statistics.json were verified byte-identical for both
    # LIBERO checkpoints -- but the un-normalisation statistics in use come from
    # config.json, not from this file. Kept as-is so the loaded statistics do not
    # silently change; if it is ever "fixed", re-verify the q01/q99 values first.
    dataset_statistics_path = os.path.join(checkpoint_id, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
        print(f"[*] Loaded dataset_statistics.json; keys: {list(norm_stats.keys())}")
    else:
        print(
            "WARNING: No local dataset_statistics.json found. "
            "This is expected for base checkpoints, not fine-tuned ones."
        )

    return processor, vla


def get_vla_action(
    vla: Any,
    processor: Any,
    base_vla_name: str,
    obs: Dict[str, Any],
    task_label: str,
    unnorm_key: str,
    center_crop: bool,
    dtype: torch.dtype,
    return_model_input_image: bool = False,
    return_prepared_inputs: bool = False,
):
    """Generate a single action from OpenVLA given a preprocessed observation.

    With ``return_model_input_image=True`` the exact uint8 array handed to the
    processor is returned alongside the action, so spatial labels can be checked
    against the image the model actually sees.
    """
    prepared = prepare_openvla_inputs(
        vla=vla, processor=processor, base_vla_name=base_vla_name, obs=obs,
        task_label=task_label, center_crop=center_crop, dtype=dtype,
    )
    action = vla.predict_action(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        pixel_values=prepared.pixel_values,
        unnorm_key=unnorm_key,
        do_sample=False,
    )
    if return_model_input_image and return_prepared_inputs:
        return action, prepared.model_input_image, prepared
    if return_model_input_image:
        return action, prepared.model_input_image
    if return_prepared_inputs:
        return action, prepared
    return action


# -----------------------------------------------------------------------------
# LIBERO action helpers
# -----------------------------------------------------------------------------
def get_libero_dummy_action() -> List[float]:
    """Return no-op dummy action used during initial stabilization."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle."""
    qw = quat[3]
    if qw > 1.0:
        qw = 1.0
    elif qw < -1.0:
        qw = -1.0
    den = math.sqrt(1.0 - qw * qw)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(qw)) / den
