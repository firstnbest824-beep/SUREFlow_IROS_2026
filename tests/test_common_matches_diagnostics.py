"""Durable parity checks for the dependency-free ``tools/common`` copies.

Real 7B inference is intentionally excluded. The tests instead compare every
deterministic helper and the deterministic processor/model-input portion of
``get_vla_action`` against the active diagnostics implementation.
"""

import os
import random
import sys

import numpy as np
import torch


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.join(_ROOT, "tools", "common")
_OPENVLA = os.path.join(_ROOT, "tools", "openvla")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import checkpoints as common_checkpoints  # noqa: E402
import image_transform as common_image  # noqa: E402
import openvla_model as common_model  # noqa: E402
from seeding import episode_seed, seed_everything  # noqa: E402
from task_resolution import resolve_task_condition  # noqa: E402

if _OPENVLA not in sys.path:
    sys.path.insert(0, _OPENVLA)

import model_input_transform as diagnostics_image  # noqa: E402
import official_task_pair_resolver as diagnostics_pairs  # noqa: E402
import run_single_vanilla_rollout as diagnostics_model  # noqa: E402


def test_action_helpers_match_diagnostics_exactly():
    rng = np.random.RandomState(123)
    actions = rng.uniform(-2.0, 2.0, size=(8, 7))
    actions[:, -1] = rng.uniform(0.0, 1.0, size=8)
    assert np.array_equal(
        common_model.normalize_gripper_action(actions.copy()),
        diagnostics_model.normalize_gripper_action(actions.copy()),
    )
    assert np.array_equal(
        common_model.invert_gripper_action(actions.copy()),
        diagnostics_model.invert_gripper_action(actions.copy()),
    )
    for quat in ([0.0, 0.0, 0.0, 1.0], [0.1, -0.2, 0.3, 0.9], [0.0, 0.0, 0.0, -1.0]):
        assert np.allclose(
            common_model.quat2axisangle(np.asarray(quat)),
            diagnostics_model.quat2axisangle(np.asarray(quat)),
            rtol=0.0,
            atol=1e-12,
        )
    assert common_model.get_libero_dummy_action() == diagnostics_model.get_libero_dummy_action()


def test_image_preprocessing_matches_diagnostics_exactly():
    rgb = np.random.RandomState(7).randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    observation = {"agentview_image": rgb}
    assert np.array_equal(
        common_image.get_libero_image(observation, 224),
        diagnostics_image.get_libero_image(observation, 224),
    )
    assert np.array_equal(common_image.rgb_to_model_input(rgb), diagnostics_image.rgb_to_model_input(rgb))


def test_seed_and_checkpoint_registry_match_diagnostics():
    keys = [(base, suite, task_id, init_id) for base in (0, 7) for suite in ("libero_spatial", "libero_object") for task_id in (0, 9) for init_id in (0, 3)]
    assert all(episode_seed(*key) == diagnostics_pairs.episode_seed(*key) for key in keys)
    assert all(
        common_checkpoints.get_checkpoint(suite).__dict__ == checkpoint
        for suite, checkpoint in diagnostics_pairs.SUITE_CHECKPOINTS.items()
    )

    seed_everything(41)
    first = (random.random(), np.random.rand(), torch.rand(4))
    seed_everything(41)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


class _Processor:
    def __init__(self):
        self.prompt = None
        self.image = None

    def __call__(self, prompt, image):
        self.prompt = prompt
        self.image = np.asarray(image)
        return {
            "input_ids": torch.tensor([[11]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224)),
        }


class _VLA:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return np.arange(7, dtype=np.float64) / 10.0


def test_prompt_and_model_input_transform_match_diagnostics():
    rgb = np.random.RandomState(19).randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    observation = {"full_image": common_image.get_libero_image({"agentview_image": rgb}, 224), "state": np.zeros(8)}
    common_processor, diagnostics_processor = _Processor(), _Processor()
    common_vla, diagnostics_vla = _VLA(), _VLA()
    common_action = common_model.get_vla_action(
        common_vla, common_processor, "openvla/openvla-7b-finetuned-libero-spatial", observation,
        "Pick Up Object", "libero_spatial", True, torch.float32,
    )
    diagnostics_action = diagnostics_model.get_vla_action(
        diagnostics_vla, diagnostics_processor, "openvla/openvla-7b-finetuned-libero-spatial", observation,
        "Pick Up Object", "libero_spatial", True, torch.float32,
    )
    assert common_processor.prompt == diagnostics_processor.prompt
    assert np.array_equal(common_processor.image, diagnostics_processor.image)
    assert np.array_equal(common_action, diagnostics_action)
    for key in ("input_ids", "attention_mask", "pixel_values"):
        assert torch.equal(common_vla.kwargs[key], diagnostics_vla.kwargs[key])
    assert common_vla.kwargs["unnorm_key"] == diagnostics_vla.kwargs["unnorm_key"]
    assert common_vla.kwargs["do_sample"] == diagnostics_vla.kwargs["do_sample"]


def test_vanilla_and_position_offset_resolve_to_distinct_bddls():
    vanilla = resolve_task_condition("libero_object", 0, "vanilla")
    perturbed = resolve_task_condition("libero_object", 0, "y0.1")
    assert vanilla.resolved_bddl_path == vanilla.vanilla_bddl_path
    assert perturbed.resolved_bddl_path != perturbed.vanilla_bddl_path
    assert perturbed.requested_condition == "y0.1"
    assert perturbed.perturbation_family == "position_offset"
