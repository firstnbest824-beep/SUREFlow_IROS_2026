"""Unit and closed-loop sanity checks for the parameter-free approach controller."""

import os
import sys

import numpy as np


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACTIONGEN = os.path.join(_ROOT, "tools", "action_generalization")
if _ACTIONGEN not in sys.path:
    sys.path.insert(0, _ACTIONGEN)

from methods.base import NoOverrideMethod  # noqa: E402
from methods.global_approach import GlobalApproachMethod  # noqa: E402


class _ObjectState:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def get_geom_state(self):
        return {"pos": self.position.copy()}


class _BaseEnv:
    def __init__(self, target_name, position):
        self.object_states_dict = {target_name: _ObjectState(position)}


class _Env:
    def __init__(self, target_name, position):
        self.env = _BaseEnv(target_name, position)


def _config(target="target_1", offset=(0.0, 0.0, 0.0), switch_distance=0.02):
    return {
        "global_approach": {
            "oracle_target": True,
            "target_object": target,
            "approach_offset_xyz": list(offset),
            "switch_distance": switch_distance,
            "kp": 4.0,
            "max_translation_command": 0.5,
            "one_way_switch": True,
        }
    }


def _action(method, env, eef, timestep=0):
    return method.predict_action(
        {}, "task", env=env, raw_observation={"robot0_eef_pos": np.asarray(eef)}, timestep=timestep,
    )


def test_direction_follows_target_on_both_x_sides():
    right = GlobalApproachMethod()
    right.setup(_config())
    assert _action(right, _Env("target_1", (0.3, 0.0, 0.0)), (0.0, 0.0, 0.0))[0] > 0

    left = GlobalApproachMethod()
    left.setup(_config())
    assert _action(left, _Env("target_1", (-0.3, 0.0, 0.0)), (0.0, 0.0, 0.0))[0] < 0


def test_closed_loop_smoke_decreases_waypoint_distance():
    method = GlobalApproachMethod()
    method.setup(_config(switch_distance=0.01))
    env = _Env("target_1", (0.3, 0.0, 0.0))
    eef = np.zeros(3)
    distances = []
    for timestep in range(5):
        action = _action(method, env, eef, timestep)
        distances.append(method.last_diagnostics["distance_to_waypoint"])
        # OSC_POSE maps normalized +/-1 translation to +/-0.05 m deltas.
        eef = eef + action[:3] * 0.05
    assert all(later < earlier for earlier, later in zip(distances, distances[1:]))


def test_perturbation_response_changes_geometry_direction():
    positive = GlobalApproachMethod()
    positive.setup(_config())
    negative = GlobalApproachMethod()
    negative.setup(_config())
    pos_action = _action(positive, _Env("target_1", (0.2, 0.0, 0.0)), (0.0, 0.0, 0.0))
    neg_action = _action(negative, _Env("target_1", (-0.2, 0.0, 0.0)), (0.0, 0.0, 0.0))
    assert pos_action[0] > 0 > neg_action[0]


def test_switch_is_one_way_latch():
    method = GlobalApproachMethod()
    method.setup(_config(switch_distance=0.05))
    env = _Env("target_1", (0.0, 0.0, 0.0))
    assert _action(method, env, (0.01, 0.0, 0.0), timestep=3) is None
    assert method.mode == "openvla"
    assert method.switch_step == 3
    assert _action(method, env, (0.5, 0.0, 0.0), timestep=4) is None
    assert method.mode == "openvla"
    assert method.last_diagnostics["controller_mode"] == "openvla"


def test_baseline_isolation_and_no_learning_state():
    assert NoOverrideMethod().predict_action({}, "task") is None
    method = GlobalApproachMethod()
    method.setup(_config())
    assert method.trainable_parameter_count == 0
    assert not hasattr(method, "parameters")
