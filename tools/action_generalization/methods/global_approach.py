"""Oracle, closed-loop global approach followed by one-way vanilla OpenVLA.

This first intervention is intentionally task-scoped and parameter-free.  It
reads the configured object's live MuJoCo position after the perturbed BDDL
and frozen init state have been applied, moves the EEF to a coarse waypoint
above it with OSC delta actions, then latches permanently to the unmodified
OpenVLA policy.  It is not a detector, planner, IK solver, or learned module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import ActionGeneralizationMethod


class GlobalApproachMethod(ActionGeneralizationMethod):
    """Closed-loop geometric approach controller with a one-way OpenVLA latch."""

    name = "global_approach"
    target_source = "oracle"
    trainable_parameter_count = 0

    def setup(self, config: Dict[str, Any]) -> None:
        settings = dict(config.get("global_approach") or {})
        if settings.get("target_source", "oracle") != self.target_source:
            raise ValueError(
                f"{self.name} requires global_approach.target_source: {self.target_source!r}"
            )
        if settings.get("one_way_switch") is not True:
            raise ValueError("global_approach v1 requires one_way_switch: true")
        target_object = settings.get("target_object")
        if not isinstance(target_object, str) or not target_object:
            raise ValueError("global_approach.target_object must name the runtime LIBERO object")
        self.target_object = target_object
        self.approach_offset_xyz = self._vector(settings.get("approach_offset_xyz"), "approach_offset_xyz")
        self.switch_distance = self._positive(settings.get("switch_distance"), "switch_distance")
        self.kp = self._positive(settings.get("kp"), "kp")
        self.max_translation_command = self._positive(
            settings.get("max_translation_command"), "max_translation_command"
        )
        if self.max_translation_command > 1.0:
            raise ValueError("max_translation_command must fit the normalized OSC input range [-1, 1]")
        self.mode = "global_approach"
        self.switch_step: Optional[int] = None
        self.switch_distance_actual: Optional[float] = None
        self.minimum_target_distance = float("inf")
        self.minimum_waypoint_distance = float("inf")
        self.reached_target_neighborhood = False
        self.last_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def _vector(value: Any, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{label} must be three finite numbers, got {value!r}")
        return array

    @staticmethod
    def _positive(value: Any, label: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{label} must be a finite positive number, got {value!r}")
        return number

    @staticmethod
    def _base_env(env: Any) -> Any:
        current = env
        seen = set()
        while hasattr(current, "env") and id(current) not in seen:
            seen.add(id(current))
            current = current.env
        return current

    def _target_position(self, env: Any) -> np.ndarray:
        """Read the actual simulator position, never a BDDL/default coordinate."""
        base_env = self._base_env(env)
        states = getattr(base_env, "object_states_dict", {})
        if self.target_object in states:
            try:
                return self._vector(states[self.target_object].get_geom_state()["pos"], "runtime target position")
            except Exception:
                pass
        sim = getattr(env, "sim", None)
        if sim is not None:
            names = [name for name in sim.model.body_names if name.startswith(self.target_object)]
            if names:
                body_id = sim.model.body_name2id(names[0])
                return self._vector(sim.data.body_xpos[body_id], "runtime target position")
        raise RuntimeError(f"could not read runtime position for target object {self.target_object!r}")

    def begin_episode(self, **kwargs: Any) -> None:
        # Reset state explicitly: a method instance is valid for exactly one episode.
        self.mode = "global_approach"
        self.switch_step = None
        self.switch_distance_actual = None
        self.minimum_target_distance = float("inf")
        self.minimum_waypoint_distance = float("inf")
        self.reached_target_neighborhood = False
        self.last_diagnostics = {}

    def _diagnostics(self, env: Any, raw_observation: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        target = self._target_position(env)
        eef = self._vector(raw_observation["robot0_eef_pos"], "robot0_eef_pos")
        waypoint = target + self.approach_offset_xyz
        error = waypoint - eef
        waypoint_distance = float(np.linalg.norm(error))
        target_distance = float(np.linalg.norm(target - eef))
        self.minimum_target_distance = min(self.minimum_target_distance, target_distance)
        self.minimum_waypoint_distance = min(self.minimum_waypoint_distance, waypoint_distance)
        return target, eef, waypoint, waypoint_distance, target_distance

    def predict_action(
        self, observation: Dict[str, Any], task_label: str, **kwargs: Any,
    ) -> Optional[np.ndarray]:
        del observation, task_label
        env = kwargs["env"]
        raw_observation = kwargs["raw_observation"]
        timestep = int(kwargs["timestep"])
        target, eef, waypoint, distance, target_distance = self._diagnostics(env, raw_observation)
        base = {
            "target_source": self.target_source,
            "target_object": self.target_object,
            "target_position_xyz": target.tolist(),
            "approach_waypoint_xyz": waypoint.tolist(),
            "current_eef_xyz": eef.tolist(),
            "error_xyz": (waypoint - eef).tolist(),
            "distance_to_waypoint": distance,
            "distance_to_target": target_distance,
            "switch_step": self.switch_step,
            "switch_distance_actual": self.switch_distance_actual,
        }
        if self.mode == "openvla":
            self.last_diagnostics = {"controller_mode": "openvla", **base, "geometry_translation_command": None}
            return None
        if distance <= self.switch_distance:
            self.mode = "openvla"  # Deliberate one-way latch: never re-enter geometry control after handoff.
            self.switch_step = timestep
            self.switch_distance_actual = distance
            self.reached_target_neighborhood = True
            base["switch_step"] = timestep
            base["switch_distance_actual"] = distance
            self.last_diagnostics = {"controller_mode": "openvla", **base, "geometry_translation_command": None}
            return None

        direction = (waypoint - eef) / distance
        magnitude = min(self.kp * distance, self.max_translation_command)
        translation = direction * magnitude
        action = np.concatenate((translation, np.zeros(3, dtype=np.float64), np.array([-1.0])))
        self.last_diagnostics = {
            "controller_mode": "global_approach", **base,
            "geometry_translation_command": translation.tolist(),
        }
        return action

    def episode_summary(self) -> Dict[str, Any]:
        return {
            "target_object": self.target_object,
            "switch_step": self.switch_step,
            "switch_distance_actual": self.switch_distance_actual,
            "minimum_target_distance": None if np.isinf(self.minimum_target_distance) else self.minimum_target_distance,
            "minimum_waypoint_distance": None if np.isinf(self.minimum_waypoint_distance) else self.minimum_waypoint_distance,
            "reached_target_neighborhood": self.reached_target_neighborhood,
            "trainable_parameter_count": self.trainable_parameter_count,
            "one_way_switch": True,
        }
