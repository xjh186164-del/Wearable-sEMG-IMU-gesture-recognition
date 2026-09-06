"""Deterministic transition rejection for streaming gesture probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .intervals import CLASS_LABELS


_EMA_NEW_WEIGHTS = (0.4, 0.6, 0.8)
_MINIMUM_CONFIDENCES = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
_CONSECUTIVE_UPDATES = (2, 3, 4)
_DISPLAY_STATES = frozenset((*CLASS_LABELS, "WARMING_UP", "TRANSITION_UNSTABLE"))
# Ignore one upward binary64 rounding step when a decimal sensor-time gap is exactly 0.5 s.
_GAP_RESET_THRESHOLD_S = math.nextafter(0.5, math.inf)


@dataclass(frozen=True)
class DecisionFilterConfig:
    """Parameters selected exclusively by validation-session calibration."""

    ema_new_weight: float
    minimum_confidence: float
    consecutive_updates: int

    def __post_init__(self) -> None:
        if (
            type(self.ema_new_weight) is not float
            or not math.isfinite(self.ema_new_weight)
            or self.ema_new_weight not in _EMA_NEW_WEIGHTS
        ):
            raise ValueError("ema_new_weight must be one calibrated value: 0.4, 0.6, or 0.8")
        if (
            type(self.minimum_confidence) is not float
            or not math.isfinite(self.minimum_confidence)
            or self.minimum_confidence not in _MINIMUM_CONFIDENCES
        ):
            raise ValueError("minimum_confidence must be in 0.50..0.95 in 0.05 increments")
        if (
            isinstance(self.consecutive_updates, bool)
            or not isinstance(self.consecutive_updates, Integral)
            or int(self.consecutive_updates) not in _CONSECUTIVE_UPDATES
        ):
            raise ValueError("consecutive_updates must be 2, 3, or 4")
        object.__setattr__(self, "consecutive_updates", int(self.consecutive_updates))


@dataclass(frozen=True)
class DecisionOutput:
    """One filter update expressed using only JSON-compatible primitives."""

    candidate_gesture_id: str
    display_state: str
    confidence: float
    smoothed_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.display_state not in _DISPLAY_STATES:
            raise ValueError("display_state is not recognized")


class RealtimeDecisionFilter:
    """Stateful EMA and consecutive-candidate filter for the fixed class order."""

    def __init__(self, config: DecisionFilterConfig, class_labels=CLASS_LABELS):
        if not isinstance(config, DecisionFilterConfig):
            raise TypeError("config must be a DecisionFilterConfig")
        if tuple(class_labels) != CLASS_LABELS:
            raise ValueError("class_labels must match the fixed trained class order")
        self._config = config
        self._class_labels = tuple(class_labels)
        self._ema: np.ndarray | None = None
        self._last_sensor_time_s: float | None = None
        self._candidate_index: int | None = None
        self._candidate_count = 0

    @property
    def config(self) -> DecisionFilterConfig:
        return self._config

    def _reset_filter_state(self) -> None:
        self._ema = None
        self._candidate_index = None
        self._candidate_count = 0

    def reset(self) -> None:
        """Clear all streaming state so the next update behaves like first use."""
        self._reset_filter_state()
        self._last_sensor_time_s = None

    @staticmethod
    def _validate_probabilities(probabilities) -> np.ndarray:
        if not isinstance(probabilities, np.ndarray):
            raise ValueError("probabilities must be a NumPy float32 array")
        if probabilities.dtype != np.float32 or probabilities.shape != (len(CLASS_LABELS),):
            raise ValueError("probabilities must have dtype float32 and shape [8]")
        if not bool(np.all(np.isfinite(probabilities))):
            raise ValueError("probabilities must be finite")
        if bool(np.any(probabilities < 0.0)) or bool(np.any(probabilities > 1.0)):
            raise ValueError("probabilities must be within [0, 1]")
        if abs(float(probabilities.sum(dtype=np.float64)) - 1.0) > 1e-4:
            raise ValueError("probabilities must sum to one within 1e-4")
        return probabilities.copy()

    def update(self, probabilities: np.ndarray, sensor_time_s: float) -> DecisionOutput:
        values = self._validate_probabilities(probabilities)
        if (
            isinstance(sensor_time_s, bool)
            or not isinstance(sensor_time_s, Real)
            or not math.isfinite(float(sensor_time_s))
        ):
            raise ValueError("sensor_time_s must be finite")
        sensor_time = float(sensor_time_s)
        if self._last_sensor_time_s is not None and sensor_time <= self._last_sensor_time_s:
            raise ValueError("sensor_time_s must be strictly increasing")

        first_after_reset = self._ema is None
        if (
            self._last_sensor_time_s is not None
            and sensor_time - self._last_sensor_time_s > _GAP_RESET_THRESHOLD_S
        ):
            self._reset_filter_state()
            first_after_reset = True
        self._last_sensor_time_s = sensor_time

        if self._ema is None:
            self._ema = values
        else:
            self._ema = (
                self._config.ema_new_weight * values
                + (1.0 - self._config.ema_new_weight) * self._ema
            ).astype(np.float32, copy=False)
        candidate_index = int(np.argmax(self._ema))
        confidence = float(self._ema[candidate_index])
        if confidence >= self._config.minimum_confidence:
            if candidate_index == self._candidate_index:
                self._candidate_count += 1
            else:
                self._candidate_index = candidate_index
                self._candidate_count = 1
        else:
            self._candidate_index = None
            self._candidate_count = 0

        if first_after_reset:
            display_state = "WARMING_UP"
        elif self._candidate_count >= self._config.consecutive_updates:
            display_state = self._class_labels[candidate_index]
        else:
            display_state = "TRANSITION_UNSTABLE"
        return DecisionOutput(
            candidate_gesture_id=self._class_labels[candidate_index],
            display_state=display_state,
            confidence=confidence,
            smoothed_probabilities=tuple(float(value) for value in self._ema),
        )
