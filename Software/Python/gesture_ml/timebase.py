"""Reconstruct the MATLAB-compatible shared sensor timebase."""

from __future__ import annotations

import numpy as np


UINT32_MODULUS = 2**32
UINT32_HALF_RANGE = 2**31


def _validated_u32_vector(timestamps) -> np.ndarray:
    values = np.asarray(timestamps)
    if values.ndim != 1 or np.iscomplexobj(values):
        raise ValueError("timestamps must be a real one-dimensional vector")
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.copy()
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values > UINT32_MODULUS - 1)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("timestamps must contain uint32-range integer values")
    return values


def unwrap_u32(timestamps) -> np.ndarray:
    """Unwrap an ESP32 micros() stream and require strict monotonicity."""
    raw = _validated_u32_vector(timestamps)
    if raw.size == 0:
        return raw
    unwrapped = np.empty(raw.shape, dtype=np.float64)
    offset = 0.0
    previous = None
    for index, value in enumerate(raw):
        if previous is not None and value < previous and previous - value > UINT32_HALF_RANGE:
            offset += UINT32_MODULUS
        unwrapped[index] = value + offset
        previous = value
    if np.any(np.diff(unwrapped) <= 0):
        raise ValueError("unwrapped timestamps must be strictly increasing")
    return unwrapped


def shared_sensor_time_axes(raw_timestamps, filtered_timestamps, imu_timestamps):
    """Match gesture_internal.shared_sensor_time_axes for three streams."""
    raw_vectors = [
        _validated_u32_vector(raw_timestamps),
        _validated_u32_vector(filtered_timestamps),
        _validated_u32_vector(imu_timestamps),
    ]
    unwrapped = [unwrap_u32(values) for values in raw_vectors]
    available = [index for index, values in enumerate(unwrapped) if values.size]
    if not available:
        return tuple(values.copy() for values in unwrapped)

    anchor = raw_vectors[available[0]][0]
    offsets = {}
    for index in available:
        delta = raw_vectors[index][0] - anchor
        offsets[index] = (delta + UINT32_HALF_RANGE) % UINT32_MODULUS - UINT32_HALF_RANGE
    origin = min(offsets.values())

    axes = []
    for index, values in enumerate(unwrapped):
        if values.size == 0:
            axes.append(values.copy())
            continue
        axes.append((values - values[0] + offsets[index] - origin) / 1e6)
    return tuple(axes)

