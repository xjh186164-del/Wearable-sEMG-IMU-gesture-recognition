"""Create fixed-shape aligned EMG and IMU windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .intervals import StableInterval


WINDOW_SECONDS = 0.5
STEP_SECONDS = 0.1
EMG_RATE_HZ = 500
IMU_RATE_HZ = 104
EMG_SAMPLES = 250
IMU_SAMPLES = 52


@dataclass(frozen=True)
class WindowBatch:
    emg: np.ndarray
    imu: np.ndarray
    labels: np.ndarray
    physical_subject_ids: np.ndarray
    acquisition_subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    block_ids: np.ndarray
    interval_ids: np.ndarray


def _empty_batch() -> WindowBatch:
    return WindowBatch(
        emg=np.empty((0, 8, EMG_SAMPLES), dtype=np.float32),
        imu=np.empty((0, 6, IMU_SAMPLES), dtype=np.float32),
        labels=np.empty(0, dtype=np.int64),
        physical_subject_ids=np.empty(0, dtype="<U1"),
        acquisition_subject_ids=np.empty(0, dtype="<U1"),
        session_ids=np.empty(0, dtype="<U1"),
        trial_ids=np.empty(0, dtype=np.int32),
        block_ids=np.empty(0, dtype=np.int16),
        interval_ids=np.empty(0, dtype="<U1"),
    )


def _validate_stream(time_s, values, channels: int, name: str):
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if time_s.ndim != 1 or values.ndim != 2 or values.shape != (time_s.size, channels):
        raise ValueError(f"{name} stream has an invalid shape")
    if (
        time_s.size == 0
        or not np.all(np.isfinite(time_s))
        or np.any(np.diff(time_s) <= 0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} stream must be finite and strictly increasing")
    return time_s, values


def _interpolate(time_s: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(grid, time_s, values[:, channel])
        for channel in range(values.shape[1])
    ])


def make_windows(
    emg_time_s,
    emg_values,
    imu_time_s,
    imu_values,
    intervals: Iterable[StableInterval],
    *,
    physical_subject_id: str,
    acquisition_subject_id: str,
    session_id: str,
    window_s: float = WINDOW_SECONDS,
    step_s: float = STEP_SECONDS,
) -> WindowBatch:
    if not np.isclose(window_s, WINDOW_SECONDS) or not np.isclose(step_s, STEP_SECONDS):
        raise ValueError("the eight_pose_v5 dataset contract requires 0.5 s windows and 0.1 s steps")
    emg_time, emg = _validate_stream(emg_time_s, emg_values, 8, "EMG")
    imu_time, imu = _validate_stream(imu_time_s, imu_values, 6, "IMU")

    emg_windows = []
    imu_windows = []
    labels = []
    physical_ids = []
    acquisition_ids = []
    session_ids = []
    trial_ids = []
    block_ids = []
    interval_ids = []
    tolerance = 1e-9

    for interval in intervals:
        available = interval.end_s - interval.start_s - window_s
        if available < -tolerance:
            continue
        count = int(np.floor(max(0.0, available) / step_s + tolerance)) + 1
        for offset_index in range(count):
            start = interval.start_s + offset_index * step_s
            emg_grid = start + np.arange(EMG_SAMPLES, dtype=np.float64) / EMG_RATE_HZ
            imu_grid = start + np.arange(IMU_SAMPLES, dtype=np.float64) / IMU_RATE_HZ
            if (
                emg_grid[0] < emg_time[0] - tolerance
                or emg_grid[-1] > emg_time[-1] + tolerance
                or imu_grid[0] < imu_time[0] - tolerance
                or imu_grid[-1] > imu_time[-1] + tolerance
            ):
                continue
            emg_windows.append(_interpolate(emg_time, emg, emg_grid).T.astype(np.float32))
            imu_windows.append(_interpolate(imu_time, imu, imu_grid).T.astype(np.float32))
            labels.append(interval.label_id)
            physical_ids.append(physical_subject_id)
            acquisition_ids.append(acquisition_subject_id)
            session_ids.append(session_id)
            trial_ids.append(interval.trial_id)
            block_ids.append(interval.block_id)
            interval_ids.append(interval.interval_id)

    if not emg_windows:
        return _empty_batch()
    return WindowBatch(
        emg=np.stack(emg_windows),
        imu=np.stack(imu_windows),
        labels=np.asarray(labels, dtype=np.int64),
        physical_subject_ids=np.asarray(physical_ids, dtype=str),
        acquisition_subject_ids=np.asarray(acquisition_ids, dtype=str),
        session_ids=np.asarray(session_ids, dtype=str),
        trial_ids=np.asarray(trial_ids, dtype=np.int32),
        block_ids=np.asarray(block_ids, dtype=np.int16),
        interval_ids=np.asarray(interval_ids, dtype=str),
    )


def concatenate_batches(batches: Iterable[WindowBatch]) -> WindowBatch:
    batches = [batch for batch in batches if batch.labels.size]
    if not batches:
        return _empty_batch()
    fields = WindowBatch.__dataclass_fields__
    return WindowBatch(**{
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in fields
    })

