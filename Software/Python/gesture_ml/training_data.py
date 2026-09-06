"""Fixed session split and train-only transforms for personalized training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EXPECTED_SESSIONS = (
    "S001_D01_R02",
    "S001_D02_R02",
    "S002_D01_R01",
    "S002_D02_R01",
    "S003_D01_R01",
    "S003_D01_R02",
    "S003_D02_R01",
    "S003_D02_R02",
    "S003_D03_R01",
    "S003_D03_R02",
    "S003_D04_R01",
    "S003_D04_R02",
    "S003_D05_R01",
    "S003_D05_R02",
    "S003_D06_R01",
    "S003_D06_R02",
    "S003_D07_R01",
    "S003_D08_R01",
    "S003_D08_R02",
    "S003_D09_R01",
    "S003_D09_R02",
)

TRAIN_SESSIONS = EXPECTED_SESSIONS[:14]
VALIDATION_SESSIONS = EXPECTED_SESSIONS[14:17]
TEST_SESSIONS = EXPECTED_SESSIONS[17:]


@dataclass(frozen=True)
class NormalizationStats:
    emg_mean: np.ndarray
    emg_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray


def fixed_session_split(session_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Validate the dataset session set and return the frozen chronological split."""
    observed = {str(value) for value in session_ids}
    expected = set(EXPECTED_SESSIONS)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"dataset session set mismatch; missing={missing}, extra={extra}")

    split = {
        "train": TRAIN_SESSIONS,
        "validation": VALIDATION_SESSIONS,
        "test": TEST_SESSIONS,
    }
    flattened = split["train"] + split["validation"] + split["test"]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("fixed session split contains overlapping membership")
    return split


def _validate_sensor_arrays(emg: np.ndarray, imu: np.ndarray):
    if emg.ndim != 3 or emg.shape[1] != 8 or emg.shape[2] == 0:
        raise ValueError(f"EMG shape must be [N, 8, samples], got {emg.shape}")
    if imu.ndim != 3 or imu.shape[1] != 6 or imu.shape[2] == 0:
        raise ValueError(f"IMU shape must be [N, 6, samples], got {imu.shape}")
    if emg.shape[0] != imu.shape[0]:
        raise ValueError("EMG and IMU shape must have the same window count")
    if not np.all(np.isfinite(emg)) or not np.all(np.isfinite(imu)):
        raise ValueError("EMG and IMU values must all be finite")


def _validate_indices(indices: np.ndarray, row_count: int) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("training indices must be a nonempty one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if int(values.min()) < 0 or int(values.max()) >= row_count:
        raise ValueError("training index is outside the available windows")
    return values


def compute_normalization(
    emg: np.ndarray,
    imu: np.ndarray,
    training_indices: np.ndarray,
    *,
    standard_deviation_floor: float = 1e-6,
) -> NormalizationStats:
    """Calculate per-channel mean/std from training windows only."""
    _validate_sensor_arrays(emg, imu)
    indices = _validate_indices(training_indices, emg.shape[0])
    if not np.isfinite(standard_deviation_floor) or standard_deviation_floor <= 0:
        raise ValueError("standard deviation floor must be positive and finite")

    selected_emg = emg[indices].astype(np.float64, copy=False)
    selected_imu = imu[indices].astype(np.float64, copy=False)
    emg_mean = selected_emg.mean(axis=(0, 2))
    imu_mean = selected_imu.mean(axis=(0, 2))
    emg_std = np.maximum(selected_emg.std(axis=(0, 2)), standard_deviation_floor)
    imu_std = np.maximum(selected_imu.std(axis=(0, 2)), standard_deviation_floor)
    return NormalizationStats(
        emg_mean=emg_mean.astype(np.float32),
        emg_std=emg_std.astype(np.float32),
        imu_mean=imu_mean.astype(np.float32),
        imu_std=imu_std.astype(np.float32),
    )


def normalize_arrays(
    emg: np.ndarray,
    imu: np.ndarray,
    stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply precomputed per-channel statistics without mutating the inputs."""
    _validate_sensor_arrays(emg, imu)
    if stats.emg_mean.shape != (8,) or stats.emg_std.shape != (8,):
        raise ValueError("EMG normalization statistics must have shape [8]")
    if stats.imu_mean.shape != (6,) or stats.imu_std.shape != (6,):
        raise ValueError("IMU normalization statistics must have shape [6]")
    for value in (stats.emg_mean, stats.emg_std, stats.imu_mean, stats.imu_std):
        if not np.all(np.isfinite(value)):
            raise ValueError("normalization statistics must all be finite")
    if np.any(stats.emg_std <= 0) or np.any(stats.imu_std <= 0):
        raise ValueError("normalization standard deviations must be positive")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_emg = (
            emg.astype(np.float32, copy=False) - stats.emg_mean[None, :, None]
        ) / stats.emg_std[None, :, None]
        normalized_imu = (
            imu.astype(np.float32, copy=False) - stats.imu_mean[None, :, None]
        ) / stats.imu_std[None, :, None]
    if not np.all(np.isfinite(normalized_emg)) or not np.all(np.isfinite(normalized_imu)):
        raise ValueError("normalized sensor arrays must contain only finite values")
    return normalized_emg.astype(np.float32, copy=False), normalized_imu.astype(np.float32, copy=False)


def inverse_sqrt_class_weights(
    labels: np.ndarray,
    training_indices: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Return inverse-square-root training frequencies normalized to mean one."""
    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if class_count <= 1:
        raise ValueError("class count must be greater than one")
    indices = _validate_indices(training_indices, values.size)
    selected = values[indices].astype(np.int64, copy=False)
    if int(selected.min()) < 0 or int(selected.max()) >= class_count:
        raise ValueError("training label is outside the class range")
    counts = np.bincount(selected, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("training subset must contain every class")
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    return weights.astype(np.float32)
