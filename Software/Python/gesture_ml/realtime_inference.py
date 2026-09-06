"""Safe checkpoint inference and calibrated live decision composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .intervals import CLASS_LABELS
from .model import DualBranchCNN
from .realtime_decision import DecisionFilterConfig, RealtimeDecisionFilter


_CHECKPOINT_SCHEMA = "personalized_dual_cnn_v1"
_CALIBRATION_SCHEMA = "realtime_decision_calibration_v2"
_VALIDATION_SESSIONS = ("S003_D06_R01", "S003_D06_R02", "S003_D07_R01")
_CHECKPOINT_FIELDS = frozenset({
    "schema_version", "model_state", "normalization", "class_weights", "class_labels",
    "split", "hyperparameters", "learned_emg_weights", "dataset_sha256",
    "dataset_manifest_sha256", "best_epoch", "best_validation_macro_f1",
})
_CALIBRATION_FIELDS = frozenset({
    "schema_version", "checkpoint_sha256", "validation_session_ids", "search_grid",
    "selected", "validation_evidence", "generation_seed",
})
_SEARCH_GRID = {
    "ema_new_weights": [0.4, 0.6, 0.8],
    "minimum_confidences": [round(0.50 + 0.05 * index, 2) for index in range(10)],
    "consecutive_updates": [2, 3, 4],
}
_EVIDENCE_FIELDS = frozenset({
    "stable_macro_f1", "transition_endpoint_compatibility",
    "median_correct_onset_latency_s", "p95_correct_onset_latency_s",
    "stable_confusion_matrix", "stable_true_support", "stable_window_count",
    "stable_missed_count", "transition_window_count",
    "compatible_transition_window_count", "unrelated_transition_window_count",
    "movement_transition_window_count", "movement_compatible_transition_window_count",
    "movement_unrelated_transition_window_count", "return_transition_window_count",
    "return_compatible_transition_window_count", "return_unrelated_transition_window_count",
    "expected_target_onset_count", "correct_onset_count", "correct_onset_latencies_s",
})
_ACCEPTANCE_THRESHOLDS = {
    "stable_macro_f1_minimum": 0.95,
    "transition_endpoint_compatibility_required": 1.0,
    "median_correct_onset_latency_s_maximum": 0.8,
    "p95_correct_onset_latency_s_maximum": 1.2,
}


def _validate_search_grid(value) -> dict:
    if not isinstance(value, dict) or set(value) != set(_SEARCH_GRID):
        raise ValueError("calibration search grid fields are malformed")
    if (
        not isinstance(value["ema_new_weights"], list)
        or any(type(item) is not float for item in value["ema_new_weights"])
        or not isinstance(value["minimum_confidences"], list)
        or any(type(item) is not float for item in value["minimum_confidences"])
        or not isinstance(value["consecutive_updates"], list)
        or any(type(item) is not int for item in value["consecutive_updates"])
        or value != _SEARCH_GRID
    ):
        raise ValueError("calibration search grid values/types are invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_real(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _json_float(value, field: str, *, nullable=False) -> float | None:
    if nullable and value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite JSON float")
    return value


def _nonnegative_int(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative JSON integer")
    return value


def _macro_f1(confusion: np.ndarray, support: np.ndarray) -> float:
    scores = []
    for class_index in range(len(CLASS_LABELS)):
        true_positive = float(confusion[class_index, class_index])
        false_positive = float(confusion[:, class_index].sum()) - true_positive
        false_negative = float(support[class_index]) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores, dtype=np.float64))


def _validate_endpoint_evidence(
    value: dict,
    *,
    expected_onsets: int,
    require_pass: bool,
) -> dict:
    """Strictly recompute endpoint, stable, and onset evidence invariants."""
    if not isinstance(value, dict) or set(value) != _EVIDENCE_FIELDS:
        raise ValueError("endpoint calibration evidence fields are malformed")
    stable_f1 = _json_float(value["stable_macro_f1"], "stable_macro_f1")
    compatibility = _json_float(
        value["transition_endpoint_compatibility"],
        "transition_endpoint_compatibility",
    )
    if not 0.0 <= stable_f1 <= 1.0 or not 0.0 <= compatibility <= 1.0:
        raise ValueError("endpoint classification metrics must be within [0, 1]")

    raw_confusion = value["stable_confusion_matrix"]
    raw_support = value["stable_true_support"]
    if (
        not isinstance(raw_confusion, list)
        or len(raw_confusion) != len(CLASS_LABELS)
        or any(
            not isinstance(row, list)
            or len(row) != len(CLASS_LABELS)
            or any(type(item) is not int or item < 0 for item in row)
            for row in raw_confusion
        )
        or not isinstance(raw_support, list)
        or len(raw_support) != len(CLASS_LABELS)
        or any(type(item) is not int or item < 0 for item in raw_support)
    ):
        raise ValueError("stable confusion/support evidence is malformed")
    confusion = np.asarray(raw_confusion, dtype=np.float64)
    support = np.asarray(raw_support, dtype=np.float64)
    stable_count = _nonnegative_int(value["stable_window_count"], "stable_window_count")
    missed = _nonnegative_int(value["stable_missed_count"], "stable_missed_count")
    if (
        stable_count <= 0
        or int(support.sum()) != stable_count
        or bool(np.any(confusion.sum(axis=1) > support))
        or missed != stable_count - int(confusion.sum())
        or not math.isclose(
            stable_f1, _macro_f1(confusion, support), rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError("stable evidence is internally inconsistent")

    counts = {
        name: _nonnegative_int(value[name], name)
        for name in (
            "transition_window_count", "compatible_transition_window_count",
            "unrelated_transition_window_count", "movement_transition_window_count",
            "movement_compatible_transition_window_count",
            "movement_unrelated_transition_window_count", "return_transition_window_count",
            "return_compatible_transition_window_count",
            "return_unrelated_transition_window_count",
        )
    }
    total = counts["transition_window_count"]
    compatible = counts["compatible_transition_window_count"]
    unrelated = counts["unrelated_transition_window_count"]
    if (
        total <= 0
        or compatible + unrelated != total
        or counts["movement_transition_window_count"]
        + counts["return_transition_window_count"] != total
        or counts["movement_compatible_transition_window_count"]
        + counts["movement_unrelated_transition_window_count"]
        != counts["movement_transition_window_count"]
        or counts["return_compatible_transition_window_count"]
        + counts["return_unrelated_transition_window_count"]
        != counts["return_transition_window_count"]
        or counts["movement_compatible_transition_window_count"]
        + counts["return_compatible_transition_window_count"] != compatible
        or counts["movement_unrelated_transition_window_count"]
        + counts["return_unrelated_transition_window_count"] != unrelated
        or not math.isclose(
            compatibility, compatible / total, rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError("transition endpoint evidence is internally inconsistent")

    expected = _nonnegative_int(value["expected_target_onset_count"], "expected_target_onset_count")
    correct = _nonnegative_int(value["correct_onset_count"], "correct_onset_count")
    latencies = value["correct_onset_latencies_s"]
    if (
        expected != expected_onsets
        or correct > expected
        or not isinstance(latencies, list)
        or len(latencies) != expected
        or any(
            item is not None
            and (type(item) is not float or not math.isfinite(item) or item < 0.0)
            for item in latencies
        )
    ):
        raise ValueError("target-onset evidence is malformed")
    finite = [item for item in latencies if item is not None]
    if correct != len(finite):
        raise ValueError("correct onset count disagrees with latency evidence")
    median = _json_float(
        value["median_correct_onset_latency_s"],
        "median_correct_onset_latency_s",
        nullable=True,
    )
    p95 = _json_float(
        value["p95_correct_onset_latency_s"],
        "p95_correct_onset_latency_s",
        nullable=True,
    )
    if correct != expected:
        if median is not None or p95 is not None:
            raise ValueError("incomplete onset evidence requires null aggregate latencies")
    else:
        expected_median = float(np.median(np.asarray(finite, dtype=np.float64)))
        expected_p95 = float(np.percentile(np.asarray(finite, dtype=np.float64), 95))
        if (
            median is None
            or p95 is None
            or not math.isclose(median, expected_median, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(p95, expected_p95, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("onset aggregate latencies disagree with detailed evidence")
    if require_pass and not (
        stable_f1 >= _ACCEPTANCE_THRESHOLDS["stable_macro_f1_minimum"]
        and compatibility == _ACCEPTANCE_THRESHOLDS["transition_endpoint_compatibility_required"]
        and correct == expected
        and median is not None
        and median <= _ACCEPTANCE_THRESHOLDS["median_correct_onset_latency_s_maximum"]
        and p95 is not None
        and p95 <= _ACCEPTANCE_THRESHOLDS["p95_correct_onset_latency_s_maximum"]
    ):
        raise ValueError("endpoint calibration evidence does not pass acceptance gates")
    return value


def _require_sha256(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a lowercase SHA-256") from error
    if value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_input(values, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != shape:
        raise ValueError(f"{name} must be finite float32 with shape {list(shape)}")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_tensor(value, expected: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError(f"checkpoint tensor {name} has the wrong shape or dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"checkpoint tensor {name} is non-finite")


def _load_checkpoint(payload: bytes, path: Path) -> dict:
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    try:
        if type(payload) is not bytes:
            raise TypeError("checkpoint payload must be exact bytes")
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"could not safely load checkpoint: {error}") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("checkpoint fields do not match the training schema")
    if checkpoint["schema_version"] != _CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema version is unsupported")
    if checkpoint["class_labels"] != list(CLASS_LABELS):
        raise ValueError("checkpoint class order does not match the trained contract")
    if not isinstance(checkpoint["model_state"], dict):
        raise ValueError("checkpoint model_state must be an object")
    expected_state = DualBranchCNN(class_count=len(CLASS_LABELS)).state_dict()
    if set(checkpoint["model_state"]) != set(expected_state):
        raise ValueError("checkpoint model state keys do not match the architecture")
    for name, expected in expected_state.items():
        _validate_tensor(checkpoint["model_state"][name], expected, name)

    normalization = checkpoint["normalization"]
    expected_normalization = {
        "emg_mean": (8,), "emg_std": (8,), "imu_mean": (6,), "imu_std": (6,),
    }
    if not isinstance(normalization, dict) or set(normalization) != set(expected_normalization):
        raise ValueError("checkpoint normalization fields do not match the training schema")
    for name, shape in expected_normalization.items():
        value = normalization[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"checkpoint normalization {name} has the wrong shape or dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"checkpoint normalization {name} is non-finite")
        if name.endswith("_std") and not bool(torch.all(value > 0).item()):
            raise ValueError(f"checkpoint normalization {name} must be positive")
    _validate_tensor(checkpoint["class_weights"], torch.ones(8, dtype=torch.float32), "class_weights")
    if not bool(torch.all(checkpoint["class_weights"] > 0).item()):
        raise ValueError("checkpoint class_weights must be positive")
    if not isinstance(checkpoint["split"], dict) or set(checkpoint["split"]) != {"train", "validation", "test"}:
        raise ValueError("checkpoint split is malformed")
    if not isinstance(checkpoint["hyperparameters"], dict):
        raise ValueError("checkpoint hyperparameters are malformed")
    if not isinstance(checkpoint["learned_emg_weights"], list) or len(checkpoint["learned_emg_weights"]) != 8:
        raise ValueError("checkpoint learned_emg_weights are malformed")
    for index, value in enumerate(checkpoint["learned_emg_weights"]):
        weight = _finite_real(value, f"learned_emg_weights[{index}]")
        if not 0.0 < weight < 1.0:
            raise ValueError("checkpoint learned_emg_weights must be in (0, 1)")
    _require_sha256(checkpoint["dataset_sha256"], "dataset_sha256")
    _require_sha256(checkpoint["dataset_manifest_sha256"], "dataset_manifest_sha256")
    if isinstance(checkpoint["best_epoch"], bool) or not isinstance(checkpoint["best_epoch"], Integral) or checkpoint["best_epoch"] <= 0:
        raise ValueError("checkpoint best_epoch is malformed")
    score = _finite_real(checkpoint["best_validation_macro_f1"], "best_validation_macro_f1")
    if not 0.0 <= score <= 1.0:
        raise ValueError("checkpoint best_validation_macro_f1 must be in [0, 1]")
    return checkpoint


@dataclass(frozen=True)
class RealtimeModelRuntime:
    """A validated immutable checkpoint ready for repeated single-window inference."""

    model: DualBranchCNN
    device: torch.device
    emg_mean: np.ndarray
    emg_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray
    checkpoint_sha256: str

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str | torch.device) -> "RealtimeModelRuntime":
        path = Path(checkpoint_path).resolve()
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        try:
            checkpoint_bytes = path.read_bytes()
        except OSError as error:
            raise ValueError(f"could not read checkpoint: {error}") from error
        checkpoint = _load_checkpoint(checkpoint_bytes, path)
        model = DualBranchCNN(class_count=len(CLASS_LABELS))
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        except RuntimeError as error:
            raise ValueError(f"checkpoint model state cannot load: {error}") from error
        model.to(requested)
        model.eval()
        normalization = checkpoint["normalization"]
        return cls(
            model=model,
            device=requested,
            emg_mean=normalization["emg_mean"].numpy().copy(),
            emg_std=normalization["emg_std"].numpy().copy(),
            imu_mean=normalization["imu_mean"].numpy().copy(),
            imu_std=normalization["imu_std"].numpy().copy(),
            checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        )

    def predict_probabilities(self, emg: np.ndarray, imu: np.ndarray) -> np.ndarray:
        emg_values = _validate_input(emg, (8, 250), "EMG")
        imu_values = _validate_input(imu, (6, 52), "IMU")
        normalized_emg = (emg_values - self.emg_mean[:, None]) / self.emg_std[:, None]
        normalized_imu = (imu_values - self.imu_mean[:, None]) / self.imu_std[:, None]
        if not bool(np.all(np.isfinite(normalized_emg))) or not bool(np.all(np.isfinite(normalized_imu))):
            raise FloatingPointError("normalization produced non-finite model inputs")
        emg_tensor = torch.from_numpy(normalized_emg[None]).to(self.device)
        imu_tensor = torch.from_numpy(normalized_imu[None]).to(self.device)
        with torch.inference_mode():
            logits = self.model(emg_tensor, imu_tensor).fused_logits
            if logits.shape != (1, len(CLASS_LABELS)) or not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("model produced non-finite logits")
            probabilities = torch.softmax(logits, dim=1)
            if not bool(torch.isfinite(probabilities).all().item()):
                raise FloatingPointError("model produced non-finite probabilities")
        result = probabilities[0].detach().to("cpu").numpy().astype(np.float32, copy=True)
        if bool(np.any(result < 0.0)) or bool(np.any(result > 1.0)) or abs(float(result.sum(dtype=np.float64)) - 1.0) > 1e-4:
            raise FloatingPointError("model probabilities violate the probability contract")
        return result


def _load_calibration_bytes(payload: bytes, checkpoint_sha256: str) -> DecisionFilterConfig:
    def reject_constant(value: str):
        raise ValueError(f"calibration contains non-finite constant {value}")

    try:
        if type(payload) is not bytes:
            raise TypeError("calibration payload must be exact bytes")
        calibration = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (TypeError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not load calibration: {error}") from error
    if not isinstance(calibration, dict) or set(calibration) != _CALIBRATION_FIELDS:
        raise ValueError("calibration fields do not match the required schema")
    if calibration["schema_version"] != _CALIBRATION_SCHEMA:
        raise ValueError("calibration schema version is unsupported")
    if _require_sha256(calibration["checkpoint_sha256"], "calibration checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("calibration checkpoint SHA-256 does not match the selected checkpoint")
    if calibration["validation_session_ids"] != list(_VALIDATION_SESSIONS):
        raise ValueError("calibration validation sessions do not match the approved split")
    _validate_search_grid(calibration["search_grid"])
    selected = calibration["selected"]
    if not isinstance(selected, dict) or set(selected) != {
        "ema_new_weight", "minimum_confidence", "consecutive_updates",
    }:
        raise ValueError("calibration selected configuration is malformed")
    if (
        type(selected["ema_new_weight"]) is not float
        or type(selected["minimum_confidence"]) is not float
        or type(selected["consecutive_updates"]) is not int
    ):
        raise ValueError("calibration selected configuration types are invalid")
    config = DecisionFilterConfig(**selected)
    _validate_endpoint_evidence(
        calibration["validation_evidence"], expected_onsets=42, require_pass=True,
    )
    seed = calibration["generation_seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("calibration generation_seed is invalid")
    return config


def _load_calibration(path: Path, checkpoint_sha256: str) -> DecisionFilterConfig:
    try:
        payload = Path(path).resolve().read_bytes()
    except OSError as error:
        raise ValueError(f"could not load calibration: {error}") from error
    return _load_calibration_bytes(payload, checkpoint_sha256)


@dataclass
class RealtimeInferenceEngine:
    """Validated live inference runtime with a persistent calibrated filter."""

    runtime: RealtimeModelRuntime
    decision_filter: RealtimeDecisionFilter

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        calibration_path: str | Path,
        device: str | torch.device,
    ) -> "RealtimeInferenceEngine":
        runtime = RealtimeModelRuntime.load(checkpoint_path, device)
        config = _load_calibration(Path(calibration_path).resolve(), runtime.checkpoint_sha256)
        return cls(runtime=runtime, decision_filter=RealtimeDecisionFilter(config, CLASS_LABELS))

    def predict(self, emg: np.ndarray, imu: np.ndarray, sequence: int, sensor_time_s: float) -> dict:
        if isinstance(sequence, bool) or not isinstance(sequence, Integral) or not 0 <= int(sequence) <= 2**64 - 1:
            raise ValueError("sequence must be an unsigned 64-bit integer")
        if self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)
        start = time.perf_counter()
        probabilities = self.runtime.predict_probabilities(emg, imu)
        if self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)
        inference_ms = (time.perf_counter() - start) * 1000.0
        output = self.decision_filter.update(probabilities, sensor_time_s)
        return {
            "sequence": int(sequence),
            "sensor_time_s": float(sensor_time_s),
            "candidate_gesture_id": output.candidate_gesture_id,
            "display_state": output.display_state,
            "confidence": float(output.confidence),
            "probabilities": [float(value) for value in probabilities],
            "inference_ms": float(inference_ms),
        }
