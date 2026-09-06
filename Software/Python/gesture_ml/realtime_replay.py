"""Causal chronological replay of complete eight_pose_v5 capture sessions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np

from .artifacts import SessionArtifacts
from .build_dataset import (
    FILTERED_HEADER,
    IMU_HEADER,
    RAW_HEADER,
    _load_numeric,
    _read_events,
    _require_header,
)
from .intervals import CLASS_LABELS, TARGET_LABELS, extract_stable_intervals
from .realtime_inference import RealtimeModelRuntime
from .timebase import shared_sensor_time_axes
from .windowing import (
    EMG_RATE_HZ,
    EMG_SAMPLES,
    IMU_RATE_HZ,
    IMU_SAMPLES,
    STEP_SECONDS,
    WINDOW_SECONDS,
)


_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class StreamingInterval:
    """One scored stable or transition interval from the v5 lifecycle."""

    kind: str
    start_s: float
    end_s: float
    label: str | None
    label_id: int | None
    transition_kind: str | None
    block_id: int
    trial_id: int
    interval_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"stable", "transition"}:
            raise ValueError("streaming interval kind must be stable or transition")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError("streaming interval bounds must be finite and increasing")


@dataclass(frozen=True)
class SensorReplayWindow:
    """One immutable causal dual-stream input in chronological sensor time."""

    session_id: str
    sequence: int
    start_s: float
    sensor_time_s: float
    emg: np.ndarray
    imu: np.ndarray
    stable_label: str | None
    stable_label_id: int | None
    transition_kind: str | None
    block_id: int | None
    trial_id: int | None
    interval_id: str | None

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("emg", self.emg, (8, EMG_SAMPLES)),
            ("imu", self.imu, (6, IMU_SAMPLES)),
        ):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f"{name} has the wrong sensor replay shape or dtype")
            if not bool(np.all(np.isfinite(value))):
                raise ValueError(f"{name} must be finite")
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


@dataclass(frozen=True)
class ReplayWindow:
    """One immutable model input and prediction in chronological sensor time."""

    session_id: str
    sequence: int
    start_s: float
    sensor_time_s: float
    emg: np.ndarray
    imu: np.ndarray
    probabilities: np.ndarray
    stable_label: str | None
    stable_label_id: int | None
    transition_kind: str | None
    block_id: int | None
    trial_id: int | None
    interval_id: str | None

    def __post_init__(self) -> None:
        arrays = (
            ("emg", self.emg, (8, EMG_SAMPLES)),
            ("imu", self.imu, (6, IMU_SAMPLES)),
            ("probabilities", self.probabilities, (len(CLASS_LABELS),)),
        )
        for name, value, shape in arrays:
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f"{name} has the wrong replay shape or dtype")
            if not bool(np.all(np.isfinite(value))):
                raise ValueError(f"{name} must be finite")
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if (
            bool(np.any(self.probabilities < 0.0))
            or bool(np.any(self.probabilities > 1.0))
            or abs(float(self.probabilities.sum(dtype=np.float64)) - 1.0) > 1e-4
        ):
            raise ValueError("replay probabilities violate the probability contract")


@dataclass(frozen=True)
class ExpectedTargetInterval:
    """Authoritative target onset retained even when no sensor window is emitted."""

    session_id: str
    interval_id: str
    label: str
    label_id: int
    block_id: int
    trial_id: int
    onset_sensor_time_s: float

    def __post_init__(self) -> None:
        if self.label not in TARGET_LABELS or self.label_id != CLASS_LABELS.index(self.label):
            raise ValueError("expected target interval label violates the fixed class order")
        if not self.interval_id or not math.isfinite(self.onset_sensor_time_s):
            raise ValueError("expected target interval identity and onset are required")


@dataclass(frozen=True)
class SensorReplaySession(Sequence):
    """Immutable sensor inputs plus lifecycle-authoritative expected onsets."""

    session_id: str
    windows: tuple[SensorReplayWindow, ...]
    expected_target_intervals: tuple[ExpectedTargetInterval, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "expected_target_intervals", tuple(self.expected_target_intervals))
        if any(item.session_id != self.session_id for item in self.windows):
            raise ValueError("sensor replay window session identity mismatch")
        if any(item.session_id != self.session_id for item in self.expected_target_intervals):
            raise ValueError("expected target interval session identity mismatch")
        identities = [item.interval_id for item in self.expected_target_intervals]
        if len(identities) != len(set(identities)):
            raise ValueError("expected target interval identities must be unique")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]

    def __iter__(self):
        return iter(self.windows)


@dataclass(frozen=True)
class ReplaySession(Sequence):
    """Immutable replay records plus lifecycle-authoritative expected onsets."""

    session_id: str
    windows: tuple[ReplayWindow, ...]
    expected_target_intervals: tuple[ExpectedTargetInterval, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "expected_target_intervals", tuple(self.expected_target_intervals))
        if any(item.session_id != self.session_id for item in self.windows):
            raise ValueError("replay window session identity mismatch")
        if any(item.session_id != self.session_id for item in self.expected_target_intervals):
            raise ValueError("expected target interval session identity mismatch")
        identities = [item.interval_id for item in self.expected_target_intervals]
        if len(identities) != len(set(identities)):
            raise ValueError("expected target interval identities must be unique")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]

    def __iter__(self):
        return iter(self.windows)


@dataclass(frozen=True)
class SessionReplayAudit:
    """Freshly reconstructed v5 identity/lifecycle evidence used by replay."""

    session_id: str
    performed_trial_ids: tuple[int, ...]
    invalid_trial_ids: tuple[int, ...]
    streaming_intervals: tuple[StreamingInterval, ...]
    invalid_trial_spans: tuple[tuple[float, float], ...]
    expected_target_intervals: tuple[ExpectedTargetInterval, ...]


def _event_bool(value) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(f"invalid event valid flag: {value!r}")


def _event_nonnegative_integer(value, field: str, row_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"event row {row_index} {field} must be an exact finite nonnegative integer")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError(
            f"event row {row_index} {field} must be an exact finite nonnegative integer",
        ) from error
    if (
        not number.is_finite()
        or number < 0
        or number != number.to_integral_value()
    ):
        raise ValueError(
            f"event row {row_index} {field} must be an exact finite nonnegative integer",
        )
    return int(number)


def _parsed_event_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    parsed = []
    previous_time = -math.inf
    for row_index, row in enumerate(rows, start=1):
        try:
            block_value = row["block_id"]
            trial_value = row["trial_id"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed event row {row_index}") from error
        block_id = _event_nonnegative_integer(block_value, "block_id", row_index)
        trial_id = _event_nonnegative_integer(trial_value, "trial_id", row_index)
        try:
            item = {
                "time_s": float(row["session_time_s"]),
                "event_type": str(row["event_type"]).strip(),
                "block_id": block_id,
                "trial_id": trial_id,
                "gesture_id": str(row["gesture_id"]).strip(),
                "valid": _event_bool(row["valid"]),
                "note": str(row.get("note", "")).strip(),
                "row": row_index,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed event row {row_index}") from error
        if not math.isfinite(item["time_s"]) or item["time_s"] < previous_time:
            raise ValueError("event times must be finite and nondecreasing")
        previous_time = item["time_s"]
        parsed.append(item)
    return parsed


def _transition_pair(trial_events: list[dict], start_type: str, end_type: str) -> tuple[dict, dict]:
    starts = [item for item in trial_events if item["event_type"] == start_type]
    ends = [item for item in trial_events if item["event_type"] == end_type]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"valid trial must contain exactly one {start_type}/{end_type} pair")
    start, end = starts[0], ends[0]
    if (
        end["row"] <= start["row"]
        or end["time_s"] <= start["time_s"]
        or start["block_id"] != end["block_id"]
        or start["trial_id"] != end["trial_id"]
        or start["gesture_id"] != end["gesture_id"]
        or not start["valid"]
        or not end["valid"]
    ):
        raise ValueError(f"mismatched {start_type}/{end_type} events")
    return start, end


def derive_streaming_intervals(
    rows: Iterable[Mapping[str, object]],
) -> tuple[StreamingInterval, ...]:
    """Derive guarded stable and movement/return scoring intervals."""
    source_rows = list(rows)
    stable = extract_stable_intervals(source_rows)
    parsed = _parsed_event_rows(source_rows)
    invalid_trials = {
        item["trial_id"] for item in parsed if item["event_type"] == "trial_invalid"
    }
    intervals = [
        StreamingInterval(
            kind="stable",
            start_s=item.start_s,
            end_s=item.end_s,
            label=item.label,
            label_id=item.label_id,
            transition_kind=None,
            block_id=item.block_id,
            trial_id=item.trial_id,
            interval_id=item.interval_id,
        )
        for item in stable
    ]
    valid_trial_ids = sorted({
        item["trial_id"]
        for item in parsed
        if item["trial_id"] > 0
        and item["trial_id"] not in invalid_trials
        and item["event_type"] == "movement_start"
    })
    for trial_id in valid_trial_ids:
        trial_events = [item for item in parsed if item["trial_id"] == trial_id]
        movement_start, movement_end = _transition_pair(
            trial_events, "movement_start", "target_hold_start",
        )
        return_start, return_end = _transition_pair(
            trial_events, "return_start", "return_end",
        )
        for transition_kind, start, end in (
            ("movement", movement_start, movement_end),
            ("return", return_start, return_end),
        ):
            intervals.append(StreamingInterval(
                kind="transition",
                start_s=start["time_s"],
                end_s=end["time_s"],
                label=None,
                label_id=None,
                transition_kind=transition_kind,
                block_id=start["block_id"],
                trial_id=trial_id,
                interval_id=f"trial-{trial_id}-{transition_kind}",
            ))
    intervals.sort(key=lambda item: (item.start_s, item.trial_id, item.interval_id))
    return tuple(intervals)


def _validate_six_file_eligibility(artifacts: SessionArtifacts) -> None:
    if not isinstance(artifacts, SessionArtifacts):
        raise TypeError("artifacts must be SessionArtifacts")
    suffixes = ("_raw.csv", "_filtered.csv", "_imu.csv", "_events.csv", "_metadata.json", "_quality.json")
    if any(
        not path.is_file() or path.name != f"{artifacts.session_id}{suffix}"
        for path, suffix in zip(artifacts.paths, suffixes)
    ):
        raise ValueError(f"session is not an exact six-file artifact set: {artifacts.session_id}")


def _load_json_object(path: Path, name: str) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"{name} contains non-finite constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not load {name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


_V5_LIFECYCLE = (
    "rest_valid_start", "rest_valid_end", "prepare", "movement_start",
    "target_hold_start", "target_hold_end", "return_start", "return_end",
    "rest_valid_start", "rest_valid_end",
)
_V5_OFFSETS_S = (0.0, 2.0, 2.0, 3.0, 4.5, 8.5, 8.5, 10.5, 10.5, 12.5)
_REPLACEMENT_NOTE = re.compile(r"replacement_trial_id=(\d+)")
_V5_ALLOWED_EVENT_TYPES = frozenset({
    "session_start", "session_end", "trial_invalid", *_V5_LIFECYCLE,
})


def _finite_equal(value, expected: float, field: str) -> None:
    try:
        actual = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{field} does not match the exact v5 duration/terminal contract")


def _audit_metadata(artifacts: SessionArtifacts, metadata: dict) -> tuple[dict, ...]:
    if (
        metadata.get("session_id") != artifacts.session_id
        or metadata.get("subject_id") != artifacts.acquisition_subject_id
    ):
        raise ValueError("metadata identity does not bind to the artifact basename")
    if metadata.get("protocol_version") != "eight_pose_v5" or metadata.get("block_count") != 2:
        raise ValueError("metadata protocol must be exact eight_pose_v5 with two blocks")
    if (
        metadata.get("completed") is not True
        or metadata.get("stopped_by_operator") is not False
        or metadata.get("terminal_status") != "completed"
    ):
        raise ValueError("metadata terminal state is not a completed v5 session")
    table = metadata.get("trial_table")
    if not isinstance(table, list) or len(table) != 14:
        raise ValueError("metadata trial table must contain exactly 14 original trials")
    normalized = []
    for expected_id, row in enumerate(table, start=1):
        if not isinstance(row, dict) or set(row) != {"trial_id", "block_id", "gesture_id", "is_repeat"}:
            raise ValueError("metadata trial table row schema is invalid")
        expected_block = 1 if expected_id <= 7 else 2
        if (
            row["trial_id"] != expected_id
            or row["block_id"] != expected_block
            or row["gesture_id"] not in TARGET_LABELS
            or row["is_repeat"] is not False
        ):
            raise ValueError("metadata trial table identity or order is invalid")
        normalized.append(dict(row))
    for block_id in (1, 2):
        labels = [row["gesture_id"] for row in normalized if row["block_id"] == block_id]
        if set(labels) != set(TARGET_LABELS) or len(labels) != len(set(labels)):
            raise ValueError("metadata trial table must schedule every target once per block")
    return tuple(normalized)


def _audit_event_lifecycle(
    session_id: str,
    parsed: list[dict],
    original_trials: tuple[dict, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, tuple[int, str]], float]:
    if not parsed:
        raise ValueError("events are empty")
    unapproved = sorted({
        item["event_type"] for item in parsed
        if item["event_type"] not in _V5_ALLOWED_EVENT_TYPES
    })
    if unapproved:
        raise ValueError(f"events contain an unapproved v5 event type: {unapproved}")
    extra_trial_zero = [
        item for item in parsed
        if item["trial_id"] == 0 and item["event_type"] != "session_end"
    ]
    if extra_trial_zero:
        raise ValueError("events contain an unapproved extra trial-zero row")
    start_rows = [item for item in parsed if item["event_type"] == "session_start"]
    end_rows = [item for item in parsed if item["event_type"] == "session_end"]
    other_terminals = [item for item in parsed if item["event_type"] == "session_stopped"]
    if len(start_rows) != 1 or start_rows[0]["row"] != 1 or start_rows[0]["time_s"] != 0.0:
        raise ValueError("events require one exact initial session_start terminal")
    if len(end_rows) != 1 or end_rows[0]["row"] != len(parsed) or other_terminals:
        raise ValueError("events require one exact final session_end terminal")
    if (
        start_rows[0]["trial_id"] != 1
        or start_rows[0]["block_id"] != original_trials[0]["block_id"]
        or start_rows[0]["gesture_id"] != original_trials[0]["gesture_id"]
        or not start_rows[0]["valid"]
    ):
        raise ValueError("session_start does not bind to metadata trial one")
    if (
        end_rows[0]["trial_id"] != 0
        or end_rows[0]["block_id"] != 0
        or end_rows[0]["gesture_id"] != ""
        or not end_rows[0]["valid"]
    ):
        raise ValueError("session_end context is invalid")

    trial_rows = [item for item in parsed if item["trial_id"] > 0 and item["event_type"] != "session_start"]
    performed_ids = tuple(dict.fromkeys(item["trial_id"] for item in trial_rows))
    contiguous_ids = []
    for item in trial_rows:
        if not contiguous_ids or contiguous_ids[-1] != item["trial_id"]:
            contiguous_ids.append(item["trial_id"])
    if tuple(contiguous_ids) != performed_ids:
        raise ValueError("trial lifecycle rows must remain contiguous and in performed order")
    if performed_ids != tuple(range(1, len(performed_ids) + 1)) or len(performed_ids) < 14:
        raise ValueError("performed trial identities must be sequential originals plus tail replacements")
    if performed_ids[:14] != tuple(range(1, 15)):
        raise ValueError("the 14 metadata trials must be performed before replacements")

    invalid_ids = []
    replacement_by_invalid: dict[int, int] = {}
    trial_context: dict[int, tuple[int, str]] = {}
    previous_end = 0.0
    for trial_id in performed_ids:
        rows = [item for item in trial_rows if item["trial_id"] == trial_id]
        base_rows = [item for item in rows if item["event_type"] != "trial_invalid"]
        if tuple(item["event_type"] for item in base_rows) != _V5_LIFECYCLE:
            raise ValueError(f"trial {trial_id} lifecycle is not the exact ten-event v5 sequence")
        block_id = base_rows[0]["block_id"]
        gesture_id = base_rows[0]["gesture_id"]
        if any(item["block_id"] != block_id or item["gesture_id"] != gesture_id for item in rows):
            raise ValueError(f"trial {trial_id} lifecycle identity is inconsistent")
        trial_context[trial_id] = (block_id, gesture_id)
        if trial_id <= 14:
            metadata_row = original_trials[trial_id - 1]
            if (block_id, gesture_id) != (metadata_row["block_id"], metadata_row["gesture_id"]):
                raise ValueError("events do not match metadata trial table identity/order")
        _finite_equal(base_rows[0]["time_s"], previous_end, f"trial {trial_id} continuity")
        for item, offset in zip(base_rows, _V5_OFFSETS_S):
            _finite_equal(
                item["time_s"], base_rows[0]["time_s"] + offset,
                f"trial {trial_id} lifecycle duration",
            )
        previous_end = base_rows[-1]["time_s"]
        invalid_markers = [item for item in rows if item["event_type"] == "trial_invalid"]
        if len(invalid_markers) > 1:
            raise ValueError("a trial may contain at most one invalid marker")
        if invalid_markers:
            marker = invalid_markers[0]
            if not base_rows[0]["row"] < marker["row"] < base_rows[-1]["row"]:
                raise ValueError("invalid marker must be strictly inside its own lifecycle boundary rows")
            match = _REPLACEMENT_NOTE.fullmatch(marker["note"])
            if marker["valid"] or match is None:
                raise ValueError("invalid trial replacement marker/note is malformed")
            replacement_id = int(match.group(1))
            invalid_ids.append(trial_id)
            replacement_by_invalid[trial_id] = replacement_id
            for item in base_rows:
                expected_valid = item["row"] < marker["row"]
                if item["valid"] is not expected_valid:
                    raise ValueError("invalid trial lifecycle valid flags do not follow the marker")
        elif any(not item["valid"] for item in base_rows):
            raise ValueError("valid trial lifecycle contains false validity")

    tail_ids = tuple(range(15, len(performed_ids) + 1))
    replacement_ids = tuple(replacement_by_invalid[trial_id] for trial_id in invalid_ids)
    if replacement_ids != tail_ids:
        raise ValueError("invalid replacement relations must reference sequential tail trials")
    for invalid_id, replacement_id in replacement_by_invalid.items():
        if replacement_id not in trial_context or trial_context[replacement_id] != trial_context[invalid_id]:
            raise ValueError("replacement trial target/block does not match its invalid source")
    _finite_equal(end_rows[0]["time_s"], previous_end, "session_end terminal")
    return performed_ids, tuple(invalid_ids), trial_context, previous_end


def _audit_quality(
    quality: dict,
    invalid_ids: tuple[int, ...],
    trial_context: dict[int, tuple[int, str]],
    terminal_time_s: float,
) -> None:
    quality_invalid = quality.get("invalid_trial_ids")
    if isinstance(quality_invalid, int) and not isinstance(quality_invalid, bool):
        quality_invalid = [quality_invalid]
    valid_counts = {
        label: sum(
            1 for trial_id, (_, gesture) in trial_context.items()
            if trial_id not in invalid_ids and gesture == label
        )
        for label in TARGET_LABELS
    }
    expected_counts = [valid_counts[label] for label in TARGET_LABELS]
    if (
        quality.get("schema_version") != "gesture_session_quality_v4"
        or quality.get("passed") is not True
        or quality.get("fail_reasons") != []
        or quality.get("target_labels") != list(TARGET_LABELS)
        or quality.get("expected_valid_trials_per_target") != 2
        or quality_invalid != list(invalid_ids)
        or quality.get("valid_trial_counts") != expected_counts
        or quality.get("target_hold_start_counts") != expected_counts
        or quality.get("target_hold_end_counts") != expected_counts
        or quality.get("valid_trial_count_by_gesture") != valid_counts
        or expected_counts != [2] * len(TARGET_LABELS)
    ):
        raise ValueError("quality identity/count evidence does not match fresh lifecycle audit")
    _finite_equal(quality.get("max_event_time_s"), terminal_time_s, "quality max_event_time_s")


def audit_replay_session(artifacts: SessionArtifacts) -> SessionReplayAudit:
    """Freshly validate metadata/events/quality before any sensor window is emitted."""
    _validate_six_file_eligibility(artifacts)
    metadata = _load_json_object(artifacts.metadata_path, "metadata")
    quality = _load_json_object(artifacts.quality_path, "quality")
    event_rows = _read_events(artifacts.events_path)
    parsed = _parsed_event_rows(event_rows)
    original_trials = _audit_metadata(artifacts, metadata)
    performed, invalid_ids, trial_context, terminal_time = _audit_event_lifecycle(
        artifacts.session_id, parsed, original_trials,
    )
    _audit_quality(quality, invalid_ids, trial_context, terminal_time)
    streaming_intervals = derive_streaming_intervals(event_rows)
    invalid_spans = _invalid_trial_spans(event_rows)
    expected = []
    for item in streaming_intervals:
        if item.kind != "stable" or item.label == "REST":
            continue
        start_index = int(math.ceil((item.start_s - _TOLERANCE_S) / STEP_SECONDS))
        onset_sensor_time_s = start_index * STEP_SECONDS + WINDOW_SECONDS
        if onset_sensor_time_s > item.end_s + _TOLERANCE_S:
            raise ValueError("target stable interval cannot contain a complete v5 replay window")
        expected.append(ExpectedTargetInterval(
            session_id=artifacts.session_id,
            interval_id=item.interval_id,
            label=str(item.label),
            label_id=int(item.label_id),
            block_id=item.block_id,
            trial_id=item.trial_id,
            onset_sensor_time_s=float(onset_sensor_time_s),
        ))
    if len(expected) != 14:
        raise ValueError("fresh lifecycle audit must yield exactly 14 valid target intervals")
    return SessionReplayAudit(
        session_id=artifacts.session_id,
        performed_trial_ids=performed,
        invalid_trial_ids=invalid_ids,
        streaming_intervals=streaming_intervals,
        invalid_trial_spans=invalid_spans,
        expected_target_intervals=tuple(expected),
    )


def _validated_stream(time_s, values, channels: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    time_values = np.asarray(time_s, dtype=np.float64)
    sensor_values = np.asarray(values, dtype=np.float64)
    if time_values.ndim != 1 or sensor_values.shape != (time_values.size, channels):
        raise ValueError(f"{name} stream has an invalid shape")
    if (
        time_values.size == 0
        or not bool(np.all(np.isfinite(time_values)))
        or not bool(np.all(np.diff(time_values) > 0.0))
        or not bool(np.all(np.isfinite(sensor_values)))
    ):
        raise ValueError(f"{name} stream must be finite and strictly increasing")
    return time_values, sensor_values


def _causal_interpolate(
    time_s: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    sensor_time_s: float,
) -> np.ndarray | None:
    causal_count = int(np.searchsorted(time_s, sensor_time_s + _TOLERANCE_S, side="right"))
    causal_time = time_s[:causal_count]
    if (
        causal_time.size == 0
        or grid[0] < causal_time[0] - _TOLERANCE_S
        or grid[-1] > causal_time[-1] + _TOLERANCE_S
    ):
        return None
    causal_values = values[:causal_count]
    result = np.column_stack([
        np.interp(grid, causal_time, causal_values[:, channel])
        for channel in range(causal_values.shape[1])
    ])
    return result.T.astype(np.float32)


def _containing_interval(
    intervals: tuple[StreamingInterval, ...],
    start_s: float,
    end_s: float,
) -> StreamingInterval | None:
    matches = [
        item for item in intervals
        if start_s >= item.start_s - _TOLERANCE_S and end_s <= item.end_s + _TOLERANCE_S
    ]
    if len(matches) > 1:
        raise ValueError("a replay window belongs to overlapping scoring intervals")
    return matches[0] if matches else None


def _invalid_trial_spans(rows: Iterable[Mapping[str, object]]) -> tuple[tuple[float, float], ...]:
    parsed = _parsed_event_rows(rows)
    invalid_ids = {
        item["trial_id"] for item in parsed if item["event_type"] == "trial_invalid"
    }
    spans = []
    for trial_id in sorted(invalid_ids):
        trial_times = [item["time_s"] for item in parsed if item["trial_id"] == trial_id]
        if not trial_times or max(trial_times) <= min(trial_times):
            raise ValueError("invalid trial does not have a complete lifecycle span")
        spans.append((min(trial_times), max(trial_times)))
    return tuple(spans)


def _overlaps_invalid_trial(
    spans: tuple[tuple[float, float], ...], start_s: float, end_s: float,
) -> bool:
    return any(
        start_s < invalid_end - _TOLERANCE_S and end_s > invalid_start + _TOLERANCE_S
        for invalid_start, invalid_end in spans
    )


def replay_sensor_session(artifacts: SessionArtifacts) -> SensorReplaySession:
    """Build every causal 0.1-second sensor window with complete stream coverage."""
    audit = audit_replay_session(artifacts)
    _require_header(artifacts.raw_path, RAW_HEADER)
    _require_header(artifacts.filtered_path, FILTERED_HEADER)
    _require_header(artifacts.imu_path, IMU_HEADER)
    raw = _load_numeric(artifacts.raw_path, (1,))
    filtered = _load_numeric(artifacts.filtered_path, tuple(range(1, 11)))
    imu = _load_numeric(artifacts.imu_path, tuple(range(1, 8)))
    _, filtered_time, imu_time = shared_sensor_time_axes(raw[:, 0], filtered[:, 0], imu[:, 0])
    emg_time, emg_values = _validated_stream(filtered_time, filtered[:, 2:10], 8, "EMG")
    imu_time, imu_values = _validated_stream(imu_time, imu[:, 1:7], 6, "IMU")
    intervals = audit.streaming_intervals
    invalid_spans = audit.invalid_trial_spans

    coverage_end = min(float(emg_time[-1]), float(imu_time[-1]))
    last_start_index = int(math.floor(
        (coverage_end - WINDOW_SECONDS + _TOLERANCE_S) / STEP_SECONDS,
    ))
    windows = []
    for start_index in range(max(-1, last_start_index) + 1):
        start_s = start_index * STEP_SECONDS
        sensor_time_s = start_s + WINDOW_SECONDS
        if _overlaps_invalid_trial(invalid_spans, start_s, sensor_time_s):
            continue
        emg_grid = start_s + np.arange(EMG_SAMPLES, dtype=np.float64) / EMG_RATE_HZ
        imu_grid = start_s + np.arange(IMU_SAMPLES, dtype=np.float64) / IMU_RATE_HZ
        emg_window = _causal_interpolate(emg_time, emg_values, emg_grid, sensor_time_s)
        imu_window = _causal_interpolate(imu_time, imu_values, imu_grid, sensor_time_s)
        if emg_window is None or imu_window is None:
            continue
        scoring_interval = _containing_interval(
            intervals, start_s, start_s + WINDOW_SECONDS,
        )
        stable = scoring_interval if scoring_interval is not None and scoring_interval.kind == "stable" else None
        transition = (
            scoring_interval
            if scoring_interval is not None and scoring_interval.kind == "transition"
            else None
        )
        windows.append(SensorReplayWindow(
            session_id=artifacts.session_id,
            sequence=len(windows),
            start_s=float(start_s),
            sensor_time_s=float(sensor_time_s),
            emg=emg_window,
            imu=imu_window,
            stable_label=None if stable is None else stable.label,
            stable_label_id=None if stable is None else stable.label_id,
            transition_kind=None if transition is None else transition.transition_kind,
            block_id=None if scoring_interval is None else scoring_interval.block_id,
            trial_id=None if scoring_interval is None else scoring_interval.trial_id,
            interval_id=None if scoring_interval is None else scoring_interval.interval_id,
        ))
    return SensorReplaySession(
        session_id=artifacts.session_id,
        windows=tuple(windows),
        expected_target_intervals=audit.expected_target_intervals,
    )


def replay_session(
    artifacts: SessionArtifacts,
    runtime: RealtimeModelRuntime,
) -> ReplaySession:
    """Map strict sensor replay windows through the eight-class runtime."""
    if not hasattr(runtime, "predict_probabilities"):
        raise TypeError("runtime must provide predict_probabilities")
    sensor_session = replay_sensor_session(artifacts)
    windows = []
    for sensor in sensor_session:
        probabilities = runtime.predict_probabilities(sensor.emg, sensor.imu)
        windows.append(ReplayWindow(
            session_id=sensor.session_id,
            sequence=sensor.sequence,
            start_s=sensor.start_s,
            sensor_time_s=sensor.sensor_time_s,
            emg=sensor.emg,
            imu=sensor.imu,
            probabilities=np.asarray(probabilities),
            stable_label=sensor.stable_label,
            stable_label_id=sensor.stable_label_id,
            transition_kind=sensor.transition_kind,
            block_id=sensor.block_id,
            trial_id=sensor.trial_id,
            interval_id=sensor.interval_id,
        ))
    return ReplaySession(
        session_id=sensor_session.session_id,
        windows=tuple(windows),
        expected_target_intervals=sensor_session.expected_target_intervals,
    )
