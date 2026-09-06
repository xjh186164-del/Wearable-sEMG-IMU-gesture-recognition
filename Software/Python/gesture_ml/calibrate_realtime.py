"""Validation-only decision calibration and locked chronological test replay."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import json
import math
import msvcrt
from numbers import Integral
import os
from pathlib import Path
import secrets
import sys
from typing import Iterable, Mapping
import warnings

import numpy as np

from .artifacts import load_sessions_by_id
from .intervals import CLASS_LABELS
from .realtime_decision import DecisionFilterConfig, RealtimeDecisionFilter
from .realtime_inference import (
    RealtimeModelRuntime,
    _load_calibration,
    _load_calibration_bytes,
    _validate_endpoint_evidence,
)
from .realtime_replay import ExpectedTargetInterval, ReplaySession, ReplayWindow, replay_session


VALIDATION_SESSIONS = ("S003_D06_R01", "S003_D06_R02", "S003_D07_R01")
TEST_SESSIONS = ("S003_D08_R01", "S003_D08_R02", "S003_D09_R01", "S003_D09_R02")
SEARCH_GRID = {
    "ema_new_weights": [0.4, 0.6, 0.8],
    "minimum_confidences": [round(0.50 + 0.05 * index, 2) for index in range(10)],
    "consecutive_updates": [2, 3, 4],
}
ACCEPTANCE_THRESHOLDS = {
    "stable_macro_f1_minimum": 0.95,
    "transition_endpoint_compatibility_required": 1.0,
    "median_correct_onset_latency_s_maximum": 0.8,
    "p95_correct_onset_latency_s_maximum": 1.2,
}
_CALIBRATION_SCHEMA = "realtime_decision_calibration_v2"
_TEST_EVALUATION_SCHEMA = "realtime_test_evaluation_v2"
_TEST_EVALUATION_FIELDS = {
    "schema_version", "calibration_sha256", "checkpoint_sha256", "class_labels",
    "test_session_ids", "selected", "acceptance_thresholds", "test_evidence", "passed",
}
_ORIGINAL_OS_CLOSE = os.close


class CalibrationTestFailure(RuntimeError):
    """The locked test evaluation completed and preserved failing evidence."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """Complete streaming metrics for one decision-filter configuration."""

    config: DecisionFilterConfig
    stable_macro_f1: float
    transition_endpoint_compatibility: float
    median_correct_onset_latency_s: float
    p95_correct_onset_latency_s: float
    stable_confusion_matrix: tuple[tuple[int, ...], ...]
    stable_true_support: tuple[int, ...]
    stable_window_count: int
    stable_missed_count: int
    transition_window_count: int
    compatible_transition_window_count: int
    unrelated_transition_window_count: int
    movement_transition_window_count: int
    movement_compatible_transition_window_count: int
    movement_unrelated_transition_window_count: int
    return_transition_window_count: int
    return_compatible_transition_window_count: int
    return_unrelated_transition_window_count: int
    expected_target_onset_count: int
    correct_onset_count: int
    correct_onset_latencies_s: tuple[float, ...]

    @property
    def eligible(self) -> bool:
        return bool(
            self.stable_macro_f1 >= ACCEPTANCE_THRESHOLDS["stable_macro_f1_minimum"]
            and self.transition_endpoint_compatibility
            == ACCEPTANCE_THRESHOLDS["transition_endpoint_compatibility_required"]
            and self.expected_target_onset_count == 42
            and self.correct_onset_count == self.expected_target_onset_count
            and math.isfinite(self.median_correct_onset_latency_s)
            and self.median_correct_onset_latency_s
            <= ACCEPTANCE_THRESHOLDS["median_correct_onset_latency_s_maximum"]
            and math.isfinite(self.p95_correct_onset_latency_s)
            and self.p95_correct_onset_latency_s
            <= ACCEPTANCE_THRESHOLDS["p95_correct_onset_latency_s_maximum"]
            and all(math.isfinite(value) for value in self.correct_onset_latencies_s)
        )


def candidate_configs() -> tuple[DecisionFilterConfig, ...]:
    """Enumerate the approved grid in stable lexicographic order."""
    return tuple(
        DecisionFilterConfig(ema, confidence, consecutive)
        for ema in SEARCH_GRID["ema_new_weights"]
        for confidence in SEARCH_GRID["minimum_confidences"]
        for consecutive in SEARCH_GRID["consecutive_updates"]
    )


def _macro_f1(confusion: np.ndarray, true_counts: np.ndarray) -> float:
    scores = []
    for class_index in range(len(CLASS_LABELS)):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(true_counts[class_index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return float(np.mean(scores, dtype=np.float64))


def _latency_metrics(latencies: list[float]) -> tuple[float, float]:
    if not latencies or not all(math.isfinite(value) for value in latencies):
        return math.inf, math.inf
    values = np.asarray(latencies, dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 95))


def evaluate_replays(
    replays_by_session: Mapping[str, Iterable[ReplayWindow]],
    config: DecisionFilterConfig,
) -> CandidateEvaluation:
    """Apply one filter causally to every complete session and score fixed labels."""
    if not isinstance(config, DecisionFilterConfig):
        raise TypeError("config must be DecisionFilterConfig")
    confusion = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    true_counts = np.zeros(len(CLASS_LABELS), dtype=np.int64)
    stable_window_count = 0
    transition_window_count = 0
    compatible_transition_window_count = 0
    transition_counts = {
        "movement": {"total": 0, "compatible": 0},
        "return": {"total": 0, "compatible": 0},
    }
    target_interval_outputs: dict[tuple[str, str], list[tuple[float, str, str]]] = {}
    expected_target_onsets: dict[tuple[str, str], tuple[float, str]] = {}

    for session_id in sorted(replays_by_session):
        replay = replays_by_session[session_id]
        if isinstance(replay, ReplaySession):
            windows = replay.windows
            if replay.session_id != session_id:
                raise ValueError("replay session identity does not match its mapping key")
            for expected in replay.expected_target_intervals:
                key = (session_id, expected.interval_id)
                expected_target_onsets[key] = (expected.onset_sensor_time_s, expected.label)
        else:
            windows = tuple(replay)
        if any(not isinstance(item, ReplayWindow) for item in windows):
            raise TypeError("replays must contain ReplayWindow records")
        if any(item.session_id != session_id for item in windows):
            raise ValueError("replay session identity does not match its mapping key")
        keys = [(item.sensor_time_s, item.sequence) for item in windows]
        if keys != sorted(keys):
            raise ValueError("replay windows must already be chronological")
        trial_targets: dict[tuple[int, int], str] = {}
        if isinstance(replay, ReplaySession):
            target_records = replay.expected_target_intervals
        else:
            target_records = tuple(item for item in windows if item.stable_label not in {None, "REST"})
        for item in target_records:
            if item.block_id is None or item.trial_id is None:
                raise ValueError("target identity requires block and trial IDs")
            key = (int(item.block_id), int(item.trial_id))
            label = str(item.label if isinstance(item, ExpectedTargetInterval) else item.stable_label)
            if key in trial_targets and trial_targets[key] != label:
                raise ValueError("trial identity maps to conflicting target labels")
            trial_targets[key] = label

        decision = RealtimeDecisionFilter(config, CLASS_LABELS)
        for item in windows:
            output = decision.update(item.probabilities, item.sensor_time_s)
            if item.stable_label is not None:
                if item.stable_label not in CLASS_LABELS or item.stable_label_id != CLASS_LABELS.index(item.stable_label):
                    raise ValueError("stable replay label does not match the fixed class order")
                true_index = int(item.stable_label_id)
                true_counts[true_index] += 1
                stable_window_count += 1
                if output.display_state in CLASS_LABELS:
                    confusion[true_index, CLASS_LABELS.index(output.display_state)] += 1
                if item.stable_label != "REST":
                    if not item.interval_id:
                        raise ValueError("target stable replay windows require interval identity")
                    target_interval_outputs.setdefault(
                        (session_id, item.interval_id), [],
                    ).append((item.sensor_time_s, item.stable_label, output.display_state))
                    if not isinstance(replay, ReplaySession):
                        expected_target_onsets.setdefault(
                            (session_id, item.interval_id),
                            (item.sensor_time_s, item.stable_label),
                        )
            if item.transition_kind is not None:
                if (
                    item.transition_kind not in transition_counts
                    or item.block_id is None
                    or item.trial_id is None
                ):
                    raise ValueError("transition window identity is incomplete")
                target_label = trial_targets.get((int(item.block_id), int(item.trial_id)))
                if target_label not in CLASS_LABELS[1:]:
                    raise ValueError("transition window cannot be bound to its trial target")
                transition_window_count += 1
                transition_counts[item.transition_kind]["total"] += 1
                compatible = (
                    output.display_state not in CLASS_LABELS
                    or output.display_state in {"REST", target_label}
                )
                if compatible:
                    compatible_transition_window_count += 1
                    transition_counts[item.transition_kind]["compatible"] += 1

    if stable_window_count == 0:
        raise ValueError("streaming evaluation requires stable windows")
    if transition_window_count == 0:
        raise ValueError("streaming evaluation requires transition windows")
    latencies = []
    for key, (onset_time, expected_label) in expected_target_onsets.items():
        outputs = target_interval_outputs.get(key, [])
        correct_time = next(
            (time_s for time_s, label, display in outputs
             if label == expected_label and display == expected_label),
            None,
        )
        if correct_time is not None:
            latency = max(0.0, float(correct_time - onset_time))
            latencies.append(float(round(latency, 12)))
        else:
            latencies.append(math.inf)
    median_latency, p95_latency = _latency_metrics(latencies)
    stable_predictions = int(confusion.sum())
    unrelated_transition_window_count = (
        transition_window_count - compatible_transition_window_count
    )
    return CandidateEvaluation(
        config=config,
        stable_macro_f1=_macro_f1(confusion, true_counts),
        transition_endpoint_compatibility=float(
            compatible_transition_window_count / transition_window_count
        ),
        median_correct_onset_latency_s=median_latency,
        p95_correct_onset_latency_s=p95_latency,
        stable_confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        stable_true_support=tuple(int(value) for value in true_counts),
        stable_window_count=stable_window_count,
        stable_missed_count=stable_window_count - stable_predictions,
        transition_window_count=transition_window_count,
        compatible_transition_window_count=compatible_transition_window_count,
        unrelated_transition_window_count=unrelated_transition_window_count,
        movement_transition_window_count=transition_counts["movement"]["total"],
        movement_compatible_transition_window_count=transition_counts["movement"]["compatible"],
        movement_unrelated_transition_window_count=(
            transition_counts["movement"]["total"]
            - transition_counts["movement"]["compatible"]
        ),
        return_transition_window_count=transition_counts["return"]["total"],
        return_compatible_transition_window_count=transition_counts["return"]["compatible"],
        return_unrelated_transition_window_count=(
            transition_counts["return"]["total"]
            - transition_counts["return"]["compatible"]
        ),
        expected_target_onset_count=len(latencies),
        correct_onset_count=sum(math.isfinite(value) for value in latencies),
        correct_onset_latencies_s=tuple(latencies),
    )


def select_candidate(evaluations: Iterable[CandidateEvaluation]) -> CandidateEvaluation:
    """Choose by approved priorities plus an explicit deterministic remainder."""
    eligible = [item for item in evaluations if item.eligible]
    if not eligible:
        raise ValueError("no calibration candidate is eligible")
    return min(eligible, key=lambda item: (
        item.median_correct_onset_latency_s,
        -item.stable_macro_f1,
        item.config.consecutive_updates,
        -item.transition_endpoint_compatibility,
        item.p95_correct_onset_latency_s,
        item.config.ema_new_weight,
        item.config.minimum_confidence,
    ))


_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_DELETE = 0x00010000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_TEMPORARY = 0x00000100
_FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_LINK_INFO_CLASS = 11
_FILE_DISPOSITION_INFO_CLASS = 4
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = (
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    )


class _FILE_LINK_INFO(ctypes.Structure):
    _fields_ = (
        ("ReplaceIfExists", wintypes.BOOLEAN),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.DWORD),
        ("FileName", wintypes.WCHAR * 1),
    )


class _FILE_DISPOSITION_INFO(ctypes.Structure):
    _fields_ = (("DeleteFile", wintypes.BOOL),)


class _IO_STATUS_BLOCK(ctypes.Structure):
    _fields_ = (("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t))


def _kernel32_file():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetFileInformationByHandle.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    )
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    return kernel32


def _ntdll_file():
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtSetInformationFile.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_IO_STATUS_BLOCK),
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    )
    ntdll.NtSetInformationFile.restype = ctypes.c_long
    ntdll.RtlNtStatusToDosError.argtypes = (ctypes.c_long,)
    ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
    return ntdll


def _last_os_error(label: str) -> OSError:
    code = ctypes.get_last_error()
    return OSError(code, f"{label}: {ctypes.FormatError(code)}")


def _close_raw_handle(kernel32, handle: int, primary: BaseException) -> None:
    if not kernel32.CloseHandle(handle):
        error = _last_os_error("CloseHandle failed")
        if not kernel32.CloseHandle(handle):
            primary.add_note(str(error))


def _close_descriptor(descriptor: int, primary: BaseException | None = None) -> bool:
    failures = []
    for _ in range(2):
        try:
            os.close(descriptor)
            return True
        except OSError as error:
            failures.append(error)
    try:
        _ORIGINAL_OS_CLOSE(descriptor)
    except OSError as fallback_error:
        failure = OSError(
            "descriptor close failed after two retries and native fallback: "
            f"{fallback_error}",
        )
        if primary is None:
            raise failure from failures[-1]
        primary.add_note(str(failure))
        return False
    diagnostic = (
        "descriptor close failed twice but native fallback closed the exact fd: "
        f"{failures[-1]}"
    )
    if primary is None:
        warnings.warn(diagnostic, ResourceWarning, stacklevel=2)
    else:
        primary.add_note(diagnostic)
    return True


def _file_identity(descriptor: int) -> tuple[int, int]:
    kernel32 = _kernel32_file()
    info = _BY_HANDLE_FILE_INFORMATION()
    handle = msvcrt.get_osfhandle(descriptor)
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise _last_os_error("GetFileInformationByHandle failed")
    return (
        int(info.dwVolumeSerialNumber),
        (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow),
    )


@dataclass
class _PublishedJsonLease:
    final_path: Path
    staging_path: Path
    descriptor: int

    @property
    def closed(self) -> bool:
        return self.descriptor < 0

    def read_bytes(self) -> bytes:
        if self.closed:
            raise ValueError("publication lease is closed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(self.descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            descriptor = self.descriptor
            if _close_descriptor(descriptor):
                self.descriptor = -1

    def __enter__(self):
        if self.closed:
            raise ValueError("publication lease is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception is None:
            self.close()
        elif not self.closed:
            descriptor = self.descriptor
            if _close_descriptor(descriptor, exception):
                self.descriptor = -1
        return False


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("owned publication write made no progress")
        offset += written
    os.fsync(descriptor)


def _create_writer_lease(final_path: Path) -> tuple[_PublishedJsonLease, tuple[int, int]]:
    final = Path(final_path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    kernel32 = _kernel32_file()
    for _ in range(128):
        staging = final.parent / f".{final.name}.{secrets.token_hex(16)}.tmp"
        handle = kernel32.CreateFileW(
            str(staging),
            _GENERIC_READ | _GENERIC_WRITE | _DELETE,
            _FILE_SHARE_READ,
            None,
            _CREATE_NEW,
            _FILE_ATTRIBUTE_TEMPORARY
            | _FILE_FLAG_DELETE_ON_CLOSE
            | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            error = ctypes.get_last_error()
            if error in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
                continue
            raise _last_os_error("CreateFileW staging creation failed")
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException as primary:
            _close_raw_handle(kernel32, handle, primary)
            raise
        lease = _PublishedJsonLease(final, staging, descriptor)
        try:
            identity = _file_identity(descriptor)
        except BaseException as primary:
            if _close_descriptor(descriptor, primary):
                lease.descriptor = -1
            raise
        return lease, identity
    raise FileExistsError("could not allocate a unique owned staging file")


def _link_owned_handle_create_only(lease: _PublishedJsonLease, final_path: Path) -> None:
    final = Path(final_path).resolve()
    encoded = final.name.encode("utf-16-le")
    size = _FILE_LINK_INFO.FileName.offset + len(encoded)
    buffer = ctypes.create_string_buffer(size)
    info = _FILE_LINK_INFO.from_buffer(buffer)
    info.ReplaceIfExists = False
    kernel32 = _kernel32_file()
    ntdll = _ntdll_file()
    parent_handle = kernel32.CreateFileW(
        str(final.parent),
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if parent_handle == _INVALID_HANDLE_VALUE:
        raise _last_os_error("CreateFileW publication parent failed")
    info.RootDirectory = parent_handle
    info.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + _FILE_LINK_INFO.FileName.offset, encoded, len(encoded))
    handle = msvcrt.get_osfhandle(lease.descriptor)
    primary = None
    try:
        io_status = _IO_STATUS_BLOCK()
        status = int(ntdll.NtSetInformationFile(
            handle,
            ctypes.byref(io_status),
            buffer,
            size,
            _FILE_LINK_INFO_CLASS,
        ))
        if status < 0:
            code = int(ntdll.RtlNtStatusToDosError(status))
            if code in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
                primary = FileExistsError(
                    f"refusing to overwrite locked artifact: {final}",
                )
            else:
                primary = OSError(
                    code,
                    f"NtSetInformationFile(FileLinkInformation) failed: {ctypes.FormatError(code)}",
                )
            raise primary
    except BaseException as error:
        if primary is None:
            primary = error
        raise
    finally:
        if not kernel32.CloseHandle(parent_handle):
            close_error = _last_os_error("publication parent CloseHandle failed")
            if not kernel32.CloseHandle(parent_handle):
                if primary is not None:
                    primary.add_note(str(close_error))
                else:
                    import warnings
                    warnings.warn(str(close_error), ResourceWarning, stacklevel=2)


def _mark_delete_on_close(descriptor: int) -> None:
    kernel32 = _kernel32_file()
    disposition = _FILE_DISPOSITION_INFO(True)
    if not kernel32.SetFileInformationByHandle(
        msvcrt.get_osfhandle(descriptor),
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise _last_os_error("SetFileInformationByHandle(FileDispositionInfo) failed")


def _reopen_descriptor(
    descriptor: int,
    desired_access: int,
    share_mode: int,
) -> int:
    kernel32 = _kernel32_file()
    kernel32.ReOpenFile.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    )
    kernel32.ReOpenFile.restype = wintypes.HANDLE
    handle = kernel32.ReOpenFile(
        msvcrt.get_osfhandle(descriptor),
        desired_access,
        share_mode,
        _FILE_FLAG_OPEN_REPARSE_POINT,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _last_os_error("ReOpenFile publication lease failed")
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException as primary:
        _close_raw_handle(kernel32, handle, primary)
        raise


def _identity_with_retry(descriptor: int) -> tuple[int, int]:
    first_error = None
    for _ in range(2):
        try:
            return _file_identity(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
            else:
                error.add_note(f"first identity query failed: {first_error}")
                raise
    raise AssertionError("unreachable identity retry state")


def _same_file_object(left_descriptor: int, right_descriptor: int) -> bool:
    return _identity_with_retry(left_descriptor) == _identity_with_retry(
        right_descriptor,
    )


def _open_strict_final_path_lease(
    final_path: Path,
    staging_path: Path,
) -> _PublishedJsonLease:
    final = Path(final_path).resolve()
    kernel32 = _kernel32_file()
    handle = kernel32.CreateFileW(
        str(final),
        _GENERIC_READ,
        _FILE_SHARE_READ,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _last_os_error("CreateFileW strict final-path lease failed")
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException as primary:
        _close_raw_handle(kernel32, handle, primary)
        raise
    return _PublishedJsonLease(final, staging_path, descriptor)


def _delete_verified_owned_alias(
    verification: _PublishedJsonLease,
    primary: BaseException,
) -> None:
    deletion_descriptor = -1
    try:
        deletion_descriptor = _reopen_descriptor(
            verification.descriptor,
            _GENERIC_READ | _DELETE,
            _FILE_SHARE_READ,
        )
        _mark_delete_on_close(deletion_descriptor)
    except OSError as cleanup_error:
        primary.add_note(f"owned final cleanup failed: {cleanup_error}")
    finally:
        if deletion_descriptor >= 0:
            _close_descriptor(deletion_descriptor, primary)


def _open_published_alias_lease(
    final_path: Path,
    staging_path: Path,
    expected_identity: tuple[int, int],
    expected_payload: bytes,
    writer: _PublishedJsonLease,
) -> _PublishedJsonLease:
    final = Path(final_path).resolve()
    kernel32 = _kernel32_file()
    handle = kernel32.CreateFileW(
        str(final),
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _last_os_error("CreateFileW published alias lease failed")
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException as primary:
        _close_raw_handle(kernel32, handle, primary)
        raise
    verification = _PublishedJsonLease(final, staging_path, descriptor)
    owned_final = False
    strict = None
    try:
        try:
            observed_identity = _identity_with_retry(descriptor)
        except OSError:
            # The create-only link was opened while the no-share-delete writer
            # still pins it, so failed metadata queries cannot make it foreign.
            owned_final = True
            raise
        if observed_identity != expected_identity:
            raise OSError("published alias does not identify the owned file object")
        owned_final = True
        observed_payload = verification.read_bytes()
        if observed_payload != expected_payload:
            raise OSError("published owned file bytes differ from the frozen payload")
        writer.close()
        strict = _open_strict_final_path_lease(final, staging_path)
        if not _same_file_object(strict.descriptor, verification.descriptor):
            raise OSError(
                "published strict alias is not the same file object as the verified owned final",
            )
        if strict.read_bytes() != expected_payload:
            raise OSError("strict publication lease bytes differ from the frozen payload")
        verification.close()
        return strict
    except BaseException as primary:
        if not writer.closed:
            writer.__exit__(type(primary), primary, primary.__traceback__)
        if strict is not None and not strict.closed:
            strict.__exit__(type(primary), primary, primary.__traceback__)
        if owned_final and not verification.closed:
            _delete_verified_owned_alias(verification, primary)
        verification.__exit__(type(primary), primary, primary.__traceback__)
        raise


def _publish_json_lease(path: Path, value: dict) -> _PublishedJsonLease:
    final = Path(path).resolve()
    payload = (
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    if final.exists():
        raise FileExistsError(f"refusing to overwrite locked artifact: {final}")
    writer, identity = _create_writer_lease(final)
    try:
        _write_all(writer.descriptor, payload)
        if writer.read_bytes() != payload:
            raise OSError("owned staging bytes differ from the frozen payload")
        _link_owned_handle_create_only(writer, final)
    except BaseException as primary:
        if not writer.closed:
            descriptor = writer.descriptor
            if _close_descriptor(descriptor, primary):
                writer.descriptor = -1
        raise
    return _open_published_alias_lease(
        final, writer.staging_path, identity, payload, writer,
    )


def _atomic_create_json(path: Path, value: dict) -> None:
    """Create one handle-bound UTF-8 JSON without overwriting any directory entry."""
    with _publish_json_lease(path, value):
        pass


def _selected_dict(config: DecisionFilterConfig) -> dict:
    return {
        "ema_new_weight": config.ema_new_weight,
        "minimum_confidence": config.minimum_confidence,
        "consecutive_updates": config.consecutive_updates,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _evidence(evaluation: CandidateEvaluation) -> dict:
    return {
        "stable_macro_f1": evaluation.stable_macro_f1,
        "transition_endpoint_compatibility": evaluation.transition_endpoint_compatibility,
        "median_correct_onset_latency_s": _finite_or_none(
            evaluation.median_correct_onset_latency_s,
        ),
        "p95_correct_onset_latency_s": _finite_or_none(
            evaluation.p95_correct_onset_latency_s,
        ),
        "stable_confusion_matrix": [list(row) for row in evaluation.stable_confusion_matrix],
        "stable_true_support": list(evaluation.stable_true_support),
        "stable_window_count": evaluation.stable_window_count,
        "stable_missed_count": evaluation.stable_missed_count,
        "transition_window_count": evaluation.transition_window_count,
        "compatible_transition_window_count": evaluation.compatible_transition_window_count,
        "unrelated_transition_window_count": evaluation.unrelated_transition_window_count,
        "movement_transition_window_count": evaluation.movement_transition_window_count,
        "movement_compatible_transition_window_count": (
            evaluation.movement_compatible_transition_window_count
        ),
        "movement_unrelated_transition_window_count": (
            evaluation.movement_unrelated_transition_window_count
        ),
        "return_transition_window_count": evaluation.return_transition_window_count,
        "return_compatible_transition_window_count": (
            evaluation.return_compatible_transition_window_count
        ),
        "return_unrelated_transition_window_count": (
            evaluation.return_unrelated_transition_window_count
        ),
        "expected_target_onset_count": evaluation.expected_target_onset_count,
        "correct_onset_count": evaluation.correct_onset_count,
        "correct_onset_latencies_s": [
            _finite_or_none(value) for value in evaluation.correct_onset_latencies_s
        ],
    }


def _validation_evidence(evaluation: CandidateEvaluation) -> dict:
    evidence = _evidence(evaluation)
    _validate_endpoint_evidence(evidence, expected_onsets=42, require_pass=True)
    return evidence


def _test_evidence(evaluation: CandidateEvaluation) -> dict:
    return _evidence(evaluation)


def _test_passed(evaluation: CandidateEvaluation) -> bool:
    return bool(
        evaluation.stable_macro_f1 >= ACCEPTANCE_THRESHOLDS["stable_macro_f1_minimum"]
        and evaluation.transition_endpoint_compatibility
        == ACCEPTANCE_THRESHOLDS["transition_endpoint_compatibility_required"]
        and evaluation.expected_target_onset_count == 56
        and evaluation.correct_onset_count == 56
        and math.isfinite(evaluation.median_correct_onset_latency_s)
        and evaluation.median_correct_onset_latency_s
        <= ACCEPTANCE_THRESHOLDS["median_correct_onset_latency_s_maximum"]
        and math.isfinite(evaluation.p95_correct_onset_latency_s)
        and evaluation.p95_correct_onset_latency_s
        <= ACCEPTANCE_THRESHOLDS["p95_correct_onset_latency_s_maximum"]
    )


def _require_seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) <= 2**32 - 1:
        raise ValueError("generation_seed must be an unsigned 32-bit integer")
    return int(value)


def _strict_json_object_bytes(payload: bytes, label: str) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"{label} contains non-finite constant {value}")

    try:
        if type(payload) is not bytes:
            raise TypeError(f"{label} payload must be exact bytes")
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (TypeError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not load {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _load_test_evaluation(
    path: str | Path,
    calibration_path: str | Path,
    checkpoint_sha256: str,
) -> dict:
    """Strictly reload and bind a complete endpoint-compatible test artifact."""
    calibration_target = Path(calibration_path).resolve()
    try:
        calibration_bytes = calibration_target.read_bytes()
        evaluation_bytes = Path(path).resolve().read_bytes()
    except OSError as error:
        raise ValueError(f"could not load locked test bindings: {error}") from error
    return _load_test_evaluation_bytes(
        evaluation_bytes, calibration_bytes, checkpoint_sha256,
    )


def _load_test_evaluation_bytes(
    payload: bytes,
    calibration_bytes: bytes,
    checkpoint_sha256: str,
) -> dict:
    frozen_config = _load_calibration_bytes(calibration_bytes, checkpoint_sha256)
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    value = _strict_json_object_bytes(payload, "locked test evaluation")
    if (
        set(value) != _TEST_EVALUATION_FIELDS
        or value["schema_version"] != _TEST_EVALUATION_SCHEMA
    ):
        raise ValueError("locked test evaluation fields/schema are invalid")
    thresholds = value["acceptance_thresholds"]
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != set(ACCEPTANCE_THRESHOLDS)
        or any(type(item) is not float for item in thresholds.values())
        or thresholds != ACCEPTANCE_THRESHOLDS
    ):
        raise ValueError("locked test acceptance threshold values/types are invalid")
    selected = value["selected"]
    if (
        not isinstance(selected, dict)
        or set(selected) != {"ema_new_weight", "minimum_confidence", "consecutive_updates"}
        or type(selected["ema_new_weight"]) is not float
        or type(selected["minimum_confidence"]) is not float
        or type(selected["consecutive_updates"]) is not int
    ):
        raise ValueError("locked test selected configuration values/types are invalid")
    if (
        type(value["calibration_sha256"]) is not str
        or value["calibration_sha256"] != calibration_hash
        or type(value["checkpoint_sha256"]) is not str
        or value["checkpoint_sha256"] != checkpoint_sha256
        or value["class_labels"] != list(CLASS_LABELS)
        or value["test_session_ids"] != list(TEST_SESSIONS)
        or selected != _selected_dict(frozen_config)
    ):
        raise ValueError("locked test evaluation bindings are invalid")
    evidence = _validate_endpoint_evidence(
        value["test_evidence"], expected_onsets=56, require_pass=False,
    )
    if type(value["passed"]) is not bool:
        raise ValueError("locked test passed flag must be a JSON bool")
    expected_passed = bool(
        evidence["stable_macro_f1"] >= ACCEPTANCE_THRESHOLDS["stable_macro_f1_minimum"]
        and evidence["transition_endpoint_compatibility"]
        == ACCEPTANCE_THRESHOLDS["transition_endpoint_compatibility_required"]
        and evidence["correct_onset_count"] == 56
        and evidence["median_correct_onset_latency_s"] is not None
        and evidence["median_correct_onset_latency_s"]
        <= ACCEPTANCE_THRESHOLDS["median_correct_onset_latency_s_maximum"]
        and evidence["p95_correct_onset_latency_s"] is not None
        and evidence["p95_correct_onset_latency_s"]
        <= ACCEPTANCE_THRESHOLDS["p95_correct_onset_latency_s_maximum"]
    )
    if value["passed"] is not expected_passed:
        raise ValueError("locked test passed flag disagrees with exact evidence")
    return value


def calibrate_realtime(
    *,
    capture_dir: str | Path,
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    test_evaluation_path: str | Path,
    device: str = "cuda",
    generation_seed: int = 20260821,
) -> dict:
    """Freeze validation calibration, reload/hash it, then lock test evidence."""
    checkpoint = Path(checkpoint_path).resolve()
    calibration_target = Path(calibration_path).resolve()
    test_target = Path(test_evaluation_path).resolve()
    if calibration_target == test_target:
        raise ValueError("calibration and test-evaluation paths must differ")
    if calibration_target.exists():
        raise FileExistsError(f"refusing to overwrite locked artifact: {calibration_target}")
    if test_target.exists():
        raise FileExistsError(f"refusing to overwrite locked artifact: {test_target}")
    seed = _require_seed(generation_seed)

    validation_artifacts = load_sessions_by_id(capture_dir, VALIDATION_SESSIONS)
    validation_by_id = {item.session_id: item for item in validation_artifacts}
    if tuple(validation_by_id) != VALIDATION_SESSIONS:
        raise ValueError("validation loader did not return the exact requested session order")

    runtime = RealtimeModelRuntime.load(checkpoint, device)
    checkpoint_hash = runtime.checkpoint_sha256
    validation_replays = {
        session_id: replay_session(validation_by_id[session_id], runtime)
        for session_id in VALIDATION_SESSIONS
    }
    candidate_evaluations = tuple(
        evaluate_replays(validation_replays, config) for config in candidate_configs()
    )
    if len(candidate_evaluations) != 90:
        raise RuntimeError("calibration did not evaluate the complete 90-candidate grid")
    selected = select_candidate(candidate_evaluations)
    calibration = {
        "schema_version": _CALIBRATION_SCHEMA,
        "checkpoint_sha256": checkpoint_hash,
        "validation_session_ids": list(VALIDATION_SESSIONS),
        "search_grid": SEARCH_GRID,
        "selected": _selected_dict(selected.config),
        "validation_evidence": _validation_evidence(selected),
        "generation_seed": seed,
    }
    with _publish_json_lease(calibration_target, calibration) as calibration_lease:
        calibration_hash = hashlib.sha256(calibration_lease.read_bytes()).hexdigest()
        calibration_bytes = calibration_lease.read_bytes()
        frozen_config = _load_calibration_bytes(calibration_bytes, checkpoint_hash)
        if frozen_config != selected.config:
            raise RuntimeError("reloaded frozen calibration differs from selected parameters")
        test_artifacts = load_sessions_by_id(capture_dir, TEST_SESSIONS)
        test_by_id = {item.session_id: item for item in test_artifacts}
        if tuple(test_by_id) != TEST_SESSIONS:
            raise ValueError("test loader did not return the exact requested session order")
        test_replays = {
            session_id: replay_session(test_by_id[session_id], runtime)
            for session_id in TEST_SESSIONS
        }
        test_metrics = evaluate_replays(test_replays, frozen_config)
        passed = _test_passed(test_metrics)
        test_evaluation = {
            "schema_version": _TEST_EVALUATION_SCHEMA,
            "calibration_sha256": calibration_hash,
            "checkpoint_sha256": checkpoint_hash,
            "class_labels": list(CLASS_LABELS),
            "test_session_ids": list(TEST_SESSIONS),
            "selected": _selected_dict(frozen_config),
            "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
            "test_evidence": _test_evidence(test_metrics),
            "passed": passed,
        }
        with _publish_json_lease(test_target, test_evaluation) as test_lease:
            frozen_test_evaluation = _load_test_evaluation_bytes(
                test_lease.read_bytes(), calibration_bytes, checkpoint_hash,
            )
        if not passed:
            raise CalibrationTestFailure(
                f"locked test evaluation failed acceptance thresholds; evidence preserved at {test_target}",
            )
        return frozen_test_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test-evaluation", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    try:
        result = calibrate_realtime(
            capture_dir=args.captures,
            checkpoint_path=args.checkpoint,
            calibration_path=args.calibration,
            test_evaluation_path=args.test_evaluation,
            device=args.device,
            generation_seed=args.generation_seed,
        )
    except CalibrationTestFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "Calibration and locked test evaluation passed: "
        f"Macro-F1={result['test_evidence']['stable_macro_f1']:.6f}, "
        "transition endpoint compatibility="
        f"{result['test_evidence']['transition_endpoint_compatibility']:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
