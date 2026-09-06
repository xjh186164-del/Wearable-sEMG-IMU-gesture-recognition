"""Strict version-one framing for loopback real-time inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import struct
from typing import Protocol

import numpy as np


PROTOCOL_VERSION = 1
WINDOW_MAGIC = b"GWIN"
SHUTDOWN_MAGIC = b"GEND"
HEADER_STRUCT = struct.Struct("<4sHQd")
SHUTDOWN_STRUCT = struct.Struct("<4sH")
EMG_VALUE_COUNT = 8 * 250
IMU_VALUE_COUNT = 6 * 52
PAYLOAD_VALUE_COUNT = EMG_VALUE_COUNT + IMU_VALUE_COUNT
REQUEST_BYTES = HEADER_STRUCT.size + PAYLOAD_VALUE_COUNT * np.dtype("<f4").itemsize
SHUTDOWN_BYTES = SHUTDOWN_STRUCT.size


class ProtocolError(ValueError):
    """A peer sent an incomplete or invalid protocol frame."""


class _Readable(Protocol):
    def recv(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class InferenceRequest:
    sequence: int
    sensor_time_s: float
    emg: np.ndarray
    imu: np.ndarray


@dataclass(frozen=True)
class ShutdownRequest:
    protocol_version: int = PROTOCOL_VERSION


def read_exact(stream: _Readable, length: int) -> bytes:
    """Return exactly *length* bytes or reject a fragmented/truncated peer."""
    if isinstance(length, bool) or not isinstance(length, Integral) or int(length) < 0:
        raise ValueError("length must be a non-negative integer")
    remaining = int(length)
    chunks: list[bytes] = []
    while remaining:
        try:
            chunk = stream.recv(remaining)
        except OSError as error:
            raise ProtocolError(f"socket read failed: {error}") from error
        if not isinstance(chunk, bytes):
            raise ProtocolError("stream returned non-bytes data")
        if not chunk:
            raise ProtocolError("unexpected EOF while reading protocol frame")
        if len(chunk) > remaining:
            raise ProtocolError("stream returned more bytes than requested")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_request(frame: bytes) -> InferenceRequest:
    """Decode one exact GWIN frame into model-layout float32 arrays."""
    if not isinstance(frame, bytes) or len(frame) != REQUEST_BYTES:
        raise ProtocolError(f"GWIN frame must be exactly {REQUEST_BYTES} bytes")
    magic, version, sequence, sensor_time_s = HEADER_STRUCT.unpack_from(frame)
    if magic != WINDOW_MAGIC:
        raise ProtocolError("request magic must be GWIN")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("request protocol version is unsupported")
    if not math.isfinite(sensor_time_s):
        raise ProtocolError("sensor time must be finite")
    payload = np.frombuffer(frame, dtype="<f4", count=PAYLOAD_VALUE_COUNT, offset=HEADER_STRUCT.size)
    if payload.shape != (PAYLOAD_VALUE_COUNT,) or not bool(np.all(np.isfinite(payload))):
        raise ProtocolError("request payload must contain only finite float32 values")
    emg = payload[:EMG_VALUE_COUNT].reshape(250, 8).T.copy()
    imu = payload[EMG_VALUE_COUNT:].reshape(52, 6).T.copy()
    return InferenceRequest(int(sequence), float(sensor_time_s), emg, imu)


def decode_shutdown(frame: bytes) -> ShutdownRequest:
    """Validate the dedicated fixed-size GEND shutdown frame."""
    if not isinstance(frame, bytes) or len(frame) != SHUTDOWN_BYTES:
        raise ProtocolError(f"GEND frame must be exactly {SHUTDOWN_BYTES} bytes")
    magic, version = SHUTDOWN_STRUCT.unpack(frame)
    if magic != SHUTDOWN_MAGIC:
        raise ProtocolError("shutdown magic must be GEND")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("shutdown protocol version is unsupported")
    return ShutdownRequest()


def read_message(stream: _Readable) -> InferenceRequest | ShutdownRequest:
    """Read one request or graceful-shutdown frame from a fragmented stream."""
    magic = read_exact(stream, 4)
    if magic == SHUTDOWN_MAGIC:
        return decode_shutdown(magic + read_exact(stream, SHUTDOWN_BYTES - len(magic)))
    if magic != WINDOW_MAGIC:
        raise ProtocolError("frame magic is neither GWIN nor GEND")
    return decode_request(magic + read_exact(stream, REQUEST_BYTES - len(magic)))


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ProtocolError(f"prediction {field} must be finite")
    return float(value)


def encode_prediction(response: dict) -> bytes:
    """Validate and serialize one compact newline-delimited prediction record."""
    if not isinstance(response, dict) or set(response) != {
        "sequence", "sensor_time_s", "candidate_gesture_id", "display_state",
        "confidence", "probabilities", "inference_ms",
    }:
        raise ProtocolError("prediction response fields do not match the protocol")
    sequence = response["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, Integral) or not 0 <= int(sequence) <= 2**64 - 1:
        raise ProtocolError("prediction sequence must be uint64")
    candidate = response["candidate_gesture_id"]
    display = response["display_state"]
    if not isinstance(candidate, str) or not candidate or not isinstance(display, str) or not display:
        raise ProtocolError("prediction labels must be non-empty strings")
    probabilities = response["probabilities"]
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 8:
        raise ProtocolError("prediction must contain eight probabilities")
    probability_values = [_finite_number(value, "probability") for value in probabilities]
    if any(value < 0.0 or value > 1.0 for value in probability_values):
        raise ProtocolError("prediction probabilities must be in [0, 1]")
    if abs(sum(probability_values) - 1.0) > 1e-4:
        raise ProtocolError("prediction probabilities must sum to one")
    confidence = _finite_number(response["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ProtocolError("prediction confidence must be in [0, 1]")
    inference_ms = _finite_number(response["inference_ms"], "inference_ms")
    if inference_ms < 0.0:
        raise ProtocolError("prediction inference_ms must be non-negative")
    record = {
        "type": "prediction",
        "protocol_version": PROTOCOL_VERSION,
        "sequence": int(sequence),
        "sensor_time_s": _finite_number(response["sensor_time_s"], "sensor_time_s"),
        "candidate_gesture_id": candidate,
        "display_state": display,
        "confidence": confidence,
        "probabilities": probability_values,
        "inference_ms": inference_ms,
    }
    try:
        return json.dumps(record, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"prediction cannot be encoded as JSON: {error}") from error
