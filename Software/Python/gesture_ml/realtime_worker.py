"""Single-client loopback worker for persistent CUDA gesture inference."""

from __future__ import annotations

import argparse
import contextlib
import json
from numbers import Integral
from pathlib import Path
import select
import socket
from typing import Callable

import numpy as np

from .intervals import CLASS_LABELS
from .realtime_inference import RealtimeInferenceEngine
from .realtime_protocol import (
    InferenceRequest,
    ProtocolError,
    ShutdownRequest,
    encode_prediction,
    read_message,
)


LOOPBACK_HOST = "127.0.0.1"


def _validate_host_port(host: str, port: int) -> tuple[str, int]:
    if host != LOOPBACK_HOST:
        raise ValueError("worker host must be exactly 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, Integral) or not 1024 <= int(port) <= 65535:
        raise ValueError("worker port must be an integer from 1024 through 65535")
    return host, int(port)


def _warm_up_engine(engine: RealtimeInferenceEngine) -> None:
    """Run exactly one unfiltered dummy window before declaring readiness."""
    runtime = engine.runtime
    runtime.predict_probabilities(
        np.zeros((8, 250), dtype=np.float32),
        np.zeros((6, 52), dtype=np.float32),
    )
    device = getattr(runtime, "device", None)
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _device_name(engine: RealtimeInferenceEngine, ready_device: str | None) -> str:
    if ready_device is not None:
        if not isinstance(ready_device, str) or not ready_device:
            raise ValueError("ready_device must be a non-empty string")
        return ready_device
    runtime_device = engine.runtime.device
    if getattr(runtime_device, "type", None) != "cuda":
        raise RuntimeError("live worker requires a CUDA runtime")
    import torch

    return str(torch.cuda.get_device_name(runtime_device))


def _ready_record(device_name: str) -> bytes:
    return json.dumps({
        "type": "ready",
        "protocol_version": 1,
        "device": device_name,
        "class_labels": list(CLASS_LABELS),
    }, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _close_socket(client: socket.socket | None) -> None:
    if client is not None:
        with contextlib.suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            client.close()


def serve_one_client(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    host: str = LOOPBACK_HOST,
    port: int,
    device: str = "cuda",
    engine_loader: Callable[[str | Path, str | Path, str], RealtimeInferenceEngine] = RealtimeInferenceEngine.load,
    ready_device: str | None = None,
) -> None:
    """Serve sequential GWIN frames until a validated GEND closes the worker.

    A bad frame or inference response terminates only its current client session;
    the already-loaded inference engine remains available for a replacement client.
    """
    host, port = _validate_host_port(host, port)
    engine = engine_loader(checkpoint_path, calibration_path, device)
    _warm_up_engine(engine)
    ready = _ready_record(_device_name(engine, ready_device))
    current: socket.socket | None = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(2)
        try:
            while True:
                readers = [listener]
                if current is not None:
                    readers.append(current)
                readable, _, _ = select.select(readers, [], [])
                if listener in readable:
                    incoming, _ = listener.accept()
                    if current is not None:
                        _close_socket(incoming)
                    else:
                        current = incoming
                        try:
                            current.sendall(ready)
                        except OSError:
                            _close_socket(current)
                            current = None
                if current is None or current not in readable:
                    continue
                try:
                    message = read_message(current)
                    if isinstance(message, ShutdownRequest):
                        return
                    if not isinstance(message, InferenceRequest):
                        raise ProtocolError("unrecognized request type")
                    prediction = engine.predict(
                        message.emg, message.imu, message.sequence, message.sensor_time_s,
                    )
                    if (
                        not isinstance(prediction, dict)
                        or prediction.get("sequence") != message.sequence
                        or prediction.get("sensor_time_s") != message.sensor_time_s
                    ):
                        raise ProtocolError("engine response does not correlate to its request")
                    current.sendall(encode_prediction(prediction))
                except (OSError, ProtocolError, ValueError, FloatingPointError):
                    _close_socket(current)
                    current = None
        finally:
            _close_socket(current)


def _port_argument(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be from 1024 through 65535")
    return port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve persistent loopback CUDA gesture inference.")
    parser.add_argument("--checkpoint", required=True, help="approved .pt checkpoint path")
    parser.add_argument("--calibration", required=True, help="approved calibration JSON path")
    parser.add_argument("--host", required=True, choices=(LOOPBACK_HOST,), help="must be 127.0.0.1")
    parser.add_argument("--port", required=True, type=_port_argument, help="loopback TCP port (1024-65535)")
    parser.add_argument("--device", required=True, choices=("cuda",), help="required CUDA device")
    arguments = parser.parse_args(argv)
    serve_one_client(
        arguments.checkpoint,
        arguments.calibration,
        host=arguments.host,
        port=arguments.port,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
