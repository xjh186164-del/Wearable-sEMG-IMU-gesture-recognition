"""Build a deterministic NPZ dataset from passed MATLAB capture sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .artifacts import DEFAULT_SUBJECT_ALIASES, SessionArtifacts, discover_sessions
from .intervals import CLASS_LABELS, extract_stable_intervals
from .timebase import shared_sensor_time_axes
from .windowing import concatenate_batches, make_windows


RAW_HEADER = ("sample", "timestamp_us", "status", *(f"ch{i}_raw" for i in range(1, 9)), "dropped")
FILTERED_HEADER = ("sample", "timestamp_us", "status", *(f"ch{i}_filtered_mV" for i in range(1, 9)), "dropped")
IMU_HEADER = ("sample", "timestamp_us", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "temp_C", "dropped")
EVENT_HEADER = ("session_time_s", "event_type", "block_id", "trial_id", "gesture_id", "valid", "note")


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.reader(handle), None)
    if row is None:
        raise ValueError(f"empty CSV: {path.name}")
    return tuple(value.strip() for value in row)


def _require_header(path: Path, expected: tuple[str, ...]):
    actual = _read_header(path)
    if actual != expected:
        raise ValueError(f"unexpected CSV schema in {path.name}: {actual}")


def _load_numeric(path: Path, usecols) -> np.ndarray:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            next(handle, None)
            nonblank_lines = (line for line in handle if line.strip())
            values = np.loadtxt(
                nonblank_lines,
                delimiter=",",
                usecols=usecols,
                dtype=np.float64,
                ndmin=2,
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read numeric CSV {path.name}: {error}") from error
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"numeric CSV is empty or non-finite: {path.name}")
    return values


def _read_events(path: Path):
    _require_header(path, EVENT_HEADER)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"events CSV is empty: {path.name}")
    return rows


def _load_session(artifacts: SessionArtifacts):
    _require_header(artifacts.raw_path, RAW_HEADER)
    _require_header(artifacts.filtered_path, FILTERED_HEADER)
    _require_header(artifacts.imu_path, IMU_HEADER)
    raw = _load_numeric(artifacts.raw_path, (1,))
    filtered = _load_numeric(artifacts.filtered_path, tuple(range(1, 11)))
    imu = _load_numeric(artifacts.imu_path, tuple(range(1, 8)))
    _, filtered_time, imu_time = shared_sensor_time_axes(raw[:, 0], filtered[:, 0], imu[:, 0])
    events = _read_events(artifacts.events_path)
    intervals = extract_stable_intervals(events)
    return make_windows(
        filtered_time,
        filtered[:, 2:10],
        imu_time,
        imu[:, 1:7],
        intervals,
        physical_subject_id=artifacts.physical_subject_id,
        acquisition_subject_id=artifacts.acquisition_subject_id,
        session_id=artifacts.session_id,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(
    capture_dir: str | Path,
    output_path: str | Path,
    aliases=None,
):
    aliases = dict(DEFAULT_SUBJECT_ALIASES if aliases is None else aliases)
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("output path must end with .npz")
    sessions, rejections = discover_sessions(capture_dir, aliases)
    if not sessions:
        raise ValueError("no eligible eight_pose_v5 sessions were found")

    batches = []
    sources = {}
    accepted_counts = {}
    for session in sessions:
        batch = _load_session(session)
        if batch.labels.size == 0:
            raise ValueError(f"eligible session produced no windows: {session.session_id}")
        batches.append(batch)
        accepted_counts[session.session_id] = int(batch.labels.size)
        sources[session.session_id] = {
            path.name: _sha256(path) for path in session.paths
        }
    combined = concatenate_batches(batches)
    arrays = {
        name: getattr(combined, name)
        for name in combined.__dataclass_fields__
    }
    _atomic_npz(output, arrays)

    class_counts = {
        label: int(np.count_nonzero(combined.labels == index))
        for index, label in enumerate(CLASS_LABELS)
    }
    manifest = {
        "schema_version": "gesture_window_dataset_v1",
        "protocol_version": "eight_pose_v5",
        "class_labels": list(CLASS_LABELS),
        "window_seconds": 0.5,
        "step_seconds": 0.1,
        "guard_seconds": 0.4,
        "emg_shape_per_window": [8, 250],
        "imu_shape_per_window": [6, 52],
        "subject_aliases": dict(sorted(aliases.items())),
        "accepted_sessions": accepted_counts,
        "rejected_sessions": [
            {"session_id": item.session_id, "reason": item.reason}
            for item in rejections
        ],
        "window_count": int(combined.labels.size),
        "class_counts": class_counts,
        "source_sha256": sources,
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def _parse_alias(values: list[str]):
    aliases = dict(DEFAULT_SUBJECT_ALIASES)
    for value in values:
        if "=" not in value:
            raise ValueError(f"alias must use ACQUISITION=PHYSICAL form: {value}")
        acquisition, physical = value.split("=", maxsplit=1)
        if not acquisition or not physical:
            raise ValueError(f"alias must use ACQUISITION=PHYSICAL form: {value}")
        aliases[acquisition] = physical
    return aliases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="ACQUISITION=PHYSICAL",
        help="map an acquisition SubjectId to the real physical user",
    )
    args = parser.parse_args(argv)
    manifest = build_dataset(args.captures, args.output, _parse_alias(args.alias))
    print(f"Built {manifest['window_count']} windows from {len(manifest['accepted_sessions'])} sessions.")
    for label in CLASS_LABELS:
        print(f"  {label}: {manifest['class_counts'][label]}")


if __name__ == "__main__":
    main()
