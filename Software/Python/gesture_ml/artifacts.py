"""Strict discovery and eligibility checks for acquisition artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping


ARTIFACT_SUFFIXES = (
    "_raw.csv",
    "_filtered.csv",
    "_imu.csv",
    "_events.csv",
    "_metadata.json",
    "_quality.json",
)
SESSION_PATTERN = re.compile(r"^(S\d{3}_D\d{2}_R\d{2})")
DEFAULT_SUBJECT_ALIASES = {"S001": "P001", "S002": "P001", "S003": "P001"}
EXPECTED_TARGET_LABELS = [
    "WRIST_UP",
    "WRIST_DOWN",
    "FOREARM_IN",
    "FOREARM_OUT",
    "ARM_UP",
    "ARM_DOWN",
    "FIST",
]


@dataclass(frozen=True)
class SessionArtifacts:
    session_id: str
    acquisition_subject_id: str
    physical_subject_id: str
    raw_path: Path
    filtered_path: Path
    imu_path: Path
    events_path: Path
    metadata_path: Path
    quality_path: Path

    @property
    def paths(self) -> tuple[Path, ...]:
        return (
            self.raw_path,
            self.filtered_path,
            self.imu_path,
            self.events_path,
            self.metadata_path,
            self.quality_path,
        )


@dataclass(frozen=True)
class Rejection:
    session_id: str
    reason: str


def _load_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON in {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object in {path.name}")
    return value


def _candidate_session_ids(capture_dir: Path) -> list[str]:
    candidates = set()
    for path in capture_dir.iterdir():
        if not path.is_file():
            continue
        for suffix in ARTIFACT_SUFFIXES:
            if path.name.endswith(suffix):
                session_id = path.name[: -len(suffix)]
                if SESSION_PATTERN.fullmatch(session_id):
                    candidates.add(session_id)
                break
    return sorted(candidates)


def _validate_session(
    root: Path,
    session_id: str,
    aliases: Mapping[str, str],
) -> tuple[SessionArtifacts | None, str | None]:
    subject_id = session_id.split("_", maxsplit=1)[0]
    paths = {suffix: root / f"{session_id}{suffix}" for suffix in ARTIFACT_SUFFIXES}
    if subject_id == "S000":
        return None, "dry-run S000 sessions are excluded"
    if not all(path.is_file() for path in paths.values()):
        return None, "session does not have exactly six artifacts"
    try:
        metadata = _load_json(paths["_metadata.json"])
        quality = _load_json(paths["_quality.json"])
    except ValueError as error:
        return None, str(error)
    if metadata.get("session_id") != session_id or metadata.get("subject_id") != subject_id:
        return None, "metadata identity does not match artifact basename"
    if metadata.get("protocol_version") != "eight_pose_v5":
        return None, "protocol is not eight_pose_v5"
    if metadata.get("block_count") != 2:
        return None, "metadata block count is not two"
    if metadata.get("completed") is not True or metadata.get("terminal_status") != "completed":
        return None, "session did not complete successfully"
    fail_reasons = quality.get("fail_reasons")
    if quality.get("passed") is not True or not isinstance(fail_reasons, list) or fail_reasons:
        return None, "quality gate did not pass cleanly"
    if quality.get("schema_version") != "gesture_session_quality_v4":
        return None, "quality schema is not gesture_session_quality_v4"
    if quality.get("target_labels") != EXPECTED_TARGET_LABELS:
        return None, "quality target labels do not match eight_pose_v5"
    if quality.get("valid_trial_counts") != [2] * len(EXPECTED_TARGET_LABELS):
        return None, "quality valid trial counts are not exactly two per target"
    if subject_id not in aliases:
        return None, "physical subject alias is missing"
    return SessionArtifacts(
        session_id=session_id,
        acquisition_subject_id=subject_id,
        physical_subject_id=str(aliases[subject_id]),
        raw_path=paths["_raw.csv"],
        filtered_path=paths["_filtered.csv"],
        imu_path=paths["_imu.csv"],
        events_path=paths["_events.csv"],
        metadata_path=paths["_metadata.json"],
        quality_path=paths["_quality.json"],
    ), None


def load_sessions_by_id(
    capture_dir: str | Path,
    session_ids,
    aliases: Mapping[str, str] | None = None,
) -> list[SessionArtifacts]:
    """Load only the exact requested sessions without enumerating the directory."""
    root = Path(capture_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"capture directory does not exist: {root}")
    identities = tuple(session_ids)
    if (
        not identities
        or any(type(item) is not str or SESSION_PATTERN.fullmatch(item) is None for item in identities)
        or len(identities) != len(set(identities))
    ):
        raise ValueError("requested session identities must be unique exact session IDs")
    alias_map = dict(DEFAULT_SUBJECT_ALIASES if aliases is None else aliases)
    loaded = []
    for session_id in identities:
        artifact, reason = _validate_session(root, session_id, alias_map)
        if artifact is None:
            raise ValueError(f"requested session {session_id} is invalid: {reason}")
        loaded.append(artifact)
    return loaded


def discover_sessions(
    capture_dir: str | Path,
    aliases: Mapping[str, str] | None = None,
) -> tuple[list[SessionArtifacts], list[Rejection]]:
    """Return eligible sessions plus explicit fail-closed rejections."""
    root = Path(capture_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"capture directory does not exist: {root}")
    aliases = dict(DEFAULT_SUBJECT_ALIASES if aliases is None else aliases)
    accepted: list[SessionArtifacts] = []
    rejected: list[Rejection] = []

    for session_id in _candidate_session_ids(root):
        artifact, reason = _validate_session(root, session_id, aliases)
        if artifact is None:
            rejected.append(Rejection(session_id, str(reason)))
        else:
            accepted.append(artifact)
    return accepted, rejected
