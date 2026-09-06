"""Turn lifecycle events into guarded stable training intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


CLASS_LABELS = (
    "REST",
    "WRIST_UP",
    "WRIST_DOWN",
    "FOREARM_IN",
    "FOREARM_OUT",
    "ARM_UP",
    "ARM_DOWN",
    "FIST",
)
TARGET_LABELS = CLASS_LABELS[1:]


@dataclass(frozen=True)
class StableInterval:
    start_s: float
    end_s: float
    label: str
    label_id: int
    block_id: int
    trial_id: int
    interval_id: str


@dataclass(frozen=True)
class _Event:
    time_s: float
    event_type: str
    block_id: int
    trial_id: int
    gesture_id: str
    valid: bool
    row: int


def _parse_bool(value) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(f"invalid event valid flag: {value!r}")


def _parse_events(rows: Iterable[Mapping[str, object]]) -> list[_Event]:
    parsed = []
    previous_time = -math.inf
    for index, row in enumerate(rows):
        try:
            time_s = float(row["session_time_s"])
            event_type = str(row["event_type"]).strip()
            block_id = int(row["block_id"])
            trial_id = int(row["trial_id"])
            gesture_id = str(row["gesture_id"]).strip()
            valid = _parse_bool(row["valid"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed event row {index + 1}") from error
        if not math.isfinite(time_s) or time_s < previous_time:
            raise ValueError("event times must be finite and nondecreasing")
        previous_time = time_s
        parsed.append(_Event(time_s, event_type, block_id, trial_id, gesture_id, valid, index))
    return parsed


def _pair_events(events: list[_Event], start_type: str, end_type: str):
    starts = [event for event in events if event.event_type == start_type]
    ends = [event for event in events if event.event_type == end_type]
    if len(starts) != len(ends):
        raise ValueError(f"unpaired {start_type}/{end_type} events")
    pairs = []
    for start, end in zip(starts, ends):
        if (
            end.row <= start.row
            or end.time_s <= start.time_s
            or start.block_id != end.block_id
            or start.trial_id != end.trial_id
            or start.gesture_id != end.gesture_id
            or not start.valid
            or not end.valid
        ):
            raise ValueError(f"mismatched {start_type}/{end_type} events")
        pairs.append((start, end))
    return pairs


def extract_stable_intervals(
    rows: Iterable[Mapping[str, object]], guard_s: float = 0.4
) -> list[StableInterval]:
    if not math.isfinite(guard_s) or guard_s < 0:
        raise ValueError("guard_s must be finite and nonnegative")
    events = _parse_events(rows)
    invalid_trials = {event.trial_id for event in events if event.event_type == "trial_invalid"}
    trial_ids = sorted({
        event.trial_id
        for event in events
        if event.trial_id > 0 and event.event_type in {
            "rest_valid_start", "rest_valid_end", "target_hold_start", "target_hold_end"
        }
    })
    intervals: list[StableInterval] = []
    label_ids = {label: index for index, label in enumerate(CLASS_LABELS)}

    for trial_id in trial_ids:
        if trial_id in invalid_trials:
            continue
        trial_events = [event for event in events if event.trial_id == trial_id]
        rest_pairs = _pair_events(trial_events, "rest_valid_start", "rest_valid_end")
        target_pairs = _pair_events(trial_events, "target_hold_start", "target_hold_end")
        for kind, pairs in (("rest", rest_pairs), ("target", target_pairs)):
            for pair_index, (start, end) in enumerate(pairs, start=1):
                label = "REST" if kind == "rest" else start.gesture_id
                if kind == "target" and label not in TARGET_LABELS:
                    raise ValueError(f"unknown target gesture: {label}")
                guarded_start = start.time_s + guard_s
                guarded_end = end.time_s - guard_s
                if guarded_end <= guarded_start:
                    raise ValueError("stable interval is too short after applying guards")
                intervals.append(StableInterval(
                    start_s=guarded_start,
                    end_s=guarded_end,
                    label=label,
                    label_id=label_ids[label],
                    block_id=start.block_id,
                    trial_id=trial_id,
                    interval_id=f"trial-{trial_id}-{kind}-{pair_index}",
                ))
    intervals.sort(key=lambda value: (value.start_s, value.trial_id, value.interval_id))
    return intervals

