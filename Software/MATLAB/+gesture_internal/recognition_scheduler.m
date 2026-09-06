function [state, status, acceptedRateHz] = recognition_scheduler( ...
        state, latestTime, emgTime, emgValues, imuTime, accel, gyro, callback)
%RECOGNITION_SCHEDULER Emit at most one newest causal window per sensor tick.

if nargin == 0
    state = initialState();
    status = state.status;
    acceptedRateHz = 0;
    return;
end
if nargin ~= 8
    error("gesture_internal:recognition_scheduler:InvalidInput", ...
        "Scheduler updates require state, histories, time, and callback.");
end
validateState(state);
if ~isa(latestTime, "double") || ~isscalar(latestTime) || ...
        ~isreal(latestTime) || ~isfinite(latestTime) || latestTime < 0
    error("gesture_internal:recognition_scheduler:InvalidLatestTime", ...
        "latestTime must be a nonnegative finite scalar double.");
end
if ~isa(callback, "function_handle") || ~isscalar(callback)
    error("gesture_internal:recognition_scheduler:InvalidCallback", ...
        "callback must be a scalar function handle.");
end

status = state.status;
acceptedRateHz = acceptedRate(state, latestTime);
if latestTime < 0.5 || ~isEligible(state, latestTime)
    return;
end
state.last_attempt_sensor_time = latestTime;
state.attempted = state.attempted + 1;
if state.sequence_exhausted
    state.error = state.error + 1;
    state.status = unavailableStatus();
    status = state.status;
    return;
end

sequence = state.next_sequence;
try
    snapshot = gesture_internal.recognition_window( ...
        emgTime, emgValues, imuTime, accel, gyro, latestTime, sequence);
catch cause
    if string(cause.identifier) == ...
            "gesture_internal:recognition_window:IncompleteCoverage"
        state.incomplete = state.incomplete + 1;
    else
        state.error = state.error + 1;
        state.status = unavailableStatus();
    end
    status = state.status;
    return;
end

state.emitted = state.emitted + 1;
if sequence == intmax("uint64")
    state.sequence_exhausted = true;
else
    state.next_sequence = sequence + 1;
end
try
    candidateStatus = callback(snapshot);
    [~, ~, valid] = gesture_internal.recognition_status_text( ...
        candidateStatus, acceptedRateHz);
    if ~valid
        error("gesture_internal:recognition_scheduler:InvalidStatus", ...
            "OnSensorWindow returned an invalid recognition status.");
    end
catch
    state.error = state.error + 1;
    state.status = unavailableStatus();
    status = state.status;
    acceptedRateHz = acceptedRate(state, latestTime);
    return;
end

if ~candidateStatus.available
    state.status = candidateStatus;
elseif ~state.has_accepted_sequence || ...
        candidateStatus.sequence > state.last_accepted_sequence
    state.status = candidateStatus;
    state.accepted = state.accepted + 1;
    state.has_accepted_sequence = true;
    state.last_accepted_sequence = candidateStatus.sequence;
end
status = state.status;
acceptedRateHz = acceptedRate(state, latestTime);
end

function state = initialState()
state = struct( ...
    "last_attempt_sensor_time", NaN, ...
    "next_sequence", uint64(1), ...
    "sequence_exhausted", false, ...
    "attempted", 0, ...
    "emitted", 0, ...
    "incomplete", 0, ...
    "error", 0, ...
    "accepted", 0, ...
    "has_accepted_sequence", false, ...
    "last_accepted_sequence", uint64(0), ...
    "status", unavailableStatus());
end

function validateState(state)
expected = string(fieldnames(initialState()));
if ~isstruct(state) || ~isscalar(state) || ...
        ~isequal(string(fieldnames(state)), expected)
    error("gesture_internal:recognition_scheduler:InvalidState", ...
        "state must be created by recognition_scheduler.");
end
end

function eligible = isEligible(state, latestTime)
if isnan(state.last_attempt_sensor_time)
    eligible = true;
    return;
end
threshold = state.last_attempt_sensor_time + 0.1;
tolerance = 8 * eps(max(1, abs(threshold)));
eligible = latestTime >= threshold - tolerance;
end

function rate = acceptedRate(state, latestTime)
if latestTime <= 0
    rate = 0;
else
    rate = state.accepted / latestTime;
end
end

function status = unavailableStatus()
status = struct( ...
    "available", false, ...
    "display_state", "RECOGNITION_UNAVAILABLE", ...
    "candidate_gesture_id", "", ...
    "confidence", NaN, ...
    "probabilities", NaN(1, 8), ...
    "inference_ms", NaN, ...
    "round_trip_ms", NaN, ...
    "sequence", uint64(0), ...
    "sensor_time_s", NaN);
end
