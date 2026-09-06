function [unwrappedTimestamp, state] = ads1298_timestamp_unwrap(rawTimestamp, state)
%ADS1298_TIMESTAMP_UNWRAP Unwrap one uint32-range ESP32 micros timestamp.

if ~isnumeric(rawTimestamp) || ~isscalar(rawTimestamp) || ...
        ~isreal(rawTimestamp) || ~isfinite(rawTimestamp) || ...
        rawTimestamp < 0 || rawTimestamp > 2^32 - 1 || ...
        rawTimestamp ~= fix(rawTimestamp)
    error("ads1298_timestamp_unwrap:InvalidTimestamp", ...
        "rawTimestamp must be an integer scalar in the uint32 range.");
end

if isempty(state)
    state = struct("lastRaw", [], "offset", 0);
elseif ~isstruct(state) || ~isfield(state, "lastRaw") || ...
        ~isfield(state, "offset")
    error("ads1298_timestamp_unwrap:InvalidState", ...
        "state must be empty or returned by ads1298_timestamp_unwrap.");
end

rawTimestamp = double(rawTimestamp);
if ~isempty(state.lastRaw) && rawTimestamp < state.lastRaw && ...
        state.lastRaw - rawTimestamp > 2^31
    state.offset = state.offset + 2^32;
end

state.lastRaw = rawTimestamp;
unwrappedTimestamp = rawTimestamp + state.offset;
end
