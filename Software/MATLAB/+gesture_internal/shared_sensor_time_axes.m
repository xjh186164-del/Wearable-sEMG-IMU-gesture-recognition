function [raw, filtered, imu] = shared_sensor_time_axes( ...
        rawTimestamp, filteredTimestamp, imuTimestamp)
%SHARED_SENSOR_TIME_AXES Reconstruct a common time axis for sensor streams.

items = {analyse(rawTimestamp), analyse(filteredTimestamp), ...
    analyse(imuTimestamp)};
available = find(cellfun(@(x) ~isempty(x.unwrapped), items));
if ~isempty(available)
    anchor = items{available(1)}.first_raw;
    offsets = NaN(1, 3);
    for k = available
        offsets(k) = mod(double(items{k}.first_raw) - ...
            double(anchor) + 2^31, 2^32) - 2^31;
    end
    origin = min(offsets(available));
    for k = available
        item = items{k};
        item.session_time_s = (item.unwrapped - item.unwrapped(1) + ...
            offsets(k) - origin) / 1e6;
        items{k} = item;
    end
end
[raw, filtered, imu] = items{:};
end

function analysis = analyse(rawTimestamp)
if ~isnumeric(rawTimestamp) || ~isreal(rawTimestamp) || ...
        ~(isvector(rawTimestamp) || isempty(rawTimestamp))
    error("gesture_internal:shared_sensor_time_axes:InvalidTimestamp", ...
        "Each timestamp input must be a real numeric vector.");
end

rawTimestamp = rawTimestamp(:);
analysis = struct("valid", false, "unwrapped", zeros(0, 1), ...
    "duration_s", NaN, "row_count", numel(rawTimestamp), ...
    "first_raw", NaN, "session_time_s", zeros(0, 1));
if isempty(rawTimestamp)
    return;
end

analysis.first_raw = rawTimestamp(1);
if any(~isfinite(rawTimestamp)) || any(rawTimestamp < 0) || ...
        any(rawTimestamp > 2^32 - 1) || ...
        any(rawTimestamp ~= fix(rawTimestamp))
    analysis.unwrapped = double(rawTimestamp);
    return;
end

analysis.unwrapped = zeros(size(rawTimestamp));
state = [];
for row = 1:numel(rawTimestamp)
    [analysis.unwrapped(row), state] = ...
        ads1298_timestamp_unwrap(rawTimestamp(row), state);
end
if any(~isfinite(analysis.unwrapped)) || ...
        any(diff(analysis.unwrapped) <= 0)
    return;
end
analysis.duration_s = ...
    (analysis.unwrapped(end) - analysis.unwrapped(1)) / 1e6;
analysis.valid = true;
end
