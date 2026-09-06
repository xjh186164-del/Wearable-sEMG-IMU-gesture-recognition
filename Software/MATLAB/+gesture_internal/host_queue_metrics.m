function metrics = host_queue_metrics(queueAges)
%HOST_QUEUE_METRICS Summarize observed host-side notification queue ages.

validShape = isempty(queueAges) || isvector(queueAges);
if ~isnumeric(queueAges) || ~isreal(queueAges) || ~validShape
    invalidInput();
end

values = double(queueAges(:));
if any(~isfinite(values))
    invalidInput();
end
if isempty(values)
    metrics = struct("max_s", NaN, "p95_s", NaN);
    return;
end

values = sort(max(values, 0));
count = numel(values);
position = 1 + (count - 1) * 0.95;
lowerIndex = floor(position);
upperIndex = ceil(position);
fraction = position - lowerIndex;
p95 = values(lowerIndex) + ...
    fraction * (values(upperIndex) - values(lowerIndex));
metrics = struct("max_s", values(end), "p95_s", p95);
end

function invalidInput()
error("gesture_internal:host_queue_metrics:InvalidInput", ...
    "queueAges must be a finite real numeric vector.");
end
