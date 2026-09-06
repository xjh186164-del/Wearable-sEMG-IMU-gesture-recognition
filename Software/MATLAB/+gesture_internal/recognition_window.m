function snapshot = recognition_window( ...
        emgTime, emgValues, imuTime, accel, gyro, latestTime, sequence)
%RECOGNITION_WINDOW Build one exact causal dual-rate recognition window.

emgTime = validateTime(emgTime, "Emg");
imuTime = validateTime(imuTime, "Imu");
emgValues = validateValues(emgValues, numel(emgTime), 8, "EmgValues");
accel = validateValues(accel, numel(imuTime), 3, "Accel");
gyro = validateValues(gyro, numel(imuTime), 3, "Gyro");

if ~isa(latestTime, "double") || ~isscalar(latestTime) || ...
        ~isreal(latestTime) || ~isfinite(latestTime)
    error("gesture_internal:recognition_window:InvalidLatestTime", ...
        "latestTime must be a finite real scalar double.");
end
if ~isa(sequence, "uint64") || ~isscalar(sequence)
    error("gesture_internal:recognition_window:InvalidSequence", ...
        "sequence must be a scalar uint64.");
end

startTime = latestTime - 0.5;
emgGrid = startTime + (0:249) / 500;
imuGrid = startTime + (0:51) / 104;
emgCausal = emgTime <= latestTime;
imuCausal = imuTime <= latestTime;
requireCoverage(emgTime(emgCausal), emgGrid);
requireCoverage(imuTime(imuCausal), imuGrid);

emgWindow = interp1( ...
    emgTime(emgCausal), emgValues(emgCausal, :), emgGrid, "linear");
imuValues = [accel, gyro];
imuWindow = interp1( ...
    imuTime(imuCausal), imuValues(imuCausal, :), imuGrid, "linear");
if any(~isfinite(emgWindow), "all") || any(~isfinite(imuWindow), "all")
    error("gesture_internal:recognition_window:IncompleteCoverage", ...
        "Sensor histories must cover every requested grid point.");
end

snapshot = struct( ...
    "sequence", sequence, ...
    "sensor_time_s", latestTime, ...
    "emg_mV", emgWindow.', ...
    "imu", imuWindow.');
end

function time = validateTime(value, name)
identifier = "gesture_internal:recognition_window:Invalid" + name + "Time";
if ~isnumeric(value) || ~isreal(value) || isempty(value) || ...
        ~isvector(value) || any(~isfinite(value), "all")
    error(identifier, "%s time must be a finite real numeric vector.", name);
end
time = double(value(:));
if any(diff(time) <= 0)
    error(identifier, "%s time must be strictly increasing.", name);
end
end

function values = validateValues(value, rowCount, channelCount, name)
identifier = "gesture_internal:recognition_window:Invalid" + name;
if ~isnumeric(value) || ~isreal(value) || ~ismatrix(value) || ...
        ~isequal(size(value), [rowCount, channelCount]) || ...
        any(~isfinite(value), "all")
    error(identifier, ...
        "%s must be a finite real numeric matrix with %d channels.", ...
        name, channelCount);
end
values = double(value);
end

function requireCoverage(time, grid)
if isempty(time) || time(1) > grid(1) || time(end) < grid(end)
    error("gesture_internal:recognition_window:IncompleteCoverage", ...
        "Sensor histories must cover every requested grid point.");
end
end
