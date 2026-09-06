classdef ProcessGestureRecognitionClient < handle
    %PROCESSGESTURERECOGNITIONCLIENT Own one Python worker and inference stream.

    properties (Access = private)
        Transport = []
        WorkerProcess = []
        OwnsWorker logical = false
        ClockFunction function_handle
        PauseFunction function_handle
        Closed logical = false
        RuntimeFailed logical = false
        InFlight logical = false
        InFlightSequence uint64 = uint64(0)
        InFlightSensorTime double = NaN
        InFlightSentAt double = NaN
        LatestValid logical = false
        LatestResponse struct = struct()
        LatestRoundTripMs double = NaN
        LatestAcceptedAt double = NaN
        AttemptedCount double = 0
        SentCount double = 0
        SkippedBusyCount double = 0
        AcceptedCount double = 0
        MalformedCount double = 0
        StaleCount double = 0
        WorkerErrorCount double = 0
        InferenceLatencies double = zeros(0, 1)
        RoundTripLatencies double = zeros(0, 1)
    end

    methods
        function obj = ProcessGestureRecognitionClient( ...
                checkpointPath, calibrationPath, pythonExecutable, ...
                port, startupTimeoutSeconds, options)
            if nargin < 6
                options = struct();
            end
            checkpointPath = validateExistingAbsoluteFile( ...
                checkpointPath, "CheckpointPath");
            calibrationPath = validateExistingAbsoluteFile( ...
                calibrationPath, "CalibrationPath");
            pythonExecutable = validateExistingAbsoluteFile( ...
                pythonExecutable, "PythonExecutable");
            port = validatePort(port);
            startupTimeoutSeconds = validateStartupTimeout( ...
                startupTimeoutSeconds);
            options = normalizeOptions(options);
            obj.ClockFunction = options.ClockFunction;
            obj.PauseFunction = options.PauseFunction;

            startInfo = workerStartInfo( ...
                checkpointPath, calibrationPath, pythonExecutable, port);
            try
                launchResult = options.LaunchFunction(startInfo);
                [obj.WorkerProcess, obj.OwnsWorker] = ...
                    recognizableLaunchLease(launchResult);
                [obj.WorkerProcess, obj.OwnsWorker] = ...
                    validateLaunchResult(launchResult);
                obj.connectAndValidate( ...
                    port, startupTimeoutSeconds, options.TransportFactory);
            catch cause
                obj.cleanupFailedConstruction();
                rethrow(cause);
            end
        end

        function status = update(obj, snapshot)
            obj.AttemptedCount = obj.AttemptedCount + 1;
            if obj.Closed || obj.RuntimeFailed
                status = unavailableStatus();
                return;
            end
            try
                nowSeconds = obj.readClock();
                resolved = obj.pollResponse(nowSeconds);
                if obj.RuntimeFailed
                    status = unavailableStatus();
                    return;
                end
                if obj.InFlight
                    obj.SkippedBusyCount = obj.SkippedBusyCount + 1;
                else
                    obj.sendSnapshot(snapshot, nowSeconds);
                end
                if ~resolved && obj.RuntimeFailed
                    status = unavailableStatus();
                else
                    status = obj.statusAt(nowSeconds);
                end
            catch
                obj.failRuntime();
                status = unavailableStatus();
            end
        end

        function value = summary(obj)
            value = struct( ...
                "attempted", obj.AttemptedCount, ...
                "sent", obj.SentCount, ...
                "skipped_busy", obj.SkippedBusyCount, ...
                "accepted", obj.AcceptedCount, ...
                "malformed", obj.MalformedCount, ...
                "stale", obj.StaleCount, ...
                "worker_error", obj.WorkerErrorCount, ...
                "inference_ms_p50", percentileValue( ...
                    obj.InferenceLatencies, 50), ...
                "inference_ms_p95", percentileValue( ...
                    obj.InferenceLatencies, 95), ...
                "round_trip_ms_p50", percentileValue( ...
                    obj.RoundTripLatencies, 50), ...
                "round_trip_ms_p95", percentileValue( ...
                    obj.RoundTripLatencies, 95));
        end

        function close(obj)
            if obj.Closed
                return;
            end
            obj.Closed = true;
            transport = obj.Transport;
            obj.Transport = [];
            if ~isempty(transport)
                try
                    transport.writeBytes(shutdownFrame());
                catch
                end
            end
            if obj.OwnsWorker
                stopOwnedWorker(obj.WorkerProcess, 3000);
            end
            if ~isempty(transport)
                try
                    transport.close();
                catch
                end
            end
            obj.WorkerProcess = [];
            obj.OwnsWorker = false;
            obj.InFlight = false;
            obj.LatestValid = false;
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function connectAndValidate(obj, port, timeoutSeconds, factory)
            startedAt = obj.readClock();
            while true
                remaining = timeoutSeconds - ...
                    (obj.readClock() - startedAt);
                if remaining < 1
                    error("ProcessGestureRecognitionClient:StartupTimeout", ...
                        "Less than one bounded connect second remains.");
                end
                if processHasExited(obj.WorkerProcess)
                    error("ProcessGestureRecognitionClient:WorkerExited", ...
                        "The Python worker exited before accepting a connection.");
                end
                try
                    transport = factory("127.0.0.1", port, remaining);
                    obj.Transport = transport;
                catch cause
                    if processHasExited(obj.WorkerProcess)
                        error("ProcessGestureRecognitionClient:WorkerExited", ...
                            "The Python worker exited during startup.");
                    end
                    if string(cause.identifier) ~= ...
                            "TcpGestureRecognitionTransport:ConnectFailed"
                        rethrow(cause);
                    end
                    elapsed = obj.readClock() - startedAt;
                    if elapsed >= timeoutSeconds
                        error("ProcessGestureRecognitionClient:StartupTimeout", ...
                            "The Python worker was not reachable before timeout.");
                    end
                    obj.PauseFunction(min(0.05, timeoutSeconds - elapsed));
                    continue;
                end
                remaining = timeoutSeconds - (obj.readClock() - startedAt);
                if remaining <= 0
                    error("ProcessGestureRecognitionClient:StartupTimeout", ...
                        "The Python worker was not ready before timeout.");
                end
                try
                    line = obj.Transport.readLine(remaining);
                    gesture_internal.recognition_ready_validate(line);
                catch cause
                    if processHasExited(obj.WorkerProcess)
                        error("ProcessGestureRecognitionClient:WorkerExited", ...
                            "The Python worker exited before its ready record.");
                    end
                    rethrow(cause);
                end
                return;
            end
        end

        function resolved = pollResponse(obj, nowSeconds)
            resolved = false;
            if ~obj.InFlight
                return;
            end
            try
                line = obj.Transport.readLineIfAvailable();
            catch
                obj.failRuntime();
                return;
            end
            if isempty(line)
                return;
            end
            resolved = true;
            expectedSequence = obj.InFlightSequence;
            expectedSensorTime = obj.InFlightSensorTime;
            sentAt = obj.InFlightSentAt;
            try
                response = gesture_internal.recognition_response_validate( ...
                    line, expectedSequence);
            catch cause
                if string(cause.identifier) == ...
                        "gesture_internal:recognition_response_validate:StaleSequence"
                    obj.StaleCount = obj.StaleCount + 1;
                else
                    if string(cause.identifier) == ...
                            "gesture_internal:recognition_response_validate:SequenceMismatch"
                        obj.failRuntime();
                        return;
                    else
                        obj.MalformedCount = obj.MalformedCount + 1;
                    end
                    obj.InFlight = false;
                    obj.LatestValid = false;
                end
                return;
            end
            if response.sensor_time_s ~= expectedSensorTime
                obj.StaleCount = obj.StaleCount + 1;
                return;
            end
            obj.InFlight = false;
            roundTripMs = (nowSeconds - sentAt) * 1000;
            if ~isfinite(roundTripMs) || roundTripMs < 0
                obj.failRuntime();
                return;
            end
            obj.AcceptedCount = obj.AcceptedCount + 1;
            obj.LatestValid = true;
            obj.LatestResponse = response;
            obj.LatestRoundTripMs = roundTripMs;
            obj.LatestAcceptedAt = nowSeconds;
            obj.InferenceLatencies(end + 1, 1) = response.inference_ms;
            obj.RoundTripLatencies(end + 1, 1) = roundTripMs;
        end

        function sendSnapshot(obj, snapshot, nowSeconds)
            try
                frame = gesture_internal.recognition_request_encode(snapshot);
                obj.Transport.writeBytes(frame);
            catch
                obj.failRuntime();
                return;
            end
            obj.InFlight = true;
            obj.InFlightSequence = snapshot.sequence;
            obj.InFlightSensorTime = double(snapshot.sensor_time_s);
            obj.InFlightSentAt = nowSeconds;
            obj.SentCount = obj.SentCount + 1;
        end

        function status = statusAt(obj, nowSeconds)
            if ~obj.LatestValid || ...
                    nowSeconds - obj.LatestAcceptedAt > 0.5
                status = unavailableStatus();
                return;
            end
            response = obj.LatestResponse;
            status = struct( ...
                "available", true, ...
                "display_state", response.display_state, ...
                "candidate_gesture_id", response.candidate_gesture_id, ...
                "confidence", response.confidence, ...
                "probabilities", response.probabilities, ...
                "inference_ms", response.inference_ms, ...
                "round_trip_ms", obj.LatestRoundTripMs, ...
                "sequence", response.sequence, ...
                "sensor_time_s", response.sensor_time_s);
        end

        function value = readClock(obj)
            value = obj.ClockFunction();
            if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
                    ~isreal(value) || ~isfinite(value)
                error("ProcessGestureRecognitionClient:InvalidClock", ...
                    "ClockFunction must return a finite real scalar.");
            end
            value = double(value);
        end

        function failRuntime(obj)
            if ~obj.RuntimeFailed
                obj.WorkerErrorCount = obj.WorkerErrorCount + 1;
            end
            obj.RuntimeFailed = true;
            obj.InFlight = false;
            obj.LatestValid = false;
            transport = obj.Transport;
            obj.Transport = [];
            if ~isempty(transport)
                try
                    transport.close();
                catch
                end
            end
        end

        function cleanupFailedConstruction(obj)
            obj.Closed = true;
            transport = obj.Transport;
            obj.Transport = [];
            if ~isempty(transport)
                try
                    transport.close();
                catch
                end
            end
            if obj.OwnsWorker && ~processHasExited(obj.WorkerProcess)
                killOwnedWorker(obj.WorkerProcess);
            end
            obj.WorkerProcess = [];
            obj.OwnsWorker = false;
        end
    end
end

function options = normalizeOptions(options)
if ~isstruct(options) || ~isscalar(options)
    error("ProcessGestureRecognitionClient:InvalidOptions", ...
        "options must be a scalar struct.");
end
defaults = struct( ...
    "LaunchFunction", @launchWorker, ...
    "TransportFactory", @createRecognitionTransport, ...
    "ClockFunction", @wallClockSeconds, ...
    "PauseFunction", @pause);
names = string(fieldnames(options));
if ~all(ismember(names, string(fieldnames(defaults))))
    error("ProcessGestureRecognitionClient:InvalidOptions", ...
        "options contains an unknown field.");
end

for name = string(fieldnames(defaults)).'
    if ~isfield(options, name)
        options.(name) = defaults.(name);
    end
end
for name = string(fieldnames(defaults)).'
    if ~isa(options.(name), "function_handle") || ...
            ~isscalar(options.(name))
        error("ProcessGestureRecognitionClient:InvalidOptions", ...
            "%s must be a scalar function handle.", name);
    end
end
end

function transport = createRecognitionTransport(host, port, connectTimeout)
transport = TcpGestureRecognitionTransport(host, port, struct( ...
    "ConnectTimeoutSeconds", connectTimeout));
end

function path = validateExistingAbsoluteFile(value, fieldName)
validText = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~validText
    invalidPath(fieldName);
end
path = string(value);
isDrivePath = ~isempty(regexp(char(path), ...
    '^[A-Za-z]:[\\/]', 'once'));
isUncPath = startsWith(path, "\\");
if strlength(path) == 0 || ~(isDrivePath || isUncPath) || ~isfile(path)
    invalidPath(fieldName);
end
end

function invalidPath(fieldName)
error("ProcessGestureRecognitionClient:Invalid" + fieldName, ...
    "%s must be an existing absolute file path.", fieldName);
end

function port = validatePort(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value ~= fix(value) || ...
        value < 1024 || value > 65535
    error("ProcessGestureRecognitionClient:InvalidPort", ...
        "port must be an integer from 1024 through 65535.");
end
port = double(value);
end

function timeout = validateStartupTimeout(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value <= 0 || value > 15
    error("ProcessGestureRecognitionClient:InvalidStartupTimeout", ...
        "startup timeout must be positive and at most 15 seconds.");
end
timeout = double(value);
end

function startInfo = workerStartInfo( ...
        checkpointPath, calibrationPath, pythonExecutable, port)
startInfo = System.Diagnostics.ProcessStartInfo();
startInfo.FileName = char(pythonExecutable);
startInfo.Arguments = char("-m gesture_ml.realtime_worker" + ...
    " --checkpoint """ + checkpointPath + """" + ...
    " --calibration """ + calibrationPath + """" + ...
    " --host 127.0.0.1 --port " + string(port) + " --device cuda");
startInfo.UseShellExecute = false;
startInfo.CreateNoWindow = true;
end

function result = launchWorker(startInfo)
process = System.Diagnostics.Process.Start(startInfo);
if isempty(process)
    error("ProcessGestureRecognitionClient:LaunchFailed", ...
        "Could not launch the Python recognition worker.");
end
result = struct("process", process, "owned", true);
end

function [process, owned] = validateLaunchResult(result)
if ~isstruct(result) || ~isscalar(result) || ...
        ~isequal(sort(string(fieldnames(result))), ...
        sort(["process"; "owned"])) || isempty(result.process) || ...
        ~isscalar(result.process) || ~isProcessLike(result.process) || ...
        ~islogical(result.owned) || ...
        ~isscalar(result.owned)
    error("ProcessGestureRecognitionClient:InvalidLaunchResult", ...
        "LaunchFunction must return scalar process/owned fields.");
end
process = result.process;
owned = result.owned;
end

function [process, owned] = recognizableLaunchLease(result)
process = [];
owned = false;
if ~isstruct(result) || ~isscalar(result) || ...
        ~isfield(result, "process") || ~isfield(result, "owned") || ...
        isempty(result.process) || ~isscalar(result.process) || ...
        ~isProcessLike(result.process) || ...
        ~islogical(result.owned) || ~isscalar(result.owned)
    return;
end
process = result.process;
owned = result.owned;
end

function valid = isProcessLike(process)
valid = isobject(process) && isscalar(process) && ...
    isprop(process, "HasExited") && ...
    ismethod(process, "WaitForExit") && ismethod(process, "Kill");
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

function frame = shutdownFrame()
version = uint16(1);
[~, ~, endian] = computer;
if endian == 'B'
    version = swapbytes(version);
end
frame = [uint8('GEND'), reshape(typecast(version, "uint8"), 1, [])];
end

function value = percentileValue(values, percentile)
if isempty(values)
    value = NaN;
    return;
end
values = sort(double(values(:)));
position = percentile / 100 * numel(values) + 0.5;
position = min(max(position, 1), numel(values));
lower = floor(position);
upper = ceil(position);
weight = position - lower;
value = values(lower) * (1 - weight) + values(upper) * weight;
end

function stopOwnedWorker(process, timeoutMilliseconds)
if isempty(process) || processHasExited(process)
    return;
end
exited = false;
try
    exited = logical(process.WaitForExit(timeoutMilliseconds));
catch
end
if ~exited && ~processHasExited(process)
    killOwnedWorker(process);
end
end

function killOwnedWorker(process)
try
    process.Kill(true);
catch
    try
        process.Kill();
    catch
    end
end
end

function exited = processHasExited(process)
if isempty(process)
    exited = true;
    return;
end
try
    exited = logical(process.HasExited);
catch
    exited = true;
end
end

function value = wallClockSeconds()
value = posixtime(datetime("now", "TimeZone", "UTC"));
end
