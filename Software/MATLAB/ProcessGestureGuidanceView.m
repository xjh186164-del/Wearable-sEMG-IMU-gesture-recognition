classdef ProcessGestureGuidanceView < handle
    %PROCESSGESTUREGUIDANCEVIEW Proxy guidance UI to a child MATLAB process.

    properties (Access = private)
        IpcDirectory string = ""
        StatePath string = ""
        CommandPath string = ""
        ReadyPath string = ""
        WorkerProcess = []
        ClockFunction function_handle
        PauseFunction function_handle
        PublishPeriodSeconds double = 0.5
        ShutdownTimeoutSeconds double = 5
        LastPublishTime double = -Inf
        Sequence double = 0
        LastSnapshot = struct()
        LastInvalidateGeneration double = 0
        InvalidatePending logical = false
        StopRequestedFlag logical = false
        Closed logical = false
    end

    methods
        function obj = ProcessGestureGuidanceView(catalog, visible, options)
            if nargin < 2
                visible = true;
            end
            if nargin < 3
                options = struct();
            end
            [mediaDirectory, visible] = validateInputs(catalog, visible);
            options = normalizeOptions(options);
            obj.IpcDirectory = options.IpcDirectory;
            obj.StatePath = fullfile(obj.IpcDirectory, "state.json");
            obj.CommandPath = fullfile(obj.IpcDirectory, "command.json");
            obj.ReadyPath = fullfile(obj.IpcDirectory, "ready.json");
            obj.ClockFunction = options.ClockFunction;
            obj.PauseFunction = options.PauseFunction;
            obj.PublishPeriodSeconds = options.PublishPeriodSeconds;
            obj.ShutdownTimeoutSeconds = options.ShutdownTimeoutSeconds;

            mkdir(obj.IpcDirectory);
            try
                gesture_internal.guidance_mailbox_write( ...
                    obj.CommandPath, struct( ...
                    "invalidate_generation", 0, ...
                    "stop_requested", false));
                obj.WorkerProcess = options.LaunchFunction( ...
                    obj.IpcDirectory, mediaDirectory, visible);
                obj.waitUntilReady(options.ReadyTimeoutSeconds);
            catch cause
                obj.cleanupFailedConstruction();
                rethrow(cause);
            end
        end

        function update(obj, snapshot)
            if obj.Closed
                return;
            end
            obj.LastSnapshot = snapshotPayload(snapshot);
            nowSeconds = obj.ClockFunction();
            if obj.Sequence == 0 || ...
                    nowSeconds - obj.LastPublishTime >= ...
                    obj.PublishPeriodSeconds
                obj.publish(false);
                obj.LastPublishTime = nowSeconds;
            end
        end

        function requested = consumeInvalidateRequested(obj)
            obj.refreshCommands();
            requested = obj.InvalidatePending;
            obj.InvalidatePending = false;
        end

        function requested = stopRequested(obj)
            obj.refreshCommands();
            if processHasExited(obj.WorkerProcess)
                obj.StopRequestedFlag = true;
            end
            requested = obj.StopRequestedFlag;
        end

        function close(obj)
            if obj.Closed
                return;
            end
            obj.Closed = true;
            try
                obj.publish(true);
            catch
                % Process cleanup remains mandatory if the mailbox is unavailable.
            end
            obj.stopWorker();
            obj.removeIpcDirectory();
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function waitUntilReady(obj, timeoutSeconds)
            startedAt = obj.ClockFunction();
            while true
                [ready, valid] = gesture_internal.guidance_mailbox_read( ...
                    obj.ReadyPath);
                if valid && isfield(ready, "ready") && ...
                        islogical(ready.ready) && isscalar(ready.ready) && ...
                        ready.ready
                    return;
                end
                if processHasExited(obj.WorkerProcess)
                    error("ProcessGestureGuidanceView:WorkerExited", ...
                        "The guidance process exited before becoming ready.");
                end
                if obj.ClockFunction() - startedAt >= timeoutSeconds
                    error("ProcessGestureGuidanceView:StartupTimeout", ...
                        "The guidance process did not become ready in time.");
                end
                obj.PauseFunction(0.05);
            end
        end

        function publish(obj, closeRequested)
            if isempty(fieldnames(obj.LastSnapshot))
                payload = defaultSnapshotPayload();
            else
                payload = obj.LastSnapshot;
            end
            obj.Sequence = obj.Sequence + 1;
            payload.sequence = obj.Sequence;
            payload.close_requested = logical(closeRequested);
            gesture_internal.guidance_mailbox_write(obj.StatePath, payload);
        end

        function refreshCommands(obj)
            if obj.Closed
                return;
            end
            [command, valid] = gesture_internal.guidance_mailbox_read( ...
                obj.CommandPath);
            if ~valid || ~isfield(command, "invalidate_generation") || ...
                    ~isfield(command, "stop_requested")
                return;
            end
            generation = command.invalidate_generation;
            stopRequestedValue = command.stop_requested;
            validGeneration = isnumeric(generation) && isscalar(generation) && ...
                isreal(generation) && isfinite(generation) && generation >= 0 && ...
                generation == fix(generation);
            validStop = islogical(stopRequestedValue) && ...
                isscalar(stopRequestedValue);
            if ~(validGeneration && validStop)
                return;
            end
            generation = double(generation);
            if generation > obj.LastInvalidateGeneration
                obj.InvalidatePending = true;
                obj.LastInvalidateGeneration = generation;
            end
            obj.StopRequestedFlag = obj.StopRequestedFlag || stopRequestedValue;
        end

        function cleanupFailedConstruction(obj)
            obj.Closed = true;
            obj.stopWorker();
            obj.removeIpcDirectory();
        end

        function stopWorker(obj)
            process = obj.WorkerProcess;
            if isempty(process) || processHasExited(process)
                return;
            end
            timeoutMilliseconds = max(0, round( ...
                obj.ShutdownTimeoutSeconds * 1000));
            exited = false;
            try
                exited = logical(process.WaitForExit(timeoutMilliseconds));
            catch
            end
            if ~exited && ~processHasExited(process)
                try
                    process.Kill(true);
                catch
                    try
                        process.Kill();
                    catch
                    end
                end
            end
        end

        function removeIpcDirectory(obj)
            directory = obj.IpcDirectory;
            if strlength(directory) == 0 || ~isfolder(directory)
                return;
            end
            try
                rmdir(directory, "s");
            catch
            end
        end
    end
end

function [mediaDirectory, visible] = validateInputs(catalog, visible)
requiredColumns = ["gesture_id", "path", "duration_s", "width", "height"];
if ~istable(catalog) || height(catalog) < 1 || ...
        ~all(ismember(requiredColumns, ...
        string(catalog.Properties.VariableNames)))
    error("ProcessGestureGuidanceView:InvalidCatalog", ...
        "catalog must contain the validated guidance media rows.");
end
paths = string(catalog.path);
directories = strings(size(paths));
for index = 1:numel(paths)
    directories(index) = string(fileparts(paths(index)));
end
directories = unique(directories);
if numel(directories) ~= 1 || strlength(directories) == 0
    error("ProcessGestureGuidanceView:InvalidCatalog", ...
        "All guidance media must share one directory.");
end
mediaDirectory = directories(1);
if ~(islogical(visible) || isnumeric(visible)) || ...
        ~isscalar(visible) || ~isreal(visible) || ~isfinite(visible) || ...
        ~ismember(double(visible), [0, 1])
    error("ProcessGestureGuidanceView:InvalidVisible", ...
        "visible must be a logical scalar.");
end
visible = logical(visible);
end

function options = normalizeOptions(options)
if ~isstruct(options) || ~isscalar(options)
    error("ProcessGestureGuidanceView:InvalidOptions", ...
        "options must be a scalar struct.");
end
defaults = struct( ...
    "IpcDirectory", string(tempname(tempdir)), ...
    "LaunchFunction", @launchWorkerProcess, ...
    "ClockFunction", @wallClockSeconds, ...
    "PauseFunction", @pause, ...
    "ReadyTimeoutSeconds", 30, ...
    "ShutdownTimeoutSeconds", 5, ...
    "PublishPeriodSeconds", 0.5);
names = string(fieldnames(options));
if ~all(ismember(names, string(fieldnames(defaults))))
    error("ProcessGestureGuidanceView:InvalidOptions", ...
        "options contains an unknown field.");
end
for name = string(fieldnames(defaults)).'
    if ~isfield(options, name)
        options.(name) = defaults.(name);
    end
end
options.IpcDirectory = validateIpcDirectory(options.IpcDirectory);
for field = ["LaunchFunction", "ClockFunction", "PauseFunction"]
    if ~isa(options.(field), "function_handle")
        error("ProcessGestureGuidanceView:InvalidOptions", ...
            "%s must be a function handle.", field);
    end
end
for field = ["ReadyTimeoutSeconds", "ShutdownTimeoutSeconds", ...
        "PublishPeriodSeconds"]
    value = options.(field);
    if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || ...
            ~isfinite(value) || value <= 0
        error("ProcessGestureGuidanceView:InvalidOptions", ...
            "%s must be a positive finite scalar.", field);
    end
    options.(field) = double(value);
end
end

function directory = validateIpcDirectory(value)
validText = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~validText || strlength(strtrim(string(value))) == 0
    error("ProcessGestureGuidanceView:InvalidOptions", ...
        "IpcDirectory must be a nonempty text scalar.");
end
directory = string(value);
tempRoot = string(tempdir);
if ~startsWith(lower(directory), lower(tempRoot))
    error("ProcessGestureGuidanceView:InvalidOptions", ...
        "IpcDirectory must be beneath tempdir.");
end
if isfile(directory) || isfolder(directory)
    error("ProcessGestureGuidanceView:InvalidOptions", ...
        "IpcDirectory must not already exist.");
end
end

function payload = snapshotPayload(snapshot)
required = ["gesture_id", "stage_name", "stage_elapsed_s", ...
    "stage_remaining_s", "block_id", "progress_text", ...
    "finished", "stopped"];
if ~isstruct(snapshot) || ~isscalar(snapshot) || ...
        ~all(isfield(snapshot, required))
    error("ProcessGestureGuidanceView:InvalidSnapshot", ...
        "snapshot does not contain the required guidance fields.");
end
payload = struct( ...
    "gesture_id", string(snapshot.gesture_id), ...
    "stage_name", string(snapshot.stage_name), ...
    "stage_elapsed_s", double(snapshot.stage_elapsed_s), ...
    "stage_remaining_s", double(snapshot.stage_remaining_s), ...
    "block_id", double(snapshot.block_id), ...
    "progress_text", string(snapshot.progress_text), ...
    "finished", logical(snapshot.finished), ...
    "stopped", logical(snapshot.stopped));
validText = isscalar(payload.gesture_id) && ~ismissing(payload.gesture_id) && ...
    isscalar(payload.stage_name) && ~ismissing(payload.stage_name) && ...
    isscalar(payload.progress_text) && ~ismissing(payload.progress_text);
numericValues = [payload.stage_elapsed_s, payload.stage_remaining_s, ...
    payload.block_id];
validNumeric = all(isfinite(numericValues)) && ...
    all(numericValues >= 0);
if ~(validText && validNumeric && isscalar(payload.finished) && ...
        isscalar(payload.stopped))
    error("ProcessGestureGuidanceView:InvalidSnapshot", ...
        "snapshot guidance fields are malformed.");
end
end

function payload = defaultSnapshotPayload()
payload = struct( ...
    "gesture_id", "REST", ...
    "stage_name", "stopped", ...
    "stage_elapsed_s", 0, ...
    "stage_remaining_s", 0, ...
    "block_id", 0, ...
    "progress_text", "", ...
    "finished", false, ...
    "stopped", true);
end

function process = launchWorkerProcess(ipcDirectory, mediaDirectory, visible)
matlabDirectory = string(fileparts(mfilename("fullpath")));
command = "try, addpath('" + escapeLiteral(matlabDirectory) + ...
    "'); gesture_guidance_worker('" + escapeLiteral(ipcDirectory) + ...
    "','" + escapeLiteral(mediaDirectory) + "'," + ...
    lower(string(logical(visible))) + ...
    "); catch cause, disp(getReport(cause,'extended')); exit(1); end; exit(0);";
executable = fullfile(matlabroot, "bin", "win64", "MATLAB.exe");
startInfo = System.Diagnostics.ProcessStartInfo();
startInfo.FileName = executable;
startInfo.Arguments = '-nosplash -r "' + command + '"';
startInfo.UseShellExecute = false;
startInfo.CreateNoWindow = false;
process = System.Diagnostics.Process.Start(startInfo);
if isempty(process)
    error("ProcessGestureGuidanceView:LaunchFailed", ...
        "Could not launch the guidance MATLAB process.");
end
end

function value = escapeLiteral(value)
value = replace(string(value), "'", "''");
end

function value = wallClockSeconds()
value = posixtime(datetime("now", "TimeZone", "UTC"));
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
