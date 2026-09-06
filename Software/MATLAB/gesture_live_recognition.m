function result = gesture_live_recognition(config, dependencies)
%GESTURE_LIVE_RECOGNITION Run BLE capture with live gesture recognition.

if nargin < 1
    config = struct();
end
if nargin < 2
    dependencies = struct();
end

resolved = normalizeConfig(config);
dependencies = normalizeDependencies(dependencies);

client = dependencies.ClientFactory( ...
    resolved.CheckpointPath, resolved.CalibrationPath, ...
    resolved.PythonExecutable, resolved.Port, ...
    resolved.WorkerStartupTimeoutSeconds);
try
    captureOptions = struct( ...
        "OutputDirectory", resolved.OutputDirectory, ...
        "OnSensorWindow", @(snapshot) client.update(snapshot));
    if resolved.HasCaptureBase
        captureOptions.CaptureBase = resolved.CaptureBase;
    end

    [rawPath, filteredPath, imuPath, captureSummary] = ...
        dependencies.CaptureFunction( ...
        resolved.DeviceIdentifier, resolved.DurationSeconds, captureOptions);
    recognitionSummary = client.summary();

    result = struct( ...
        "rawPath", string(rawPath), ...
        "filteredPath", string(filteredPath), ...
        "imuPath", string(imuPath), ...
        "captureSummary", captureSummary, ...
        "recognitionSummary", recognitionSummary);
catch primary
    closeClientIgnoringFailure(client);
    rethrow(primary);
end
client.close();
end

function resolved = normalizeConfig(config)
if ~isstruct(config) || ~isscalar(config)
    error("gesture_live_recognition:InvalidConfig", ...
        "config must be a scalar struct.");
end

allowedFields = [ ...
    "DeviceIdentifier", "DurationSeconds", "OutputDirectory", ...
    "CaptureBase", "CheckpointPath", "CalibrationPath", ...
    "PythonExecutable", "Port", "WorkerStartupTimeoutSeconds"];
unknownFields = setdiff(string(fieldnames(config)), allowedFields);
if ~isempty(unknownFields)
    error("gesture_live_recognition:UnknownConfigField", ...
        "Unknown config field: %s", unknownFields(1));
end

matlabDirectory = fileparts(mfilename("fullpath"));
projectRoot = string(fileparts(matlabDirectory));
resolved = struct( ...
    "DeviceIdentifier", "SensorBiShe-EMG", ...
    "DurationSeconds", Inf, ...
    "OutputDirectory", fullfile(projectRoot, "captures"), ...
    "HasCaptureBase", false, ...
    "CaptureBase", "", ...
    "CheckpointPath", fullfile(projectRoot, "training_runs", ...
        "p001_dual_cnn", "best_model.pt"), ...
    "CalibrationPath", fullfile(projectRoot, "realtime_configs", ...
        "p001_dual_cnn_endpoint_v2.json"), ...
    "PythonExecutable", fullfile(projectRoot, ".venv-gesture", ...
        "Scripts", "python.exe"), ...
    "Port", 8765, ...
    "WorkerStartupTimeoutSeconds", 15);

if isfield(config, "DeviceIdentifier")
    resolved.DeviceIdentifier = normalizeNonblankText( ...
        config.DeviceIdentifier, "DeviceIdentifier");
end
if isfield(config, "DurationSeconds")
    resolved.DurationSeconds = normalizeDuration(config.DurationSeconds);
end
if isfield(config, "OutputDirectory")
    resolved.OutputDirectory = normalizeOutputDirectory( ...
        config.OutputDirectory, projectRoot);
end
if isfield(config, "CaptureBase")
    resolved.HasCaptureBase = true;
    resolved.CaptureBase = normalizeCaptureBase(config.CaptureBase);
end
if isfield(config, "CheckpointPath")
    resolved.CheckpointPath = validateExistingAbsoluteFile( ...
        config.CheckpointPath, "CheckpointPath");
end
if isfield(config, "CalibrationPath")
    resolved.CalibrationPath = validateExistingAbsoluteFile( ...
        config.CalibrationPath, "CalibrationPath");
end
if isfield(config, "PythonExecutable")
    resolved.PythonExecutable = validateExistingAbsoluteFile( ...
        config.PythonExecutable, "PythonExecutable");
end
if isfield(config, "Port")
    resolved.Port = normalizePort(config.Port);
end
if isfield(config, "WorkerStartupTimeoutSeconds")
    resolved.WorkerStartupTimeoutSeconds = normalizeStartupTimeout( ...
        config.WorkerStartupTimeoutSeconds);
end

resolved.CheckpointPath = validateExistingAbsoluteFile( ...
    resolved.CheckpointPath, "CheckpointPath");
resolved.CalibrationPath = validateExistingAbsoluteFile( ...
    resolved.CalibrationPath, "CalibrationPath");
resolved.PythonExecutable = validateExistingAbsoluteFile( ...
    resolved.PythonExecutable, "PythonExecutable");
resolved.OutputDirectory = normalizeOutputDirectory( ...
    resolved.OutputDirectory, projectRoot);
end

function dependencies = normalizeDependencies(dependencies)
if ~isstruct(dependencies) || ~isscalar(dependencies)
    error("gesture_live_recognition:InvalidDependencies", ...
        "dependencies must be a scalar struct.");
end
defaults = struct( ...
    "ClientFactory", @createClient, ...
    "CaptureFunction", @ads1298_emg_ble_live);
unknownFields = setdiff(string(fieldnames(dependencies)), ...
    string(fieldnames(defaults)));
if ~isempty(unknownFields)
    error("gesture_live_recognition:InvalidDependencies", ...
        "dependencies contains an unknown field.");
end
for field = string(fieldnames(defaults)).'
    if ~isfield(dependencies, field)
        dependencies.(field) = defaults.(field);
    end
    if ~isa(dependencies.(field), "function_handle") || ...
            ~isscalar(dependencies.(field))
        error("gesture_live_recognition:InvalidDependencies", ...
            "%s must be a scalar function handle.", field);
    end
end
end

function client = createClient( ...
        checkpointPath, calibrationPath, pythonExecutable, port, timeout)
client = ProcessGestureRecognitionClient( ...
    checkpointPath, calibrationPath, pythonExecutable, port, timeout);
end

function closeClientIgnoringFailure(client)
try
    client.close();
catch
end
end

function value = normalizeNonblankText(value, fieldName)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~valid
    invalidField(fieldName, "%s must be a nonblank text scalar.");
end
value = strtrim(string(value));
if strlength(value) == 0
    invalidField(fieldName, "%s must be a nonblank text scalar.");
end
end

function value = normalizeDuration(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || isnan(value) || value <= 0
    error("gesture_live_recognition:InvalidDurationSeconds", ...
        "DurationSeconds must be positive finite or Inf.");
end
value = double(value);
end

function value = normalizeOutputDirectory(value, projectRoot)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~valid
    error("gesture_live_recognition:InvalidOutputDirectory", ...
        "OutputDirectory must be a nonblank text scalar.");
end
value = string(value);
if strlength(strtrim(value)) == 0
    error("gesture_live_recognition:InvalidOutputDirectory", ...
        "OutputDirectory must be a nonblank text scalar.");
end
if ~isAbsolutePath(value)
    value = fullfile(projectRoot, value);
end
try
    value = string(System.IO.Path.GetFullPath(char(value)));
catch
    error("gesture_live_recognition:InvalidOutputDirectory", ...
        "OutputDirectory is not a valid path.");
end
if isfile(value)
    error("gesture_live_recognition:InvalidOutputDirectory", ...
        "OutputDirectory must not name an existing file.");
end
end

function value = normalizeCaptureBase(value)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~valid
    invalidCaptureBase();
end
value = string(value);
if isempty(regexp(char(value), '^[A-Za-z0-9_-]+$', 'once'))
    invalidCaptureBase();
end
end

function invalidCaptureBase()
error("gesture_live_recognition:InvalidCaptureBase", ...
    "CaptureBase must contain only letters, digits, underscores, or hyphens.");
end

function value = validateExistingAbsoluteFile(value, fieldName)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~valid
    invalidPath(fieldName);
end
value = string(value);
if strlength(value) == 0 || ~isAbsolutePath(value) || ~isfile(value)
    invalidPath(fieldName);
end
end

function absolute = isAbsolutePath(value)
absolute = ~isempty(regexp(char(value), '^[A-Za-z]:[\\/]', 'once')) || ...
    startsWith(value, "\\");
end

function invalidPath(fieldName)
error("gesture_live_recognition:Invalid" + fieldName, ...
    "%s must be an existing absolute file path.", fieldName);
end

function value = normalizePort(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value ~= fix(value) || ...
        value < 1024 || value > 65535
    error("gesture_live_recognition:InvalidPort", ...
        "Port must be an integer from 1024 through 65535.");
end
value = double(value);
end

function value = normalizeStartupTimeout(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value <= 0 || value > 15
    error("gesture_live_recognition:InvalidWorkerStartupTimeoutSeconds", ...
        "WorkerStartupTimeoutSeconds must be positive and at most 15.");
end
value = double(value);
end

function invalidField(fieldName, message)
error("gesture_live_recognition:Invalid" + fieldName, message, fieldName);
end
