function options = ads1298_capture_options( ...
        captureOptions, defaultOutputDirectory, defaultCaptureBase)
%ADS1298_CAPTURE_OPTIONS Validate and normalize optional capture hooks.

if ~isstruct(captureOptions) || ~isscalar(captureOptions)
    error("ads1298_capture_options:InvalidOptions", ...
        "captureOptions must be a scalar struct.");
end

allowedFields = [ ...
    "OutputDirectory", "CaptureBase", "OnTimeUpdate", ...
    "OnSensorWindow", "StopRequested"];
unknownFields = setdiff(string(fieldnames(captureOptions)), allowedFields);
if ~isempty(unknownFields)
    error("ads1298_capture_options:UnknownOption", ...
        "Unknown capture option: %s", unknownFields(1));
end

options = struct( ...
    "OutputDirectory", normalizeOutputDirectory(defaultOutputDirectory), ...
    "CaptureBase", normalizeCaptureBase(defaultCaptureBase), ...
    "OnTimeUpdate", [], ...
    "OnSensorWindow", [], ...
    "StopRequested", @() false);

if isfield(captureOptions, "OutputDirectory")
    options.OutputDirectory = normalizeOutputDirectory( ...
        captureOptions.OutputDirectory);
end
if isfield(captureOptions, "CaptureBase")
    options.CaptureBase = normalizeCaptureBase(captureOptions.CaptureBase);
end
if isfield(captureOptions, "OnTimeUpdate")
    options.OnTimeUpdate = normalizeCallback( ...
        captureOptions.OnTimeUpdate, "OnTimeUpdate");
end
if isfield(captureOptions, "OnSensorWindow")
    options.OnSensorWindow = normalizeCallback( ...
        captureOptions.OnSensorWindow, "OnSensorWindow");
end
if isfield(captureOptions, "StopRequested")
    stopRequested = normalizeCallback( ...
        captureOptions.StopRequested, "StopRequested");
    if ~isempty(stopRequested)
        options.StopRequested = stopRequested;
    end
end
end

function outputDirectory = normalizeOutputDirectory(value)
if ~(ischar(value) || (isstring(value) && isscalar(value)))
    error("ads1298_capture_options:InvalidOutputDirectory", ...
        "OutputDirectory must be a nonempty text scalar.");
end
outputDirectory = string(value);
if ~isscalar(outputDirectory) || strlength(outputDirectory) == 0
    error("ads1298_capture_options:InvalidOutputDirectory", ...
        "OutputDirectory must be a nonempty text scalar.");
end
end

function captureBase = normalizeCaptureBase(value)
if ~(ischar(value) || (isstring(value) && isscalar(value)))
    error("ads1298_capture_options:InvalidCaptureBase", ...
        "CaptureBase must contain only letters, digits, underscores, or hyphens.");
end
captureBase = string(value);
if ~isscalar(captureBase) || isempty(regexp(char(captureBase), ...
        '^[A-Za-z0-9_-]+$', 'once'))
    error("ads1298_capture_options:InvalidCaptureBase", ...
        "CaptureBase must contain only letters, digits, underscores, or hyphens.");
end
end

function callback = normalizeCallback(value, fieldName)
if isempty(value)
    callback = [];
elseif isa(value, "function_handle") && isscalar(value)
    callback = value;
else
    error("ads1298_capture_options:InvalidCallback", ...
        "%s must be empty or a scalar function handle.", fieldName);
end
end
