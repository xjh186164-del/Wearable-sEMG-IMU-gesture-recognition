function result = gesture_guided_acquisition(config)
%GESTURE_GUIDED_ACQUISITION Run one guided EMG/IMU acquisition session.

[config, seedWasGenerated] = normalizeConfig(config);
if seedWasGenerated
    fprintf("Auto-generated Seed: %.0f\n", config.Seed);
end
result = gesture_internal.run_guided_session(config, ...
    @ads1298_emg_ble_live, ...
    @(catalog) ProcessGestureGuidanceView(catalog, true), ...
    @gesture_session_quality);
end

function [config, seedWasGenerated] = normalizeConfig(config)
if ~isstruct(config) || ~isscalar(config)
    error("gesture_guided_acquisition:InvalidConfig", ...
        "config must be a scalar struct.");
end

requiredFields = ["SubjectId", "SessionId", "OperatorCode", "SleeveSize"];
optionalFields = ["DeviceIdentifier", "ArmSide", "Seed", "BlockCount", ...
    "CaptureDirectory", "MediaDirectory", "ElectrodeLayoutVersion", ...
    "PcbOrientationVersion"];
providedFields = string(fieldnames(config));
seedWasGenerated = ~ismember("Seed", providedFields);
unknownFields = setdiff(providedFields, [requiredFields, optionalFields], ...
    "stable");
if ~isempty(unknownFields)
    error("gesture_guided_acquisition:UnknownConfigField", ...
        "Unknown config field: %s", unknownFields(1));
end
missingFields = setdiff(requiredFields, providedFields, "stable");
if ~isempty(missingFields)
    error("gesture_guided_acquisition:MissingConfigField", ...
        "Missing required config field: %s", missingFields(1));
end

matlabDirectory = string(fileparts(mfilename("fullpath")));
projectDirectory = string(fileparts(matlabDirectory));
resolved = struct( ...
    "SubjectId", "", ...
    "SessionId", "", ...
    "OperatorCode", "", ...
    "SleeveSize", "", ...
    "DeviceIdentifier", "SensorBiShe-EMG", ...
    "ArmSide", "left", ...
    "Seed", NaN, ...
    "BlockCount", 2, ...
    "CaptureDirectory", fullfile(projectDirectory, "captures"), ...
    "MediaDirectory", fullfile(matlabDirectory, "gesture_media"), ...
    "ElectrodeLayoutVersion", "layout_v1", ...
    "PcbOrientationVersion", "orientation_v1");
for index = 1:numel(providedFields)
    field = providedFields(index);
    resolved.(field) = config.(field);
end
resolved.SubjectId = requiredText(resolved.SubjectId, "SubjectId");
resolved.SessionId = requiredText(resolved.SessionId, "SessionId");
resolved.OperatorCode = requiredText(resolved.OperatorCode, "OperatorCode");
resolved.SleeveSize = requiredText(resolved.SleeveSize, "SleeveSize");
resolved.DeviceIdentifier = requiredText( ...
    resolved.DeviceIdentifier, "DeviceIdentifier");
resolved.ArmSide = requiredText(resolved.ArmSide, "ArmSide");
resolved.CaptureDirectory = requiredText( ...
    resolved.CaptureDirectory, "CaptureDirectory");
resolved.MediaDirectory = requiredText( ...
    resolved.MediaDirectory, "MediaDirectory");
resolved.ElectrodeLayoutVersion = requiredText( ...
    resolved.ElectrodeLayoutVersion, "ElectrodeLayoutVersion");
resolved.PcbOrientationVersion = requiredText( ...
    resolved.PcbOrientationVersion, "PcbOrientationVersion");

if isempty(regexp(resolved.SubjectId, "^S\d{3}$", "once"))
    error("gesture_guided_acquisition:InvalidSubjectId", ...
        "SubjectId must use the pseudonymous S### format.");
end
tokens = regexp(resolved.SessionId, "^(S\d{3})_D\d{2}_R\d{2}$", ...
    "tokens", "once");
if isempty(tokens)
    error("gesture_guided_acquisition:InvalidSessionId", ...
        "SessionId must use the S###_D##_R## format.");
end
if resolved.SubjectId ~= string(tokens{1})
    error("gesture_guided_acquisition:SessionSubjectMismatch", ...
        "The SubjectId embedded in SessionId must match SubjectId.");
end
if resolved.ArmSide ~= "left"
    error("gesture_guided_acquisition:ArmNotAllowed", ...
        "Only the left arm is permitted by this acquisition protocol.");
end
if seedWasGenerated
    resolved.Seed = gesture_internal.generate_protocol_seed();
end
if ~isNonnegativeInteger(resolved.Seed)
    error("gesture_guided_acquisition:InvalidSeed", ...
        "Seed must be a nonnegative integer scalar.");
end
if ~isNonnegativeInteger(resolved.BlockCount) || resolved.BlockCount ~= 2
    error("gesture_guided_acquisition:InvalidBlockCount", ...
        "BlockCount must equal 2 for eight_pose_v5.");
end
if ~isfolder(resolved.CaptureDirectory)
    error("gesture_guided_acquisition:InvalidCaptureDirectory", ...
        "CaptureDirectory must be an existing directory.");
end
if ~isfolder(resolved.MediaDirectory)
    error("gesture_guided_acquisition:InvalidMediaDirectory", ...
        "MediaDirectory must be an existing directory.");
end

resolved.Seed = double(resolved.Seed);
resolved.BlockCount = double(resolved.BlockCount);
config = resolved;
end

function value = requiredText(value, fieldName)
validCharacterVector = ischar(value) && isrow(value);
validStringScalar = isstring(value) && isscalar(value) && ~ismissing(value);
if ~(validCharacterVector || validStringScalar)
    error("gesture_guided_acquisition:InvalidConfig", ...
        "%s must be a nonempty text scalar.", fieldName);
end
value = strtrim(string(value));
if ismissing(value) || strlength(value) == 0
    error("gesture_guided_acquisition:InvalidConfig", ...
        "%s must be a nonempty text scalar.", fieldName);
end
end

function tf = isNonnegativeInteger(value)
tf = isnumeric(value) && isscalar(value) && isreal(value) && ...
    isfinite(value) && value >= 0 && value == fix(value);
end
