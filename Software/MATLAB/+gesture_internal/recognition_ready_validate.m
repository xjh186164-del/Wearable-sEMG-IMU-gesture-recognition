function ready = recognition_ready_validate(line)
%RECOGNITION_READY_VALIDATE Strictly validate one protocol-v1 ready JSON line.

decoded = decodeLine(line);
expectedFields = ["type", "protocol_version", "device", "class_labels"];
labels = recognitionClassLabels();
if ~isstruct(decoded) || ~isscalar(decoded) || ...
        ~isequal(sort(string(fieldnames(decoded))), sort(expectedFields.')) || ...
        ~isExactText(decoded.type, "ready") || ...
        ~isExactProtocolVersion(decoded.protocol_version) || ...
        ~isExactText(decoded.device, ...
        "NVIDIA GeForce RTX 5050 Laptop GPU")
    invalidReady();
end
try
    actualLabels = textArray(decoded.class_labels);
catch
    invalidReady();
end
if ~isequal(actualLabels, labels)
    invalidReady();
end
ready = struct( ...
    "type", "ready", ...
    "protocol_version", uint16(1), ...
    "device", "NVIDIA GeForce RTX 5050 Laptop GPU", ...
    "class_labels", labels);
end

function decoded = decodeLine(line)
if ~isTextScalar(line)
    error("gesture_internal:recognition_ready_validate:InvalidJson", ...
        "ready line must be a text scalar.");
end
try
    decoded = jsondecode(char(string(line)));
catch
    error("gesture_internal:recognition_ready_validate:InvalidJson", ...
        "ready line is not valid JSON.");
end
end

function value = textArray(raw)
if iscell(raw)
    if any(~cellfun(@isTextScalar, raw), "all")
        error("invalid");
    end
    value = reshape(string(raw), 1, []);
elseif isstring(raw) && isvector(raw) && ~any(ismissing(raw))
    value = reshape(raw, 1, []);
else
    error("invalid");
end
end

function valid = isExactProtocolVersion(value)
valid = isnumeric(value) && ~islogical(value) && isscalar(value) && ...
    isreal(value) && isfinite(value) && value == 1;
end

function valid = isExactText(value, expected)
valid = isTextScalar(value) && string(value) == expected;
end

function valid = isTextScalar(value)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
end

function labels = recognitionClassLabels()
labels = ["REST", "WRIST_UP", "WRIST_DOWN", "FOREARM_IN", ...
    "FOREARM_OUT", "ARM_UP", "ARM_DOWN", "FIST"];
end

function invalidReady()
error("gesture_internal:recognition_ready_validate:InvalidReady", ...
    "ready record violates the exact protocol-v1 contract.");
end
