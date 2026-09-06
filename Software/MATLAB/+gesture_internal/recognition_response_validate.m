function response = recognition_response_validate(line, expectedSequence)
%RECOGNITION_RESPONSE_VALIDATE Validate and correlate one v1 prediction line.

if ~isa(expectedSequence, "uint64") || ~isscalar(expectedSequence)
    error("gesture_internal:recognition_response_validate:InvalidExpectedSequence", ...
        "expectedSequence must be a uint64 scalar.");
end
[decoded, actualSequence] = decodeLine(line);
expectedFields = ["type", "protocol_version", "sequence", ...
    "sensor_time_s", "candidate_gesture_id", "display_state", ...
    "confidence", "probabilities", "inference_ms"];
if ~isstruct(decoded) || ~isscalar(decoded) || ...
        ~isequal(sort(string(fieldnames(decoded))), sort(expectedFields.')) || ...
        ~isExactText(decoded.type, "prediction") || ...
        ~isExactProtocolVersion(decoded.protocol_version)
    invalidResponse();
end

if actualSequence < expectedSequence
    error("gesture_internal:recognition_response_validate:StaleSequence", ...
        "prediction sequence is older than the expected request.");
elseif actualSequence ~= expectedSequence
    error("gesture_internal:recognition_response_validate:SequenceMismatch", ...
        "prediction sequence does not match the expected request.");
end

labels = recognitionClassLabels();
displayStates = [labels, "WARMING_UP", "TRANSITION_UNSTABLE"];
if ~isTextMember(decoded.candidate_gesture_id, labels) || ...
        ~isTextMember(decoded.display_state, displayStates) || ...
        ~isFiniteScalar(decoded.sensor_time_s) || ...
        ~isBoundedScalar(decoded.confidence, 0, 1) || ...
        ~isBoundedScalar(decoded.inference_ms, 0, Inf)
    invalidResponse();
end
probabilities = decoded.probabilities;
if ~isnumeric(probabilities) || islogical(probabilities) || ...
        ~isreal(probabilities) || ~isvector(probabilities) || ...
        numel(probabilities) ~= 8
    invalidResponse();
end
probabilities = reshape(double(probabilities), 1, []);
if ~all(isfinite(probabilities)) || any(probabilities < 0) || ...
        any(probabilities > 1) || abs(sum(probabilities) - 1) > 1e-4
    invalidResponse();
end

response = struct( ...
    "type", "prediction", ...
    "protocol_version", uint16(1), ...
    "sequence", actualSequence, ...
    "sensor_time_s", double(decoded.sensor_time_s), ...
    "candidate_gesture_id", string(decoded.candidate_gesture_id), ...
    "display_state", string(decoded.display_state), ...
    "confidence", double(decoded.confidence), ...
    "probabilities", probabilities, ...
    "inference_ms", double(decoded.inference_ms));
end

function [decoded, sequence] = decodeLine(line)
if ~isTextScalar(line)
    error("gesture_internal:recognition_response_validate:InvalidJson", ...
        "prediction line must be a text scalar.");
end
raw = char(string(line));
[sequence, tokenExtent] = parseSequence(raw);
normalized = [raw(1:tokenExtent(1) - 1), '0', ...
    raw(tokenExtent(2) + 1:end)];
try
    decoded = jsondecode(normalized);
catch
    error("gesture_internal:recognition_response_validate:InvalidJson", ...
        "prediction line is not valid JSON.");
end
end

function [value, tokenExtent] = parseSequence(raw)
keyPattern = '"((?:\\.|[^"\\])*)"\s*:';
[~, ~, ~, ~, keyTokens] = regexp(raw, keyPattern);
sequenceKeyCount = 0;
literalSequenceKeyCount = 0;
for index = 1:numel(keyTokens)
    encodedKey = keyTokens{index}{1};
    try
        decodedKey = jsondecode(['"', encodedKey, '"']);
    catch
        continue;
    end
    if ischar(decodedKey) && strcmp(decodedKey, 'sequence')
        sequenceKeyCount = sequenceKeyCount + 1;
        literalSequenceKeyCount = literalSequenceKeyCount + ...
            double(strcmp(encodedKey, 'sequence'));
    end
end
if sequenceKeyCount ~= 1 || literalSequenceKeyCount ~= 1
    invalidSequenceOrJson(raw);
end

valuePattern = '"sequence"\s*:\s*([^,}\s]+)';
[~, ~, tokenExtents] = regexp(raw, valuePattern);
if numel(tokenExtents) ~= 1
    invalidSequenceOrJson(raw);
end
tokenExtent = tokenExtents{1}(1, :);
token = raw(tokenExtent(1):tokenExtent(2));
if isempty(regexp(token, '^(0|[1-9][0-9]*)$', 'once'))
    invalidResponse();
end
value = decimalUint64(token);
end

function value = decimalUint64(token)
value = uint64(0);
maximum = intmax("uint64");
ten = uint64(10);
for index = 1:numel(token)
    digit = uint64(double(token(index)) - double('0'));
    if value > idivide(maximum - digit, ten, "floor")
        invalidResponse();
    end
    value = value * ten + digit;
end
end

function invalidSequenceOrJson(raw)
try
    jsondecode(raw);
catch
    error("gesture_internal:recognition_response_validate:InvalidJson", ...
        "prediction line is not valid JSON.");
end
invalidResponse();
end

function valid = isExactProtocolVersion(value)
valid = isnumeric(value) && ~islogical(value) && isscalar(value) && ...
    isreal(value) && isfinite(value) && value == 1;
end

function valid = isTextMember(value, allowed)
valid = isTextScalar(value) && ismember(string(value), allowed);
end

function valid = isExactText(value, expected)
valid = isTextScalar(value) && string(value) == expected;
end

function valid = isTextScalar(value)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
end

function valid = isFiniteScalar(value)
valid = isnumeric(value) && ~islogical(value) && isscalar(value) && ...
    isreal(value) && isfinite(value);
end

function valid = isBoundedScalar(value, minimum, maximum)
valid = isFiniteScalar(value) && value >= minimum && value <= maximum;
end

function labels = recognitionClassLabels()
labels = ["REST", "WRIST_UP", "WRIST_DOWN", "FOREARM_IN", ...
    "FOREARM_OUT", "ARM_UP", "ARM_DOWN", "FIST"];
end

function invalidResponse()
error("gesture_internal:recognition_response_validate:InvalidResponse", ...
    "prediction record violates the exact protocol-v1 contract.");
end
