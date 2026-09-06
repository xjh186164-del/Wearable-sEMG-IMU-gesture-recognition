function guidance_mailbox_write(path, payload)
%GUIDANCE_MAILBOX_WRITE Atomically replace a UTF-8 JSON mailbox.

path = validatedPath(path);
if ~isstruct(payload) || ~isscalar(payload)
    error("gesture_internal:guidance_mailbox_write:InvalidPayload", ...
        "payload must be a scalar struct.");
end

[parent, name, extension] = fileparts(path);
if strlength(parent) == 0
    parent = string(pwd);
end
if ~isfolder(parent)
    error("gesture_internal:guidance_mailbox_write:InvalidPath", ...
        "The mailbox parent directory must exist.");
end
[~, uniqueToken] = fileparts(tempname(parent));
temporaryPath = fullfile(parent, "." + name + extension + "." + ...
    string(uniqueToken) + ".tmp");
temporaryCleanup = onCleanup(@() deleteExactFile(temporaryPath));

bytes = unicode2native(jsonencode(payload), "UTF-8");
[fileId, message] = fopen(temporaryPath, "wb");
if fileId < 0
    error("gesture_internal:guidance_mailbox_write:WriteFailed", ...
        "Could not create mailbox temporary file: %s", message);
end
fileCleanup = onCleanup(@() closeFile(fileId));
written = fwrite(fileId, bytes, "uint8");
if written ~= numel(bytes)
    error("gesture_internal:guidance_mailbox_write:WriteFailed", ...
        "Could not write the complete mailbox payload.");
end
fclose(fileId);
fileCleanup = []; %#ok<NASGU>

maximumPublishAttempts = 51;
retryDelaySeconds = 0.01;
moved = false;
message = "";
for attempt = 1:maximumPublishAttempts
    [moved, message] = movefile(temporaryPath, path, "f");
    if moved
        break;
    end
    if attempt < maximumPublishAttempts
        pause(retryDelaySeconds);
    end
end
if ~moved
    error("gesture_internal:guidance_mailbox_write:PublishFailed", ...
        "Could not publish mailbox: %s", message);
end
temporaryCleanup = []; %#ok<NASGU>
end

function path = validatedPath(value)
validChar = ischar(value) && isrow(value);
validString = isstring(value) && isscalar(value) && ~ismissing(value);
if ~(validChar || validString)
    error("gesture_internal:guidance_mailbox_write:InvalidPath", ...
        "path must be a nonempty text scalar.");
end
path = strtrim(string(value));
if strlength(path) == 0
    error("gesture_internal:guidance_mailbox_write:InvalidPath", ...
        "path must be a nonempty text scalar.");
end
end

function closeFile(fileId)
try
    fclose(fileId);
catch
end
end

function deleteExactFile(path)
try
    if isfile(path)
        delete(path);
    end
catch
end
end
