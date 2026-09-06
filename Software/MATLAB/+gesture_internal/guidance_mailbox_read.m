function [payload, valid] = guidance_mailbox_read(path)
%GUIDANCE_MAILBOX_READ Read a complete scalar-struct JSON mailbox if present.

path = validatedPath(path);
payload = struct();
valid = false;
if ~isfile(path)
    return;
end

try
    candidate = jsondecode(fileread(path));
catch
    return;
end
if ~isstruct(candidate) || ~isscalar(candidate)
    return;
end
payload = candidate;
valid = true;
end

function path = validatedPath(value)
validChar = ischar(value) && isrow(value);
validString = isstring(value) && isscalar(value) && ~ismissing(value);
if ~(validChar || validString)
    error("gesture_internal:guidance_mailbox_read:InvalidPath", ...
        "path must be a nonempty text scalar.");
end
path = strtrim(string(value));
if strlength(path) == 0
    error("gesture_internal:guidance_mailbox_read:InvalidPath", ...
        "path must be a nonempty text scalar.");
end
end
