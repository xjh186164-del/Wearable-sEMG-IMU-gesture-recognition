function write_session_metadata(path, metadata)
%WRITE_SESSION_METADATA Atomically replace persisted session metadata JSON.

if ~((ischar(path) && isrow(path)) || ...
        (isstring(path) && isscalar(path) && ~ismissing(path))) || ...
        ~isstruct(metadata) || ~isscalar(metadata)
    error("gesture_internal:MetadataWriteFailed", ...
        "Metadata path and scalar structure are required.");
end
path = string(path);
directory = string(fileparts(path));
if strlength(strtrim(path)) == 0 || strlength(directory) == 0 || ...
        ~isfolder(directory)
    error("gesture_internal:MetadataWriteFailed", ...
        "Metadata output directory must exist.");
end

temporaryPath = string(tempname(directory)) + ".tmp";
cleanup = onCleanup(@() deleteIfPresent(temporaryPath));
try
    fileId = fopen(temporaryPath, "w", "n", "UTF-8");
    if fileId < 0
        error("gesture_internal:MetadataWriteFailed", ...
            "Could not create temporary metadata file.");
    end
    fileCleanup = onCleanup(@() closeFileSafely(fileId));
    fprintf(fileId, "%s\n", jsonencode(metadata, PrettyPrint=true));
    fclose(fileId);
    delete(fileCleanup);
    [moved, message] = movefile(temporaryPath, path, "f");
    if ~moved
        error("gesture_internal:MetadataWriteFailed", "%s", message);
    end
    delete(cleanup);
catch exception
    if exception.identifier == "gesture_internal:MetadataWriteFailed"
        rethrow(exception);
    end
    error("gesture_internal:MetadataWriteFailed", ...
        "Could not atomically write metadata JSON: %s", ...
        exception.message);
end
end

function deleteIfPresent(path)
if isfile(path)
    delete(path);
end
end

function closeFileSafely(fileId)
try
    fclose(fileId);
catch
end
end
