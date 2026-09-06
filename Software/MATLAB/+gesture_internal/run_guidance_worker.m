function run_guidance_worker(ipcDirectory, view, options)
%RUN_GUIDANCE_WORKER Exchange guidance snapshots and operator commands.

if nargin < 3
    options = struct();
end
ipcDirectory = validateDirectory(ipcDirectory);
options = normalizeOptions(options);
statePath = fullfile(ipcDirectory, "state.json");
commandPath = fullfile(ipcDirectory, "command.json");
readyPath = fullfile(ipcDirectory, "ready.json");
viewCleanup = onCleanup(@() view.close());

gesture_internal.guidance_mailbox_write(readyPath, struct("ready", true));
lastSequence = 0;
invalidateGeneration = 0;
stopRequested = false;
lastPublishedGeneration = 0;
lastPublishedStop = false;
iteration = 0;

while iteration < options.MaxIterations
    iteration = iteration + 1;
    [state, valid] = gesture_internal.guidance_mailbox_read(statePath);
    if valid
        [state, valid] = validatedState(state);
    end
    if valid && state.sequence > lastSequence
        lastSequence = state.sequence;
        if state.close_requested
            break;
        end
        view.update(state.snapshot);
    end

    if view.consumeInvalidateRequested()
        invalidateGeneration = invalidateGeneration + 1;
    end
    stopRequested = stopRequested || logical(view.stopRequested());
    if invalidateGeneration ~= lastPublishedGeneration || ...
            stopRequested ~= lastPublishedStop
        gesture_internal.guidance_mailbox_write(commandPath, struct( ...
            "invalidate_generation", invalidateGeneration, ...
            "stop_requested", stopRequested));
        lastPublishedGeneration = invalidateGeneration;
        lastPublishedStop = stopRequested;
    end

    if iteration < options.MaxIterations
        options.PauseFunction(options.PollPeriodSeconds);
    end
end
end

function [state, valid] = validatedState(candidate)
required = ["sequence", "gesture_id", "stage_name", ...
    "stage_elapsed_s", "stage_remaining_s", "block_id", ...
    "progress_text", "finished", "stopped", "close_requested"];
valid = isstruct(candidate) && isscalar(candidate) && ...
    all(isfield(candidate, required));
state = struct();
if ~valid
    return;
end

sequence = candidate.sequence;
numericValues = [candidate.stage_elapsed_s, ...
    candidate.stage_remaining_s, candidate.block_id];
valid = isnumeric(sequence) && isscalar(sequence) && isreal(sequence) && ...
    isfinite(sequence) && sequence >= 0 && sequence == fix(sequence) && ...
    isnumeric(numericValues) && isreal(numericValues) && ...
    all(isfinite(numericValues)) && all(numericValues >= 0) && ...
    islogical(candidate.finished) && isscalar(candidate.finished) && ...
    islogical(candidate.stopped) && isscalar(candidate.stopped) && ...
    islogical(candidate.close_requested) && ...
    isscalar(candidate.close_requested);
if ~valid
    return;
end
gestureId = string(candidate.gesture_id);
stageName = string(candidate.stage_name);
progressText = string(candidate.progress_text);
valid = isscalar(gestureId) && ~ismissing(gestureId) && ...
    isscalar(stageName) && ~ismissing(stageName) && ...
    isscalar(progressText) && ~ismissing(progressText);
if ~valid
    return;
end
state.sequence = double(sequence);
state.close_requested = logical(candidate.close_requested);
state.snapshot = struct( ...
    "gesture_id", gestureId, ...
    "stage_name", stageName, ...
    "stage_elapsed_s", double(candidate.stage_elapsed_s), ...
    "stage_remaining_s", double(candidate.stage_remaining_s), ...
    "block_id", double(candidate.block_id), ...
    "progress_text", progressText, ...
    "finished", logical(candidate.finished), ...
    "stopped", logical(candidate.stopped));
end

function options = normalizeOptions(options)
if ~isstruct(options) || ~isscalar(options)
    error("gesture_internal:run_guidance_worker:InvalidOptions", ...
        "options must be a scalar struct.");
end
defaults = struct("PauseFunction", @pause, ...
    "PollPeriodSeconds", 0.05, "MaxIterations", Inf);
names = string(fieldnames(options));
if ~all(ismember(names, string(fieldnames(defaults))))
    error("gesture_internal:run_guidance_worker:InvalidOptions", ...
        "options contains an unknown field.");
end
for name = string(fieldnames(defaults)).'
    if ~isfield(options, name)
        options.(name) = defaults.(name);
    end
end
if ~isa(options.PauseFunction, "function_handle") || ...
        ~isnumeric(options.PollPeriodSeconds) || ...
        ~isscalar(options.PollPeriodSeconds) || ...
        ~isreal(options.PollPeriodSeconds) || ...
        ~isfinite(options.PollPeriodSeconds) || ...
        options.PollPeriodSeconds <= 0 || ...
        ~isnumeric(options.MaxIterations) || ...
        ~isscalar(options.MaxIterations) || ...
        ~isreal(options.MaxIterations) || ...
        isnan(options.MaxIterations) || options.MaxIterations <= 0 || ...
        (~isinf(options.MaxIterations) && ...
        options.MaxIterations ~= fix(options.MaxIterations))
    error("gesture_internal:run_guidance_worker:InvalidOptions", ...
        "Worker options are malformed.");
end
options.PollPeriodSeconds = double(options.PollPeriodSeconds);
options.MaxIterations = double(options.MaxIterations);
end

function directory = validateDirectory(value)
validText = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~validText
    invalidDirectory();
end
directory = strtrim(string(value));
if strlength(directory) == 0 || ~isfolder(directory)
    invalidDirectory();
end
end

function invalidDirectory()
error("gesture_internal:run_guidance_worker:InvalidDirectory", ...
    "ipcDirectory must be an existing directory.");
end
