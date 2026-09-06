function [state, snapshot, events] = gesture_protocol_advance( ...
        state, sessionTimeSeconds, command)
if ~isnumeric(sessionTimeSeconds) || ~isscalar(sessionTimeSeconds) || ...
        ~isreal(sessionTimeSeconds) || ~isfinite(sessionTimeSeconds)
    error("gesture_protocol_advance:InvalidTime", ...
        "sessionTimeSeconds must be a finite real numeric scalar.");
end
if sessionTimeSeconds < state.lastTime
    error("gesture_protocol_advance:NonmonotonicTime", ...
        "sessionTimeSeconds must not move backwards.");
end
if ~(ischar(command) || (isstring(command) && isscalar(command)))
    error("gesture_protocol_advance:InvalidCommand", ...
        "command must be none, invalidate_current, or stop.");
end
command = string(command);
if ~ismember(command, ["none", "invalidate_current", "stop"])
    error("gesture_protocol_advance:InvalidCommand", ...
        "command must be none, invalidate_current, or stop.");
end

events = emptyEvents();
state.lastTime = double(sessionTimeSeconds);
if state.finished || state.stopped
    snapshot = makeSnapshot(state, sessionTimeSeconds);
    return;
end

if ~state.started
    state.started = true;
    [blockId, trialId, gestureId, valid] = currentTrialContext(state);
    events = appendEvent(events, state.stageStartTime, "session_start", ...
        blockId, trialId, gestureId, valid, "");
    events = appendEvent(events, state.stageStartTime, ...
        entryEvent(state.currentStageIndex), blockId, trialId, ...
        gestureId, valid, "");
end

while ~state.finished && ~state.stopped
    if state.currentStageIndex == 0
        boundaryTime = state.stageStartTime + ...
            state.protocol.breakDurationSeconds;
        if sessionTimeSeconds < boundaryTime
            break;
        end

        events = appendEvent(events, boundaryTime, "break_end", ...
            state.lastBreakBlock, 0, "", true, "");
        state.stageStartTime = boundaryTime;
        state.currentStageIndex = 1;
        [blockId, trialId, gestureId, valid] = currentTrialContext(state);
        events = appendEvent(events, boundaryTime, ...
            entryEvent(state.currentStageIndex), blockId, trialId, ...
            gestureId, valid, "");
        continue;
    end

    boundaryTime = state.stageStartTime + ...
        state.protocol.stageDurationsSeconds(state.currentStageIndex);
    if sessionTimeSeconds < boundaryTime
        break;
    end

    [blockId, trialId, gestureId, valid] = currentTrialContext(state);
    exitName = exitEvent(state.currentStageIndex);
    if exitName ~= ""
        events = appendEvent(events, boundaryTime, exitName, blockId, ...
            trialId, gestureId, valid, "");
    end
    state.stageStartTime = boundaryTime;

    if state.currentStageIndex < numel(state.protocol.stageNames)
        state.currentStageIndex = state.currentStageIndex + 1;
        entryName = entryEvent(state.currentStageIndex);
        if entryName ~= ""
            events = appendEvent(events, boundaryTime, entryName, ...
                blockId, trialId, gestureId, valid, "");
        end
        continue;
    end

    completedIndex = state.currentTrialIndex;
    completedTrial = state.trialQueue(completedIndex, :);
    state.currentTrialIndex = completedIndex + 1;
    if shouldStartBreak(state, completedIndex, completedTrial)
        state.lastBreakBlock = completedTrial.block_id;
        state.currentStageIndex = 0;
        events = appendEvent(events, boundaryTime, "break_start", ...
            completedTrial.block_id, 0, "", true, "");
    elseif state.currentTrialIndex <= height(state.trialQueue)
        state.currentStageIndex = 1;
        [blockId, trialId, gestureId, valid] = currentTrialContext(state);
        events = appendEvent(events, boundaryTime, ...
            entryEvent(state.currentStageIndex), blockId, trialId, ...
            gestureId, valid, "");
    else
        state.finished = true;
        events = appendEvent(events, boundaryTime, "session_end", ...
            0, 0, "", true, "");
    end
end

if command == "invalidate_current" && ~state.finished
    [state, invalidEvent] = invalidateCurrentTrial(state, ...
        sessionTimeSeconds);
    events = [events; invalidEvent];
end

if command == "stop" && ~state.finished
    [blockId, trialId, gestureId, valid] = currentTrialContext(state);
    events = appendEvent(events, sessionTimeSeconds, "session_stopped", ...
        blockId, trialId, gestureId, valid, "");
    state.stopped = true;
end

snapshot = makeSnapshot(state, sessionTimeSeconds);
end

function [state, event] = invalidateCurrentTrial(state, sessionTimeSeconds)
event = emptyEvents();
if state.currentStageIndex == 0 || ...
        state.currentTrialIndex > height(state.trialQueue)
    return;
end

trial = state.trialQueue(state.currentTrialIndex, :);
if ismember(trial.trial_id, state.invalidTrialIds)
    return;
end

state.invalidTrialIds(end + 1, 1) = trial.trial_id;
replacement = trial;
replacement.trial_id = max(state.trialQueue.trial_id) + 1;
replacement.is_repeat = true;
state.trialQueue = [state.trialQueue; replacement];
note = "replacement_trial_id=" + string(replacement.trial_id);
event = appendEvent(event, sessionTimeSeconds, "trial_invalid", ...
    trial.block_id, trial.trial_id, trial.gesture_id, false, note);
end

function tf = shouldStartBreak(state, completedIndex, completedTrial)
if completedTrial.is_repeat || ...
        ~ismember(completedTrial.block_id, state.protocol.breakAfterBlocks) || ...
        state.lastBreakBlock == completedTrial.block_id
    tf = false;
    return;
end

later = state.trialQueue(completedIndex + 1:end, :);
laterOriginalInBlock = ~later.is_repeat & ...
    later.block_id == completedTrial.block_id;
tf = ~any(laterOriginalInBlock);
end

function snapshot = makeSnapshot(state, sessionTimeSeconds)
if state.stopped
    stageName = "stopped";
    gestureId = "";
    trialId = 0;
    blockId = 0;
    elapsed = 0;
    remaining = 0;
    progressText = "Stopped";
elseif state.finished
    stageName = "finished";
    gestureId = "";
    trialId = 0;
    blockId = 0;
    elapsed = 0;
    remaining = 0;
    progressText = "Finished";
elseif state.currentStageIndex == 0
    stageName = "break";
    gestureId = "";
    trialId = 0;
    blockId = state.lastBreakBlock;
    elapsed = sessionTimeSeconds - state.stageStartTime;
    remaining = state.protocol.breakDurationSeconds - elapsed;
    progressText = "Break after block " + string(blockId);
else
    trial = state.trialQueue(state.currentTrialIndex, :);
    stageName = state.protocol.stageNames(state.currentStageIndex);
    gestureId = trial.gesture_id;
    trialId = trial.trial_id;
    blockId = trial.block_id;
    elapsed = sessionTimeSeconds - state.stageStartTime;
    remaining = state.protocol.stageDurationsSeconds( ...
        state.currentStageIndex) - elapsed;
    progressText = "Trial " + string(state.currentTrialIndex) + ...
        "/" + string(height(state.trialQueue));
end

snapshot = struct( ...
    "finished", state.finished, ...
    "stopped", state.stopped, ...
    "stage_name", stageName, ...
    "gesture_id", gestureId, ...
    "trial_id", double(trialId), ...
    "block_id", double(blockId), ...
    "stage_elapsed_s", double(elapsed), ...
    "stage_remaining_s", double(remaining), ...
    "progress_text", progressText);
end

function [blockId, trialId, gestureId, valid] = currentTrialContext(state)
if state.currentTrialIndex > height(state.trialQueue)
    blockId = 0;
    trialId = 0;
    gestureId = "";
    valid = true;
    return;
end

trial = state.trialQueue(state.currentTrialIndex, :);
blockId = trial.block_id;
trialId = trial.trial_id;
gestureId = trial.gesture_id;
valid = ~ismember(trialId, state.invalidTrialIds);
end

function name = entryEvent(stageIndex)
entryEvents = ["rest_valid_start", "prepare", "movement_start", ...
    "target_hold_start", "return_start", "rest_valid_start"];
name = entryEvents(stageIndex);
end

function name = exitEvent(stageIndex)
exitEvents = ["rest_valid_end", "", "", ...
    "target_hold_end", "return_end", "rest_valid_end"];
name = exitEvents(stageIndex);
end

function events = appendEvent(events, sessionTimeSeconds, eventType, ...
        blockId, trialId, gestureId, valid, note)
row = table(double(sessionTimeSeconds), string(eventType), ...
    double(blockId), double(trialId), string(gestureId), logical(valid), ...
    string(note), 'VariableNames', events.Properties.VariableNames);
events = [events; row];
end

function events = emptyEvents()
events = table(zeros(0,1), strings(0,1), zeros(0,1), zeros(0,1), ...
    strings(0,1), false(0,1), strings(0,1), 'VariableNames', ...
    {'session_time_s','event_type','block_id','trial_id', ...
     'gesture_id','valid','note'});
end
