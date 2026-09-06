function [renderedGestureId, clipTime] = guidance_video_route( ...
        stageName, gestureId, stageElapsedSeconds)
if ~(ischar(stageName) || (isstring(stageName) && isscalar(stageName))) || ...
        ~(ischar(gestureId) || (isstring(gestureId) && isscalar(gestureId))) || ...
        ~isnumeric(stageElapsedSeconds) || ~isscalar(stageElapsedSeconds) || ...
        ~isreal(stageElapsedSeconds) || ~isfinite(stageElapsedSeconds) || ...
        stageElapsedSeconds < 0
    error("gesture_internal:InvalidGuidanceRoute", ...
        "Stage, gesture ID, and elapsed time must be valid scalars.");
end

stageName = string(stageName);
gestureId = string(gestureId);
targetStages = ["prepare", "movement", "target_hold", "return"];
restStages = ["baseline", "recovery", "break", "finished", "stopped"];
if ismember(stageName, targetStages)
    if ismissing(gestureId) || strlength(gestureId) == 0
        error("gesture_internal:InvalidGuidanceRoute", ...
            "Target stages require a gesture ID.");
    end
    offsets = dictionary(targetStages, [0, 1, 2.5, 6.5]);
    renderedGestureId = gestureId;
    clipTime = offsets(stageName) + double(stageElapsedSeconds);
elseif ismember(stageName, restStages)
    renderedGestureId = "REST";
    clipTime = double(stageElapsedSeconds);
else
    error("gesture_internal:InvalidGuidanceRoute", ...
        "Unknown guidance stage: %s", stageName);
end
end
