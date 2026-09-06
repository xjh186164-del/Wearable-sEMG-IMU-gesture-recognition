function gesture_guidance_worker(ipcDirectory, mediaDirectory, visible)
%GESTURE_GUIDANCE_WORKER Own guidance video decode in a child MATLAB process.

if nargin < 3
    visible = true;
end
protocol = gesture_protocol_create(0, 2);
catalog = gesture_media_catalog(mediaDirectory, protocol.classLabels);
view = GestureGuidanceView(catalog, visible);
gesture_internal.run_guidance_worker(ipcDirectory, view);
end
