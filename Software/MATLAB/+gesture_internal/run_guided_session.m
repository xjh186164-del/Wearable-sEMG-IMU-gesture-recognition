function result = run_guided_session(config, captureFunction, ...
        viewFactory, qualityFunction)
%RUN_GUIDED_SESSION Compose a guided session around injected edge services.

protocol = gesture_protocol_create(config.Seed, config.BlockCount);
catalog = gesture_media_catalog(config.MediaDirectory, protocol.classLabels);
session = gesture_session_create(config, protocol);
view = viewFactory(catalog);
viewCleanup = onCleanup(@() view.close());
state = gesture_protocol_start(protocol);
finished = false;
stoppedByOperator = false;
stopEventPersisted = false;
hasSensorTime = false;
lastSensorTime = NaN;
lastSnapshot = [];

options = struct( ...
    "OutputDirectory", config.CaptureDirectory, ...
    "CaptureBase", config.SessionId, ...
    "OnTimeUpdate", @onSensorTime, ...
    "StopRequested", @stopRequested);
try
    [rawPath, filteredPath, imuPath, captureSummary] = ...
        captureFunction(config.DeviceIdentifier, Inf, options);
catch cause
    appendAcquisitionError(cause);
    persistAcquisitionErrorMetadata();
    rethrow(cause);
end

session.rawPath = string(rawPath);
session.filteredPath = string(filteredPath);
session.imuPath = string(imuPath);
if ~finished
    recordOperatorStop();
end
session.metadata.completed = logical(finished);
session.metadata.stopped_by_operator = logical(stoppedByOperator);
if finished
    session.metadata.terminal_status = "completed";
else
    session.metadata.terminal_status = "stopped_by_operator";
end
gesture_internal.write_session_metadata( ...
    session.metadataPath, session.metadata);
quality = qualityFunction(session, captureSummary, protocol);

result = struct( ...
    "session", session, ...
    "captureSummary", captureSummary, ...
    "quality", quality, ...
    "completed", logical(finished), ...
    "stoppedByOperator", logical(stoppedByOperator));

    function onSensorTime(sensorTime)
        if isnumeric(sensorTime) && isscalar(sensorTime) && ...
                isreal(sensorTime) && isfinite(sensorTime)
            hasSensorTime = true;
            lastSensorTime = double(sensorTime);
        end

        command = "none";
        if view.consumeInvalidateRequested()
            command = "invalidate_current";
        end
        [state, snapshot, events] = gesture_protocol_advance( ...
            state, sensorTime, command);
        if ~isempty(events)
            gesture_append_events(session.eventsPath, events);
        end
        view.update(snapshot);
        lastSnapshot = snapshot;
        finished = logical(snapshot.finished);
    end

    function requested = stopRequested()
        if finished
            requested = true;
            return;
        end

        requested = logical(view.stopRequested());
        if ~requested
            return;
        end

        stoppedByOperator = true;
        recordOperatorStop();
    end

    function recordOperatorStop()
        stoppedByOperator = true;
        if ~hasSensorTime || stopEventPersisted
            return;
        end
        [state, snapshot, events] = gesture_protocol_advance( ...
            state, lastSensorTime, "stop");
        if ~isempty(events)
            gesture_append_events(session.eventsPath, events);
        end
        stopEventPersisted = true;
        view.update(snapshot);
        lastSnapshot = snapshot;
    end

    function appendAcquisitionError(cause)
        if ~hasSensorTime || ~isfile(session.eventsPath)
            return;
        end

        blockId = 0;
        trialId = 0;
        gestureId = "";
        if ~isempty(lastSnapshot)
            blockId = double(lastSnapshot.block_id);
            trialId = double(lastSnapshot.trial_id);
            gestureId = string(lastSnapshot.gesture_id);
        end
        note = "identifier=" + string(cause.identifier) + ...
            "; message=" + string(cause.message);
        errorEvent = table(double(lastSensorTime), "acquisition_error", ...
            blockId, trialId, gestureId, false, note, ...
            'VariableNames', {'session_time_s', 'event_type', 'block_id', ...
             'trial_id', 'gesture_id', 'valid', 'note'});
        try
            gesture_append_events(session.eventsPath, errorEvent);
        catch
            % The capture exception is authoritative and must be preserved.
        end
    end

    function persistAcquisitionErrorMetadata()
        session.metadata.completed = false;
        session.metadata.stopped_by_operator = false;
        session.metadata.terminal_status = "acquisition_error";
        try
            gesture_internal.write_session_metadata( ...
                session.metadataPath, session.metadata);
        catch
            % The capture exception is authoritative and must be preserved.
        end
    end
end
