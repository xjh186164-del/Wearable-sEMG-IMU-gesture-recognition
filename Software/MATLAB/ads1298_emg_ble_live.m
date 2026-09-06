function [rawOutputFile, filteredOutputFile, imuOutputFile, summary] = ...
        ads1298_emg_ble_live( ...
        deviceIdentifier, durationSeconds, captureOptions)
%ADS1298_EMG_BLE_LIVE Display synchronized EMG and IMU received over BLE.
%   ADS1298_EMG_BLE_LIVE connects to "SensorBiShe-EMG" by default.
%   ADS1298_EMG_BLE_LIVE(DEVICE, SECONDS) accepts a BLE name or address.

if nargin < 1
    deviceIdentifier = "SensorBiShe-EMG";
end
if nargin < 2
    durationSeconds = Inf;
end
if nargin < 3
    captureOptions = struct();
end

deviceIdentifier = string(deviceIdentifier);
if ~isscalar(deviceIdentifier) || ...
        strlength(strtrim(deviceIdentifier)) == 0
    error("ads1298_emg_ble_live:InvalidDevice", ...
        "deviceIdentifier must be a nonempty BLE name or address.");
end
deviceIdentifier = strtrim(deviceIdentifier);
if ~isnumeric(durationSeconds) || ~isscalar(durationSeconds) || ...
        ~isreal(durationSeconds) || isnan(durationSeconds) || ...
        durationSeconds <= 0
    error("ads1298_emg_ble_live:InvalidDuration", ...
        "durationSeconds must be a positive scalar or Inf.");
end
durationSeconds = double(durationSeconds);

serviceUuid = "7b1e0001-6f9b-4c3d-8a6e-2f4a5b6c7d80";
dataCharacteristicUuid = "7b1e0002-6f9b-4c3d-8a6e-2f4a5b6c7d80";
sampleRateHz = 500;
channelCount = 8;
displayYLimitsMillivolts = [-0.2, 0.2];
filterState = ads1298_emg_filter_create(sampleRateHz, channelCount);
setupClock = tic;

bleConnection = [];
dataCharacteristic = [];

matlabDir = fileparts(mfilename("fullpath"));
projectRoot = fileparts(matlabDir);
defaultCaptureDir = fullfile(projectRoot, "captures");
timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
defaultCaptureBase = sprintf("ads1298_ble_live_%s", timestamp);
captureBaseSupplied = isstruct(captureOptions) && ...
    isfield(captureOptions, "CaptureBase");
options = ads1298_capture_options( ...
    captureOptions, defaultCaptureDir, defaultCaptureBase);
recognitionEnabled = ~isempty(options.OnSensorWindow);
captureDir = options.OutputDirectory;
if ~isfolder(captureDir)
    [created, message] = mkdir(captureDir);
    if ~created
        error("ads1298_emg_ble_live:OutputDirectoryFailed", ...
            "Could not create capture directory %s: %s", ...
            captureDir, message);
    end
end

captureBase = options.CaptureBase;
[rawOutputFile, filteredOutputFile, imuOutputFile] = ...
    outputPaths(captureDir, captureBase);
if captureBaseSupplied
    if isfile(rawOutputFile) || isfile(filteredOutputFile) || ...
            isfile(imuOutputFile)
        error("ads1298_emg_ble_live:OutputExists", ...
            "Capture output already exists for base %s in %s.", ...
            captureBase, captureDir);
    end
else
    suffix = 1;
    while isfile(rawOutputFile) || isfile(filteredOutputFile) || ...
            isfile(imuOutputFile)
        captureBase = sprintf( ...
            "ads1298_ble_live_%s_%02d", timestamp, suffix);
        [rawOutputFile, filteredOutputFile, imuOutputFile] = ...
            outputPaths(captureDir, captureBase);
        suffix = suffix + 1;
    end
end

[rawFileId, message] = fopen(char(rawOutputFile), "wt");
if rawFileId < 0
    error("ads1298_emg_ble_live:OutputFileFailed", ...
        "Could not create raw output file %s: %s", rawOutputFile, message);
end
rawFileCleanup = onCleanup(@() closeOutputFile(rawFileId));

[filteredFileId, message] = fopen(char(filteredOutputFile), "wt");
if filteredFileId < 0
    error("ads1298_emg_ble_live:OutputFileFailed", ...
        "Could not create filtered output file %s: %s", ...
        filteredOutputFile, message);
end
filteredFileCleanup = onCleanup(@() closeOutputFile(filteredFileId));

[imuFileId, message] = fopen(char(imuOutputFile), "wt");
if imuFileId < 0
    error("ads1298_emg_ble_live:OutputFileFailed", ...
        "Could not create IMU output file %s: %s", imuOutputFile, message);
end
imuFileCleanup = onCleanup(@() closeOutputFile(imuFileId));

fprintf(rawFileId, "%s\n", [ ...
    'sample,timestamp_us,status,ch1_raw,ch2_raw,ch3_raw,' ...
    'ch4_raw,ch5_raw,ch6_raw,ch7_raw,ch8_raw,dropped']);
fprintf(filteredFileId, "%s\n", [ ...
    'sample,timestamp_us,status,ch1_filtered_mV,ch2_filtered_mV,' ...
    'ch3_filtered_mV,ch4_filtered_mV,ch5_filtered_mV,' ...
    'ch6_filtered_mV,ch7_filtered_mV,ch8_filtered_mV,dropped']);
fprintf(imuFileId, "%s\n", ...
    'sample,timestamp_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_C,dropped');

figureHandle = figure( ...
    "Name", "ADS1298 BLE Live EMG - Filtered", ...
    "NumberTitle", "off", "Color", "white");
figureCleanup = onCleanup(@() closePlotFigure(figureHandle));
layout = tiledlayout(figureHandle, 4, 2, ...
    "TileSpacing", "compact", "Padding", "compact");
axesHandles = gobjects(channelCount, 1);
lineHandles = gobjects(channelCount, 1);
for channel = 1:channelCount
    axesHandles(channel) = nexttile(layout);
    lineHandles(channel) = plot(axesHandles(channel), NaN, NaN, ...
        "LineWidth", 0.8);
    grid(axesHandles(channel), "on");
    title(axesHandles(channel), sprintf("CH%d", channel));
    ylabel(axesHandles(channel), "mV");
    ylim(axesHandles(channel), displayYLimitsMillivolts);
end
xlabel(axesHandles(7), "Time (s)");
xlabel(axesHandles(8), "Time (s)");
linkaxes(axesHandles, "x");
filterDescription = ...
    "20-240 Hz + 48-52 Hz band-stop + 148-152 Hz band-stop";
statusTitle = sgtitle(layout, ...
    "Waiting for BLE EMG and IMU data... | Display: " + ...
    filterDescription);
recognitionTextPeriodSeconds = 0.2;
if recognitionEnabled
    recognitionGeometry = gesture_internal.recognition_status_layout();
    layout.OuterPosition = ...
        recognitionGeometry.emg_layout_outer_position;
    [recognitionSchedulerState, recognitionStatus, ...
        recognitionAcceptedRateHz] = ...
        gesture_internal.recognition_scheduler();
    [recognitionText, recognitionColor] = ...
        gesture_internal.recognition_status_text( ...
        recognitionStatus, recognitionAcceptedRateHz);
    recognitionTextHandle = annotation(figureHandle, "textbox", ...
        recognitionGeometry.status_textbox_position, ...
        "String", recognitionText, "Color", recognitionColor, ...
        "HorizontalAlignment", "center", ...
        "VerticalAlignment", "middle", "LineStyle", "none", ...
        "FontWeight", "bold", "FontSize", 11);
    recognitionTextClock = tic;
end

imuFigureHandle = figure( ...
    "Name", "LSM6DSOX Live IMU - BLE", ...
    "NumberTitle", "off", "Color", "white");
imuFigureCleanup = onCleanup(@() closePlotFigure(imuFigureHandle));
imuLayout = tiledlayout(imuFigureHandle, 2, 1, ...
    "TileSpacing", "compact", "Padding", "compact");
accelerationAxes = nexttile(imuLayout);
accelerationLines = plot(accelerationAxes, NaN, NaN, ...
    NaN, NaN, NaN, NaN, "LineWidth", 1.0);
grid(accelerationAxes, "on");
title(accelerationAxes, "Acceleration");
ylabel(accelerationAxes, "Acceleration (g)");
legend(accelerationAxes, ["X", "Y", "Z"], "Location", "northeast");

angularRateAxes = nexttile(imuLayout);
angularRateLines = plot(angularRateAxes, NaN, NaN, ...
    NaN, NaN, NaN, NaN, "LineWidth", 1.0);
grid(angularRateAxes, "on");
title(angularRateAxes, "Angular rate");
ylabel(angularRateAxes, "Angular rate (deg/s)");
xlabel(angularRateAxes, "Time (s)");
legend(angularRateAxes, ["X", "Y", "Z"], "Location", "northeast");
linkaxes([accelerationAxes, angularRateAxes], "x");
imuStatusTitle = sgtitle(imuLayout, "Waiting for BLE IMU data...");

millivoltsPerCount = 2.4 / (6 * 2^23) * 1000;
windowSeconds = 10;
plotPeriodSeconds = 1 / 2;
blePacketCount = 0;
bleReadErrors = 0;
lastBleReadError = "none";

historyTime = zeros(0, 1);
historyFilteredMillivolts = zeros(0, channelCount);
imuHistoryTime = zeros(0, 1);
imuHistoryAcceleration = zeros(0, 3);
imuHistoryAngularRate = zeros(0, 3);

receivedRows = 0;
receivedImuRows = 0;
invalidPackets = 0;
packetGaps = 0;
sequenceGaps = 0;
imuSequenceGaps = 0;
deviceDropped = 0;
imuDeviceDropped = 0;
deviceDroppedBaseline = [];
imuDeviceDroppedBaseline = [];
lastPacketSequence = [];
lastSequence = [];
lastImuSequence = [];
emgTimestampState = [];
imuTimestampState = [];
sessionOriginTimestamp = [];
rowsAtLastRateUpdate = 0;
imuRowsAtLastRateUpdate = 0;
emgHostQueueAges = zeros(0, 1);
imuHostQueueAges = zeros(0, 1);

drawnow;
fprintf("[BLE] Files and plots are ready (%.2f s).\n", toc(setupClock));

try
    bleConnection = ble(char(deviceIdentifier));
    dataCharacteristic = characteristic(bleConnection, ...
        char(serviceUuid), char(dataCharacteristicUuid));
catch cause
    closeBleConnection(dataCharacteristic, bleConnection);
    error("ads1298_emg_ble_live:ConnectionFailed", ...
        "Could not connect to BLE device %s: %s", ...
        deviceIdentifier, cause.message);
end
bleCleanup = onCleanup(@() ...
    closeBleConnection(dataCharacteristic, bleConnection));
fprintf("[BLE] Connected to %s (%.2f s).\n", ...
    deviceIdentifier, toc(setupClock));

try
    subscribe(dataCharacteristic, "notification");
    fprintf("[BLE] Notifications enabled (%.2f s).\n", toc(setupClock));
catch cause
    error("ads1298_emg_ble_live:SubscribeFailed", ...
        "Could not subscribe to BLE data from %s: %s", ...
        deviceIdentifier, cause.message);
end

fprintf("[BLE] Live capture is ready.\n");

startupTimeoutSeconds = 20;
startupClock = tic;
captureElapsedSeconds = 0;
plotClock = tic;
rateClock = tic;
notificationBatchSize = 16;

while isgraphics(figureHandle) && isgraphics(imuFigureHandle) && ...
        ~options.StopRequested() && ...
        ((receivedRows == 0 || receivedImuRows == 0) || ...
        captureElapsedSeconds < durationSeconds)
    pendingPackets = cell(notificationBatchSize, 1);
    pendingReceiptTimestamps = cell(notificationBatchSize, 1);
    packetsRead = 0;
    for notificationIndex = 1:notificationBatchSize
        if receivedRows > 0 && receivedImuRows > 0 && ...
                captureElapsedSeconds >= durationSeconds
            break;
        end
        try
            [notification, receiptTimestamp] = read( ...
                dataCharacteristic, "oldest");
            if ~isempty(notification)
                packetsRead = packetsRead + 1;
                blePacketCount = blePacketCount + 1;
                if blePacketCount == 1
                    fprintf("[BLE] First notification received (%d bytes).\n", ...
                        numel(notification));
                end
                pendingPackets{packetsRead} = uint8(notification);
                pendingReceiptTimestamps{packetsRead} = receiptTimestamp;
            end
        catch cause
            bleReadErrors = bleReadErrors + 1;
            lastBleReadError = string(cause.identifier) + ": " + ...
                string(cause.message);
            break;
        end
    end
    pendingPackets = pendingPackets(1:packetsRead);
    pendingReceiptTimestamps = pendingReceiptTimestamps(1:packetsRead);
    if ~isempty(pendingPackets)
        emgPacketValues = cell(0, 1);
        imuPacketValues = cell(0, 1);
        for packetIndex = 1:numel(pendingPackets)
            processingTimestamp = datetime("now");
            queueAgeSeconds = max(0, seconds(processingTimestamp - ...
                pendingReceiptTimestamps{packetIndex}));
            try
                [recordType, values, packetSequence] = ...
                    ads1298_ble_decode_packet(pendingPackets{packetIndex});
            catch cause
                if cause.identifier == "ads1298_ble_decode_packet:InvalidPacket"
                    invalidPackets = invalidPackets + 1;
                    continue;
                end
                rethrow(cause);
            end

            if ~isempty(lastPacketSequence)
                packetDelta = mod(packetSequence - lastPacketSequence, 2^16);
                if packetDelta > 1 && packetDelta < 2^15
                    packetGaps = packetGaps + packetDelta - 1;
                end
            end
            lastPacketSequence = packetSequence;

            if recordType == "emg"
                emgPacketValues{end + 1, 1} = values; %#ok<AGROW>
                emgHostQueueAges(end + 1, 1) = queueAgeSeconds; %#ok<AGROW>
            elseif recordType == "imu"
                imuPacketValues{end + 1, 1} = values; %#ok<AGROW>
                imuHostQueueAges(end + 1, 1) = queueAgeSeconds; %#ok<AGROW>
            end
        end

        if ~isempty(emgPacketValues)
            values = vertcat(emgPacketValues{:});
            if isempty(deviceDroppedBaseline)
                deviceDroppedBaseline = values(1, 12);
            end
            values(:, 12) = mod( ...
                values(:, 12) - deviceDroppedBaseline, 2^32);
            rowCount = size(values, 1);
            batchTime = zeros(rowCount, 1);
            for rowIndex = 1:rowCount
                currentSequence = values(rowIndex, 1);
                if ~isempty(lastSequence)
                    sequenceDelta = mod(currentSequence - lastSequence, 2^32);
                    if sequenceDelta > 1 && sequenceDelta < 2^31
                        sequenceGaps = sequenceGaps + sequenceDelta - 1;
                    end
                end
                lastSequence = currentSequence;
                [unwrappedTimestamp, emgTimestampState] = ...
                    ads1298_timestamp_unwrap( ...
                    values(rowIndex, 2), emgTimestampState);
                if isempty(sessionOriginTimestamp)
                    sessionOriginTimestamp = unwrappedTimestamp;
                end
                batchTime(rowIndex) = ...
                    (unwrappedTimestamp - sessionOriginTimestamp) / 1e6;
            end

            rawFormat = ...
                "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f," + ...
                "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f";
            rawLines = compose(rawFormat, ...
                values(:, 1), values(:, 2), values(:, 3), ...
                values(:, 4), values(:, 5), values(:, 6), ...
                values(:, 7), values(:, 8), values(:, 9), ...
                values(:, 10), values(:, 11), values(:, 12));
            fprintf(rawFileId, "%s\n", strjoin(rawLines, newline));

            rawMillivolts = values(:, 4:11) * millivoltsPerCount;
            [filteredMillivolts, filterState] = ...
                ads1298_emg_filter_step(rawMillivolts, filterState);
            fprintf(filteredFileId, "%s", ...
                ads1298_format_filtered_rows(values, filteredMillivolts));
            historyTime = [historyTime; batchTime]; %#ok<AGROW>
            historyFilteredMillivolts = [historyFilteredMillivolts; ...
                filteredMillivolts]; %#ok<AGROW>
            receivedRows = receivedRows + rowCount;
            deviceDropped = values(end, 12);
        end

        if ~isempty(imuPacketValues)
            values = vertcat(imuPacketValues{:});
            if isempty(imuDeviceDroppedBaseline)
                imuDeviceDroppedBaseline = values(1, 10);
            end
            values(:, 10) = mod( ...
                values(:, 10) - imuDeviceDroppedBaseline, 2^32);
            rowCount = size(values, 1);
            batchImuTime = zeros(rowCount, 1);
            for rowIndex = 1:rowCount
                currentSequence = values(rowIndex, 1);
                if ~isempty(lastImuSequence)
                    sequenceDelta = mod( ...
                        currentSequence - lastImuSequence, 2^32);
                    if sequenceDelta > 1 && sequenceDelta < 2^31
                        imuSequenceGaps = imuSequenceGaps + sequenceDelta - 1;
                    end
                end
                lastImuSequence = currentSequence;
                [unwrappedTimestamp, imuTimestampState] = ...
                    ads1298_timestamp_unwrap( ...
                    values(rowIndex, 2), imuTimestampState);
                if isempty(sessionOriginTimestamp)
                    sessionOriginTimestamp = unwrappedTimestamp;
                end
                batchImuTime(rowIndex) = ...
                    (unwrappedTimestamp - sessionOriginTimestamp) / 1e6;
            end

            convertedImu = ads1298_imu_convert(values);
            fprintf(imuFileId, "%s", ads1298_format_imu_rows(convertedImu));
            imuHistoryTime = [imuHistoryTime; batchImuTime]; %#ok<AGROW>
            imuHistoryAcceleration = [imuHistoryAcceleration; ...
                convertedImu(:, 3:5)]; %#ok<AGROW>
            imuHistoryAngularRate = [imuHistoryAngularRate; ...
                convertedImu(:, 6:8)]; %#ok<AGROW>
            receivedImuRows = receivedImuRows + rowCount;
            imuDeviceDropped = values(end, 10);
        end

        latestTime = gesture_internal.latest_shared_time( ...
            historyTime, imuHistoryTime);
        if ~isempty(latestTime) && latestTime > captureElapsedSeconds
            captureElapsedSeconds = latestTime;
            if ~isempty(options.OnTimeUpdate)
                options.OnTimeUpdate(captureElapsedSeconds);
            end
            keepEmg = historyTime >= latestTime - windowSeconds;
            historyTime = historyTime(keepEmg);
            historyFilteredMillivolts = ...
                historyFilteredMillivolts(keepEmg, :);
            keepImu = imuHistoryTime >= latestTime - windowSeconds;
            imuHistoryTime = imuHistoryTime(keepImu);
            imuHistoryAcceleration = imuHistoryAcceleration(keepImu, :);
            imuHistoryAngularRate = imuHistoryAngularRate(keepImu, :);
        end
        if recognitionEnabled && ~isempty(latestTime)
            [recognitionSchedulerState, recognitionStatus, ...
                recognitionAcceptedRateHz] = ...
                gesture_internal.recognition_scheduler( ...
                recognitionSchedulerState, latestTime, ...
                historyTime, historyFilteredMillivolts, ...
                imuHistoryTime, imuHistoryAcceleration, ...
                imuHistoryAngularRate, options.OnSensorWindow);
        end
    else
        pause(0.001);
    end

    if recognitionEnabled && ...
            toc(recognitionTextClock) >= recognitionTextPeriodSeconds && ...
            isgraphics(recognitionTextHandle)
        try
            [recognitionText, recognitionColor, formatValid] = ...
                gesture_internal.recognition_status_text( ...
                recognitionStatus, recognitionAcceptedRateHz);
            if ~formatValid
                recognitionSchedulerState.error = ...
                    recognitionSchedulerState.error + 1;
                recognitionStatus = unavailableRecognitionStatus();
                [recognitionText, recognitionColor] = ...
                    gesture_internal.recognition_status_text( ...
                    recognitionStatus, 0);
            end
        catch
            recognitionSchedulerState.error = ...
                recognitionSchedulerState.error + 1;
            recognitionStatus = unavailableRecognitionStatus();
            recognitionAcceptedRateHz = 0;
            recognitionText = ...
                "Recognition Status: RECOGNITION UNAVAILABLE";
            recognitionColor = [0.8, 0.1, 0.1];
        end
        recognitionTextHandle.String = recognitionText;
        recognitionTextHandle.Color = recognitionColor;
        drawnow limitrate;
        recognitionTextClock = tic;
    end

    if toc(plotClock) >= plotPeriodSeconds && ...
            isgraphics(figureHandle) && isgraphics(imuFigureHandle)
        rateElapsed = toc(rateClock);
        measuredRate = (receivedRows - rowsAtLastRateUpdate) / rateElapsed;
        measuredImuRate = ...
            (receivedImuRows - imuRowsAtLastRateUpdate) / rateElapsed;
        rowsAtLastRateUpdate = receivedRows;
        imuRowsAtLastRateUpdate = receivedImuRows;
        rateClock = tic;

        if ~isempty(historyTime)
            for channel = 1:channelCount
                set(lineHandles(channel), "XData", historyTime, ...
                    "YData", historyFilteredMillivolts(:, channel));
            end
        end
        if ~isempty(imuHistoryTime)
            for axisIndex = 1:3
                set(accelerationLines(axisIndex), ...
                    "XData", imuHistoryTime, ...
                    "YData", imuHistoryAcceleration(:, axisIndex));
                set(angularRateLines(axisIndex), ...
                    "XData", imuHistoryTime, ...
                    "YData", imuHistoryAngularRate(:, axisIndex));
            end
        end

        latestTime = gesture_internal.latest_shared_time( ...
            historyTime, imuHistoryTime);
        if ~isempty(latestTime)
            if latestTime < windowSeconds
                xLimits = [0, windowSeconds];
            else
                xLimits = [latestTime - windowSeconds, latestTime];
            end
            xlim(axesHandles(1), xLimits);
            xlim(accelerationAxes, xLimits);
        end

        statusTitle.String = sprintf([ ...
            'BLE EMG: %d rows, %.1f SPS | Sample gaps: %.0f | ' ...
            'Packet gaps: %.0f | Invalid: %d | Device drops: %.0f | ' ...
            'BLE packets: %.0f | %s'], ...
            receivedRows, measuredRate, sequenceGaps, packetGaps, ...
            invalidPackets, deviceDropped, blePacketCount, ...
            filterDescription);
        imuStatusTitle.String = sprintf([ ...
            'BLE IMU: %d rows, %.1f SPS | Sample gaps: %.0f | ' ...
            'Device drops: %.0f | BLE read errors: %.0f'], ...
            receivedImuRows, measuredImuRate, imuSequenceGaps, ...
            imuDeviceDropped, bleReadErrors);
        drawnow limitrate;
        plotClock = tic;
    end

    if (receivedRows == 0 || receivedImuRows == 0) && ...
            toc(startupClock) >= startupTimeoutSeconds
        noDataMessage = ...
            "Synchronized ADS1298 and LSM6DSOX BLE data did not " + ...
            "arrive from %s within %.0f seconds. BLE packets: " + ...
            "%.0f; BLE read errors: %.0f; Last BLE read error: %s";
        error("ads1298_emg_ble_live:NoData", ...
            noDataMessage, deviceIdentifier, startupTimeoutSeconds, ...
            blePacketCount, bleReadErrors, lastBleReadError);
    end
end

if isgraphics(figureHandle) || isgraphics(imuFigureHandle)
    drawnow;
end

emgHostQueueMetrics = ...
    gesture_internal.host_queue_metrics(emgHostQueueAges);
imuHostQueueMetrics = ...
    gesture_internal.host_queue_metrics(imuHostQueueAges);
summary = struct( ...
    "duration_s", captureElapsedSeconds, ...
    "emg_rows", receivedRows, ...
    "imu_rows", receivedImuRows, ...
    "emg_sequence_gaps", sequenceGaps, ...
    "imu_sequence_gaps", imuSequenceGaps, ...
    "packet_gaps", packetGaps, ...
    "invalid_packets", invalidPackets, ...
    "emg_dropped_delta", deviceDropped, ...
    "imu_dropped_delta", imuDeviceDropped, ...
    "ble_read_errors", bleReadErrors, ...
    "emg_host_queue_age_max_s", emgHostQueueMetrics.max_s, ...
    "emg_host_queue_age_p95_s", emgHostQueueMetrics.p95_s, ...
    "imu_host_queue_age_max_s", imuHostQueueMetrics.max_s, ...
    "imu_host_queue_age_p95_s", imuHostQueueMetrics.p95_s);
if recognitionEnabled
    summary.recognition_attempted = recognitionSchedulerState.attempted;
    summary.recognition_emitted = recognitionSchedulerState.emitted;
    summary.recognition_incomplete = recognitionSchedulerState.incomplete;
    summary.recognition_error = recognitionSchedulerState.error;
end

end

function [rawPath, filteredPath, imuPath] = outputPaths(directory, base)
rawPath = string(fullfile(directory, base + "_raw.csv"));
filteredPath = string(fullfile(directory, base + "_filtered.csv"));
imuPath = string(fullfile(directory, base + "_imu.csv"));
end

function closeBleConnection(dataCharacteristic, bleConnection)
try
    if ~isempty(dataCharacteristic)
        unsubscribe(dataCharacteristic);
    end
catch
end
try
    if ~isempty(bleConnection) && isvalid(bleConnection)
        delete(bleConnection);
    end
catch
end
end

function closeOutputFile(fileId)
try
    fclose(fileId);
catch
end
end

function closePlotFigure(figureHandle)
try
    if isgraphics(figureHandle)
        delete(figureHandle);
    end
catch
end
end

function status = unavailableRecognitionStatus()
status = struct( ...
    "available", false, ...
    "display_state", "RECOGNITION_UNAVAILABLE", ...
    "candidate_gesture_id", "", ...
    "confidence", NaN, ...
    "probabilities", NaN(1, 8), ...
    "inference_ms", NaN, ...
    "round_trip_ms", NaN, ...
    "sequence", uint64(0), ...
    "sensor_time_s", NaN);
end
