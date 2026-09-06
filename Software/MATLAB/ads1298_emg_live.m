function [rawOutputFile, filteredOutputFile, imuOutputFile] = ...
        ads1298_emg_live(portName, durationSeconds)
%ADS1298_EMG_LIVE Display synchronized filtered EMG and six-axis IMU data.
%   RAWFILE = ADS1298_EMG_LIVE("COM5") returns the untouched ADS1298 CSV.
%   [RAWFILE, FILTEREDFILE, IMUFILE] = ADS1298_EMG_LIVE("COM5", 5)
%   also returns aligned filtered-EMG and physical-unit IMU CSV files.

if nargin < 1
    portName = "COM5";
end
if nargin < 2
    durationSeconds = Inf;
end

portName = string(portName);
if ~isscalar(portName) || strlength(strtrim(portName)) == 0
    error("ads1298_emg_live:InvalidPort", ...
        "portName must be a nonempty character vector or string scalar.");
end
portName = strtrim(portName);

if ~isnumeric(durationSeconds) || ~isscalar(durationSeconds) || ...
        ~isreal(durationSeconds) || isnan(durationSeconds) || durationSeconds <= 0
    error("ads1298_emg_live:InvalidDuration", ...
        "durationSeconds must be a positive scalar or Inf.");
end
durationSeconds = double(durationSeconds);

sampleRateHz = 500;
channelCount = 8;
displayYLimitsMillivolts = [-0.2, 0.2];
filterState = ads1298_emg_filter_create(sampleRateHz, channelCount);

try
    serialConnection = serialport(char(portName), 115200, "Timeout", 1);
catch cause
    error("ads1298_emg_live:PortOpenFailed", ...
        "Could not open serial port %s: %s", portName, cause.message);
end
serialCleanup = onCleanup(@() closeSerialPort(serialConnection));

matlabDir = fileparts(mfilename("fullpath"));
projectRoot = fileparts(matlabDir);
captureDir = fullfile(projectRoot, "captures");
if ~isfolder(captureDir)
    [created, message] = mkdir(captureDir);
    if ~created
        error("ads1298_emg_live:OutputDirectoryFailed", ...
            "Could not create capture directory %s: %s", captureDir, message);
    end
end

timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
captureBase = sprintf("ads1298_live_%s", timestamp);
[rawOutputFile, filteredOutputFile, imuOutputFile] = ...
    outputPaths(captureDir, captureBase);
suffix = 1;
while isfile(rawOutputFile) || isfile(filteredOutputFile) || ...
        isfile(imuOutputFile)
    captureBase = sprintf("ads1298_live_%s_%02d", timestamp, suffix);
    [rawOutputFile, filteredOutputFile, imuOutputFile] = ...
        outputPaths(captureDir, captureBase);
    suffix = suffix + 1;
end

[rawFileId, message] = fopen(char(rawOutputFile), "wt");
if rawFileId < 0
    error("ads1298_emg_live:OutputFileFailed", ...
        "Could not create raw output file %s: %s", rawOutputFile, message);
end
rawFileCleanup = onCleanup(@() closeOutputFile(rawFileId));

[filteredFileId, message] = fopen(char(filteredOutputFile), "wt");
if filteredFileId < 0
    error("ads1298_emg_live:OutputFileFailed", ...
        "Could not create filtered output file %s: %s", ...
        filteredOutputFile, message);
end
filteredFileCleanup = onCleanup(@() closeOutputFile(filteredFileId));

[imuFileId, message] = fopen(char(imuOutputFile), "wt");
if imuFileId < 0
    error("ads1298_emg_live:OutputFileFailed", ...
        "Could not create IMU output file %s: %s", imuOutputFile, message);
end
imuFileCleanup = onCleanup(@() closeOutputFile(imuFileId));

rawCsvHeader = ['sample,timestamp_us,status,ch1_raw,ch2_raw,ch3_raw,' ...
    'ch4_raw,ch5_raw,ch6_raw,ch7_raw,ch8_raw,dropped'];
filteredCsvHeader = [ ...
    'sample,timestamp_us,status,ch1_filtered_mV,ch2_filtered_mV,' ...
    'ch3_filtered_mV,ch4_filtered_mV,ch5_filtered_mV,ch6_filtered_mV,' ...
    'ch7_filtered_mV,ch8_filtered_mV,dropped'];
imuCsvHeader = ...
    'sample,timestamp_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_C,dropped';
fprintf(rawFileId, "%s\n", rawCsvHeader);
fprintf(filteredFileId, "%s\n", filteredCsvHeader);
fprintf(imuFileId, "%s\n", imuCsvHeader);

figureHandle = figure( ...
    "Name", "ADS1298 Live EMG - Filtered", ...
    "NumberTitle", "off", ...
    "Color", "white");
figureCleanup = onCleanup(@() closePlotFigure(figureHandle));
layout = tiledlayout(figureHandle, 4, 2, ...
    "TileSpacing", "compact", "Padding", "compact");
axesHandles = gobjects(8, 1);
lineHandles = gobjects(8, 1);
for channel = 1:8
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
    "Waiting for synchronized EMG and IMU data... | Display: " + ...
    filterDescription);

imuFigureHandle = figure( ...
    "Name", "LSM6DSOX Live IMU", ...
    "NumberTitle", "off", ...
    "Color", "white");
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
imuStatusTitle = sgtitle(imuLayout, "Waiting for LSM6DSOX data...");

millivoltsPerCount = 2.4 / (6 * 2^23) * 1000;
windowSeconds = 10;
plotPeriodSeconds = 1 / 20;
newlineCharacter = newline;

receiveBuffer = "";
historyTime = zeros(0, 1);
historyFilteredMillivolts = zeros(0, channelCount);
imuHistoryTime = zeros(0, 1);
imuHistoryAcceleration = zeros(0, 3);
imuHistoryAngularRate = zeros(0, 3);

receivedRows = 0;
receivedImuRows = 0;
invalidRows = 0;
sequenceGaps = 0;
imuSequenceGaps = 0;
deviceDropped = 0;
imuDeviceDropped = 0;
lastSequence = [];
lastImuSequence = [];
emgTimestampState = [];
imuTimestampState = [];
sessionOriginTimestamp = [];
rowsAtLastRateUpdate = 0;
imuRowsAtLastRateUpdate = 0;

flush(serialConnection, "input");
startupTimeoutSeconds = 15;
startupClock = tic;
captureClock = [];
plotClock = tic;
rateClock = tic;

while isgraphics(figureHandle) && isgraphics(imuFigureHandle) && ...
        (isempty(captureClock) || toc(captureClock) < durationSeconds)
    availableBytes = serialConnection.NumBytesAvailable;
    if availableBytes > 0
        incomingBytes = read(serialConnection, availableBytes, "uint8");
        receiveBuffer = receiveBuffer + string(char(incomingBytes(:).'));

        lastNewline = find(char(receiveBuffer) == newlineCharacter, 1, "last");
        if ~isempty(lastNewline)
            bufferCharacters = char(receiveBuffer);
            completeText = bufferCharacters(1:lastNewline);
            receiveBuffer = string(bufferCharacters(lastNewline + 1:end));
            completeLines = regexp(completeText, "\r?\n", "split");
            if isempty(completeLines{end})
                completeLines(end) = [];
            end

            maximumRows = numel(completeLines);
            batchTime = zeros(maximumRows, 1);
            batchValues = zeros(maximumRows, 12);
            outputLines = cell(maximumRows, 1);
            batchImuTime = zeros(maximumRows, 1);
            batchImuRaw = zeros(maximumRows, 10);
            validInBatch = 0;
            imuValidInBatch = 0;

            for lineIndex = 1:maximumRows
                originalLine = completeLines{lineIndex};
                [recordType, values] = ads1298_parse_line(originalLine);
                if recordType == "invalid"
                    invalidRows = invalidRows + 1;
                    continue;
                elseif recordType == "valid"
                    validInBatch = validInBatch + 1;
                    receivedRows = receivedRows + 1;
                    outputLines{validInBatch} = originalLine;
                    batchValues(validInBatch, :) = values;

                    currentSequence = values(1);
                    if ~isempty(lastSequence)
                        sequenceDelta = mod(currentSequence - lastSequence, 2^32);
                        if sequenceDelta > 1 && sequenceDelta < 2^31
                            sequenceGaps = sequenceGaps + sequenceDelta - 1;
                        end
                    end
                    lastSequence = currentSequence;

                    [unwrappedTimestamp, emgTimestampState] = ...
                        ads1298_timestamp_unwrap(values(2), emgTimestampState);
                    if isempty(sessionOriginTimestamp)
                        sessionOriginTimestamp = unwrappedTimestamp;
                    end
                    batchTime(validInBatch) = ...
                        (unwrappedTimestamp - sessionOriginTimestamp) / 1e6;
                    deviceDropped = values(12);
                elseif recordType == "imu"
                    imuValidInBatch = imuValidInBatch + 1;
                    receivedImuRows = receivedImuRows + 1;
                    batchImuRaw(imuValidInBatch, :) = values;

                    currentSequence = values(1);
                    if ~isempty(lastImuSequence)
                        sequenceDelta = mod( ...
                            currentSequence - lastImuSequence, 2^32);
                        if sequenceDelta > 1 && sequenceDelta < 2^31
                            imuSequenceGaps = ...
                                imuSequenceGaps + sequenceDelta - 1;
                        end
                    end
                    lastImuSequence = currentSequence;

                    [unwrappedTimestamp, imuTimestampState] = ...
                        ads1298_timestamp_unwrap(values(2), imuTimestampState);
                    if isempty(sessionOriginTimestamp)
                        sessionOriginTimestamp = unwrappedTimestamp;
                    end
                    batchImuTime(imuValidInBatch) = ...
                        (unwrappedTimestamp - sessionOriginTimestamp) / 1e6;
                    imuDeviceDropped = values(10);
                end
            end

            if validInBatch > 0
                validValues = batchValues(1:validInBatch, :);
                batchRawMillivolts = ...
                    validValues(:, 4:11) * millivoltsPerCount;
                [batchFilteredMillivolts, filterState] = ...
                    ads1298_emg_filter_step(batchRawMillivolts, filterState);
                filteredPayload = ads1298_format_filtered_rows( ...
                    validValues, batchFilteredMillivolts);

                payload = strjoin( ...
                    outputLines(1:validInBatch), newlineCharacter);
                fprintf(rawFileId, "%s\n", payload);
                fprintf(filteredFileId, "%s", filteredPayload);

                historyTime = [historyTime; ...
                    batchTime(1:validInBatch)]; %#ok<AGROW>
                historyFilteredMillivolts = [historyFilteredMillivolts; ...
                    batchFilteredMillivolts]; %#ok<AGROW>
            end

            if imuValidInBatch > 0
                convertedImu = ads1298_imu_convert( ...
                    batchImuRaw(1:imuValidInBatch, :));
                fprintf(imuFileId, "%s", ...
                    ads1298_format_imu_rows(convertedImu));
                imuHistoryTime = [imuHistoryTime; ...
                    batchImuTime(1:imuValidInBatch)]; %#ok<AGROW>
                imuHistoryAcceleration = [imuHistoryAcceleration; ...
                    convertedImu(:, 3:5)]; %#ok<AGROW>
                imuHistoryAngularRate = [imuHistoryAngularRate; ...
                    convertedImu(:, 6:8)]; %#ok<AGROW>
            end

            latestTime = latestAvailableTime(historyTime, imuHistoryTime);
            if ~isempty(latestTime)
                keepEmg = historyTime >= latestTime - windowSeconds;
                historyTime = historyTime(keepEmg);
                historyFilteredMillivolts = ...
                    historyFilteredMillivolts(keepEmg, :);
                keepImu = imuHistoryTime >= latestTime - windowSeconds;
                imuHistoryTime = imuHistoryTime(keepImu);
                imuHistoryAcceleration = imuHistoryAcceleration(keepImu, :);
                imuHistoryAngularRate = imuHistoryAngularRate(keepImu, :);
            end

            if isempty(captureClock) && receivedRows > 0 && receivedImuRows > 0
                captureClock = tic;
            end
        end
    else
        pause(0.001);
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
            for channel = 1:8
                set(lineHandles(channel), ...
                    "XData", historyTime, ...
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

        latestTime = latestAvailableTime(historyTime, imuHistoryTime);
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
            'EMG: %d rows, %.1f SPS | Invalid: %d | Gaps: %.0f | ' ...
            'Dropped: %.0f | Display: %s'], ...
            receivedRows, measuredRate, invalidRows, sequenceGaps, ...
            deviceDropped, filterDescription);
        imuStatusTitle.String = sprintf( ...
            'IMU: %d rows, %.1f SPS | Gaps: %.0f | Dropped: %.0f', ...
            receivedImuRows, measuredImuRate, ...
            imuSequenceGaps, imuDeviceDropped);
        drawnow limitrate;
        plotClock = tic;
    end

    if isempty(captureClock) && toc(startupClock) >= startupTimeoutSeconds
        error("ads1298_emg_live:NoData", ...
            ["Synchronized ADS1298 and LSM6DSOX data did not arrive from " ...
            "%s within %.0f seconds."], portName, startupTimeoutSeconds);
    end
end

if isgraphics(figureHandle) || isgraphics(imuFigureHandle)
    drawnow;
end
end

function [rawPath, filteredPath, imuPath] = outputPaths(directory, base)
rawPath = string(fullfile(directory, base + "_raw.csv"));
filteredPath = string(fullfile(directory, base + "_filtered.csv"));
imuPath = string(fullfile(directory, base + "_imu.csv"));
end

function latestTime = latestAvailableTime(emgTime, imuTime)
if isempty(emgTime) && isempty(imuTime)
    latestTime = [];
elseif isempty(emgTime)
    latestTime = imuTime(end);
elseif isempty(imuTime)
    latestTime = emgTime(end);
else
    latestTime = max(emgTime(end), imuTime(end));
end
end

function closeSerialPort(serialConnection)
try
    if isvalid(serialConnection)
        delete(serialConnection);
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
