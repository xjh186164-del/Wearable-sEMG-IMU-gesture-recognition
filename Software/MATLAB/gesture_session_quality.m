function quality = gesture_session_quality(session, captureSummary, protocol)
%GESTURE_SESSION_QUALITY Calculate and atomically persist session QA metrics.

[paths, targetLabels, blockCount, targetHoldDuration] = ...
    validateInputs(session, protocol);
limits = qualityLimits();
quality = initialQuality(limits, targetLabels, blockCount);
reasons = strings(0, 1);
captureSummaryValid = validateCaptureSummary(captureSummary);
if ~captureSummaryValid
    reasons = addReasons(reasons, "CAPTURE_SUMMARY_INVALID");
end

[raw, rawIssue] = readNumericCsv(paths.raw, rawSchema(), "RAW");
[filtered, filteredIssue] = readNumericCsv( ...
    paths.filtered, filteredSchema(), "FILTERED");
[imu, imuIssue] = readNumericCsv(paths.imu, imuSchema(), "IMU");
[events, eventIssue] = readEventCsv(paths.events);
[metadata, metadataIssue] = readMetadata(paths.metadata);
reasons = addReasons(reasons, ...
    [rawIssue; filteredIssue; imuIssue; eventIssue; metadataIssue]);
if isempty(metadataIssue)
    reasons = addReasons(reasons, ...
        metadataReasons(metadata, session, protocol));
end

[rawTime, filteredTime, imuTime] = ...
    gesture_internal.shared_sensor_time_axes( ...
    columnOrEmpty(raw, 2), columnOrEmpty(filtered, 2), ...
    columnOrEmpty(imu, 2));
rawSessionTime = rawTime.session_time_s;
filteredSessionTime = filteredTime.session_time_s;
imuSessionTime = imuTime.session_time_s;

quality.emg_row_count = rowCount(raw);
quality.filtered_emg_row_count = rowCount(filtered);
quality.imu_row_count = rowCount(imu);
[quality, captureReasons] = auditCaptureSummary(quality, ...
    captureSummary, captureSummaryValid, rawSessionTime, imuSessionTime, ...
    limits);
reasons = addReasons(reasons, captureReasons);
reasons = addReasons(reasons, rawFilteredParityReasons(raw, filtered));

[quality.emg_rate_hz, rateReason] = sampleRate( ...
    rawTime, limits.emg_rate_min_hz, limits.emg_rate_max_hz, "EMG");
reasons = addReasons(reasons, rateReason);
[quality.imu_rate_hz, rateReason] = sampleRate( ...
    imuTime, limits.imu_rate_min_hz, limits.imu_rate_max_hz, "IMU");
reasons = addReasons(reasons, rateReason);

[quality.emg_sequence_gaps, sequenceCorrupt] = ...
    sequenceMetrics(columnOrEmpty(raw, 1));
if sequenceCorrupt
    reasons = addReasons(reasons, "EMG_SEQUENCE_CORRUPT");
end
if isfinite(quality.emg_sequence_gaps) && ...
        quality.emg_sequence_gaps > limits.max_sequence_gaps
    reasons = addReasons(reasons, "EMG_SEQUENCE_GAP");
end
[quality.imu_sequence_gaps, sequenceCorrupt] = ...
    sequenceMetrics(columnOrEmpty(imu, 1));
if sequenceCorrupt
    reasons = addReasons(reasons, "IMU_SEQUENCE_CORRUPT");
end
if isfinite(quality.imu_sequence_gaps) && ...
        quality.imu_sequence_gaps > limits.max_sequence_gaps
    reasons = addReasons(reasons, "IMU_SEQUENCE_GAP");
end

[quality.emg_dropped_start, quality.emg_dropped_end, emgCsvDrop, ...
    emgDropCorrupt] = droppedMetrics(columnOrEmpty(raw, 12));
[quality.emg_dropped_delta, quality.emg_dropped_csv_delta, ...
    dropReasons] = resolveDroppedDelta(captureSummary, ...
    captureSummaryValid, "emg_dropped_delta", emgCsvDrop, ...
    emgDropCorrupt, "EMG", limits);
reasons = addReasons(reasons, dropReasons);
quality.emg_dropped_summary_delta = quality.emg_dropped_delta;
[quality.imu_dropped_start, quality.imu_dropped_end, imuCsvDrop, ...
    imuDropCorrupt] = droppedMetrics(columnOrEmpty(imu, 10));
[quality.imu_dropped_delta, quality.imu_dropped_csv_delta, ...
    dropReasons] = resolveDroppedDelta(captureSummary, ...
    captureSummaryValid, "imu_dropped_delta", imuCsvDrop, ...
    imuDropCorrupt, "IMU", limits);
reasons = addReasons(reasons, dropReasons);
quality.imu_dropped_summary_delta = quality.imu_dropped_delta;

if ~isempty(raw)
    rawChannels = raw(:, 4:11);
    quality.clipping_fraction_per_channel = ...
        sum(abs(rawChannels) >= 2^23 - 1, 1) / size(rawChannels, 1);
    if any(quality.clipping_fraction_per_channel > ...
            limits.max_clipping_fraction)
        reasons = addReasons(reasons, "EMG_CLIPPING");
    end
end

if ~isempty(filtered) && filteredTime.valid
    [quality.robust_rms_mV_per_channel, fullBlockCount] = ...
        robustRms(filteredSessionTime, filtered(:, 4:11));
    quality.emg_rms_full_block_count = fullBlockCount;
end
if quality.emg_rms_full_block_count < 1
    reasons = addReasons(reasons, "EMG_RMS_INSUFFICIENT");
end

if ~isempty(events)
    reasons = addReasons(reasons, ...
        auditTrialLifecycle(events, protocol, limits));
    [restIntervals, invalidTrialIds, restEventsCorrupt] = ...
        validRestIntervals(events, targetLabels);
    quality.invalid_trial_ids = reshape(invalidTrialIds, [], 1);
    reasons = addReasons(reasons, terminalReasons(events, protocol, ...
        limits.lifecycle_duration_tolerance_s));
    [quality.max_event_time_s, quality.shared_sensor_coverage_end_s, ...
        coverageReasons] = eventCoverage( ...
        events, rawSessionTime, imuSessionTime, limits);
    reasons = addReasons(reasons, coverageReasons);
    if restEventsCorrupt
        reasons = addReasons(reasons, "REST_EVENTS_CORRUPT");
    end
    [quality.valid_trial_counts, quality.target_hold_start_counts, ...
        quality.target_hold_end_counts, quality.valid_trial_count_by_gesture, ...
        trialStructureInvalid, targetEventsCorrupt, holdDurationInvalid] = ...
        validTrialCounts(events, invalidTrialIds, targetLabels, ...
        targetHoldDuration, limits.target_hold_duration_tolerance_s);
    if targetEventsCorrupt
        reasons = addReasons(reasons, "TARGET_EVENTS_CORRUPT");
    end
    if holdDurationInvalid
        reasons = addReasons(reasons, "TARGET_HOLD_DURATION_INVALID");
    end
    if trialStructureInvalid || ...
            any(quality.valid_trial_counts ~= blockCount)
        reasons = addReasons(reasons, "MISSING_VALID_TRIAL");
    end
else
    restIntervals = zeros(0, 2);
    reasons = addReasons(reasons, "SESSION_START_COUNT_INVALID");
    reasons = addReasons(reasons, "SESSION_END_COUNT_INVALID");
    reasons = addReasons(reasons, "MISSING_VALID_TRIAL");
end

restRows = intervalMask(imuSessionTime, restIntervals);
quality.rest_imu_sample_count = sum(restRows);
if isempty(imu) || ~imuTime.valid || ~any(restRows)
    reasons = addReasons(reasons, "REST_IMU_MISSING");
else
    accelerationMagnitude = sqrt(sum(imu(restRows, 3:5).^2, 2));
    gyroMagnitude = sqrt(sum(imu(restRows, 6:8).^2, 2));
    quality.rest_accel_median_g = median(accelerationMagnitude);
    quality.rest_gyro_median_dps = median(gyroMagnitude);
    if quality.rest_accel_median_g < limits.rest_accel_min_g || ...
            quality.rest_accel_median_g > limits.rest_accel_max_g
        reasons = addReasons(reasons, "REST_ACCEL_OUT_OF_RANGE");
    end
    if quality.rest_gyro_median_dps > limits.rest_gyro_max_dps
        reasons = addReasons(reasons, "REST_GYRO_EXCESS");
    end
end

quality.fail_reasons = reasons;
quality.passed = isempty(reasons);
writeQualityAtomically(paths.quality, quality);
end

function [paths, targetLabels, blockCount, targetHoldDuration] = ...
        validateInputs(session, protocol)
sessionFields = ["rawPath", "filteredPath", "imuPath", ...
    "eventsPath", "metadataPath", "qualityPath", "base"];
if ~isstruct(session) || ~isscalar(session) || ...
        ~all(isfield(session, sessionFields)) || ...
        ~isstruct(protocol) || ~isscalar(protocol) || ...
        ~all(isfield(protocol, ["targetLabels", "blockCount", ...
        "stageNames", "stageDurationsSeconds", "trialTable", ...
        "version", "seed", "breakAfterBlocks", ...
        "breakDurationSeconds"]))
    error("gesture_session_quality:InvalidInput", ...
        "session, captureSummary, and protocol do not use the required schema.");
end

paths = struct( ...
    "raw", requiredPath(session.rawPath), ...
    "filtered", requiredPath(session.filteredPath), ...
    "imu", requiredPath(session.imuPath), ...
    "events", requiredPath(session.eventsPath), ...
    "metadata", requiredPath(session.metadataPath), ...
    "quality", requiredPath(session.qualityPath));
sessionBase = requiredPath(session.base);
if isempty(regexp(sessionBase, "^S\d{3}_D\d{2}_R\d{2}$", "once"))
    error("gesture_session_quality:InvalidInput", ...
        "session.base must use the S###_D##_R## format.");
end
qualityDirectory = fileparts(paths.quality);
if strlength(qualityDirectory) == 0 || ~isfolder(qualityDirectory)
    error("gesture_session_quality:InvalidInput", ...
        "The quality output directory must already exist.");
end
targetLabels = string(protocol.targetLabels);
if ~isvector(targetLabels) || isempty(targetLabels) || ...
        any(ismissing(targetLabels) | strlength(strtrim(targetLabels)) == 0) || ...
        numel(unique(targetLabels)) ~= numel(targetLabels)
    error("gesture_session_quality:InvalidInput", ...
        "protocol.targetLabels must contain unique nonempty labels.");
end
targetLabels = reshape(targetLabels, 1, []);
blockCount = protocol.blockCount;
if ~isnumeric(blockCount) || ~isscalar(blockCount) || ...
        ~isreal(blockCount) || ~isfinite(blockCount) || ...
        blockCount ~= 2
    error("gesture_session_quality:InvalidInput", ...
        "protocol.blockCount must equal 2 for eight_pose_v5.");
end
blockCount = double(blockCount);
breakAfterBlocks = protocol.breakAfterBlocks;
breakDuration = protocol.breakDurationSeconds;
validBreakContract = isnumeric(breakAfterBlocks) && ...
    isreal(breakAfterBlocks) && isvector(breakAfterBlocks) && ...
    isempty(breakAfterBlocks) && ...
    isnumeric(breakDuration) && isscalar(breakDuration) && ...
    isreal(breakDuration) && isfinite(breakDuration) && ...
    breakDuration == 0;
if ~validBreakContract
    error("gesture_session_quality:InvalidInput", ...
        "protocol must define valid break blocks and duration.");
end
if ~isTextScalar(protocol.version) || ...
        string(protocol.version) ~= "eight_pose_v5" || ...
        ~isnumeric(protocol.seed) || ~isscalar(protocol.seed) || ...
        ~isreal(protocol.seed) || ~isfinite(protocol.seed) || ...
        protocol.seed < 0 || protocol.seed ~= fix(protocol.seed)
    error("gesture_session_quality:InvalidInput", ...
        "protocol.version and protocol.seed must be valid scalars.");
end
stageNames = reshape(string(protocol.stageNames), 1, []);
stageDurations = protocol.stageDurationsSeconds;
targetIndex = find(stageNames == "target_hold");
if numel(targetIndex) ~= 1 || ~isnumeric(stageDurations) || ...
        ~isreal(stageDurations) || ~isvector(stageDurations) || ...
        numel(stageDurations) ~= numel(stageNames) || ...
        any(~isfinite(stageDurations) | stageDurations < 0) || ...
        stageDurations(targetIndex) <= 0
    error("gesture_session_quality:InvalidInput", ...
        "protocol must define one positive target_hold duration.");
end
targetHoldDuration = double(stageDurations(targetIndex));
validateOriginalSchedule(protocol.trialTable, targetLabels, blockCount);
end

function validateOriginalSchedule(schedule, targetLabels, blockCount)
required = ["trial_id", "block_id", "gesture_id", "is_repeat"];
if ~istable(schedule) || ...
        ~all(ismember(required, string(schedule.Properties.VariableNames))) || ...
        height(schedule) ~= blockCount * numel(targetLabels)
    error("gesture_session_quality:InvalidInput", ...
        "protocol.trialTable must contain the complete original schedule.");
end
trialIds = schedule.trial_id;
blockIds = schedule.block_id;
gestures = string(schedule.gesture_id);
repeats = schedule.is_repeat;
valid = isnumeric(trialIds) && iscolumn(trialIds) && ...
    all(isfinite(trialIds) & trialIds > 0 & trialIds == fix(trialIds)) && ...
    numel(unique(trialIds)) == height(schedule) && ...
    isnumeric(blockIds) && iscolumn(blockIds) && ...
    all(isfinite(blockIds) & blockIds >= 1 & ...
    blockIds <= blockCount & blockIds == fix(blockIds)) && ...
    iscolumn(gestures) && all(ismember(gestures, targetLabels)) && ...
    (islogical(repeats) || isnumeric(repeats)) && iscolumn(repeats) && ...
    all(repeats == 0);
if ~valid
    error("gesture_session_quality:InvalidInput", ...
        "protocol.trialTable must contain valid original trial identities.");
end
end

function valid = validateCaptureSummary(summary)
required = ["duration_s", "emg_rows", "imu_rows", ...
    "emg_sequence_gaps", "imu_sequence_gaps", "packet_gaps", ...
    "invalid_packets", "emg_dropped_delta", "imu_dropped_delta", ...
    "ble_read_errors", "emg_host_queue_age_max_s", ...
    "emg_host_queue_age_p95_s", "imu_host_queue_age_max_s", ...
    "imu_host_queue_age_p95_s"];
valid = isstruct(summary) && isscalar(summary) && ...
    all(isfield(summary, required));
if ~valid
    return;
end
integerFields = ["emg_rows", "imu_rows", "emg_sequence_gaps", ...
    "imu_sequence_gaps", "packet_gaps", "invalid_packets", ...
    "emg_dropped_delta", "imu_dropped_delta", "ble_read_errors"];
for field = required
    value = summary.(field);
    if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || ...
            ~isfinite(value) || value < 0
        valid = false;
        return;
    end
    if ismember(field, integerFields) && value ~= fix(value)
        valid = false;
        return;
    end
end
end

function [metadata, issue] = readMetadata(path)
metadata = struct();
issue = strings(0, 1);
if ~isfile(path)
    issue = "METADATA_FILE_MISSING";
    return;
end
try
    metadata = jsondecode(fileread(path));
catch
    issue = "METADATA_FILE_MALFORMED";
    return;
end
required = ["subject_id", "session_id", "arm_side", ...
    "operator_code", "sleeve_size", "device_identifier", ...
    "protocol_version", "protocol_seed", "block_count", ...
    "trial_table", "created_at_utc", "emg_rate_hz", "imu_rate_hz", ...
    "ads1298_gain", "accelerometer_range_g", "gyroscope_range_dps", ...
    "electrode_layout_version", "pcb_orientation_version", ...
    "acquisition_software_version", "matlab_version", "completed", ...
    "stopped_by_operator", "terminal_status"];
valid = isstruct(metadata) && isscalar(metadata) && ...
    isequal(sort(string(fieldnames(metadata))), sort(required(:)));
if valid
    textFields = ["subject_id", "session_id", "arm_side", ...
        "operator_code", "sleeve_size", "device_identifier", ...
        "protocol_version", "created_at_utc", ...
        "electrode_layout_version", "pcb_orientation_version", ...
        "acquisition_software_version", "matlab_version", ...
        "terminal_status"];
    numericFields = ["protocol_seed", "block_count", "emg_rate_hz", ...
        "imu_rate_hz", "ads1298_gain", "accelerometer_range_g", ...
        "gyroscope_range_dps"];
    valid = all(arrayfun(@(field) ...
        isTextScalar(metadata.(field)), textFields)) && ...
        all(arrayfun(@(field) isFiniteNumericScalar( ...
        metadata.(field)), numericFields)) && ...
        islogical(metadata.completed) && isscalar(metadata.completed) && ...
        islogical(metadata.stopped_by_operator) && ...
        isscalar(metadata.stopped_by_operator) && ...
        validMetadataTrialTable(metadata.trial_table);
end
if ~valid
    metadata = struct();
    issue = "METADATA_FILE_MALFORMED";
end
end

function valid = isTextScalar(value)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if valid
    valid = strlength(strtrim(string(value))) > 0;
end
end

function valid = isFiniteNumericScalar(value)
valid = isnumeric(value) && isscalar(value) && isreal(value) && ...
    isfinite(value);
end

function valid = validMetadataTrialTable(trialTable)
required = ["trial_id", "block_id", "gesture_id", "is_repeat"];
valid = isstruct(trialTable) && isvector(trialTable) && ...
    ~isempty(trialTable) && ...
    isequal(string(fieldnames(trialTable)), required(:));
if ~valid
    return;
end
for index = 1:numel(trialTable)
    row = trialTable(index);
    valid = isFiniteNumericScalar(row.trial_id) && ...
        row.trial_id > 0 && row.trial_id == fix(row.trial_id) && ...
        isFiniteNumericScalar(row.block_id) && ...
        row.block_id > 0 && row.block_id == fix(row.block_id) && ...
        isTextScalar(row.gesture_id) && ...
        islogical(row.is_repeat) && isscalar(row.is_repeat);
    if ~valid
        return;
    end
end
end

function reasons = metadataReasons(metadata, session, protocol)
reasons = strings(0, 1);
if ~metadata.completed
    reasons = addReasons(reasons, "SESSION_INCOMPLETE");
end
if metadata.stopped_by_operator
    reasons = addReasons(reasons, "SESSION_STOPPED_BY_OPERATOR");
end
if string(metadata.terminal_status) ~= "completed"
    reasons = addReasons(reasons, "TERMINAL_STATUS_NOT_COMPLETED");
end
if string(metadata.protocol_version) ~= "eight_pose_v5" || ...
        string(metadata.protocol_version) ~= string(protocol.version)
    reasons = addReasons(reasons, ...
        "METADATA_PROTOCOL_VERSION_MISMATCH");
end
if double(metadata.protocol_seed) ~= double(protocol.seed)
    reasons = addReasons(reasons, "METADATA_PROTOCOL_SEED_MISMATCH");
end
if double(metadata.block_count) ~= 2 || ...
        double(metadata.block_count) ~= double(protocol.blockCount)
    reasons = addReasons(reasons, "METADATA_BLOCK_COUNT_MISMATCH");
end
sessionBase = string(session.base);
if string(metadata.session_id) ~= sessionBase
    reasons = addReasons(reasons, "METADATA_SESSION_ID_MISMATCH");
end
subjectToken = regexp(char(sessionBase), "^(S\d{3})_", ...
    "tokens", "once");
if isempty(subjectToken) || ...
        string(metadata.subject_id) ~= string(subjectToken{1})
    reasons = addReasons(reasons, "METADATA_SUBJECT_ID_MISMATCH");
end
if ~metadataTrialTableMatches(metadata.trial_table, protocol.trialTable)
    reasons = addReasons(reasons, "METADATA_TRIAL_TABLE_MISMATCH");
end
end

function matches = metadataTrialTableMatches(actual, expected)
matches = numel(actual) == height(expected);
if ~matches
    return;
end
actual = actual(:);
actualTrialIds = reshape([actual.trial_id], [], 1);
actualBlockIds = reshape([actual.block_id], [], 1);
actualGestures = string({actual.gesture_id}).';
actualRepeats = reshape([actual.is_repeat], [], 1);
matches = isequal(double(actualTrialIds), double(expected.trial_id)) && ...
    isequal(double(actualBlockIds), double(expected.block_id)) && ...
    isequal(actualGestures, string(expected.gesture_id)) && ...
    isequal(logical(actualRepeats), logical(expected.is_repeat));
end

function [quality, reasons] = auditCaptureSummary(quality, summary, ...
        summaryValid, rawSessionTime, imuSessionTime, limits)
reasons = strings(0, 1);
if ~isempty(rawSessionTime) && ~isempty(imuSessionTime)
    quality.reconstructed_capture_duration_s = ...
        min(rawSessionTime(end), imuSessionTime(end));
end
if ~summaryValid
    return;
end

quality.capture_duration_s = double(summary.duration_s);
quality.capture_emg_rows = double(summary.emg_rows);
quality.capture_imu_rows = double(summary.imu_rows);
quality.capture_emg_sequence_gaps = ...
    double(summary.emg_sequence_gaps);
quality.capture_imu_sequence_gaps = ...
    double(summary.imu_sequence_gaps);
quality.capture_packet_gaps = double(summary.packet_gaps);
quality.capture_invalid_packets = double(summary.invalid_packets);
quality.capture_ble_read_errors = double(summary.ble_read_errors);
quality.emg_host_queue_age_max_s = ...
    double(summary.emg_host_queue_age_max_s);
quality.emg_host_queue_age_p95_s = ...
    double(summary.emg_host_queue_age_p95_s);
quality.imu_host_queue_age_max_s = ...
    double(summary.imu_host_queue_age_max_s);
quality.imu_host_queue_age_p95_s = ...
    double(summary.imu_host_queue_age_p95_s);

if quality.capture_emg_rows ~= quality.emg_row_count
    reasons = addReasons(reasons, ...
        "CAPTURE_EMG_ROW_COUNT_MISMATCH");
end
if quality.capture_imu_rows ~= quality.imu_row_count
    reasons = addReasons(reasons, ...
        "CAPTURE_IMU_ROW_COUNT_MISMATCH");
end
durationError = abs(quality.capture_duration_s - ...
    quality.reconstructed_capture_duration_s);
durationScale = max([1, quality.capture_duration_s, ...
    quality.reconstructed_capture_duration_s]);
if ~isfinite(durationError) || durationError > ...
        limits.summary_duration_tolerance_s + 10 * eps(durationScale)
    reasons = addReasons(reasons, "CAPTURE_DURATION_MISMATCH");
end

faultFields = ["emg_sequence_gaps", "imu_sequence_gaps", ...
    "packet_gaps", "invalid_packets", "ble_read_errors"];
faultReasons = ["CAPTURE_EMG_SEQUENCE_GAP", ...
    "CAPTURE_IMU_SEQUENCE_GAP", "CAPTURE_PACKET_GAP", ...
    "CAPTURE_INVALID_PACKET", "CAPTURE_BLE_READ_ERROR"];
for index = 1:numel(faultFields)
    if summary.(faultFields(index)) > 0
        reasons = addReasons(reasons, faultReasons(index));
    end
end
end

function reasons = rawFilteredParityReasons(raw, filtered)
reasons = strings(0, 1);
if size(raw, 1) ~= size(filtered, 1)
    reasons = addReasons(reasons, ...
        "RAW_FILTERED_ROW_COUNT_MISMATCH");
    return;
end
if isempty(raw)
    return;
end
rawColumns = [1, 2, 3, 12];
filteredColumns = [1, 2, 3, 12];
mismatchReasons = ["RAW_FILTERED_SAMPLE_MISMATCH", ...
    "RAW_FILTERED_TIMESTAMP_MISMATCH", ...
    "RAW_FILTERED_STATUS_MISMATCH", ...
    "RAW_FILTERED_DROPPED_MISMATCH"];
for index = 1:numel(rawColumns)
    if ~isequal(raw(:, rawColumns(index)), ...
            filtered(:, filteredColumns(index)))
        reasons = addReasons(reasons, mismatchReasons(index));
    end
end
end

function reasons = terminalReasons(events, protocol, tolerance)
reasons = strings(0, 1);
startRows = find(events.event_type == "session_start");
endRows = find(events.event_type == "session_end");
if numel(startRows) ~= 1
    reasons = addReasons(reasons, "SESSION_START_COUNT_INVALID");
end
if numel(endRows) ~= 1
    reasons = addReasons(reasons, "SESSION_END_COUNT_INVALID");
end
if any(ismember(events.event_type, ...
        ["session_stopped", "acquisition_error"]))
    reasons = addReasons(reasons, "SESSION_ABORT_EVENT_PRESENT");
end
if isscalar(startRows) && isscalar(endRows)
    coreTypes = ["rest_valid_start", "rest_valid_end", "prepare", ...
        "movement_start", "target_hold_start", "target_hold_end", ...
        "return_start", "return_end"];
    coreRows = find(ismember(events.event_type, coreTypes));
    firstTrial = protocol.trialTable(1, :);
    startIdentityValid = events.block_id(startRows) == ...
        firstTrial.block_id && events.trial_id(startRows) == ...
        firstTrial.trial_id && events.gesture_id(startRows) == ...
        string(firstTrial.gesture_id) && events.valid(startRows) && ...
        events.note(startRows) == "";
    endIdentityValid = events.block_id(endRows) == 0 && ...
        events.trial_id(endRows) == 0 && ...
        events.gesture_id(endRows) == "" && events.valid(endRows) && ...
        events.note(endRows) == "";
    boundaryValid = ~isempty(coreRows) && ...
        abs(events.session_time_s(startRows) - ...
        events.session_time_s(coreRows(1))) <= tolerance && ...
        abs(events.session_time_s(endRows) - ...
        events.session_time_s(coreRows(end))) <= tolerance;
    orderValid = startRows == 1 && endRows == height(events) && ...
        startRows < endRows && ...
        events.session_time_s(startRows) <= ...
        events.session_time_s(endRows) && startIdentityValid && ...
        endIdentityValid && boundaryValid;
    if ~orderValid
        reasons = addReasons(reasons, ...
            "TERMINAL_EVENT_ORDER_INVALID");
    end
end
end

function reasons = auditTrialLifecycle(events, protocol, limits)
reasons = strings(0, 1);
schedule = protocol.trialTable;
originalIds = double(schedule.trial_id(:));
expectedIds = originalIds;
expectedBlocks = double(schedule.block_id(:));
expectedGestures = string(schedule.gesture_id(:));
isReplacement = false(size(expectedIds));
invalidParentIds = zeros(0, 1);
invalidRows = find(events.event_type == "trial_invalid");
replacementInvalid = false;
maximumOriginalId = max(originalIds);

for relationIndex = 1:numel(invalidRows)
    row = invalidRows(relationIndex);
    parentId = events.trial_id(row);
    parentIndex = find(expectedIds == parentId);
    [replacementId, noteValid] = parsedReplacementId(events.note(row));
    relationValid = isscalar(parentIndex) && noteValid && ...
        replacementId == maximumOriginalId + relationIndex && ...
        ~ismember(parentId, invalidParentIds) && ...
        ~ismember(replacementId, expectedIds) && ...
        ~events.valid(row);
    if relationValid
        relationValid = events.block_id(row) == ...
            expectedBlocks(parentIndex) && ...
            events.gesture_id(row) == expectedGestures(parentIndex);
    end
    if ~relationValid
        replacementInvalid = true;
        continue;
    end
    invalidParentIds(end + 1, 1) = parentId; %#ok<AGROW>
    expectedIds(end + 1, 1) = replacementId; %#ok<AGROW>
    expectedBlocks(end + 1, 1) = expectedBlocks(parentIndex); %#ok<AGROW>
    expectedGestures(end + 1, 1) = ...
        expectedGestures(parentIndex); %#ok<AGROW>
    isReplacement(end + 1, 1) = true; %#ok<AGROW>
end

expectedTypes = ["rest_valid_start", "rest_valid_end", "prepare", ...
    "movement_start", "target_hold_start", "target_hold_end", ...
    "return_start", "return_end", "rest_valid_start", ...
    "rest_valid_end"];
coreMask = ismember(events.event_type, unique(expectedTypes));
appearanceIds = unique(events.trial_id(coreMask), "stable");
lifecycleInvalid = false;
if ~isequal(appearanceIds(:), expectedIds)
    lifecycleInvalid = true;
    if ~isempty(invalidRows) || ...
            any(~ismember(appearanceIds, originalIds))
        replacementInvalid = true;
    end
end

expectedCoreIds = repelem(expectedIds, numel(expectedTypes));
expectedCoreTypes = repmat(expectedTypes(:), numel(expectedIds), 1);
if ~isequal(events.trial_id(coreMask), expectedCoreIds) || ...
        ~isequal(events.event_type(coreMask), expectedCoreTypes)
    lifecycleInvalid = true;
end

durations = double(reshape(protocol.stageDurationsSeconds, 1, []));
expectedOffsets = [0, durations(1), durations(1), ...
    sum(durations(1:2)), sum(durations(1:3)), ...
    sum(durations(1:4)), sum(durations(1:4)), ...
    sum(durations(1:5)), sum(durations(1:5)), sum(durations)];
for expectedIndex = 1:numel(expectedIds)
    trialId = expectedIds(expectedIndex);
    rows = find(coreMask & events.trial_id == trialId);
    if numel(rows) ~= numel(expectedTypes) || ...
            ~isequal(events.event_type(rows).', expectedTypes)
        lifecycleInvalid = true;
        continue;
    end
    identityValid = all(events.block_id(rows) == ...
        expectedBlocks(expectedIndex)) && ...
        all(events.gesture_id(rows) == expectedGestures(expectedIndex));
    if ~identityValid
        lifecycleInvalid = true;
        if isReplacement(expectedIndex)
            replacementInvalid = true;
        end
    end
    actualOffsets = events.session_time_s(rows).' - ...
        events.session_time_s(rows(1));
    if any(abs(actualOffsets - expectedOffsets) > ...
            limits.lifecycle_duration_tolerance_s)
        lifecycleInvalid = true;
    end

    invalidRow = invalidRows(events.trial_id(invalidRows) == trialId);
    if isempty(invalidRow)
        if any(~events.valid(rows))
            lifecycleInvalid = true;
        end
    elseif isscalar(invalidRow)
        positionValid = rows(1) < invalidRow && invalidRow < rows(end) && ...
            events.session_time_s(invalidRow) >= ...
            events.session_time_s(rows(1)) - ...
            limits.lifecycle_duration_tolerance_s && ...
            events.session_time_s(invalidRow) <= ...
            events.session_time_s(rows(end)) + ...
            limits.lifecycle_duration_tolerance_s;
        if ~positionValid
            replacementInvalid = true;
        end
        if any(~events.valid(rows(rows < invalidRow))) || ...
                any(events.valid(rows(rows > invalidRow)))
            lifecycleInvalid = true;
        end
    else
        replacementInvalid = true;
    end
end

[breakInvalid, boundaryInvalid] = auditBreakStateMachine( ...
    events, protocol, expectedIds, coreMask, ...
    limits.lifecycle_duration_tolerance_s);
lifecycleInvalid = lifecycleInvalid || boundaryInvalid;

if lifecycleInvalid
    reasons = addReasons(reasons, "TRIAL_LIFECYCLE_INVALID");
end
if replacementInvalid
    reasons = addReasons(reasons, "REPLACEMENT_RELATION_INVALID");
end
if breakInvalid
    reasons = addReasons(reasons, "BREAK_EVENTS_INVALID");
end
end

function [breakInvalid, boundaryInvalid] = auditBreakStateMachine( ...
        events, protocol, expectedIds, coreMask, tolerance)
schedule = protocol.trialTable;
originalCount = height(schedule);
breakBlocks = double(reshape(protocol.breakAfterBlocks, [], 1));
startRows = find(events.event_type == "break_start");
endRows = find(events.event_type == "break_end");
breakInvalid = numel(startRows) ~= numel(breakBlocks) || ...
    numel(endRows) ~= numel(breakBlocks);
boundaryInvalid = false;

for blockId = reshape(breakBlocks, 1, [])
    blockStarts = startRows(events.block_id(startRows) == blockId);
    blockEnds = endRows(events.block_id(endRows) == blockId);
    if ~isscalar(blockStarts) || ~isscalar(blockEnds)
        breakInvalid = true;
        continue;
    end
    startRow = blockStarts(1);
    endRow = blockEnds(1);
    identityValid = events.trial_id(startRow) == 0 && ...
        events.trial_id(endRow) == 0 && ...
        events.gesture_id(startRow) == "" && ...
        events.gesture_id(endRow) == "" && ...
        events.valid(startRow) && events.valid(endRow) && ...
        events.note(startRow) == "" && events.note(endRow) == "";
    lastOriginalIndex = find(schedule.block_id == blockId, 1, "last");
    if isempty(lastOriginalIndex) || lastOriginalIndex >= originalCount
        breakInvalid = true;
        continue;
    end
    parentId = schedule.trial_id(lastOriginalIndex);
    nextId = schedule.trial_id(lastOriginalIndex + 1);
    parentRows = find(coreMask & events.trial_id == parentId);
    nextRows = find(coreMask & events.trial_id == nextId);
    if isempty(parentRows) || isempty(nextRows)
        breakInvalid = true;
        continue;
    end
    positionValid = startRow == parentRows(end) + 1 && ...
        endRow == startRow + 1 && nextRows(1) == endRow + 1;
    timingValid = abs(events.session_time_s(startRow) - ...
        events.session_time_s(parentRows(end))) <= tolerance && ...
        abs(events.session_time_s(endRow) - ...
        events.session_time_s(startRow) - ...
        protocol.breakDurationSeconds) <= tolerance && ...
        abs(events.session_time_s(nextRows(1)) - ...
        events.session_time_s(endRow)) <= tolerance;
    if ~identityValid || ~positionValid || ~timingValid
        breakInvalid = true;
    end
end

for index = 1:(numel(expectedIds) - 1)
    previousRows = find(coreMask & events.trial_id == expectedIds(index));
    nextRows = find(coreMask & events.trial_id == expectedIds(index + 1));
    if isempty(previousRows) || isempty(nextRows)
        boundaryInvalid = true;
        continue;
    end
    breakExpected = index <= originalCount && ...
        ismember(schedule.block_id(index), breakBlocks) && ...
        index == find(schedule.block_id == schedule.block_id(index), ...
        1, "last");
    if ~breakExpected && abs(events.session_time_s(nextRows(1)) - ...
            events.session_time_s(previousRows(end))) > tolerance
        boundaryInvalid = true;
    end
end
end

function [replacementId, valid] = parsedReplacementId(note)
replacementId = NaN;
token = regexp(char(string(note)), ...
    '^replacement_trial_id=(\d+)$', 'tokens', 'once');
valid = ~isempty(token);
if valid
    replacementId = str2double(token{1});
    valid = isfinite(replacementId) && replacementId > 0 && ...
        replacementId == fix(replacementId);
end
end

function [maxEventTime, sharedCoverageEnd, reasons] = eventCoverage( ...
        events, rawSessionTime, imuSessionTime, limits)
maxEventTime = max(events.session_time_s);
sharedCoverageEnd = NaN;
reasons = strings(0, 1);
if isempty(rawSessionTime) || isempty(imuSessionTime) || ...
        ~isfinite(rawSessionTime(end)) || ~isfinite(imuSessionTime(end))
    reasons = addReasons(reasons, "EVENT_SENSOR_COVERAGE_INVALID");
    return;
end
sharedCoverageEnd = min(rawSessionTime(end), imuSessionTime(end));
if maxEventTime > sharedCoverageEnd + ...
        limits.event_coverage_tolerance_s + ...
        10 * eps(max(1, sharedCoverageEnd))
    reasons = addReasons(reasons, "EVENT_OUT_OF_SENSOR_COVERAGE");
end
end

function path = requiredPath(value)
if ~(ischar(value) && isrow(value)) && ...
        ~(isstring(value) && isscalar(value) && ~ismissing(value))
    error("gesture_session_quality:InvalidInput", ...
        "Session paths must be nonempty text scalars.");
end
path = string(value);
if strlength(strtrim(path)) == 0
    error("gesture_session_quality:InvalidInput", ...
        "Session paths must be nonempty text scalars.");
end
end

function limits = qualityLimits()
limits = struct( ...
    "emg_rate_min_hz", 475, "emg_rate_max_hz", 525, ...
    "imu_rate_min_hz", 98.8, "imu_rate_max_hz", 109.2, ...
    "max_sequence_gaps", 0, "max_dropped_delta", 0, ...
    "max_clipping_fraction", 0.001, ...
    "rest_accel_min_g", 0.85, "rest_accel_max_g", 1.15, ...
    "rest_gyro_max_dps", 5.0, ...
    "target_hold_duration_tolerance_s", 1e-6, ...
    "lifecycle_duration_tolerance_s", 1e-6, ...
    "event_coverage_tolerance_s", 1 / 104, ...
    "summary_duration_tolerance_s", 1 / 104);
end

function quality = initialQuality(limits, targetLabels, blockCount)
countByGesture = struct();
for label = targetLabels
    countByGesture.(matlab.lang.makeValidName(char(label))) = 0;
end
quality = struct( ...
    "schema_version", "gesture_session_quality_v4", ...
    "passed", false, ...
    "fail_reasons", strings(0, 1), ...
    "limits", limits, ...
    "emg_row_count", 0, ...
    "filtered_emg_row_count", 0, ...
    "imu_row_count", 0, ...
    "emg_rate_hz", NaN, ...
    "imu_rate_hz", NaN, ...
    "emg_sequence_gaps", NaN, ...
    "imu_sequence_gaps", NaN, ...
    "capture_duration_s", NaN, ...
    "reconstructed_capture_duration_s", NaN, ...
    "capture_emg_rows", NaN, ...
    "capture_imu_rows", NaN, ...
    "capture_emg_sequence_gaps", NaN, ...
    "capture_imu_sequence_gaps", NaN, ...
    "capture_packet_gaps", NaN, ...
    "capture_invalid_packets", NaN, ...
    "capture_ble_read_errors", NaN, ...
    "emg_host_queue_age_max_s", NaN, ...
    "emg_host_queue_age_p95_s", NaN, ...
    "imu_host_queue_age_max_s", NaN, ...
    "imu_host_queue_age_p95_s", NaN, ...
    "emg_dropped_start", NaN, ...
    "emg_dropped_end", NaN, ...
    "emg_dropped_delta", NaN, ...
    "emg_dropped_csv_delta", NaN, ...
    "emg_dropped_summary_delta", NaN, ...
    "imu_dropped_start", NaN, ...
    "imu_dropped_end", NaN, ...
    "imu_dropped_delta", NaN, ...
    "imu_dropped_csv_delta", NaN, ...
    "imu_dropped_summary_delta", NaN, ...
    "clipping_fraction_per_channel", NaN(1, 8), ...
    "robust_rms_mV_per_channel", NaN(1, 8), ...
    "emg_rms_full_block_count", 0, ...
    "rest_accel_median_g", NaN, ...
    "rest_gyro_median_dps", NaN, ...
    "rest_imu_sample_count", 0, ...
    "max_event_time_s", NaN, ...
    "shared_sensor_coverage_end_s", NaN, ...
    "invalid_trial_ids", zeros(0, 1), ...
    "target_labels", targetLabels, ...
    "expected_valid_trials_per_target", blockCount, ...
    "valid_trial_counts", zeros(1, numel(targetLabels)), ...
    "target_hold_start_counts", zeros(1, numel(targetLabels)), ...
    "target_hold_end_counts", zeros(1, numel(targetLabels)), ...
    "valid_trial_count_by_gesture", countByGesture);
end

function schema = rawSchema()
schema = ["sample", "timestamp_us", "status", "ch1_raw", ...
    "ch2_raw", "ch3_raw", "ch4_raw", "ch5_raw", "ch6_raw", ...
    "ch7_raw", "ch8_raw", "dropped"];
end

function schema = filteredSchema()
schema = ["sample", "timestamp_us", "status", ...
    "ch1_filtered_mV", "ch2_filtered_mV", "ch3_filtered_mV", ...
    "ch4_filtered_mV", "ch5_filtered_mV", "ch6_filtered_mV", ...
    "ch7_filtered_mV", "ch8_filtered_mV", "dropped"];
end

function schema = imuSchema()
schema = ["sample", "timestamp_us", "ax_g", "ay_g", "az_g", ...
    "gx_dps", "gy_dps", "gz_dps", "temp_C", "dropped"];
end

function [data, issue] = readNumericCsv(path, schema, prefix)
data = zeros(0, numel(schema));
issue = strings(0, 1);
if ~isfile(path)
    issue = prefix + "_FILE_MISSING";
    return;
end
try
    values = readtable(path, TextType="string", ...
        VariableNamingRule="preserve");
catch
    issue = prefix + "_FILE_MALFORMED";
    return;
end
if ~isequal(string(values.Properties.VariableNames), schema)
    issue = prefix + "_FILE_MALFORMED";
    return;
end
if height(values) == 0
    issue = prefix + "_FILE_EMPTY";
    return;
end
data = zeros(height(values), numel(schema));
for column = 1:numel(schema)
    [data(:, column), valid] = numericValues(values{:, column});
    if ~valid
        data = zeros(0, numel(schema));
        issue = prefix + "_FILE_MALFORMED";
        return;
    end
end
uint32Columns = unique([1, 2, numel(schema)]);
if any(any(data(:, uint32Columns) < 0 | ...
        data(:, uint32Columns) > 2^32 - 1 | ...
        data(:, uint32Columns) ~= fix(data(:, uint32Columns))))
    data = zeros(0, numel(schema));
    issue = prefix + "_FILE_MALFORMED";
    return;
end
if prefix == "FILTERED" && ...
        any(abs(data(:, 4:11)) > sqrt(realmax("double")), "all")
    data = zeros(0, numel(schema));
    issue = "FILTERED_FILE_MALFORMED";
end
end

function [numeric, valid] = numericValues(values)
if isnumeric(values) || islogical(values)
    numeric = double(values);
else
    numeric = str2double(string(values));
end
numeric = reshape(numeric, [], 1);
valid = isreal(numeric) && all(isfinite(numeric));
end

function [events, issue] = readEventCsv(path)
events = table();
issue = strings(0, 1);
schema = ["session_time_s", "event_type", "block_id", "trial_id", ...
    "gesture_id", "valid", "note"];
if ~isfile(path)
    issue = "EVENTS_FILE_MISSING";
    return;
end
try
    values = readtable(path, TextType="string", ...
        VariableNamingRule="preserve");
catch
    issue = "EVENTS_FILE_MALFORMED";
    return;
end
if ~isequal(string(values.Properties.VariableNames), schema)
    issue = "EVENTS_FILE_MALFORMED";
    return;
end
if height(values) == 0
    issue = "EVENTS_FILE_EMPTY";
    return;
end
[sessionTime, validTime] = numericValues(values.session_time_s);
[blockId, validBlock] = numericValues(values.block_id);
[trialId, validTrial] = numericValues(values.trial_id);
[validFlag, validLogical] = logicalValues(values.valid);
eventType = string(values.event_type);
gestureId = string(values.gesture_id);
note = string(values.note);
gestureId(ismissing(gestureId)) = "";
note(ismissing(note)) = "";
allowedTypes = ["session_start", "session_end", "session_stopped", ...
    "acquisition_error", "rest_valid_start", "rest_valid_end", ...
    "prepare", "movement_start", "target_hold_start", ...
    "target_hold_end", "return_start", "return_end", ...
    "trial_invalid", "break_start", "break_end"];
requiresGesture = ismember(eventType, ["rest_valid_start", ...
    "rest_valid_end", "prepare", "movement_start", ...
    "target_hold_start", "target_hold_end", "return_start", ...
    "return_end", "trial_invalid"]);
valid = validTime && validBlock && validTrial && validLogical && ...
    all(sessionTime >= 0) && all(diff(sessionTime) >= 0) && ...
    all(blockId >= 0 & blockId == fix(blockId)) && ...
    all(trialId >= 0 & trialId == fix(trialId)) && ...
    ~any(ismissing(eventType) | strlength(strtrim(eventType)) == 0) && ...
    all(ismember(eventType, allowedTypes)) && ...
    ~any(requiresGesture & strlength(strtrim(gestureId)) == 0);
if ~valid
    issue = "EVENTS_FILE_MALFORMED";
    return;
end
events = table(sessionTime, eventType, blockId, trialId, gestureId, ...
    validFlag, note, 'VariableNames', cellstr(schema));
end

function [logicalValue, valid] = logicalValues(values)
if islogical(values)
    logicalValue = values;
    valid = true;
elseif isnumeric(values)
    valid = all(values == 0 | values == 1);
    logicalValue = logical(values);
else
    text = lower(strtrim(string(values)));
    valid = all(ismember(text, ["true", "false", "1", "0"]));
    logicalValue = text == "true" | text == "1";
end
logicalValue = reshape(logicalValue, [], 1);
end

function [rate, issue] = sampleRate(analysis, minimum, maximum, prefix)
rate = NaN;
issue = strings(0, 1);
if analysis.row_count < 2 || ~analysis.valid || ...
        analysis.duration_s < 1
    if analysis.row_count >= 2 && ~analysis.valid
        issue = prefix + "_TIMESTAMP_CORRUPT";
    else
        issue = prefix + "_RATE_INSUFFICIENT";
    end
    return;
end
rate = (analysis.row_count - 1) / analysis.duration_s;
if rate < minimum || rate > maximum
    issue = prefix + "_RATE_OUT_OF_RANGE";
end
end

function [gaps, corrupt] = sequenceMetrics(samples)
if isempty(samples)
    gaps = NaN;
    corrupt = false;
    return;
end
deltas = mod(diff(samples), 2^32);
corrupt = any(deltas == 0 | deltas >= 2^31);
validGaps = deltas > 1 & deltas < 2^31;
gaps = sum(deltas(validGaps) - 1);
end

function [startValue, endValue, delta, corrupt] = droppedMetrics(dropped)
if isempty(dropped)
    startValue = NaN;
    endValue = NaN;
    delta = NaN;
    corrupt = false;
    return;
end
startValue = dropped(1);
endValue = dropped(end);
steps = mod(diff(dropped), 2^32);
corrupt = any(steps >= 2^31);
delta = mod(endValue - startValue, 2^32);
end

function [effective, csvDelta, reasons] = resolveDroppedDelta( ...
        summary, summaryValid, field, csvDelta, csvCorrupt, prefix, limits)
reasons = strings(0, 1);
if csvCorrupt
    reasons = addReasons(reasons, prefix + "_DROPPED_CORRUPT");
end
if summaryValid
    candidate = summary.(field);
    effective = double(candidate);
    if isfinite(csvDelta) && effective ~= csvDelta
        reasons = addReasons(reasons, prefix + "_DROPPED_MISMATCH");
    end
else
    effective = NaN;
end
if isfinite(effective) && effective > limits.max_dropped_delta
    reasons = addReasons(reasons, prefix + "_DROPPED");
end
end

function [rmsValues, fullBlockCount] = robustRms(timeSeconds, channels)
rmsValues = NaN(1, size(channels, 2));
fullBlockCount = 0;
if numel(timeSeconds) < 2 || any(diff(timeSeconds) <= 0)
    return;
end
expectedRateHz = 500;
minimumSamples = floor(0.95 * expectedRateHz);
minimumCoverageSeconds = 0.98;
maximumGapSeconds = 1.5 / expectedRateHz;
lastCandidate = floor(max(timeSeconds));
blockRms = zeros(0, size(channels, 2));
for blockStart = 0:lastCandidate
    inBlock = timeSeconds >= blockStart & ...
        timeSeconds < blockStart + 1;
    if ~any(inBlock)
        continue;
    end
    blockTimes = timeSeconds(inBlock);
    dense = numel(blockTimes) >= minimumSamples && ...
        blockTimes(end) - blockTimes(1) >= minimumCoverageSeconds && ...
        all(diff(blockTimes) <= maximumGapSeconds + 10 * eps(1));
    if dense
        currentRms = stableRms(channels(inBlock, :));
        if any(~isfinite(currentRms))
            rmsValues(:) = NaN;
            fullBlockCount = 0;
            return;
        end
        blockRms(end + 1, :) = currentRms; %#ok<AGROW>
    end
end
fullBlockCount = size(blockRms, 1);
if fullBlockCount > 0
    rmsValues = median(blockRms, 1);
end
end

function values = stableRms(channels)
scale = max(abs(channels), [], 1);
values = zeros(1, size(channels, 2));
nonzero = scale > 0;
if any(nonzero)
    normalized = channels(:, nonzero) ./ scale(nonzero);
    values(nonzero) = scale(nonzero) .* ...
        sqrt(mean(normalized.^2, 1));
end
end

function [intervals, invalidIds, corrupt] = ...
        validRestIntervals(events, targetLabels)
invalidIds = unique(events.trial_id(events.event_type == "trial_invalid"));
eligible = events.valid & ~ismember(events.trial_id, invalidIds);
restRows = find(eligible & ismember(events.event_type, ...
    ["rest_valid_start", "rest_valid_end"]));
targetRows = find(eligible & ismember(events.event_type, ...
    ["target_hold_start", "target_hold_end"]));
intervals = zeros(0, 2);
corrupt = false;
if isempty(restRows)
    return;
end
keys = string(events.trial_id(restRows)) + char(31) + ...
    events.gesture_id(restRows);
groupKeys = unique(keys, "stable");
for key = reshape(groupKeys, 1, [])
    rows = restRows(keys == key);
    [~, order] = sort(events.session_time_s(rows), "ascend");
    rows = rows(order);
    types = events.event_type(rows);
    trialId = events.trial_id(rows(1));
    restGesture = events.gesture_id(rows(1));
    targetGestures = unique(events.gesture_id(targetRows( ...
        events.trial_id(targetRows) == trialId)));
    identityValid = trialId > 0 && trialId == fix(trialId) && ...
        ismember(restGesture, targetLabels) && ...
        isscalar(targetGestures) && ...
        ismember(targetGestures, targetLabels) && ...
        targetGestures == restGesture;
    groupValid = identityValid && mod(numel(rows), 2) == 0 && ...
        all(types(1:2:end) == "rest_valid_start") && ...
        all(types(2:2:end) == "rest_valid_end") && ...
        all(events.session_time_s(rows(1:2:end)) < ...
        events.session_time_s(rows(2:2:end)));
    if ~groupValid
        corrupt = true;
        continue;
    end
    intervals = [intervals; ...
        events.session_time_s(rows(1:2:end)), ...
        events.session_time_s(rows(2:2:end))]; %#ok<AGROW>
end
end

function [counts, startCounts, endCounts, byGesture, invalidStructure, ...
        targetEventsCorrupt, holdDurationInvalid] = validTrialCounts( ...
        events, invalidIds, targetLabels, targetHoldDuration, ...
        durationTolerance)
counts = zeros(1, numel(targetLabels));
startCounts = zeros(1, numel(targetLabels));
endCounts = zeros(1, numel(targetLabels));
byGesture = struct();
holdEvent = ismember(events.event_type, ...
    ["target_hold_start", "target_hold_end"]) & ...
    ~ismember(events.trial_id, invalidIds);
invalidStructure = any(holdEvent & (~events.valid | ...
    events.trial_id <= 0 | ~ismember(events.gesture_id, targetLabels)));
holdDurationInvalid = false;
badTrialIds = zeros(0, 1);
for trialId = reshape(unique(events.trial_id(holdEvent)), 1, [])
    gestures = unique(events.gesture_id(holdEvent & ...
        events.trial_id == trialId));
    if numel(gestures) ~= 1 || ~ismember(gestures, targetLabels)
        badTrialIds(end + 1, 1) = trialId; %#ok<AGROW>
    end
end
targetEventsCorrupt = ~isempty(badTrialIds);
invalidStructure = invalidStructure || targetEventsCorrupt;
eligibleForCounts = events.valid & ...
    ~ismember(events.trial_id, invalidIds) & ...
    ~ismember(events.trial_id, badTrialIds);
eligibleForRawCounts = events.valid & ...
    ~ismember(events.trial_id, invalidIds);
for index = 1:numel(targetLabels)
    label = targetLabels(index);
    isGesture = events.gesture_id == label;
    starts = eligibleForCounts & isGesture & ...
        events.event_type == "target_hold_start";
    ends = eligibleForCounts & isGesture & ...
        events.event_type == "target_hold_end";
    startCounts(index) = sum(eligibleForRawCounts & isGesture & ...
        events.event_type == "target_hold_start");
    endCounts(index) = sum(eligibleForRawCounts & isGesture & ...
        events.event_type == "target_hold_end");
    trialIds = unique(events.trial_id(starts | ends));
    for trialId = reshape(trialIds, 1, [])
        startNumber = sum(starts & events.trial_id == trialId);
        endNumber = sum(ends & events.trial_id == trialId);
        startRow = find(starts & events.trial_id == trialId);
        endRow = find(ends & events.trial_id == trialId);
        ordered = startNumber == 1 && endNumber == 1 && ...
            events.session_time_s(startRow) < events.session_time_s(endRow);
        durationValid = ordered && abs( ...
            events.session_time_s(endRow) - ...
            events.session_time_s(startRow) - targetHoldDuration) <= ...
            durationTolerance;
        if ordered && ~durationValid
            holdDurationInvalid = true;
        end
        if durationValid
            counts(index) = counts(index) + 1;
        else
            invalidStructure = true;
        end
    end
    byGesture.(matlab.lang.makeValidName(char(label))) = counts(index);
end
end

function mask = intervalMask(timeSeconds, intervals)
mask = false(size(timeSeconds));
for index = 1:size(intervals, 1)
    mask = mask | (timeSeconds >= intervals(index, 1) & ...
        timeSeconds <= intervals(index, 2));
end
end

function count = rowCount(data)
count = size(data, 1);
end

function column = columnOrEmpty(data, index)
if isempty(data)
    column = zeros(0, 1);
else
    column = data(:, index);
end
end

function reasons = addReasons(reasons, additions)
additions = reshape(string(additions), [], 1);
additions = additions(~ismissing(additions) & strlength(additions) > 0);
for addition = additions.'
    if ~any(reasons == addition)
        reasons(end + 1, 1) = addition; %#ok<AGROW>
    end
end
end

function writeQualityAtomically(path, quality)
directory = string(fileparts(path));
temporaryPath = string(tempname(directory)) + ".tmp";
cleanup = onCleanup(@() deleteIfPresent(temporaryPath));
try
    fileId = fopen(temporaryPath, "w", "n", "UTF-8");
    if fileId < 0
        error("gesture_session_quality:WriteFailed", ...
            "Could not create the temporary quality file.");
    end
    fileCleanup = onCleanup(@() closeFileSafely(fileId));
    payload = jsonencode(quality, PrettyPrint=true);
    fprintf(fileId, "%s\n", payload);
    fclose(fileId);
    delete(fileCleanup);
    [moved, message] = movefile(temporaryPath, path, "f");
    if ~moved
        error("gesture_session_quality:WriteFailed", "%s", message);
    end
    delete(cleanup);
catch exception
    if exception.identifier == "gesture_session_quality:WriteFailed"
        rethrow(exception);
    end
    error("gesture_session_quality:WriteFailed", ...
        "Could not atomically write quality JSON: %s", exception.message);
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
