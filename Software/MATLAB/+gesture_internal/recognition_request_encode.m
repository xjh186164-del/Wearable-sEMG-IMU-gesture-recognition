function frame = recognition_request_encode(snapshot)
%RECOGNITION_REQUEST_ENCODE Encode one exact protocol-v1 GWIN request.

identifier = "gesture_internal:recognition_request_encode:InvalidSnapshot";
required = ["sequence", "sensor_time_s", "emg_mV", "imu"];
if ~isstruct(snapshot) || ~isscalar(snapshot) || ...
        ~isequal(sort(string(fieldnames(snapshot))), sort(required.'))
    error(identifier, "snapshot must contain only the required fields.");
end
if ~isa(snapshot.sequence, "uint64") || ~isscalar(snapshot.sequence)
    error(identifier, "sequence must be a uint64 scalar.");
end
sensorTime = snapshot.sensor_time_s;
if ~isnumeric(sensorTime) || ~isscalar(sensorTime) || ...
        ~isreal(sensorTime) || ~isfinite(sensorTime)
    error(identifier, "sensor_time_s must be a finite real scalar.");
end
emg = validateSensorArray(snapshot.emg_mV, [8, 250], "emg_mV", ...
    identifier);
imu = validateSensorArray(snapshot.imu, [6, 52], "imu", identifier);

frame = [uint8('GWIN'), ...
    littleEndianBytes(uint16(1)), ...
    littleEndianBytes(snapshot.sequence), ...
    littleEndianBytes(double(sensorTime)), ...
    littleEndianBytes(emg(:)), ...
    littleEndianBytes(imu(:))];
frame = reshape(uint8(frame), 1, []);
if numel(frame) ~= 9270
    error(identifier, "encoded request has the wrong byte count.");
end
end

function values = validateSensorArray(value, expectedSize, name, identifier)
if ~isnumeric(value) || ~isreal(value) || issparse(value) || ...
        ~isequal(size(value), expectedSize) || ...
        ~all(isfinite(value), "all")
    error(identifier, "%s must be a finite real full array of size %s.", ...
        name, mat2str(expectedSize));
end
values = single(value);
if ~all(isfinite(values), "all")
    error(identifier, "%s cannot be represented as finite float32.", name);
end
end

function bytes = littleEndianBytes(value)
[~, ~, endian] = computer;
if endian == 'B'
    value = swapbytes(value);
elseif endian ~= 'L'
    error("gesture_internal:recognition_request_encode:UnsupportedEndian", ...
        "The host byte order is not recognized.");
end
bytes = reshape(typecast(value(:), "uint8"), 1, []);
end
