function [recordType, rows, packetSequence] = ...
        ads1298_ble_decode_packet(packet)
%ADS1298_BLE_DECODE_PACKET Decode one SensorBiShe BLE notification.
%   Protocol v2 preserves every sensor value while storing sequence,
%   timestamp, status, and drop counters once per packet. Per-sample
%   sequence numbers and timestamps are reconstructed at the configured
%   500 Hz EMG and 104 Hz IMU rates.

if ~(isnumeric(packet) || islogical(packet)) || ~isreal(packet) || ...
        ~isvector(packet) || isempty(packet)
    invalidPacket("Packet must be a nonempty real numeric byte vector.");
end

numericPacket = double(packet(:).');
if any(~isfinite(numericPacket)) || any(numericPacket ~= fix(numericPacket)) || ...
        any(numericPacket < 0 | numericPacket > 255)
    invalidPacket("Packet elements must be integer bytes from 0 through 255.");
end
bytes = uint8(numericPacket);

headerBytes = 8;
if numel(bytes) < headerBytes
    invalidPacket("Packet is shorter than the 8-byte header.");
end
if bytes(1) ~= hex2dec("A5") || bytes(2) ~= hex2dec("5A")
    invalidPacket("Packet magic is invalid.");
end
if bytes(3) ~= 2
    invalidPacket("Packet protocol version is not supported.");
end

recordCode = bytes(4);
recordCount = double(bytes(5));
sampleBytes = double(bytes(6));
packetSequence = decodeU16(bytes, 7);
if recordCount < 1
    invalidPacket("Packet must contain at least one sample.");
end

if recordCode == 1
    recordType = "emg";
    expectedSampleBytes = 24;
    metadataBytes = 15;
    columnCount = 12;
elseif recordCode == 2
    recordType = "imu";
    expectedSampleBytes = 14;
    metadataBytes = 12;
    columnCount = 10;
else
    invalidPacket("Packet record type is not supported.");
end
if sampleBytes ~= expectedSampleBytes
    invalidPacket("Packet sample size does not match its record type.");
end
if numel(bytes) ~= headerBytes + metadataBytes + ...
        recordCount * sampleBytes
    invalidPacket("Packet length does not match its count and sample size.");
end

metadataOffset = headerBytes + 1;
baseSequence = decodeU32(bytes, metadataOffset);
baseTimestamp = decodeU32(bytes, metadataOffset + 4);
rows = zeros(recordCount, columnCount);

if recordCode == 1
    status = decodeU24(bytes, metadataOffset + 8);
    dropped = decodeU32(bytes, metadataOffset + 11);
    samplePeriodUs = 1e6 / 500;
    payloadOffset = metadataOffset + metadataBytes;
    for recordIndex = 1:recordCount
        zeroBasedIndex = recordIndex - 1;
        rows(recordIndex, 1) = mod(baseSequence + zeroBasedIndex, 2^32);
        rows(recordIndex, 2) = mod(baseTimestamp + ...
            round(zeroBasedIndex * samplePeriodUs), 2^32);
        rows(recordIndex, 3) = status;
        sampleOffset = payloadOffset + zeroBasedIndex * sampleBytes;
        for channel = 1:8
            rows(recordIndex, 3 + channel) = decodeI24(bytes, ...
                sampleOffset + (channel - 1) * 3);
        end
        rows(recordIndex, 12) = dropped;
    end
else
    dropped = decodeU32(bytes, metadataOffset + 8);
    samplePeriodUs = 1e6 / 104;
    payloadOffset = metadataOffset + metadataBytes;
    for recordIndex = 1:recordCount
        zeroBasedIndex = recordIndex - 1;
        rows(recordIndex, 1) = mod(baseSequence + zeroBasedIndex, 2^32);
        rows(recordIndex, 2) = mod(baseTimestamp + ...
            round(zeroBasedIndex * samplePeriodUs), 2^32);
        sampleOffset = payloadOffset + zeroBasedIndex * sampleBytes;
        for valueIndex = 1:7
            rows(recordIndex, 2 + valueIndex) = decodeI16(bytes, ...
                sampleOffset + (valueIndex - 1) * 2);
        end
        rows(recordIndex, 10) = dropped;
    end
end
end

function value = decodeU16(bytes, offset)
value = double(bytes(offset)) + double(bytes(offset + 1)) * 2^8;
end

function value = decodeI16(bytes, offset)
value = decodeU16(bytes, offset);
if value >= 2^15
    value = value - 2^16;
end
end

function value = decodeU24(bytes, offset)
value = double(bytes(offset)) + double(bytes(offset + 1)) * 2^8 + ...
    double(bytes(offset + 2)) * 2^16;
end

function value = decodeI24(bytes, offset)
value = decodeU24(bytes, offset);
if value >= 2^23
    value = value - 2^24;
end
end

function value = decodeU32(bytes, offset)
value = double(bytes(offset)) + double(bytes(offset + 1)) * 2^8 + ...
    double(bytes(offset + 2)) * 2^16 + ...
    double(bytes(offset + 3)) * 2^24;
end

function invalidPacket(messageText)
error("ads1298_ble_decode_packet:InvalidPacket", "%s", messageText);
end
