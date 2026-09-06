function [recordType, values] = ads1298_parse_line(line)
%ADS1298_PARSE_LINE Classify and parse one ESP32 ADS1298 CSV record.

line = strtrim(string(line));
values = zeros(1, 0);

if ~isscalar(line)
    recordType = "invalid";
    return;
end

if strlength(line) == 0 || startsWith(line, "#") || startsWith(line, "sample,")
    recordType = "ignored";
    return;
end

fields = split(line, ",");
if startsWith(line, "I,")
    if numel(fields) ~= 11
        recordType = "invalid";
        return;
    end

    parsedValues = str2double(fields(2:end)).';
    if any(~isfinite(parsedValues))
        recordType = "invalid";
        return;
    end

    recordType = "imu";
    values = parsedValues;
    return;
elseif startsWith(line, "I")
    recordType = "invalid";
    return;
end

if numel(fields) ~= 12
    recordType = "invalid";
    return;
end

parsedValues = str2double(fields).';
if any(~isfinite(parsedValues))
    recordType = "invalid";
    return;
end

recordType = "valid";
values = parsedValues;
end
