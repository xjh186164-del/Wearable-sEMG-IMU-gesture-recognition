function payload = ads1298_format_filtered_rows(rawValues, filteredMillivolts)
%ADS1298_FORMAT_FILTERED_ROWS Format aligned filtered ADS1298 CSV rows.

if ~isnumeric(rawValues) || ~isreal(rawValues) || ~ismatrix(rawValues) || ...
        size(rawValues, 2) ~= 12 || any(~isfinite(rawValues), "all")
    error("ads1298_format_filtered_rows:InvalidRawValues", ...
        "rawValues must be a finite real N-by-12 matrix.");
end
if ~isnumeric(filteredMillivolts) || ~isreal(filteredMillivolts) || ...
        ~ismatrix(filteredMillivolts) || ...
        size(filteredMillivolts, 2) ~= 8 || ...
        any(~isfinite(filteredMillivolts), "all")
    error("ads1298_format_filtered_rows:InvalidFilteredValues", ...
        "filteredMillivolts must be a finite real N-by-8 matrix.");
end
if size(rawValues, 1) ~= size(filteredMillivolts, 1)
    error("ads1298_format_filtered_rows:RowCountMismatch", ...
        "Raw and filtered row counts must match.");
end

if isempty(rawValues)
    payload = '';
    return;
end

outputRows = [double(rawValues(:, 1:3)), ...
    double(filteredMillivolts), double(rawValues(:, 12))];
format = ["%.0f,%.0f,%.0f", repmat(",%.9f", 1, 8), ",%.0f\n"];
payload = sprintf(char(format), outputRows.');
end
