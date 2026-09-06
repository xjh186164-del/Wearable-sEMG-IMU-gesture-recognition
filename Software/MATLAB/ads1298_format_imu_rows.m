function payload = ads1298_format_imu_rows(rows)
%ADS1298_FORMAT_IMU_ROWS Format converted LSM6DSOX CSV rows.

if ~isnumeric(rows) || ~isreal(rows) || ~ismatrix(rows) || ...
        size(rows, 2) ~= 10 || any(~isfinite(rows), "all")
    error("ads1298_format_imu_rows:InvalidInput", ...
        "rows must be a finite real numeric matrix with 10 columns.");
end

if isempty(rows)
    payload = '';
    return;
end

rows = double(rows);
format = ['%.0f,%.0f', repmat(',%.6f', 1, 7), ',%.0f\n'];
payload = sprintf(format, rows.');
end
