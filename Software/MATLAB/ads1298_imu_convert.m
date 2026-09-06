function converted = ads1298_imu_convert(raw)
%ADS1298_IMU_CONVERT Convert LSM6DSOX raw rows to physical units.

if ~isnumeric(raw) || ~isreal(raw) || ~ismatrix(raw) || ...
        size(raw, 2) ~= 10 || any(~isfinite(raw), "all")
    error("ads1298_imu_convert:InvalidInput", ...
        "raw must be a finite real numeric matrix with 10 columns.");
end

raw = double(raw);
accelerationGPerCount = 0.122 / 1000;
angularRateDpsPerCount = 17.50 / 1000;

converted = [raw(:, 1:2), ...
    raw(:, 3:5) * accelerationGPerCount, ...
    raw(:, 6:8) * angularRateDpsPerCount, ...
    raw(:, 9) / 256 + 25, raw(:, 10)];
end
