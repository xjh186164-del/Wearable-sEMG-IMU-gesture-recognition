function latestTime = latest_shared_time(emgTime, imuTime)
%LATEST_SHARED_TIME Latest session time covered by both sensor streams.

if isempty(emgTime) || isempty(imuTime)
    latestTime = [];
    return;
end

latestTime = min(double(emgTime(end)), double(imuTime(end)));
end
