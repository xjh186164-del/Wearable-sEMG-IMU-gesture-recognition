function catalog = gesture_media_catalog(mediaDirectory, classLabels)
if ~(ischar(mediaDirectory) || ...
        (isstring(mediaDirectory) && isscalar(mediaDirectory)))
    error("gesture_media_catalog:InvalidInput", ...
        "mediaDirectory must be a character vector or string scalar.");
end
if ~(isstring(classLabels) || ischar(classLabels) || iscellstr(classLabels))
    error("gesture_media_catalog:InvalidInput", ...
        "classLabels must contain eight text labels.");
end

classLabels = string(classLabels);
classLabels = classLabels(:);
canonicalLabels = ["REST"; "WRIST_UP"; "WRIST_DOWN"; ...
    "FOREARM_IN"; "FOREARM_OUT"; "ARM_UP"; "ARM_DOWN"; "FIST"];
expectedCount = numel(canonicalLabels);
if numel(classLabels) ~= expectedCount || any(ismissing(classLabels)) || ...
        any(strlength(classLabels) == 0) || ...
        numel(unique(classLabels)) ~= expectedCount
    error("gesture_media_catalog:InvalidInput", ...
        "classLabels must contain eight unique, nonempty labels.");
end
if ~isequal(classLabels, canonicalLabels)
    error("gesture_media_catalog:InvalidInput", ...
        "classLabels must use the canonical eight_pose_v5 order.");
end

mediaDirectory = string(mediaDirectory);
expectedNames = canonicalLabels + ".mp4";
if ~isfolder(mediaDirectory)
    error("gesture_media_catalog:MissingMedia", ...
        "Media directory does not exist: %s", mediaDirectory);
end
listing = dir(mediaDirectory);
listing = listing(~[listing.isdir]);
actualNames = string({listing.name}).';
mp4Names = actualNames(endsWith(lower(actualNames), ".mp4"));
missingNames = setdiff(expectedNames, mp4Names, "stable");
if ~isempty(missingNames)
    error("gesture_media_catalog:MissingMedia", ...
        "Required media file is missing: %s", missingNames(1));
end
extraNames = setdiff(mp4Names, expectedNames, "stable");
if ~isempty(extraNames) || numel(mp4Names) ~= expectedCount
    error("gesture_media_catalog:InvalidMedia", ...
        "Media directory must contain exactly the eight canonical MP4 files.");
end

paths = strings(expectedCount, 1);
durations = zeros(expectedCount, 1);
widths = zeros(expectedCount, 1);
heights = zeros(expectedCount, 1);
for index = 1:numel(classLabels)
    paths(index) = string(fullfile(mediaDirectory, ...
        classLabels(index) + ".mp4"));
    if ~isfile(paths(index))
        error("gesture_media_catalog:MissingMedia", ...
            "Required media file is missing: %s", paths(index));
    end


    sampleEntries = mp4VideoSampleEntries(paths(index));
    if ~isempty(sampleEntries) && ...
            any(~ismember(sampleEntries, ["avc1", "avc3"]))
        error("gesture_media_catalog:InvalidMedia", ...
            "Media video codec must use an H.264/AVC avc1 or avc3 " + ...
            "sample entry: %s", paths(index));
    end

    try
        reader = VideoReader(char(paths(index)));
        durations(index) = double(reader.Duration);
        widths(index) = double(reader.Width);
        heights(index) = double(reader.Height);
        frameRate = double(reader.FrameRate);
    catch cause
        error("gesture_media_catalog:UnreadableMedia", ...
            "Cannot read media file %s: %s", paths(index), cause.message);
    end

    if isempty(sampleEntries)
        error("gesture_media_catalog:InvalidMedia", ...
            "Media must expose an H.264/AVC avc1 or avc3 sample entry: %s", ...
            paths(index));
    end
    validFrameRate = isfinite(frameRate) && ...
        (abs(frameRate - 25) <= 0.05 || abs(frameRate - 30) <= 0.05);
    minimumDuration = 2.5;
    if classLabels(index) == "REST"
        minimumDuration = 2.0;
    end
    if ~isfinite(durations(index)) || ...
            durations(index) + 1e-6 < minimumDuration || ...
            ~validFrameRate || widths(index) < 1280 || ...
            heights(index) < 720 || widths(index) <= heights(index)
        error("gesture_media_catalog:InvalidMedia", ...
            "Media must be H.264, landscape, at least 1280-by-720, " + ...
            "25 or 30 fps (including 29.97), and at least %.1f " + ...
            "seconds long: %s", minimumDuration, paths(index));
    end
end

catalog = table(classLabels, paths, durations, widths, heights, ...
    'VariableNames', ...
    {'gesture_id', 'path', 'duration_s', 'width', 'height'});
end

function sampleEntries = mp4VideoSampleEntries(path)
sampleEntries = strings(0, 1);
fileId = fopen(path, "r");
if fileId < 0
    return;
end
cleanup = onCleanup(@() fclose(fileId));
bytes = fread(fileId, Inf, "*uint8");
if numel(bytes) < 8
    return;
end

topLevel = parseMp4Boxes(bytes, 1, numel(bytes));
moovBoxes = boxesOfType(topLevel, "moov");
for moov = reshape(moovBoxes, 1, [])
    tracks = boxesOfType(parseMp4Boxes(bytes, ...
        moov.dataStart, moov.endIndex), "trak");
    for track = reshape(tracks, 1, [])
        mediaBoxes = boxesOfType(parseMp4Boxes(bytes, ...
            track.dataStart, track.endIndex), "mdia");
        for media = reshape(mediaBoxes, 1, [])
            mediaChildren = parseMp4Boxes(bytes, ...
                media.dataStart, media.endIndex);
            handlers = boxesOfType(mediaChildren, "hdlr");
            if isempty(handlers) || ...
                    handlers(1).dataStart + 11 > handlers(1).endIndex
                continue;
            end
            handlerType = fourCc(bytes( ...
                handlers(1).dataStart + 8:handlers(1).dataStart + 11));
            if handlerType ~= "vide"
                continue;
            end
            mediaInfo = boxesOfType(mediaChildren, "minf");
            for info = reshape(mediaInfo, 1, [])
                sampleTables = boxesOfType(parseMp4Boxes(bytes, ...
                    info.dataStart, info.endIndex), "stbl");
                for sampleTable = reshape(sampleTables, 1, [])
                    descriptions = boxesOfType(parseMp4Boxes(bytes, ...
                        sampleTable.dataStart, sampleTable.endIndex), ...
                        "stsd");
                    for description = reshape(descriptions, 1, [])
                        entryStart = description.dataStart + 8;
                        entries = parseMp4Boxes(bytes, entryStart, ...
                            description.endIndex);
                        sampleEntries = [sampleEntries; ...
                            string({entries.type}).']; %#ok<AGROW>
                    end
                end
            end
        end
    end
end
sampleEntries = unique(sampleEntries, "stable");
end

function selected = boxesOfType(boxes, type)
if isempty(boxes)
    selected = boxes;
else
    selected = boxes(string({boxes.type}) == string(type));
end
end

function boxes = parseMp4Boxes(bytes, firstIndex, lastIndex)
template = struct("type", "", "dataStart", 0, "endIndex", 0);
boxes = repmat(template, 0, 1);
position = double(firstIndex);
lastIndex = double(lastIndex);
while position + 7 <= lastIndex
    boxSize = readBigEndian(bytes(position:position + 3));
    boxType = fourCc(bytes(position + 4:position + 7));
    headerSize = 8;
    if boxSize == 1
        if position + 15 > lastIndex
            return;
        end
        boxSize = readBigEndian(bytes(position + 8:position + 15));
        headerSize = 16;
    elseif boxSize == 0
        boxSize = lastIndex - position + 1;
    end
    boxEnd = position + boxSize - 1;
    if boxSize < headerSize || ~isfinite(boxEnd) || ...
            boxEnd > lastIndex
        return;
    end
    box = struct( ...
        "type", boxType, ...
        "dataStart", position + headerSize, ...
        "endIndex", boxEnd);
    boxes(end + 1, 1) = box; %#ok<AGROW>
    position = boxEnd + 1;
end
end

function value = readBigEndian(bytes)
value = 0;
for index = 1:numel(bytes)
    value = value * 256 + double(bytes(index));
end
end

function value = fourCc(bytes)
value = string(char(reshape(bytes, 1, [])));
end
