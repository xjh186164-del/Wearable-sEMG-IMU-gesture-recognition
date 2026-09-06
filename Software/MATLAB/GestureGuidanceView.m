classdef GestureGuidanceView < handle
    properties (Access = private)
        Catalog
        Figure
        Axes
        ImageHandle
        PoseLabel
        InstructionLabel
        ProgressLabel
        CountdownLabel
        Reader
        CurrentGesture = ""
        InvalidateRequestedFlag = false
        StopRequestedFlag = false
    end

    methods
        function obj = GestureGuidanceView(catalog, visible)
            if ~istable(catalog) || ~isequal( ...
                    string(catalog.Properties.VariableNames), ...
                    ["gesture_id", "path", "duration_s", "width", "height"])
                error("GestureGuidanceView:InvalidCatalog", ...
                    "catalog must be a validated gesture media catalog.");
            end
            if ~islogical(visible) || ~isscalar(visible)
                error("GestureGuidanceView:InvalidVisible", ...
                    "visible must be a logical scalar.");
            end

            obj.Catalog = catalog;
            visibility = "off";
            if visible
                visibility = "on";
            end

            obj.Figure = uifigure( ...
                "Name", "Gesture Acquisition Guidance", ...
                "Visible", visibility, ...
                "Position", [100, 100, 1000, 720], ...
                "CloseRequestFcn", @(source, event) ...
                    obj.onWindowClose(source, event));
            layout = uigridlayout(obj.Figure, [5, 2]);
            layout.RowHeight = {"1x", 45, 55, 35, 48};
            layout.ColumnWidth = {"1x", "1x"};
            layout.Padding = [12, 12, 12, 12];
            layout.RowSpacing = 8;

            obj.Axes = uiaxes(layout);
            obj.Axes.Layout.Row = 1;
            obj.Axes.Layout.Column = [1, 2];
            obj.Axes.Visible = "off";
            obj.ImageHandle = image(obj.Axes, ...
                zeros(720, 1280, 3, "uint8"));
            axis(obj.Axes, "image");

            obj.PoseLabel = uilabel(layout, ...
                "Text", "Ready to Start", ...
                "FontSize", 26, ...
                "FontWeight", "bold", ...
                "HorizontalAlignment", "center");
            obj.PoseLabel.Layout.Row = 2;
            obj.PoseLabel.Layout.Column = [1, 2];

            obj.InstructionLabel = uilabel(layout, ...
                "Text", "Forearm horizontal, hand relaxed", ...
                "FontSize", 20, ...
                "HorizontalAlignment", "center");
            obj.InstructionLabel.Layout.Row = 3;
            obj.InstructionLabel.Layout.Column = [1, 2];

            obj.ProgressLabel = uilabel(layout, ...
                "Text", "", ...
                "FontSize", 16, ...
                "HorizontalAlignment", "left");
            obj.ProgressLabel.Layout.Row = 4;
            obj.ProgressLabel.Layout.Column = 1;

            obj.CountdownLabel = uilabel(layout, ...
                "Text", "", ...
                "FontSize", 20, ...
                "FontWeight", "bold", ...
                "HorizontalAlignment", "right");
            obj.CountdownLabel.Layout.Row = 4;
            obj.CountdownLabel.Layout.Column = 2;

            invalidateButton = uibutton(layout, "push", ...
                "Text", "Mark Trial Invalid and Repeat", ...
                "FontSize", 17, ...
                "ButtonPushedFcn", @(source, event) ...
                    obj.onInvalidate(source, event));
            invalidateButton.Layout.Row = 5;
            invalidateButton.Layout.Column = 1;

            stopButton = uibutton(layout, "push", ...
                "Text", "Safe Stop", ...
                "FontSize", 17, ...
                "ButtonPushedFcn", @(source, event) ...
                    obj.onStop(source, event));
            stopButton.Layout.Row = 5;
            stopButton.Layout.Column = 2;
        end

        function update(obj, snapshot)
            if isempty(obj.Figure) || ~isvalid(obj.Figure)
                return;
            end

            stageName = string(snapshot.stage_name);
            if snapshot.finished
                stageName = "finished";
                instruction = "Acquisition Complete";
            elseif snapshot.stopped
                stageName = "stopped";
                instruction = "Acquisition Stopped";
            else
                instruction = obj.instructionForStage(stageName);
            end

            [gestureId, clipTime] = ...
                gesture_internal.guidance_video_route(stageName, ...
                string(snapshot.gesture_id), ...
                double(snapshot.stage_elapsed_s));

            obj.PoseLabel.Text = obj.poseName(gestureId);
            obj.InstructionLabel.Text = instruction;
            if double(snapshot.block_id) > 0
                obj.ProgressLabel.Text = "Block " + ...
                    string(snapshot.block_id) + " · " + ...
                    string(snapshot.progress_text);
            else
                obj.ProgressLabel.Text = string(snapshot.progress_text);
            end
            obj.CountdownLabel.Text = obj.countdownForStage(stageName, ...
                double(snapshot.stage_remaining_s));

            obj.renderFrame(gestureId, clipTime);
            drawnow limitrate;
        end

        function requested = consumeInvalidateRequested(obj)
            requested = obj.InvalidateRequestedFlag;
            obj.InvalidateRequestedFlag = false;
        end

        function requested = stopRequested(obj)
            requested = obj.StopRequestedFlag;
        end

        function close(obj)
            obj.StopRequestedFlag = true;
            if ~isempty(obj.Figure) && isvalid(obj.Figure)
                delete(obj.Figure);
            end
            obj.Figure = [];
            obj.Reader = [];
            obj.CurrentGesture = "";
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function renderFrame(obj, gestureId, clipTime)
            if gestureId ~= obj.CurrentGesture || isempty(obj.Reader)
                row = find(obj.Catalog.gesture_id == gestureId, 1);
                if isempty(row)
                    error("GestureGuidanceView:UnknownGesture", ...
                        "Gesture is absent from the media catalog: %s", ...
                        gestureId);
                end
                obj.Reader = VideoReader(char(obj.Catalog.path(row)));
                obj.CurrentGesture = gestureId;
            end

            clipTime = max(0, double(clipTime));
            framePeriod = 1 / max(1, double(obj.Reader.FrameRate));
            latestTime = max(0, double(obj.Reader.Duration) - framePeriod);
            clipTime = min(clipTime, latestTime);
            obj.Reader.CurrentTime = clipTime;
            frame = readFrame(obj.Reader);
            obj.ImageHandle.CData = frame;
        end

        function onInvalidate(obj, ~, ~)
            obj.InvalidateRequestedFlag = true;
        end

        function onStop(obj, ~, ~)
            obj.StopRequestedFlag = true;
        end

        function onWindowClose(obj, source, ~)
            obj.StopRequestedFlag = true;
            if isvalid(source)
                delete(source);
            end
            obj.Figure = [];
            obj.Reader = [];
            obj.CurrentGesture = "";
        end
    end

    methods (Static, Access = private)
        function instruction = instructionForStage(stageName)
            instructionByStage = dictionary( ...
                ["baseline", "prepare", "movement", "target_hold", ...
                 "return", "recovery", "break"], ...
                ["Hold the relaxed pose still", ...
                 "Watch the next gesture and get ready", ...
                 "Move to the target pose now", ...
                 "Hold the target pose still", ...
                 "Return to the relaxed forearm position", ...
                 "Hold the relaxed pose still", ...
                 "Rest and keep the device in position"]);
            if isKey(instructionByStage, stageName)
                instruction = instructionByStage(stageName);
            else
                instruction = "Waiting for acquisition";
            end
        end

        function countdown = countdownForStage(stageName, remainingSeconds)
            if ismember(stageName, ["finished", "stopped"])
                countdown = "";
                return;
            end
            stageTextByStage = dictionary( ...
                ["baseline", "prepare", "movement", "target_hold", ...
                 "return", "recovery", "break"], ...
                ["RELAXED HOLD", "PREPARE", "MOVE", "HOLD STILL", ...
                 "RETURN TO REST", "RELAXED HOLD", "BREAK"]);
            if ~isKey(stageTextByStage, stageName)
                countdown = "";
                return;
            end
            countdown = stageTextByStage(stageName) + " · " + ...
                sprintf("Remaining %.1f s", max(0, remainingSeconds));
        end

        function name = poseName(gestureId)
            nameByGesture = dictionary( ...
                ["REST", "WRIST_UP", "WRIST_DOWN", "FOREARM_IN", ...
                 "FOREARM_OUT", "ARM_UP", "ARM_DOWN", "FIST"], ...
                ["Relaxed Forearm", "Wrist Up", "Wrist Down", ...
                 "Forearm Inward Rotation", ...
                 "Forearm Outward Rotation", "Arm Up", "Arm Down", ...
                 "Make a Fist"]);
            if isKey(nameByGesture, gestureId)
                name = nameByGesture(gestureId);
            else
                name = gestureId;
            end
        end
    end
end
