classdef TcpGestureRecognitionTransport < handle
    %TCPGESTURERECOGNITIONTRANSPORT Buffered newline transport over tcpclient.

    properties (Access = private)
        Adapter = []
        Buffer = zeros(1, 0, "uint8")
        ClockFunction function_handle
        PauseFunction function_handle
        PollPeriodSeconds double = 0.01
        MaximumRecordBytes double = 65536
        Closed logical = false
    end

    methods
        function obj = TcpGestureRecognitionTransport(host, port, options)
            if nargin < 3
                options = struct();
            end
            host = validateHost(host);
            port = validatePort(port);
            options = normalizeOptions(options);
            obj.ClockFunction = options.ClockFunction;
            obj.PauseFunction = options.PauseFunction;
            obj.PollPeriodSeconds = options.PollPeriodSeconds;
            obj.MaximumRecordBytes = options.MaximumRecordBytes;
            try
                adapter = options.AdapterFactory(host, port, ...
                    options.ConnectTimeoutSeconds, ...
                    options.OperationTimeoutSeconds);
            catch cause
                error("TcpGestureRecognitionTransport:ConnectFailed", ...
                    "Could not connect the TCP adapter: %s", cause.message);
            end
            if isempty(adapter) || ~isscalar(adapter) || ...
                    ~isprop(adapter, "NumBytesAvailable") || ...
                    ~isprop(adapter, "Timeout")
                error("TcpGestureRecognitionTransport:InvalidAdapter", ...
                    "AdapterFactory must return one tcpclient-like adapter.");
            end
            obj.Adapter = adapter;
            try
                obj.Adapter.Timeout = options.OperationTimeoutSeconds;
                configureTerminator(obj.Adapter, "LF");
            catch cause
                obj.close();
                error("TcpGestureRecognitionTransport:AdapterFailure", ...
                    "Could not configure the TCP adapter: %s", cause.message);
            end
        end

        function writeBytes(obj, bytes)
            obj.requireOpen();
            if ~isa(bytes, "uint8") || ~isvector(bytes)
                error("TcpGestureRecognitionTransport:InvalidBytes", ...
                    "bytes must be a uint8 vector.");
            end
            try
                write(obj.Adapter, reshape(bytes, 1, []), "uint8");
            catch cause
                error("TcpGestureRecognitionTransport:WriteFailed", ...
                    "TCP write failed: %s", cause.message);
            end
        end

        function line = readLineIfAvailable(obj)
            obj.requireOpen();
            line = obj.extractBufferedLine();
            if ~isempty(line)
                return;
            end
            try
                available = obj.Adapter.NumBytesAvailable;
                if ~isnumeric(available) || ~isscalar(available) || ...
                        ~isreal(available) || ~isfinite(available) || ...
                        available < 0 || available ~= fix(available)
                    error("TcpGestureRecognitionTransport:AdapterContract", ...
                        "NumBytesAvailable is malformed.");
                end
                if available > 0
                    capacity = obj.MaximumRecordBytes + 1 - ...
                        numel(obj.Buffer);
                    readCount = min(double(available), capacity);
                    bytes = read(obj.Adapter, readCount, "uint8");
                    if ~isa(bytes, "uint8") || numel(bytes) ~= readCount
                        error("TcpGestureRecognitionTransport:AdapterContract", ...
                            "Adapter read returned the wrong byte payload.");
                    end
                    obj.Buffer = [obj.Buffer, reshape(bytes, 1, [])];
                    if ~any(obj.Buffer == uint8(10)) && ...
                            numel(obj.Buffer) > obj.MaximumRecordBytes
                        obj.failOversizedRecord();
                    end
                end
            catch cause
                if ismember(string(cause.identifier), [ ...
                        "TcpGestureRecognitionTransport:AdapterContract", ...
                        "TcpGestureRecognitionTransport:RecordTooLarge"])
                    rethrow(cause);
                end
                error("TcpGestureRecognitionTransport:ReadFailed", ...
                    "TCP read failed: %s", cause.message);
            end
            line = obj.extractBufferedLine();
        end

        function line = readLine(obj, timeoutSeconds)
            obj.requireOpen();
            if ~isnumeric(timeoutSeconds) || ~isscalar(timeoutSeconds) || ...
                    ~isreal(timeoutSeconds) || ~isfinite(timeoutSeconds) || ...
                    timeoutSeconds < 0
                error("TcpGestureRecognitionTransport:InvalidTimeout", ...
                    "timeoutSeconds must be a nonnegative finite scalar.");
            end
            timeoutSeconds = double(timeoutSeconds);
            startedAt = obj.readClock();
            while true
                line = obj.readLineIfAvailable();
                if ~isempty(line)
                    return;
                end
                elapsed = obj.readClock() - startedAt;
                if elapsed >= timeoutSeconds
                    error("TcpGestureRecognitionTransport:ReadTimeout", ...
                        "No complete newline record arrived before timeout.");
                end
                obj.PauseFunction(min(obj.PollPeriodSeconds, ...
                    timeoutSeconds - elapsed));
            end
        end

        function close(obj)
            if obj.Closed
                return;
            end
            obj.Closed = true;
            adapter = obj.Adapter;
            obj.Adapter = [];
            obj.Buffer = zeros(1, 0, "uint8");
            if isempty(adapter)
                return;
            end
            try
                if ismethod(adapter, "close")
                    close(adapter);
                elseif ismethod(adapter, "delete")
                    delete(adapter);
                end
            catch
                % Closing remains idempotent even after an adapter failure.
            end
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function line = extractBufferedLine(obj)
            newline = find(obj.Buffer == uint8(10), 1, "first");
            if isempty(newline)
                line = strings(0, 1);
                return;
            end
            if newline - 1 > obj.MaximumRecordBytes
                obj.failOversizedRecord();
            end
            bytes = obj.Buffer(1:newline - 1);
            obj.Buffer(1:newline) = [];
            line = string(native2unicode(bytes, "UTF-8"));
        end

        function failOversizedRecord(obj)
            obj.Buffer = zeros(1, 0, "uint8");
            obj.close();
            error("TcpGestureRecognitionTransport:RecordTooLarge", ...
                "A newline-delimited record exceeded MaximumRecordBytes.");
        end

        function value = readClock(obj)
            value = obj.ClockFunction();
            if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || ...
                    ~isfinite(value)
                error("TcpGestureRecognitionTransport:InvalidClock", ...
                    "ClockFunction must return a finite scalar.");
            end
            value = double(value);
        end

        function requireOpen(obj)
            if obj.Closed
                error("TcpGestureRecognitionTransport:Closed", ...
                    "The TCP transport is closed.");
            end
        end
    end
end

function host = validateHost(value)
valid = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
if ~valid || strlength(strtrim(string(value))) == 0
    error("TcpGestureRecognitionTransport:InvalidHost", ...
        "host must be a nonempty text scalar.");
end
host = string(value);
end

function port = validatePort(value)
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value ~= fix(value) || ...
        value < 1 || value > 65535
    error("TcpGestureRecognitionTransport:InvalidPort", ...
        "port must be an integer from 1 through 65535.");
end
port = double(value);
end

function options = normalizeOptions(options)
if ~isstruct(options) || ~isscalar(options)
    error("TcpGestureRecognitionTransport:InvalidOptions", ...
        "options must be a scalar struct.");
end
adapterFactoryProvided = isfield(options, "AdapterFactory");
defaults = struct( ...
    "AdapterFactory", [], ...
    "TcpClientConstructor", @tcpclient, ...
    "ClockFunction", @wallClockSeconds, ...
    "PauseFunction", @pause, ...
    "PollPeriodSeconds", 0.01, ...
    "ConnectTimeoutSeconds", 5, ...
    "OperationTimeoutSeconds", 0.05, ...
    "MaximumRecordBytes", 65536);
names = string(fieldnames(options));
if ~all(ismember(names, string(fieldnames(defaults))))
    error("TcpGestureRecognitionTransport:InvalidOptions", ...
        "options contains an unknown field.");
end
for name = string(fieldnames(defaults)).'
    if ~isfield(options, name)
        options.(name) = defaults.(name);
    end
end
if ~adapterFactoryProvided
    constructor = options.TcpClientConstructor;
    options.AdapterFactory = @(host, port, connectTimeout, operationTimeout) ...
        createTcpAdapter(host, port, connectTimeout, operationTimeout, ...
        constructor);
end
for field = ["AdapterFactory", "TcpClientConstructor", ...
        "ClockFunction", "PauseFunction"]
    if ~isa(options.(field), "function_handle")
        error("TcpGestureRecognitionTransport:InvalidOptions", ...
            "%s must be a function handle.", field);
    end
end
value = options.PollPeriodSeconds;
if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || ...
        ~isfinite(value) || value <= 0
    error("TcpGestureRecognitionTransport:InvalidOptions", ...
        "PollPeriodSeconds must be a positive finite scalar.");
end
options.PollPeriodSeconds = double(value);
for field = ["ConnectTimeoutSeconds", "OperationTimeoutSeconds"]
    value = options.(field);
    if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
            ~isreal(value) || ~isfinite(value) || value <= 0
        error("TcpGestureRecognitionTransport:InvalidOptions", ...
            "%s must be a positive finite scalar.", field);
    end
    options.(field) = double(value);
end
value = options.MaximumRecordBytes;
if ~isnumeric(value) || islogical(value) || ~isscalar(value) || ...
        ~isreal(value) || ~isfinite(value) || value <= 0 || ...
        value ~= fix(value) || value > flintmax
    error("TcpGestureRecognitionTransport:InvalidOptions", ...
        "MaximumRecordBytes must be a positive finite integer.");
end
options.MaximumRecordBytes = double(value);
end

function adapter = createTcpAdapter(host, port, connectTimeout, ...
        operationTimeout, constructor)
adapter = constructor(char(host), port, ...
    "ConnectTimeout", connectTimeout, "Timeout", operationTimeout);
end

function value = wallClockSeconds()
value = posixtime(datetime("now", "TimeZone", "UTC"));
end
