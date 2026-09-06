function seed = generate_protocol_seed()
try
    generator = System.Security.Cryptography.RandomNumberGenerator.Create();
    cleanupGenerator = onCleanup(@() generator.Dispose());
    bytes = NET.createArray("System.Byte", 4);
    generator.GetBytes(bytes);
    seed = double(typecast(uint8(bytes), "uint32"));
catch cause
    failure = MException("gesture_guided_acquisition:SeedGenerationFailed", ...
        "Unable to generate an automatic protocol Seed.");
    failure = addCause(failure, cause);
    throwAsCaller(failure);
end
end
