# Authoritative gesture demonstration media

This directory contains the eight approved demonstration videos derived from
reviewed human recordings. The approved bytes were produced and audited during
v3 and are reused byte-for-byte by `eight_pose_v5`; this protocol-only upgrade
requires no re-recording and no re-encoding. Do not replace them with synthetic
motion, placeholder, or test media.

An approved recording session must provide these exact files:

- `REST.mp4`
- `WRIST_UP.mp4`
- `WRIST_DOWN.mp4`
- `FOREARM_IN.mp4`
- `FOREARM_OUT.mp4`
- `ARM_UP.mp4`
- `ARM_DOWN.mp4`
- `FIST.mp4`

Each installed file is a silent H.264 MP4 in landscape orientation at
1280×720, 30 fps, 255 frames, and approximately 8.5 seconds. The exact target
timeline is `READY` from 0-1 s, `MOVE` from 1-2.5 s, `HOLD STILL` from
2.5-6.5 s, and `RETURN TO REST` from 6.5-8.5 s. The target endpoint must remain
stable throughout `HOLD STILL`; during that stage, `ARM_DOWN.mp4` must visibly
point toward the floor rather than horizontally. The final segment must show
the participant returning to the horizontal neutral pose.

`REST.mp4` shows `READY` from 0-1 s and a horizontal, wrist-neutral, relaxed
pose labelled `HOLD: RELAX` from 1-8.5 s. The acquisition program uses the
first two seconds of a rest presentation for each 2 s relaxed hold.

`FIST.mp4` must begin with the forearm horizontal, wrist neutral, and fingers
relaxed; close to a complete fist during `MOVE`, with the thumb outside the
four fingers; hold the endpoint for the complete 4 s `HOLD STILL` stage without
wrist, forearm, or arm movement; then return to the relaxed pose.

`gesture_video_manifest.csv` records the v3-produced stage endpoints and duration.
`output_hashes_sha256_v3.csv` records the SHA-256 digest of every installed
MP4 and is the integrity reference for preflight checks and the protocol
freeze.

The automated MATLAB tests create synthetic MP4 fixtures in temporary
directories and never use this production directory.
