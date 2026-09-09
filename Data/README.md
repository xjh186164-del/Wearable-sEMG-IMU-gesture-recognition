# Data, trained models and evaluation results

This directory contains the accepted recordings, derived window dataset,
selected model checkpoint, controlled ablation study and real-time decision
configuration used to produce the results reported for the wearable sEMG--IMU
gesture-recognition system.

## Important participant note

The dataset contains recordings from **one physical participant**. Acquisition
identifiers `S001`, `S002` and `S003` are legacy session aliases and all map to
the canonical participant identifier `P001`; they must not be interpreted as
three different participants. The reported results therefore represent
personalised, session-separated recognition rather than participant-independent
validation.

The files use pseudonymous identifiers and contain no participant name or
email address. They still contain recording timestamps and study metadata and
must therefore not be treated as fully anonymous. Confirm the applicable
consent, institutional and data-protection requirements before making this
directory public.

## Directory structure

```text
Data/
├── captures/                     # 21 accepted guided-recording sessions
├── datasets/
│   ├── p001_windows.npz
│   └── p001_windows.manifest.json
├── training_runs/
│   ├── p001_dual_cnn/
│   │   ├── best_model.pt
│   │   ├── history.csv
│   │   ├── metrics.json
│   │   └── split.json
│   └── p001_ablation_20260908/  # Five variants, three seeds and post-hoc results
├── realtime_configs/
│   ├── p001_dual_cnn_endpoint_v2.json
│   └── p001_dual_cnn_endpoint_v2.test_evaluation.json
├── checksums.sha256
└── README.md
```

## Accepted recording sessions

The `captures` directory contains 21 sessions that passed the automated
`gesture_session_quality_v4` quality gate. Together they contain approximately
62.98 minutes of simultaneous eight-channel sEMG and six-axis IMU data. The
nominal sampling rates are 500 Hz for sEMG and 104 Hz for the accelerometer and
gyroscope.

Each accepted session has exactly six files sharing the same session basename:

| Suffix | Contents |
|---|---|
| `_raw.csv` | ADS1298 samples in raw ADC codes, device timestamps, status and dropped-sample counter |
| `_filtered.csv` | Eight digitally filtered sEMG channels in mV |
| `_imu.csv` | Acceleration in g, angular velocity in degrees/s and temperature in degrees Celsius |
| `_events.csv` | Gesture-protocol events and authoritative temporal labels |
| `_metadata.json` | Acquisition settings, session identity and protocol information |
| `_quality.json` | Sampling, continuity, clipping, IMU plausibility and event-coverage checks |

In session identifiers such as `S003_D08_R02`, `Dxx` denotes the acquisition
date index and `Rxx` denotes a complete removal and refitting performed on that
date. The accepted data therefore include multiple dates and repeated donning
cycles for the same physical participant.

The principal CSV schemas are:

```text
sample,timestamp_us,status,ch1_raw,...,ch8_raw,dropped
sample,timestamp_us,status,ch1_filtered_mV,...,ch8_filtered_mV,dropped
sample,timestamp_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_C,dropped
session_time_s,event_type,block_id,trial_id,gesture_id,valid,note
```

The recording protocol contains the following eight classes in their fixed
model order:

```text
REST
WRIST_UP
WRIST_DOWN
FOREARM_IN
FOREARM_OUT
ARM_UP
ARM_DOWN
FIST
```

## Derived window dataset

`datasets/p001_windows.npz` contains 12,936 paired windows generated from the
21 accepted sessions. Each window covers 0.5 s and successive candidate
windows are separated by 0.1 s. The stored input shapes are:

- sEMG: `8 x 250` samples;
- IMU: `6 x 52` samples.

Only windows fully contained within guarded stable intervals are retained.
Movement, return transitions, invalid trials, sequence gaps and windows that
cross label boundaries are excluded. The accompanying manifest records the
accepted sessions, rejected sessions, subject aliases, class counts, source
file SHA-256 values and dataset-generation parameters.

The session-level split used for the reported experiment is:

| Subset | Sessions | Windows |
|---|---:|---:|
| Training | 14 | 8,624 |
| Validation | 3 | 1,848 |
| Test | 4 | 2,464 |

The training subset contains 4,704 REST windows and 1,176 windows for each of
the seven active gestures. Normalisation parameters were calculated from the
training sessions only.

## Trained model artifacts

`training_runs/p001_dual_cnn` contains the selected class-dependent fusion
model and the records needed to interpret it:

- `best_model.pt`: selected PyTorch checkpoint;
- `history.csv`: epoch-by-epoch training loss, validation loss and validation
  Macro-F1;
- `metrics.json`: validation/test metrics, per-class results, confusion
  matrices and learned fusion weights;
- `split.json`: exact session partition and dataset/checksum linkage.

The selected checkpoint was obtained at epoch 8. On the four held-out test
sessions it achieved 99.68% accuracy, 99.55% balanced accuracy and 99.64%
Macro-F1.

## Controlled modality and fusion ablation

`training_runs/p001_ablation_20260908` contains a post-hoc diagnostic of the
selected checkpoint and an independently trained comparison using three fixed
random seeds. All variants use the same dataset and session-level split.

| Retrained variant | Parameters | Accuracy, mean +/- SD | Macro-F1, mean +/- SD | Target trials |
|---|---:|---:|---:|---:|
| sEMG only | 59,080 | 84.97 +/- 2.53% | 85.33 +/- 2.89% | 154/168 |
| IMU only | 21,416 | 95.08 +/- 0.81% | 94.09 +/- 0.96% | 165/168 |
| Fixed 50:50 fusion | 80,496 | 99.68 +/- 0.25% | 99.64 +/- 0.34% | 168/168 |
| Learned fusion, neutral start | 80,504 | 99.69 +/- 0.23% | 99.65 +/- 0.32% | 168/168 |
| Learned fusion, physics-informed start | 80,504 | 99.70 +/- 0.02% | 99.67 +/- 0.03% | 168/168 |

The three-seed results support multimodal fusion over either isolated sensor
branch. The differences among the three fused configurations are small and do
not demonstrate that trainable class-dependent fusion is superior to fixed
50:50 fusion on this dataset. Neutral initialisation also produced high fused
performance without converging to the physics-informed weight pattern.

The four heads of the original selected checkpoint were also evaluated without
retraining. Their test Macro-F1 scores were 83.86% for the sEMG head, 91.50%
for the IMU head, 99.43% for fixed 50:50 fusion and 99.64% for its learned
fusion. These are diagnostic outputs of one jointly trained checkpoint and
must not be treated as a controlled comparison; the independently retrained
results above provide that comparison.

The result directory contains all 15 selected checkpoints together with their
histories, exact splits, window predictions, per-session and target-trial
metrics. `aggregate_summary.csv` is the concise cross-seed result table,
`run_summary.csv` contains one row per training run, and
`ablation_results.json` retains the complete machine-readable evaluation.

Successive test windows overlap by 80%, so the 2,464 window predictions are not
independent observations. The grouped files report results for each held-out
session and for the 56 target trials in every run.

## Real-time configuration

`realtime_configs/p001_dual_cnn_endpoint_v2.json` stores the
validation-selected post-processing configuration used by the live MATLAB and
Python pipeline. It uses an exponential moving average weight of 0.4 for the
new probability vector, a confidence threshold of 0.50 and two consecutive
matching outputs. The test-evaluation JSON records the corresponding held-out
replay results.

## Excluded material

This release intentionally excludes:

- informal BLE checks and live-recognition demonstrations;
- zero-length or interrupted capture attempts;
- session `S003_D07_R02`, which did not pass the quality gate;
- the separate `quarantine` directory containing early, failed or superseded
  recordings.

These files were not used to train, validate or test the selected model and
are omitted to prevent them from being mistaken for accepted experimental
data.

## Integrity and storage

`checksums.sha256` records the SHA-256 digest of every distributed file except
the checksum file itself. Because the capture CSVs, NPZ dataset and PyTorch
checkpoints are large data or binary files, they are tracked with Git Large
File Storage rather than ordinary Git history.

No data-reuse licence is granted by this README. Add an explicit repository
data licence only after ownership, participant consent and institutional
release conditions have been confirmed.
