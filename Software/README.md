# Software

This directory contains the software used by the battery-powered wearable
surface-electromyography (sEMG) and inertial-measurement-unit (IMU) gesture
recognition system. It includes the production embedded firmware, the MATLAB
acquisition and visualisation software, and the Python machine-learning
pipeline.

## Directory structure

```text
Software/
├── Firmware/
│   └── esp32c3_ads1298_rld_csv/
│       ├── esp32c3_ads1298_rld_csv.ino
│       ├── lsm6dsox_spi.cpp
│       └── lsm6dsox_spi.h
├── MATLAB/
│   ├── +gesture_internal/
│   ├── gesture_media/
│   └── *.m
├── Python/
│   ├── gesture_ml/
│   │   ├── ablation.py
│   │   ├── variants.py
│   │   └── ...
│   ├── pyproject.toml
│   ├── requirements-data.txt
│   ├── requirements-cuda.txt
│   ├── uv.lock
│   └── verify_cuda.py
└── README.md
```

## System software flow

1. The ESP32-C3 firmware samples eight ADS1298 channels at 500 samples/s and
   the LSM6DSOX accelerometer and gyroscope at 104 samples/s.
2. Device timestamps, sequence numbers and dropped-sample counters are added
   before the measurements are transmitted using the version 2 binary BLE
   protocol.
3. MATLAB receives and decodes the BLE notifications, filters and displays the
   sEMG signals, displays the IMU measurements, and writes recording files.
4. For real-time recognition, MATLAB sends paired 0.5 s sensor windows to a
   persistent local Python worker over TCP.
5. Python applies the saved normalisation parameters, evaluates the trained
   dual-branch convolutional neural network, and returns class probabilities to
   MATLAB.

## 1. Embedded firmware

The production sketch is located at:

```text
Firmware/esp32c3_ads1298_rld_csv/esp32c3_ads1298_rld_csv.ino
```

It configures:

- ADS1298: eight channels, 500 samples/s, gain 6 and right-leg-drive enabled;
- LSM6DSOX accelerometer: 104 Hz and +/-4 g;
- LSM6DSOX gyroscope: 104 Hz and +/-500 degrees/s;
- BLE device name: `SensorBiShe-EMG`;
- BLE protocol version: 2;
- requested BLE MTU: 247 bytes.

Open the `.ino` file in the Arduino IDE with all three firmware files in the
same sketch directory. Select `ESP32C3 Dev Module` and enable `USB CDC On
Boot`, then compile and upload the sketch using the Espressif ESP32 Arduino
core. USB is used for programming and startup diagnostics; measurement samples
are transmitted through BLE during normal operation.

## 2. MATLAB acquisition and live recognition

The principal MATLAB entry points are:

- `ads1298_emg_ble_live`: BLE acquisition, filtering, visualisation and file
  recording;
- `gesture_guided_acquisition`: video-guided labelled-session acquisition;
- `gesture_live_recognition`: acquisition with live Python model inference.

From the repository root, add the MATLAB directory to the search path:

```matlab
repositoryRoot = pwd;
addpath(fullfile(repositoryRoot, "Software", "MATLAB"));
```

To verify that the wearable device is visible and run a short BLE acquisition:

```matlab
blelist
[rawFile, filteredFile, imuFile] = ...
    ads1298_emg_ble_live("SensorBiShe-EMG", 10);
```

For a guided recording, supply an anonymous subject identifier and an explicit
output directory. The following directory names are examples and can be
adapted to the final repository data layout:

```matlab
config = struct( ...
    "SubjectId", "S001", ...
    "SessionId", "S001_D01_R01", ...
    "OperatorCode", "OperatorCode01", ...
    "SleeveSize", "CUSTOM_V1", ...
    "ArmSide", "left", ...
    "BlockCount", 2, ...
    "CaptureDirectory", fullfile(repositoryRoot, "Data", "captures"));

result = gesture_guided_acquisition(config);
disp(result.quality);
```

The MATLAB code requires access to the `ble` and `blelist` interfaces and to
the signal-processing functions used by the digital filter implementation.

## 3. Python machine-learning pipeline

The Python package supports:

- construction of timestamp-aligned sEMG--IMU datasets;
- session-level training, validation and test partitioning;
- training and evaluation of the dual-branch convolutional neural network;
- post-hoc branch evaluation and controlled retraining of five ablation
  variants;
- calibration and replay of the real-time decision filter;
- persistent TCP inference for the MATLAB client.

The software was developed using Python 3.12. Create a virtual environment
outside the tracked source directories and install the package from the
repository root:

```powershell
py -3.12 -m venv .venv-gesture
.\.venv-gesture\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gesture\Scripts\python.exe -m pip install -r .\Software\Python\requirements-data.txt
.\.venv-gesture\Scripts\python.exe -m pip install -r .\Software\Python\requirements-cuda.txt
.\.venv-gesture\Scripts\python.exe -m pip install -e .\Software\Python
```

`requirements-cuda.txt` pins the PyTorch configuration used for GPU training.
Users who do not need to reproduce the original CUDA environment may install a
PyTorch build suitable for their own operating system and hardware.

An example dataset-build command is:

```powershell
.\.venv-gesture\Scripts\gesture-build-dataset.exe `
  --captures .\Data\captures `
  --output .\Data\datasets\p001_windows.npz
```

An example training command is:

```powershell
.\.venv-gesture\Scripts\gesture-train.exe `
  --dataset .\Data\datasets\p001_windows.npz `
  --output-dir .\Data\training_runs\p001_dual_cnn `
  --seed 20260821
```

To reproduce the additional modality and fusion comparison, run:

```powershell
.\.venv-gesture\Scripts\gesture-ablation.exe `
  --dataset .\Data\datasets\p001_windows.npz `
  --baseline-checkpoint .\Data\training_runs\p001_dual_cnn\best_model.pt `
  --output-dir .\Data\training_runs\p001_ablation_reproduction `
  --device cuda `
  --seeds 20260821,20260822,20260823
```

The command first evaluates the existing selected checkpoint with its sEMG
head, IMU head, fixed 50:50 fusion and learned fusion. It then independently
trains `emg_only`, `imu_only`, `fixed_50_50`, `learned_neutral` and
`learned_physics` variants for each seed. Each run stores its selected
checkpoint, epoch history, exact split, window-level predictions and grouped
session/trial metrics. Aggregate outputs are written to
`aggregate_summary.csv`, `run_summary.csv` and `ablation_results.json`.

The published run is retained under
`Data/training_runs/p001_ablation_20260908`. Reproduction results should be
written to a different directory, as in the example, to avoid overwriting the
reported artifacts.

## 4. Live recognition

Live recognition requires a trained checkpoint and the selected real-time
calibration file. These binary/data artifacts are not stored in this Software
directory. Pass their final locations explicitly when they are placed in the
repository Data directory:

```matlab
config = struct( ...
    "DurationSeconds", 60, ...
    "OutputDirectory", fullfile(repositoryRoot, "Data", "captures"), ...
    "CheckpointPath", fullfile(repositoryRoot, "Data", ...
        "training_runs", "p001_dual_cnn", "best_model.pt"), ...
    "CalibrationPath", fullfile(repositoryRoot, "Data", ...
        "realtime_configs", "p001_dual_cnn_endpoint_v2.json"), ...
    "PythonExecutable", fullfile(repositoryRoot, ...
        ".venv-gesture", "Scripts", "python.exe"));

result = gesture_live_recognition(config);
disp(result.recognitionSummary);
```

## Safety notice

This system is an engineering research prototype and is not a medical device.
When electrodes are attached to a person, operate the wearable electronics
from the battery and disconnect the USB cable. Do not create a conductive path
from the wearer to a non-isolated computer, bench supply, oscilloscope or other
mains-powered equipment.
