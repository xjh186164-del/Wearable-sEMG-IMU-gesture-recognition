# Wearable sEMG--IMU Gesture Recognition System

This repository contains the hardware designs, embedded firmware, host
software, accepted recordings and trained model for a battery-powered wearable
gesture-recognition prototype. The system combines eight-channel surface
electromyography (sEMG) with six-axis inertial sensing and transmits both sensor
streams to a host computer using Bluetooth Low Energy (BLE).

The project was developed as an engineering research prototype. It is not a
medical device and must not be used for diagnosis, clinical monitoring or
safety-critical control.

## System overview

The wearable system consists of:

- a custom four-layer ADS1298-based acquisition PCB;
- an LSM6DSOX three-axis accelerometer and three-axis gyroscope;
- an ESP32-C3 microcontroller and BLE link;
- eight bipolar electrode modules and a separate right-leg-drive electrode;
- independently adjustable electrode, right-leg-drive and electronics straps;
- a rechargeable LP603449 lithium-polymer battery;
- MATLAB acquisition, filtering, visualisation and recording software;
- a Python dual-branch convolutional neural network for real-time recognition.

The production firmware samples sEMG at 500 Hz and the IMU at 104 Hz. MATLAB
constructs temporally corresponding 0.5 s windows, and the Python model combines
the two sensor modalities using class-dependent trainable late fusion.

## Reported results

Twenty-one accepted guided-recording sessions provided approximately 62.98
minutes of simultaneous sensor data. The derived dataset contains 12,936 paired
windows representing eight classes. A session-level split assigned 14 sessions
to training, three to validation and four to held-out testing.

On 2,464 held-out test windows, the selected personalised model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 99.68% |
| Balanced accuracy | 99.55% |
| Macro-F1 | 99.64% |

These measurements demonstrate personalised recognition for one physical
participant under the tested conditions. They do not establish equivalent
performance for unseen users, alternative electrode placements or uncontrolled
operating environments.

## Repository structure

```text
.
├── Hardware/
│   ├── Enclosure and Armband/     # Editable Fusion 360 mechanical designs
│   └── Schematic and PCB/         # EasyEDA project, schematic, BOM/CPL and Gerber archive
├── Software/
│   ├── Firmware/                  # Production ESP32-C3 firmware only
│   ├── MATLAB/                    # BLE acquisition, filtering and live display
│   ├── Python/                    # Dataset, training and inference package
│   └── README.md
├── Data/
│   ├── captures/                  # 21 accepted six-artifact sessions
│   ├── datasets/                  # Paired 0.5 s window dataset and manifest
│   ├── training_runs/             # Selected checkpoint and evaluation records
│   ├── realtime_configs/          # Validation-selected decision configuration
│   ├── checksums.sha256
│   └── README.md
├── .gitattributes
├── .gitignore
└── README.md
```

Detailed software instructions are provided in
[`Software/README.md`](Software/README.md). Dataset structure, participant
scope, session splitting and artifact descriptions are provided in
[`Data/README.md`](Data/README.md).

## Hardware files

`Hardware/Schematic and PCB` contains the editable EasyEDA Pro project, the
complete schematic image, the manufacturing BOM/CPL and a Gerber archive. The
mechanical directory contains the editable Fusion 360 designs for the electrode
strap, right-leg-drive strap, PCB strap, enclosure and lid.

The editable source files are retained in addition to manufacturing or preview
exports so that later revisions can be traced to the submitted design.

## Software quick start

### 1. Clone the repository with Git LFS

Large measurements and binary design files are tracked using Git Large File
Storage. Install Git LFS before cloning or pulling the complete repository:

```powershell
git lfs install
git clone <repository-url>
cd <repository-directory>
git lfs pull
```

### 2. Flash the wearable firmware

Open the following sketch in the Arduino IDE:

```text
Software/Firmware/esp32c3_ads1298_rld_csv/esp32c3_ads1298_rld_csv.ino
```

Select `ESP32C3 Dev Module`, enable `USB CDC On Boot`, and compile using the
Espressif ESP32 Arduino core. Keep the `.ino`, `.cpp` and `.h` files in the same
sketch directory.

### 3. Prepare the Python environment

The machine-learning software was developed with Python 3.12. From the
repository root:

```powershell
py -3.12 -m venv .venv-gesture
.\.venv-gesture\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gesture\Scripts\python.exe -m pip install -r .\Software\Python\requirements-data.txt
.\.venv-gesture\Scripts\python.exe -m pip install -r .\Software\Python\requirements-cuda.txt
.\.venv-gesture\Scripts\python.exe -m pip install -e .\Software\Python
```

The CUDA requirements reproduce the PyTorch configuration used during model
training. A different compatible PyTorch build may be required on other
hardware.

### 4. Run MATLAB acquisition

Start MATLAB from the repository root and add the acquisition code to the path:

```matlab
repositoryRoot = pwd;
addpath(fullfile(repositoryRoot, "Software", "MATLAB"));
blelist
[rawFile, filteredFile, imuFile] = ...
    ads1298_emg_ble_live("SensorBiShe-EMG", 10);
```

Refer to the Software README for guided acquisition, model training and live
recognition examples.

## Data integrity and reproducibility

`Data/checksums.sha256` contains a SHA-256 digest for every distributed data or
model artifact and for the Data README. The dataset manifest records the source
hashes and window-generation parameters. The model split file independently
records the dataset and manifest hashes, while the real-time configuration
records the selected checkpoint hash.

The published capture set contains only the 21 sessions accepted by the
automated quality gate. Informal demonstrations, failed attempts, quarantined
recordings and the rejected session `S003_D07_R02` are intentionally excluded.

## Participant and data-use notice

The acquisition identifiers `S001`, `S002` and `S003` all refer to the same
physical participant and map to canonical identifier `P001`. The recordings
are pseudonymised rather than guaranteed anonymous and include session
timestamps and technical metadata. Confirm participant consent, institutional
requirements, data ownership and an appropriate data licence before making the
repository public or redistributing the measurements.

## Safety notice

When electrodes are attached to a person, power the wearable system from its
battery and disconnect USB. Do not connect the wearer through unisolated USB,
an oscilloscope, a bench supply or other mains-powered equipment. Inspect the
battery, cables, electrodes and enclosure before use and stop immediately if
the wearer experiences discomfort, skin irritation or unusual heating.

## Licence

No software, hardware or data licence has yet been assigned in this repository.
A suitable licence should be selected only after the project ownership and
institutional release conditions have been confirmed.
