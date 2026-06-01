# AFiNET: Explainable Deep Learning for Atrial Fibrillation Detection

AFiNET is an explainable deep learning framework for automated atrial fibrillation (AF) detection from electrocardiogram (ECG) recordings. The project combines high-performance neural network models with interpretable visualization techniques to support clinical decision-making and improve transparency in AI-assisted cardiovascular diagnostics.

---

## Overview

Atrial fibrillation (AF) remains a major public health concern worldwide, including in the Philippines. Although long-term ambulatory ECG monitoring is considered the gold standard for AF diagnosis, the large volume of recorded data can make manual review time-consuming and resource-intensive.

AFiNET addresses this challenge through an automated and explainable AF detection pipeline. The project benchmarks multiple deep learning architectures on a dataset containing more than 1.2 million ECG segments and incorporates visualization techniques that allow clinicians to inspect the ECG regions influencing model predictions.

By integrating explainable artificial intelligence (XAI) methods such as Grad-CAM, AFiNET provides both accurate classification performance and clinically interpretable results.

---

## Features

### Deep Learning Benchmarking

AFiNET evaluates the performance of three deep learning architectures for AF detection:

* DDNN
* RawECGNet
* CTRhythm

Among the evaluated models, **DDNN** achieved the highest performance with a peak **Matthews Correlation Coefficient (MCC) of 0.8039**.

### Explainable AI Integration

* Grad-CAM-based heatmap visualization
* Perturbation-based explanation validation framework
* Identification of ECG waveform regions influencing model predictions
* Improved transparency and interpretability for clinical users

### Clinical Application

The accompanying desktop application provides:

* Automated AF detection from ECG recordings
* Visualization of diagnostic heatmaps
* Support for common Holter monitoring file formats
* Automated PDF report generation for clinical documentation

---

## Supported File Formats

The application supports the following ECG file formats:

| Format          | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `.dat` + `.hea` | WFDB-compatible recordings (both files must be present in the same directory) |
| `.h5`           | HDF5 ECG recordings                                                           |

---

## Repository Structure

```text
.
├── afinet_app.py
├── ddnn_FINAL.pth
└── README.md
```

### Required Files

Ensure the following files are located in the same directory before running the application:

* `afinet_app.py`
* `ddnn_FINAL.pth`

---

## Requirements

### Python Version

AFiNET supports:

* Python 3.9
* Python 3.10
* Python 3.11
* Python 3.12

### Dependencies

Install PyTorch (CPU version):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
pip install numpy scipy matplotlib wfdb h5py
```

---

## Running the Application

Open a terminal in the project directory and run:

```bash
python afinet_app.py
```

On macOS, you may need to use:

```bash
python3 afinet_app.py
```

If the model loads successfully, a confirmation message will appear in the application's left panel.

If the model fails to load, verify that:

* `ddnn_FINAL.pth` exists in the project directory
* The filename matches exactly
* All required dependencies have been installed

---

## Using the Application

### 1. Load an ECG Recording

Click **Load ECG File** and select an ECG recording.

Supported formats:

* `.dat` / `.hea` (WFDB format)
* `.h5` (HDF5 format)

For WFDB recordings, both the `.dat` and `.hea` files must be located in the same folder.

### 2. Run Analysis

Click **Run Analysis** to begin automated AF detection.

The application will:

1. Process the ECG recording
2. Generate AF predictions
3. Create Grad-CAM heatmaps
4. Display interpretable visualizations
5. Enable PDF report generation

---

## Research Contributions

This project contributes to the field of AI-assisted cardiovascular diagnostics through:

* Large-scale benchmarking of deep learning models for AF detection
* Evaluation using Matthews Correlation Coefficient (MCC) as the primary performance metric
* Development of a lead-agnostic ECG classification approach
* Integration of explainable AI techniques for clinical interpretability
* Creation of a practical desktop application for real-world deployment

---

## Platform Compatibility

AFiNET was primarily developed and tested on macOS.

Expected compatibility:

| Platform | Status                |
| -------- | --------------------- |
| macOS    | Tested                |
| Windows  | Not officially tested |
| Linux    | Not officially tested |

While the application should function on Windows and Linux systems, compatibility on these platforms has not been formally validated.

---

## Academic Context

AFiNET was developed as part of an undergraduate thesis focused on improving the accessibility, transparency, and clinical usability of AI-driven cardiovascular diagnostics.
