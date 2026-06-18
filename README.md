# DSP_Project2

## EEG-Based Diagnosis of Alzheimer's Disease and Frontotemporal Dementia using TCN-LSTM

This project reproduces the methodology proposed in the paper:

> **An Explainable and Efficient Deep Learning Framework for EEG-Based Diagnosis of Alzheimer's Disease and Frontotemporal Dementia**  
> Waqar Khan et al., Frontiers in Medicine, 2025

📄 Paper: https://pubmed.ncbi.nlm.nih.gov/40735445/

---

## Overview

Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD) are among the most common neurodegenerative disorders. Early diagnosis is crucial for slowing disease progression and improving patients' quality of life.

This project implements an EEG-based deep learning framework for dementia classification using:

- Modified Relative Band Power (RBP) feature extraction
- Temporal Convolutional Network (TCN)
- Long Short-Term Memory (LSTM)
- SHAP Explainable Artificial Intelligence (XAI)

The framework classifies:

- Alzheimer's Disease (AD)
- Frontotemporal Dementia (FTD)
- Healthy Controls (HC)

using resting-state EEG recordings.

---

# Dataset

## Dataset Source

The dataset used in this project is publicly available:

**A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG**

- DOI: https://doi.org/10.3390/data8060095
- OpenNeuro: https://openneuro.org/datasets/ds004504

---

## Subjects

| Class | Number of Subjects |
|---------|---------:|
| Alzheimer's Disease (AD) | 36 |
| Frontotemporal Dementia (FTD) | 23 |
| Healthy Controls (HC) | 29 |
| Total | 88 |

---

## EEG Recording Configuration

| Item | Value |
|--------|--------|
| EEG Channels | 19 |
| Sampling Rate | 500 Hz |
| Recording State | Eyes Closed Resting State |
| Original Frequency Range | 0.5 – 60 Hz |
| Recording Position | Sitting |

---

# Methodology

The complete workflow of the proposed framework is shown below.

```text
Raw EEG Signals
       │
       ▼
Preprocessing
       │
       ├── Butterworth Bandpass Filter
       ├── Artifact Subspace Reconstruction (ASR)
       └── Independent Component Analysis (ICA)
       │
       ▼
Feature Engineering
       │
       ├── Power Spectral Density (PSD)
       ├── Modified Relative Band Power (RBP)
       └── Epoch Segmentation
       │
       ▼
Feature Matrix
       │
       ▼
Normalization
       │
       ▼
TCN-LSTM Hybrid Network
       │
       ▼
Classification
       │
       ├── AD vs HC
       ├── FTD vs HC
       ├── AD + FTD vs HC
       └── AD vs FTD vs HC
       │
       ▼
SHAP Explainability
```

---

# EEG Preprocessing

To improve signal quality and remove unwanted artifacts, the following preprocessing pipeline is applied.

## 1. Butterworth Bandpass Filter

Frequency range:

```text
0.5 Hz – 45 Hz
```

Purpose:

- Remove baseline drift
- Remove high-frequency noise
- Preserve useful neural activity

---

## 2. Artifact Subspace Reconstruction (ASR)

ASR detects and reconstructs corrupted EEG segments.

Parameters:

```text
Window Length = 0.5 s
Threshold = 17
```

Purpose:

- Remove transient artifacts
- Preserve brain activity information

---

## 3. Independent Component Analysis (ICA)

The EEG signals are decomposed into independent components.

Artifact components are automatically identified using:

```text
EEGLAB ICLabel
```

Removed components:

- Eye movement artifacts
- Jaw muscle artifacts

---

# Feature Engineering

## Power Spectral Density (PSD)

Power Spectral Density is estimated using the Welch Method.

PSD provides the power distribution across EEG frequency bands.

---

## Modified Relative Band Power (RBP)

The paper proposes a modified Relative Band Power approach using six EEG frequency bands.

### Frequency Bands

| Band | Frequency Range |
|--------|--------|
| Delta | 0.5 – 4 Hz |
| Theta | 4 – 8 Hz |
| Alpha | 8 – 16 Hz |
| Zaeta | 16 – 24 Hz |
| Beta | 24 – 30 Hz |
| Gamma | 30 – 45 Hz |

---

## Epoch Segmentation (Segment-Level)

EEG signals are segmented into overlapping windows.

```text
Epoch Length = 6 Seconds
Overlap = 50%
```

---

## Feature Matrix

For each epoch, six RBP features are extracted:

```text
[
 Delta,
 Theta,
 Alpha,
 Zaeta,
 Beta,
 Gamma
]
```

Input shape:

```text
(6, 1)
```

---

# Proposed Deep Learning Model

The proposed model combines:

- Temporal Convolutional Network (TCN)
- Long Short-Term Memory (LSTM)

to learn both temporal patterns and long-range dependencies from EEG features.

---

## Temporal Convolutional Network (TCN)

The TCN component is responsible for:

- Temporal feature extraction
- Long-range dependency modeling
- Efficient sequence learning

Key techniques:

- Dilated Convolutions
- Residual Connections
- Spatial Dropout
- Batch Normalization

---

## Long Short-Term Memory (LSTM)

The LSTM component captures sequential dependencies from TCN-extracted features.

Configuration:

```text
LSTM Units = 64
```

---

## Fully Connected Layers

```text
Dense(128)
Dropout(0.2)

Dense(192)
Dropout(0.2)

Dense(256)
Dropout(0.2)

Output Layer
```

Output classes:

```text
AD
FTD
HC
```

---

# Model Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| TCN Filters | 32 |
| Kernel Size | 7 |
| Dilation Rate | 1 |
| TCN Blocks | 2 |
| LSTM Units | 64 |
| Dropout Rate | 0.2 |
| Early Stopping | Enabled |

---

# Model Complexity

| Metric | Value |
|----------|---------|
| Total Parameters | 131,587 |
| Trainable Parameters | 131,331 |
| Non-Trainable Parameters | 256 |
| Memory Usage | 514 KB |

The lightweight architecture makes it suitable for real-time deployment and edge medical devices.

---

# Classification Tasks

The model is evaluated on four classification tasks.

## 1. Multi-Class Classification

```text
AD vs FTD vs HC
```

Accuracy:

```text
80.34%
```

---

## 2. Binary Classification

```text
AD + FTD vs HC
```

Accuracy:

```text
99.80%
```

---

## 3. Binary Classification

```text
AD vs HC
```

Accuracy:

```text
99.74%
```

---

## 4. Binary Classification

```text
FTD vs HC
```

Accuracy:

```text
99.70%
```

---

# Results

| Classification Task | Accuracy |
|---------------------|-----------|
| AD vs HC | 99.74% |
| FTD vs HC | 99.70% |
| AD + FTD vs HC | 99.80% |
| AD vs FTD vs HC | 80.34% |

---

# Explainable AI (XAI)

To improve model transparency, SHAP (SHapley Additive Explanations) is incorporated into the framework.

SHAP is used to:

- Analyze feature importance
- Explain model predictions
- Identify critical EEG frequency bands

---

## Important EEG Features

According to the SHAP analysis, the most influential features are:

1. Beta Band (24–30 Hz)
2. Zaeta Band (16–24 Hz)

These bands contribute significantly to distinguishing:

- Alzheimer's Disease
- Frontotemporal Dementia
- Healthy Controls

---

# Project Structure

```text
DSP_Project2/
│
├── ds004504/
│   ├── derivative/
│   ├── sub-001/
│   ├── ...
│   └── participants.tsv
│ 
└── reproduction.py
```

---

# References

```bibtex
@article{khan2025eeg,
  title={An Explainable and Efficient Deep Learning Framework for EEG-Based Diagnosis of Alzheimer's Disease and Frontotemporal Dementia},
  author={Khan, Waqar and others},
  journal={Frontiers in Medicine},
  volume={12},
  pages={1590201},
  year={2025},
  doi={10.3389/fmed.2025.1590201}
}
```

---

# Acknowledgements

This project is a reproduction study based on the methodology proposed by Khan et al. (2025). All credit for the original research, dataset design, and model architecture belongs to the original authors.
