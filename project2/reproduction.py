"""
=============================================================================
Reproduction of:
  "An explainable and efficient deep learning framework for
   EEG-based diagnosis of Alzheimer's disease and frontotemporal dementia"
  Khan et al., Frontiers in Medicine, 2025
  DOI: 10.3389/fmed.2025.1590201

Architecture: Modified RBP feature extraction → TCN + LSTM → SHAP (XAI)
Acceleration: Parallelized preprocessing (joblib) + vectorized numpy PSD
=============================================================================

Requirements (install once):
    pip install mne torch scikit-learn imbalanced-learn shap joblib numpy scipy pandas matplotlib

Dataset:
    Download from: https://doi.org/10.3390/data8060095
    or OpenNeuro:  https://openneuro.org/datasets/ds004504/versions/1.0.5
    Expected structure:
        data/
          AD/   subject_*.set  (or .edf / .fif)
          FTD/  subject_*.set
          HC/   subject_*.set   (Healthy Controls)
=============================================================================
"""

import os
import warnings
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import scipy.signal as signal
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION  (edit paths here)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("ds004504")          # root folder containing AD/, FTD/, HC/
USE_DERIVATIVES = True
OUTPUT_DIR  = Path("outputs_SHAP2")       # all figures + CSVs saved here
OUTPUT_DIR.mkdir(exist_ok=True)

SFREQ       = 500                   # Hz  (paper: sampled at 500 Hz)
N_CHANNELS  = 19                    # paper: 19 electrodes
EPOCH_SEC   = 6.0                   # paper: 6-second epochs
OVERLAP     = 0.5                   # paper: 50 % overlap
BP_LOW      = 0.5                   # Butterworth bandpass low  (Hz)
BP_HIGH     = 45.0                  # Butterworth bandpass high (Hz)
BP_ORDER    = 5

# Modified RBP frequency bands (paper Table / Section 3.3)
FREQ_BANDS  = {
    "Delta": (0.5,  4.0),
    "Theta": (4.0,  8.0),
    "Alpha": (8.0, 16.0),
    "Zaeta": (16.0, 24.0),
    "Beta":  (24.0, 30.0),
    "Gamma": (30.0, 45.0),
}
N_BANDS = len(FREQ_BANDS)           # 6 features per epoch

# Train / Val / Test split (paper: 80 / 10 / 10)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

# Hyperparameters (paper Table 1 + Section 3.6)
N_FILTERS   = 32
KERNEL_SIZE = 7
DILATION    = 1
DROPOUT_TCN = 0.2
LSTM_UNITS  = 64
DENSE_UNITS = [128, 192, 256]
DROPOUT_DENSE = 0.2
BATCH_SIZE  = 32
LR          = 1e-4
N_CLASSES   = 3                     # AD, FTD, HC

# Acceleration
N_JOBS      = -1                    # -1 = use all CPU cores

CLASS_NAMES = ["Alzheimer", "Frontotemporal", "Control"]
LABEL_MAP   = {"A": 0, "F": 1, "C": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PREPROCESSING  (per-subject, runs in parallel)
# ─────────────────────────────────────────────────────────────────────────────

def butterworth_bandpass(data: np.ndarray, low: float, high: float,
                         fs: float, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.
    data: (n_channels, n_samples)
    """
    nyq  = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def asr_simple(data: np.ndarray, fs: float,
               win_sec: float = 0.5, threshold: float = 17.0) -> np.ndarray:
    """
    Simplified Artifact Subspace Reconstruction (ASR).
    Paper: 0.5-s windows, deviation threshold = 17.
    Reconstructed segments are replaced by their channel-wise mean across
    the non-artifact windows (fast vectorised version).
    data: (n_channels, n_samples)
    """
    win_samples = int(win_sec * fs)
    n_ch, n_samp = data.shape
    n_wins = n_samp // win_samples

    # Reshape into windows: (n_wins, n_ch, win_samples)
    data_wins = data[:, :n_wins * win_samples].reshape(n_ch, n_wins, win_samples)
    data_wins = data_wins.transpose(1, 0, 2)          # (n_wins, n_ch, win_samples)

    # Std per window: (n_wins, n_ch)
    stds = data_wins.std(axis=-1)
    # Mark windows where ANY channel exceeds threshold
    bad_wins = (stds > threshold).any(axis=1)          # (n_wins,)

    if bad_wins.all():
        return data                                     # nothing to replace

    # Clean mean template: average of good windows per channel/sample
    good_template = data_wins[~bad_wins].mean(axis=0)  # (n_ch, win_samples)

    # Replace bad windows
    data_wins[bad_wins] = good_template[np.newaxis]

    # Reconstruct
    cleaned = data_wins.transpose(1, 0, 2).reshape(n_ch, n_wins * win_samples)

    # Append any leftover samples unchanged
    remainder = data[:, n_wins * win_samples:]
    return np.concatenate([cleaned, remainder], axis=-1)


def ica_artifact_removal(data: np.ndarray) -> np.ndarray:
    try:
        import mne
        info  = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(data.shape[0])],
            sfreq=SFREQ, ch_types="eeg")
        raw   = mne.io.RawArray(data, info, verbose=False)
        ica   = mne.preprocessing.ICA(
            n_components=0.999999,   # ← 原本是 data.shape[0]
            method="fastica",
            random_state=42,
            verbose=False)
        ica.fit(raw, verbose=False)
        raw_clean = ica.apply(raw.copy(), verbose=False)
        return raw_clean.get_data()
    except Exception:
        return data


def preprocess_subject(raw_data: np.ndarray, fs: float = SFREQ,
                       skip_ica_asr: bool = False) -> np.ndarray:
    data = butterworth_bandpass(raw_data, BP_LOW, BP_HIGH, fs, BP_ORDER)
    if not skip_ica_asr:
        data = asr_simple(data, fs)
        data = ica_artifact_removal(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  –  Modified Relative Band Power (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rbp_epochs(data: np.ndarray,
                       fs: float = SFREQ,
                       epoch_sec: float = EPOCH_SEC,
                       overlap: float = OVERLAP) -> np.ndarray:
    """
    Vectorised Modified RBP feature extraction.
    Paper Section 3.3 & Eq. (1)-(4).

    Steps per epoch:
      1. Welch PSD  →  PSD(f)
      2. Total PSD  =  sum over 0.5-45 Hz
      3. RBP_b      =  sum(PSD in band_b) / Total PSD   [per channel]
      4. Epoch RBP  =  mean over channels

    Input:  data  (n_channels, n_samples)
    Output: features  (n_epochs, n_bands)   – column order: Delta,Theta,Alpha,Zaeta,Beta,Gamma
    """
    n_ch, n_samp    = data.shape
    epoch_samples   = int(epoch_sec * fs)
    step_samples    = int(epoch_samples * (1 - overlap))

    # Build epoch start indices (vectorised)
    starts = np.arange(0, n_samp - epoch_samples + 1, step_samples)
    n_epochs = len(starts)
    if n_epochs == 0:
        return np.empty((0, N_BANDS))

    # Stack all epochs: (n_epochs, n_ch, epoch_samples)
    idx     = starts[:, None] + np.arange(epoch_samples)[None, :]  # (n_epochs, epoch_samples)
    epochs  = data[:, idx]                                           # (n_ch, n_epochs, epoch_samples)
    epochs  = epochs.transpose(1, 0, 2)                              # (n_epochs, n_ch, epoch_samples)

    # Welch PSD for every epoch × channel simultaneously
    # signal.welch operates on the last axis
    freqs, psd = signal.welch(
        epochs.reshape(-1, epoch_samples),   # (n_epochs*n_ch, epoch_samples)
        fs=fs,
        nperseg=min(epoch_samples, 256),
        noverlap=128,
        axis=-1
    )
    # psd: (n_epochs * n_ch, n_freqs)
    psd = psd.reshape(n_epochs, n_ch, -1)   # (n_epochs, n_ch, n_freqs)

    # Frequency mask for total power (0.5 – 45 Hz)
    total_mask = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    total_psd  = psd[:, :, total_mask].sum(axis=-1)          # (n_epochs, n_ch)
    total_psd  = np.where(total_psd == 0, 1e-12, total_psd)  # avoid div/0

    # Per-band RBP: (n_epochs, n_ch, n_bands)
    band_rbp = np.zeros((n_epochs, n_ch, N_BANDS))
    for b_idx, (b_name, (b_lo, b_hi)) in enumerate(FREQ_BANDS.items()):
        mask = (freqs >= b_lo) & (freqs < b_hi)
        band_rbp[:, :, b_idx] = psd[:, :, mask].sum(axis=-1) / total_psd

    # Average over channels: (n_epochs, n_bands)  [Eq. 4]
    epoch_rbp = band_rbp.mean(axis=1)
    return epoch_rbp.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING  (parallel across subjects)
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_eeg(filepath: Path) -> np.ndarray:
    """
    Load a raw EEG file.
    Supported: .npy  /  .npz  /  .csv  /  .edf  (via MNE)
    Returns: (n_channels, n_samples)
    """
    ext = filepath.suffix.lower()

    if ext == ".npy":
        arr = np.load(filepath)
        return arr if arr.ndim == 2 and arr.shape[0] <= 64 else arr.T

    if ext == ".npz":
        d = np.load(filepath)
        arr = d[list(d.keys())[0]]
        return arr if arr.ndim == 2 and arr.shape[0] <= 64 else arr.T

    if ext == ".csv":
        arr = pd.read_csv(filepath, header=None).values
        return arr.T if arr.shape[0] > arr.shape[1] else arr

    # Try MNE for .set / .edf / .fif
    try:
        import mne
        raw = mne.io.read_raw(str(filepath), preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)
        return raw.get_data()
    except Exception as e:
        raise ValueError(f"Cannot load {filepath}: {e}")


def _process_one_subject(filepath: Path, label: int,
                         fs: float, epoch_sec: float,
                         overlap: float) -> tuple:
    try:
        raw   = load_raw_eeg(filepath)
        clean = preprocess_subject(raw, fs, skip_ica_asr=USE_DERIVATIVES)  # ← 加這個
        feats = compute_rbp_epochs(clean, fs, epoch_sec, overlap)
        n_ep  = len(feats)
        labels = np.full(n_ep, label, dtype=np.int64)
        return feats, labels
    except Exception as e:
        print(f"  [WARN] Skipping {filepath.name}: {e}")
        return np.empty((0, N_BANDS), dtype=np.float32), np.empty(0, dtype=np.int64)


def build_dataset(data_dir: Path,
                  fs: float = SFREQ,
                  epoch_sec: float = EPOCH_SEC,
                  overlap: float = OVERLAP,
                  n_jobs: int = N_JOBS,
                  use_derivatives: bool = True) -> tuple:

    participants_file = data_dir / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError(f"找不到 participants.tsv：{participants_file}")

    participants = pd.read_csv(participants_file, sep="\t")

    # 欄位名稱統一轉小寫
    participants.columns = participants.columns.str.lower().str.strip()
    print("[DEBUG] columns:", participants.columns.tolist())
    print(participants.head(3))

    tasks = []
    for _, row in participants.iterrows():
        sub_id = str(row["participant_id"]).strip()
        group  = str(row["group"]).strip()

        if group not in LABEL_MAP:
            print(f"  [WARN] 未知 group '{group}'，跳過 {sub_id}")
            continue
        label = LABEL_MAP[group]

        if use_derivatives:
            eeg_dir = data_dir / "derivatives" / sub_id / "eeg"
        else:
            eeg_dir = data_dir / sub_id / "eeg"

        if not eeg_dir.exists():
            print(f"  [WARN] 找不到 EEG 資料夾：{eeg_dir}，跳過")
            continue

        set_files = list(eeg_dir.glob("*.set"))
        if not set_files:
            print(f"  [WARN] {eeg_dir} 中沒有 .set 檔案，跳過")
            continue

        tasks.append((set_files[0], label))

    if not tasks:
        raise FileNotFoundError(
            f"在 {data_dir} 下找不到任何 EEG .set 檔案。"
            "請確認資料集已正確下載且路徑正確。")

    print(f"[INFO] 找到 {len(tasks)} 個受試者")
    worker = partial(_process_one_subject,
                     fs=fs, epoch_sec=epoch_sec, overlap=overlap)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(worker)(fp, lbl) for fp, lbl in tasks
    )
    elapsed = time.time() - t0
    print(f"[INFO] 預處理完成，耗時 {elapsed:.1f} 秒")

    X_list = [r[0] for r in results if len(r[0]) > 0]
    y_list = [r[1] for r in results if len(r[1]) > 0]
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"[INFO] Dataset: {X.shape[0]} epochs × {X.shape[1]} features  |  "
          f"classes {np.bincount(y).tolist()}")
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# 4.  NORMALISATION & SPLITTING  (paper Section 3.4)
# ─────────────────────────────────────────────────────────────────────────────

def min_max_normalise(X_train: np.ndarray,
                      X_val:   np.ndarray,
                      X_test:  np.ndarray) -> tuple:
    """Eq. (5): χ* = (χ − µ_min) / (µ_max − µ_min)"""
    mu_min = X_train.min(axis=0)
    mu_max = X_train.max(axis=0)
    denom  = np.where(mu_max - mu_min == 0, 1.0, mu_max - mu_min)

    def norm(X): return (X - mu_min) / denom

    return norm(X_train), norm(X_val), norm(X_test), mu_min, mu_max


def split_dataset(X: np.ndarray, y: np.ndarray,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio:   float = VAL_RATIO,
                  seed: int = 42) -> tuple:
    """Stratified 80 / 10 / 10 split (paper Section 3.4)."""
    from sklearn.model_selection import train_test_split
    rest = 1.0 - train_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=rest, stratify=y, random_state=seed)
    val_frac = val_ratio / rest
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL  –  TCN + LSTM  (paper Section 3.5 & Table 1)
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """
    One TCN residual block with dilated causal convolutions.
    Paper Eq. (6)-(8):
        H^l        = σ( W^l * X + b^l )
        H^l_t      = Σ W^l_i · X_{t−d·i} + b^l   (dilated)
        H^l_res    = H^l + X                        (residual)
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation           # causal padding

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Residual projection if channel dims differ
        self.residual = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

        self._pad = pad

    def _causal_trim(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future-looking padding to enforce causality."""
        return x[:, :, :-self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        out = self.conv1(x)
        out = self._causal_trim(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self._causal_trim(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + res


class TCN_LSTM(nn.Module):
    """
    Lightweight TCN-LSTM hybrid.
    Input shape:  (batch, seq_len=6, 1)   [6 RBP features treated as a sequence]
    Architecture (paper Table 1):
        Conv1D(32) → BN → ReLU → Dropout(0.2)  ⎫  TCN Block 1
        Conv1D(32) → BN → ReLU                  ⎭  + residual
        Conv1D(32) → BN → ReLU → Dropout(0.2)  ⎫  TCN Block 2
        Conv1D(32) → BN → ReLU                  ⎭  + residual
        LSTM(64)
        Dense(128) → Dropout(0.2)
        Dense(192) → Dropout(0.2)
        Dense(256) → Dropout(0.2)
        Dense(n_classes)
    Total params ≈ 131 587  (paper: 131 587)
    """
    def __init__(self, n_features: int = N_BANDS,
                 n_filters:  int = N_FILTERS,
                 kernel:     int = KERNEL_SIZE,
                 dilation:   int = DILATION,
                 dropout_tcn: float = DROPOUT_TCN,
                 lstm_units: int = LSTM_UNITS,
                 dense_units: list = DENSE_UNITS,
                 dropout_dense: float = DROPOUT_DENSE,
                 n_classes:  int = N_CLASSES):
        super().__init__()

        # 2 TCN blocks
        self.tcn1 = TCNBlock(1, n_filters, kernel, dilation, dropout_tcn)
        self.tcn2 = TCNBlock(n_filters, n_filters, kernel, dilation, dropout_tcn)

        # LSTM – expects (batch, seq, features)
        self.lstm = nn.LSTM(n_filters, lstm_units, batch_first=True)

        # Dense head
        layers = []
        in_dim = lstm_units
        for units in dense_units:
            layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(dropout_dense)]
            in_dim = units
        layers.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq=6, 1)  →  conv wants (batch, channels, seq)
        out = x.permute(0, 2, 1)         # (batch, 1, 6)
        out = self.tcn1(out)             # (batch, 32, 6)
        out = self.tcn2(out)             # (batch, 32, 6)
        out = out.permute(0, 2, 1)       # (batch, 6, 32)  for LSTM
        out, _ = self.lstm(out)          # (batch, 6, 64)
        out = out[:, -1, :]             # last time-step   (batch, 64)
        return self.head(out)            # (batch, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test,
                 batch_size: int = BATCH_SIZE) -> tuple:
    """Convert numpy arrays → PyTorch DataLoaders.
    Paper input shape: (6, 1) per sample."""
    def to_tensor(X, y):
        Xt = torch.from_numpy(X).unsqueeze(-1)         # (N, 6, 1)
        yt = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(Xt, yt)

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val,   y_val)
    test_ds  = to_tensor(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                n_epochs:     int = 100,
                patience:     int = 15) -> list:
    """
    Training with Adam(lr=1e-4) + CrossEntropyLoss.
    Early stopping on validation loss (paper Section 3.6).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                v_loss    += criterion(logits, yb).item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        t_loss /= t_total
        v_loss /= v_total

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | "
                  f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
                  f"val loss {v_loss:.4f} acc {v_acc:.4f}")

        # Early stopping
        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   class_names: list = CLASS_NAMES) -> dict:
    """Returns full classification metrics (paper Section 4.1)."""
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())
            all_probs.extend(probs)

    y_true  = np.array(all_true)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average=None)
    except Exception:
        auc = [float("nan")] * len(class_names)

    print(f"\n{'═'*55}")
    print(f"  TEST ACCURACY : {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"{'═'*55}\n")

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc=auc, confusion_matrix=cm,
                y_true=y_true, y_pred=y_pred, y_probs=y_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SMOTE BALANCING  (paper Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                seed: int = 42) -> tuple:
    """SMOTE oversampling on training set only (paper Section 4.3)."""
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] SMOTE: {np.bincount(y_train).tolist()} → {np.bincount(y_res).tolist()}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 8.  K-FOLD CROSS-VALIDATION  (paper Section 4.4)
# ─────────────────────────────────────────────────────────────────────────────

def kfold_cv(X: np.ndarray, y: np.ndarray,
             k: int = 5, n_epochs: int = 50) -> pd.DataFrame:
    """5-fold CV matching paper Table 10 & 11."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Simple normalisation within fold
        mu = X_tr.min(axis=0)
        mx = X_tr.max(axis=0)
        dn = np.where(mx - mu == 0, 1.0, mx - mu)
        X_tr = (X_tr - mu) / dn
        X_te = (X_te - mu) / dn

        # Mini val split from training
        val_size = max(1, int(0.1 * len(X_tr)))
        X_val_f, y_val_f = X_tr[:val_size], y_tr[:val_size]
        X_tr_f, y_tr_f   = X_tr[val_size:], y_tr[val_size:]

        tr_l, vl_l, te_l = make_loaders(
            X_tr_f, y_tr_f, X_val_f, y_val_f, X_te, y_te)

        n_cls = len(np.unique(y))
        model = TCN_LSTM(n_classes=n_cls).to(DEVICE)
        train_model(model, tr_l, vl_l, n_epochs=n_epochs, patience=10)

        # Training acc on full training set
        tr_all_l, _, _ = make_loaders(X_tr, y_tr, X_val_f, y_val_f, X_te, y_te)
        model.eval()
        c, tot = 0, 0
        with torch.no_grad():
            for Xb, yb in tr_all_l:
                c   += (model(Xb.to(DEVICE)).argmax(1).cpu() == yb).sum().item()
                tot += len(yb)
        tr_acc = c / tot

        te_res = evaluate_model(model, te_l)
        rows.append({
            "K-value": fold,
            "Training accuracy (%)": round(tr_acc * 100, 2),
            "Test accuracy (%)":     round(te_res["accuracy"] * 100, 2),
        })
        print(f"  Fold {fold}: train {tr_acc*100:.2f}%  test {te_res['accuracy']*100:.2f}%")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  VISUALISATION  (confusion matrix, ROC, training curves)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_roc(y_true: np.ndarray, y_probs: np.ndarray,
             class_names: list, title: str, save_path: Path):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes = sorted(set(y_true))
    y_bin   = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(class_names[:y_bin.shape[1]]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_history(history: dict, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SHAP  –  Explainable AI  (paper Section 5)
# ─────────────────────────────────────────────────────────────────────────────

def run_shap(model: nn.Module,
             X_test: np.ndarray,
             X_background: np.ndarray,
             band_names: list,
             class_names: list,
             save_dir: Path):
    model.eval()
    model.cpu()

    # Background sample (100 random points)
    bg_idx = np.random.choice(len(X_background),
                               min(100, len(X_background)), replace=False)
    bg = torch.from_numpy(X_background[bg_idx]).unsqueeze(-1).float()

    # ← 改用 GradientExplainer，對 TCN/LSTM 更相容
    explainer = shap.GradientExplainer(model, bg)

    # Test sample (up to 200 points for speed)
    te_idx = np.random.choice(len(X_test),
                               min(200, len(X_test)), replace=False)
    X_te_t = torch.from_numpy(X_test[te_idx]).unsqueeze(-1).float()

    shap_values = explainer.shap_values(X_te_t)
    # shap_values: list of (n_samples, seq_len=6, 1) per class

    # Squeeze the trailing dim
    shap_arr = [sv.squeeze(-1) for sv in shap_values]  # each: (n, 6)

    for cls_idx, cls_name in enumerate(class_names):
        sv = shap_arr[cls_idx]           # (n_samples, 6)
        mean_abs = np.abs(sv).mean(0)    # (6,) global importance

        # ── Bar chart ──
        fig, ax = plt.subplots(figsize=(7, 4))
        sorted_idx = np.argsort(mean_abs)
        ax.barh([band_names[i] for i in sorted_idx],
                mean_abs[sorted_idx], color="crimson")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"Global Feature Importance for Class: {cls_name}")
        for i, v in enumerate(mean_abs[sorted_idx]):
            ax.text(v, i, f" +{v:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        path = save_dir / f"shap_bar_{cls_name.lower()}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved → {path}")

        # ── Summary beeswarm ──
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(sv, features=X_test[te_idx],
                          feature_names=band_names,
                          show=False, plot_type="dot", color_bar=True,
                          max_display=N_BANDS)
        plt.title(f"SHAP Summary Plot for Class: {cls_name}")
        plt.tight_layout()
        path = save_dir / f"shap_summary_{cls_name.lower()}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved → {path}")

    model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  BINARY CLASSIFICATION HELPER  (paper Section 3.7)
# ─────────────────────────────────────────────────────────────────────────────

def run_binary_task(X: np.ndarray, y: np.ndarray,
                    positive_classes: list,
                    task_name: str,
                    n_epochs: int = 100) -> dict:
    """
    Run one of the four binary/multi-class tasks in the paper.
    positive_classes: class indices to merge as class 0;
                      remaining becomes class 1.
    """
    print(f"\n{'─'*55}")
    print(f"  Task: {task_name}")
    print(f"{'─'*55}")

    # Remap labels
    y_bin = np.where(np.isin(y, positive_classes), 0, 1)
    n_cls = len(np.unique(y_bin))

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y_bin, seed=42)
    X_train, X_val, X_test, *_ = min_max_normalise(X_train, X_val, X_test)

    tr_l, vl_l, te_l = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

    model   = TCN_LSTM(n_classes=n_cls).to(DEVICE)
    history = train_model(model, tr_l, vl_l, n_epochs=n_epochs, patience=15)
    metrics = evaluate_model(model, te_l,
                             class_names=[task_name.split("vs")[0].strip(),
                                          task_name.split("vs")[1].strip()])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        [task_name.split("vs")[0].strip(), task_name.split("vs")[1].strip()],
        title=f"Confusion Matrix – {task_name}",
        save_path=OUTPUT_DIR / f"cm_{task_name.replace(' ','_')}.png")

    plot_roc(
        metrics["y_true"], metrics["y_probs"],
        [task_name.split("vs")[0].strip(), task_name.split("vs")[1].strip()],
        title=f"ROC – {task_name}",
        save_path=OUTPUT_DIR / f"roc_{task_name.replace(' ','_')}.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SYNTHETIC DATA DEMO  (runs without the real dataset)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_subjects_per_class: int = 10,
                                seed: int = 0) -> tuple:
    """
    Generates plausible synthetic EEG RBP features for a quick end-to-end test.
    Real experiment: replace with build_dataset(DATA_DIR).
    """
    rng = np.random.default_rng(seed)
    print("[INFO] Using SYNTHETIC data (real dataset not found)")

    # Class-specific band power profiles (simplified clinical intuition)
    # AD:  ↑ delta/theta, ↓ beta
    # FTD: ↑ beta/zaeta (frontal disruption)
    # HC:  balanced
    profiles = {
        0: np.array([0.30, 0.25, 0.15, 0.12, 0.10, 0.08]),   # AD
        1: np.array([0.10, 0.12, 0.18, 0.25, 0.20, 0.15]),   # FTD
        2: np.array([0.15, 0.18, 0.22, 0.18, 0.15, 0.12]),   # HC
    }

    X_list, y_list = [], []
    n_epochs_per_subject = 120
    for label, profile in profiles.items():
        for _ in range(n_subjects_per_class):
            noise = rng.normal(0, 0.03, (n_epochs_per_subject, N_BANDS))
            feats = np.clip(profile + noise, 0, 1)
            # Renormalise rows to sum to ~1 (RBP property)
            feats = feats / feats.sum(axis=1, keepdims=True)
            X_list.append(feats.astype(np.float32))
            y_list.append(np.full(n_epochs_per_subject, label, dtype=np.int64))

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"[INFO] Synthetic dataset: {X.shape}  classes {np.bincount(y).tolist()}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TCN-LSTM EEG Dementia Classifier  –  Khan et al. 2025")
    print("=" * 60)

    # ── Load data ──
    if DATA_DIR.exists() and (DATA_DIR / "participants.tsv").exists():
        X, y = build_dataset(DATA_DIR, use_derivatives=USE_DERIVATIVES)
    else:
        print(f"[WARN] '{DATA_DIR}' not found. Falling back to synthetic data.")
        X, y = generate_synthetic_dataset(n_subjects_per_class=12)

    band_names = list(FREQ_BANDS.keys())   # ['Delta','Theta','Alpha','Zaeta','Beta','Gamma']

    # ─────────────────────────────────────────────────────────────
    # TASK 1: Alzheimer vs. Frontotemporal vs. Healthy  (Table 3)
    # ─────────────────────────────────────────────────────────────
    print("\n[TASK 1] Multi-class: AD vs FTD vs HC")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    X_train, X_val, X_test, mu_min, mu_max = min_max_normalise(
        X_train, X_val, X_test)

    tr_l, vl_l, te_l = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test)

    model   = TCN_LSTM(n_classes=3).to(DEVICE)
    history = train_model(model, tr_l, vl_l, n_epochs=150, patience=20)
    metrics = evaluate_model(model, te_l, CLASS_NAMES)

    plot_training_history(history,
        save_path=OUTPUT_DIR / "training_history_multiclass.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES,
        title="Confusion Matrix – AD vs FTD vs HC",
        save_path=OUTPUT_DIR / "cm_multiclass.png")
    plot_roc(metrics["y_true"], metrics["y_probs"], CLASS_NAMES,
        title="ROC – AD vs FTD vs HC",
        save_path=OUTPUT_DIR / "roc_multiclass.png")

    # ─────────────────────────────────────────────────────────────
    # TASK 2: (AD + FTD) vs Healthy  (Table 4)
    # ─────────────────────────────────────────────────────────────
    run_binary_task(X, y, positive_classes=[0, 1],
                    task_name="AD+FTD vs Healthy", n_epochs=100)

    # ─────────────────────────────────────────────────────────────
    # TASK 3: AD vs Healthy  (Table 5)
    # ─────────────────────────────────────────────────────────────
    mask = np.isin(y, [0, 2])
    run_binary_task(X[mask], y[mask], positive_classes=[0],
                    task_name="AD vs Healthy", n_epochs=100)

    # ─────────────────────────────────────────────────────────────
    # TASK 4: FTD vs Healthy  (Table 6)
    # ─────────────────────────────────────────────────────────────
    mask = np.isin(y, [1, 2])
    run_binary_task(X[mask], y[mask], positive_classes=[1],
                    task_name="FTD vs Healthy", n_epochs=100)

    # ─────────────────────────────────────────────────────────────
    # SMOTE  (Section 4.3)
    # ─────────────────────────────────────────────────────────────
    print("\n[SMOTE] Balancing training data…")
    X_sm, y_sm = apply_smote(X_train, y_train)
    tr_sm_l, _, _ = make_loaders(
        X_sm, y_sm, X_val, y_val, X_test, y_test)
    model_sm = TCN_LSTM(n_classes=3).to(DEVICE)
    train_model(model_sm, tr_sm_l, vl_l, n_epochs=100, patience=15)
    print("[SMOTE] Evaluation after balancing:")
    evaluate_model(model_sm, te_l, CLASS_NAMES)

    # ─────────────────────────────────────────────────────────────
    # K-FOLD CV  (Section 4.4)
    # ─────────────────────────────────────────────────────────────
    print("\n[K-FOLD] 5-fold cross-validation (multi-class)…")
    cv_df = kfold_cv(X, y, k=5, n_epochs=50)
    cv_df.to_csv(OUTPUT_DIR / "kfold_results.csv", index=False)

    # ─────────────────────────────────────────────────────────────
    # SHAP  (Section 5)
    # ─────────────────────────────────────────────────────────────
    print("\n[SHAP] Computing explanations…")
    try:
        run_shap(model, X_test, X_train, band_names, CLASS_NAMES, OUTPUT_DIR)
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}\n"
              "  Install shap and try again: pip install shap")

    print(f"\n[DONE] All outputs saved to → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()