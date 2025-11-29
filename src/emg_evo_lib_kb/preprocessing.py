#%% Imports
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#%% Data Load Functions

# Load a single emg file
def load_emg_file(path, emg_column_names):
    """
    Load a single EMG text file into a DataFrame.

    Assumes:
    - whitespace-delimited
    - 10 columns: Time, ch1..ch8, Class
    """
    df = pd.read_csv(path, delim_whitespace=True, header=0, names=emg_column_names)
    return df

# Iterate through folder of emg files
def iter_emg_files(root, pattern = "*.txt"):
    """
    Recursively find all EMG text files under a root directory.
    """
    return sorted(root.rglob(pattern))

#%% Segment Function
def segment_gestures(df, valid_classes):
    """
    Segment a continuous EMG recording into gesture segments.

    Each segment is a contiguous run of the same non-zero class.
    We ignore class 0 (unmarked / in-between).

    Returns:
        List of (segment_df, class_label)
    """
    segments = []

    current_label = 0
    start_idx = None

    class_series = df["Class"].to_numpy()

    for i, label in enumerate(class_series):
        if label != current_label:
            # End of a gesture segment
            if current_label in valid_classes and start_idx is not None:
                end_idx = i  # exclusive
                seg = df.iloc[start_idx:end_idx]
                segments.append((seg, int(current_label)))

            # Start of a new gesture segment
            if label in valid_classes:
                start_idx = i
            else:
                start_idx = None

            current_label = label

    # Handle segment that might continue until end of file
    if current_label in valid_classes and start_idx is not None:
        seg = df.iloc[start_idx:]
        segments.append((seg, int(current_label)))

    # segments is a List of (segment_df, class_label)
    return segments

#%% Returns sampling rate from time column
def estimate_sampling_rate(time_ms):
    """
    Estimate sampling rate [Hz] from time column in milliseconds.
    """
    if len(time_ms) < 2:
        return 1.0  # fallback, won't matter much for FFT shape
    dt_ms = np.median(np.diff(time_ms))
    if dt_ms <= 0:
        return 1.0
    fs = 1000.0 / dt_ms  # convert ms -> s
    return fs

#%% Function that computes the features of 1 EMG channel signal (=x)
def channel_features(x, fs):
    """
    Compute 8 features for a single EMG channel signal x:

    1. RMS  (Root Mean Square)
    2. MAV  (Mean Absolute Value)
    3. WL   (Waveform Length)
    4. ZC   (Zero Crossings with threshold)
    5. SSC  (Slope Sign Changes with threshold)
    6. VAR  (Variance)
    7. MNF  (Mean Frequency, from power spectrum)
    8. MDF  (Median Frequency, from power spectrum)

    Returns:
        np.array of shape (8,)
    """
    x = x.astype(float)
    N = len(x)

    if N < 3:
        # Degenerate case; return zeros
        return np.zeros(8, dtype=float)

    # --- Time-domain features ---

    # 1. RMS
    rms = np.sqrt(np.mean(x**2))

    # 2. MAV (Mean absolute value)
    mav = np.mean(np.abs(x))

    # 3. WL (Waveform length)
    wl = np.sum(np.abs(np.diff(x)))

    # Set small threshold relative to signal amplitude
    thr = 0.01 * np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 0.0

    # 4. ZC (Zero crossings)
    x1 = x[:-1]
    x2 = x[1:]
    sign_change = (x1 * x2) < 0
    above_thr = (np.abs(x1 - x2) > thr)
    zc = np.sum(sign_change & above_thr)

    # 5. SSC (Slope sign changes)
    x_prev = x[:-2]
    x_curr = x[1:-1]
    x_next = x[2:]
    ssc_cond = ((x_curr - x_prev) * (x_curr - x_next) > 0)
    ssc_thr = (np.abs(x_curr - x_prev) > thr) | (np.abs(x_curr - x_next) > thr)
    ssc = np.sum(ssc_cond & ssc_thr)

    # 6. VAR (Variance)
    var = np.var(x)

    # --- Frequency-domain features via FFT ---

    # rFFT (one-sided spectrum)
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2  # power spectrum
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Avoid division by zero
    total_power = np.sum(P)
    if total_power <= 0:
        mnf = 0.0
        mdf = 0.0
    else:
        # 7. MNF (Mean Frequency)
        mnf = np.sum(freqs * P) / total_power

        # 8. MDF (Median Frequency)
        cumulative_power = np.cumsum(P)
        mdf_idx = np.searchsorted(cumulative_power, 0.5 * total_power)
        if mdf_idx >= len(freqs):
            mdf_idx = len(freqs) - 1
        mdf = freqs[mdf_idx]

    return np.array([rms, mav, wl, zc, ssc, var, mnf, mdf], dtype=float)


#%% Feature Extraction (64-D per segment)
def compute_segment_features(segment):
    """
    Compute features for a single gesture segment.

    For each of the 8 EMG channels, we compute:

        1. RMS  (Root Mean Square)
        2. MAV  (Mean Absolute Value)
        3. WL   (Waveform Length)
        4. ZC   (Zero Crossings)
        5. SSC  (Slope Sign Changes)
        6. VAR  (Variance)
        7. MNF  (Mean Frequency)
        8. MDF  (Median Frequency)

    That is 8 features × 8 channels = 64 features.
    Features are concatenated channel-wise:
        [feat_ch1_1..8, feat_ch2_1..8, ..., feat_ch8_1..8]
    """
    ch_cols = [f"ch{i}" for i in range(1, 9)]
    data = segment[ch_cols].to_numpy()  # shape (T, 8)
    time_ms = segment["Time"].to_numpy()

    fs = estimate_sampling_rate(time_ms)
    n_channels = data.shape[1]

    feat_list = []
    for ch in range(n_channels):
        x = data[:, ch]
        feats_ch = channel_features(x, fs)
        feat_list.append(feats_ch)

    features = np.concatenate(feat_list, axis=0)  # shape (8_channels * 8_features = 64,)
    return features

#%% Function to build the full feature dataset

def build_feature_dataset(root, emg_column_names, valid_classes,
                          file_pattern="*.txt", verbose=True):
    """
    Build the full feature dataset from all EMG text files.

    Steps:
    - Find all files under `root` matching `file_pattern`.
    - For each file:
        - Load data
        - Segment into gestures
        - Extract 64-dim features for each segment

    Returns:
        X : np.ndarray of shape (N_samples, 64)
        y : np.ndarray of shape (N_samples,)
        meta : list of dicts with metadata per sample
               (e.g., file path, segment index, subject ID)
    """
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    meta: List[Dict] = []

    files = iter_emg_files(root, pattern=file_pattern)
    if verbose:
        print(f"Found {len(files)} EMG files under {root}")

    for file_idx, path in enumerate(files):
        df = load_emg_file(path, emg_column_names)
        segments = segment_gestures(df, valid_classes)

        if verbose:
            print(f"[{file_idx+1}/{len(files)}] {path.name}: {len(segments)} gesture segments")

        # Try to infer subject from parent folder name (optional)
        subject_id = path.parent.name

        for seg_idx, (segment_df, label) in enumerate(segments):
            feat = compute_segment_features(segment_df)
            all_features.append(feat)
            all_labels.append(label)
            meta.append(
                {
                    "file_path": str(path),
                    "subject_id": subject_id,
                    "segment_index": seg_idx,
                    "label": label,
                    "num_samples": len(segment_df),
                }
            )

    X = np.vstack(all_features) if all_features else np.empty((0, 0))
    y = np.array(all_labels, dtype=int)

    if verbose:
        print(f"Total segments: {len(y)}")
        print(f"Feature matrix shape: {X.shape}")  # should be (N_segments, 64)

    return X, y, meta