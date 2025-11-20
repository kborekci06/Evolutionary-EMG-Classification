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