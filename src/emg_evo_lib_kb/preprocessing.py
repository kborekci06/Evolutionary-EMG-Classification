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
    df = pd.read_csv(path, delim_whitespace=True, header=None, names=emg_column_names)
    return df

# Iterate through folder of emg files
def iter_emg_files(root, pattern = "*.txt"):
    """
    Recursively find all EMG text files under a root directory.
    """
    return sorted(root.rglob(pattern))

#%%