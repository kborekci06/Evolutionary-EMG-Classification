#%% Imports
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import neat

from preprocessing import build_feature_dataset
from control_nn import plot_confusion_matrix
# %% Load data and split for NEAT

def load_data_and_split_neat(root, emg_column_names, valid_classes,
                             test_size = 0.2, val_size = 0.1, 
                             random_state = 42):
    """
    Load EMG features and labels (classes 1-6),
    then split into train, val, and test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y, meta = build_feature_dataset(root, emg_column_names, valid_classes, verbose=True)

    print("\nUnique labels in y:", np.unique(y))
    print("Total samples:", len(y))

    # First: train + temp (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Second: temp -> val + test (we can skip stratify here to avoid tiny class issues)
    val_ratio = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_ratio,
        stratify=None,
        random_state=random_state,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

#%% NEAT Evaluation Helper Functions

def evaluate_genome(genome, config, X, y, valid_classes):
    """
    Compute classification accuracy of a single genome on (X, y).

    Assumes:
        - X: shape (N_samples, 64)
        - y: labels in {1,2,3,4,5,6}
        - Genome corresponds to a FeedForwardNetwork with 64 inputs, 6 outputs.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    correct = 0
    total = len(y)

    for xi, yi in zip(X, y):
        output = net.activate(xi.tolist())   # 6 raw outputs
        pred_idx = int(np.argmax(output))   # 0..5
        pred_label = valid_classes[pred_idx]  # map index -> class label (1..6)
        if pred_label == yi:
            correct += 1

    return correct / total if total > 0 else 0.0
