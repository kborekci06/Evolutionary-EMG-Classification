#%% Imports
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocessing import build_feature_dataset

#%% Data Loading & Splitting

def load_data_and_split(root, emg_column_names, valid_classes, test_size = 0.2, val_size = 0.1, random_state = 42):
    """
    Load features & labels using preprocessing.build_feature_dataset
    and split into train, validation, and test sets.

    Args:
        root: dataset root path (same as DATA_ROOT in preprocessing.py)
        test_size: fraction of data to hold out for final test
        val_size: fraction of remaining train to use for validation
        random_state: for reproducible splitting

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y, meta = build_feature_dataset(root, emg_column_names, valid_classes, verbose=True)

    print("\nUnique labels in y:", np.unique(y))
    print("Total samples:", len(y))

    # First: train + temp_test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then split temp into val + test
    val_ratio = val_size / (1.0 - test_size)  # fraction of X_temp to use as val
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

#%% Model Definition Multi Layer Perceptron (NN for Classification)

def build_mlp_classifier(input_dim, hidden_layers = (64, 32), random_state = 42):
    """
    Build a standard neural network classifier (MLP) with backprop,
    wrapped in a Pipeline that includes feature scaling.

    Args:
        input_dim: number of input features (should be 64 here)
        hidden_layers: sizes of hidden layers, e.g. (64, 32)
        random_state: for reproducible weight initialization

    Returns:
        A scikit-learn Pipeline: StandardScaler -> MLPClassifier
    """

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-3,             # L2 regularization
        batch_size="auto",
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,  # internal validation (inside training set)
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", mlp),
        ]
    )

    return pipeline

