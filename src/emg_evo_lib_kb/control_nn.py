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

from emg_evo_lib_kb import build_feature_dataset

#%% Data Loading & Splitting

def load_data_and_split(root, emg_column_names, valid_classes, test_size = 0.2, val_size = 0.1, random_state = 42, min_samples_per_class = 3):
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

#%% Function to Plot Confusion Matrix

def plot_confusion_matrix(cm, class_labels, title = "Confusion Matrix"):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: confusion matrix (n_classes x n_classes)
        class_labels: list of labels (e.g., [1,2,3,4,5,6,7])
        title: plot title
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%% Function for Training & Evaluation

def train_and_evaluate(root, emg_column_names, valid_classes, hidden_layers = (64, 32)):
    """
    Full training + evaluation pipeline for the control neural network.

    Steps:
        - Load dataset using preprocessing.build_feature_dataset
        - Split into train/val/test
        - Train MLP (with StandardScaler)
        - Evaluate on val and test sets
        - Print metrics and plot confusion matrix on test set
    """
    # 1. Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_and_split(root, emg_column_names, valid_classes)

    input_dim = X_train.shape[1]
    print(f"\nInput feature dimension: {input_dim}")

    # 2. Build model
    model = build_mlp_classifier(input_dim=input_dim, hidden_layers=hidden_layers)

    # 3. Train model (on train only)
    print("\nTraining MLP classifier...")
    model.fit(X_train, y_train)

    # 4. Evaluate on validation set
    print("\n--- Validation Performance ---")
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    # 5. Evaluate on test set (final hold-out)
    print("\n--- Test Performance ---")
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    print("Classification report (Test):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=valid_classes,
            digits=3,
        )
    )

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=valid_classes)
    plot_confusion_matrix(cm, class_labels=valid_classes, title="MLP Confusion Matrix (Test Set)")
