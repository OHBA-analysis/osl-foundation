"""
This script is for training the linear AR model on the parcel time courses
of the Cam-CAN resting-state dataset.
"""

import os
import pickle
from typing import Tuple

import numpy as np
from tqdm.auto import trange
from sklearn.linear_model import SGDRegressor

from osl_dynamics.data import load_tfrecord_dataset


# ---------- Helper functions ---------- #
def load_data(
    data_path: str, overwrite: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    data_path : str
        Path where the data should be loaded/saved.
    overwrite : bool
        Whether to overwrite existing data.

    Returns
    -------
    X_train : np.ndarray
        Training features. Shape is (n_train_samples, sequence_length).
    X_test : np.ndarray
        Testing features. Shape is (n_test_samples, sequence_length).
    y_train : np.ndarray
        Training targets. Shape is (n_train_samples,).
    y_test : np.ndarray
        Testing targets. Shape is (n_test_samples,).
    """
    if overwrite or not os.path.exists(data_path):
        tfrecord_dir = "../../data/tfrecords"
        train_data, val_data = load_tfrecord_dataset(
            tfrecord_dir,
            batch_size=16,
            buffer_size=2000,
            drop_last_batch=True,
            concatenate=True,
        )

        d_train = []
        for d in train_data:
            d_train.append(d["data"].numpy())
        d_train = np.concatenate(d_train, axis=0)
        d_train = np.transpose(d_train, (2, 0, 1))

        X_train = d_train[:, :, :-1]
        y_train = d_train[:, :, -1]

        d_test = []
        for d in val_data:
            d_test.append(d["data"].numpy())
        d_test = np.concatenate(d_test, axis=0)
        d_test = np.transpose(d_test, (2, 0, 1))

        X_test = d_test[:, :, :-1]
        y_test = d_test[:, :, -1]

        # Standardize the data
        X_train_mean = X_train.mean(axis=1, keepdims=True)
        X_train_std = X_train.std(axis=1, keepdims=True)
        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

        y_train_mean = y_train.mean(axis=1, keepdims=True)
        y_train_std = y_train.std(axis=1, keepdims=True)
        y_train = (y_train - y_train_mean) / y_train_std
        y_test = (y_test - y_train_mean) / y_train_std

        # Save the data
        np.savez(
            data_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_train_mean=X_train_mean,
            X_train_std=X_train_std,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
        )

    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    return X_train, X_test, y_train, y_test


# ---------- Load data ---------- #
data_path = "../../data/ar_data.npz"
X_train, X_test, y_train, y_test = load_data(data_path)

# ---------- Train AR model ---------- #
ar_model_dir = "../../models/ar_model"
os.makedirs(ar_model_dir, exist_ok=True)
models = []
n_channels = X_train.shape[0]
for i in trange(n_channels, desc="Training linear AR models"):
    model = SGDRegressor(
        penalty="l2",
        alpha=0.0001,
        learning_rate="invscaling",
        max_iter=20,
        verbose=1,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train[i], y_train[i])
    models.append(model)

# Save the models
with open(f"{ar_model_dir}/models.pkl", "wb") as f:
    pickle.dump(models, f)

# ---------- Get accuracies and standard errors ---------- #
scores = []
se = []
for i in range(n_channels):
    model = models[i]
    residual = (y_train[i] - model.predict(X_train[i])) ** 2
    residual = np.mean(residual)
    residual = np.sqrt(residual)
    se.append(residual)

    score = model.score(X_test[i], y_test[i])
    scores.append(score)
scores = np.array(scores)
se = np.array(se)

np.savez(
    f"{ar_model_dir}/accuracy.npz",
    scores=scores,
    se=se,
)
