from glob import glob
from typing import Tuple

import numpy as np

from osl_dynamics.data import Data, load_tfrecord_dataset


def load_tfrecord_train_val_data(
    data_dir: str,
    batch_size: int,
    buffer_size: int = 2000,
    drop_last_batch: bool = True,
    concatenate: bool = True,
):
    """Load train/validation TFRecord datasets with common defaults."""
    return load_tfrecord_dataset(
        data_dir,
        batch_size=batch_size,
        buffer_size=buffer_size,
        drop_last_batch=drop_last_batch,
        concatenate=concatenate,
    )


def extract_data_tensor_from_batch(batch):
    """Extract the main `data` tensor from a dataset batch structure."""
    if isinstance(batch, dict):
        return batch["data"]
    if isinstance(batch, (tuple, list)):
        for item in batch:
            if isinstance(item, dict) and "data" in item:
                return item["data"]
        return batch[0]
    return batch


def dataset_shape_matches(
    train_dataset,
    expected_seq_len: int,
    expected_n_channels: int,
) -> Tuple[bool, Tuple[int, int]]:
    """
    Check if a TF dataset's data tensor matches expected (seq_len, n_channels).

    Returns
    -------
    matches : bool
    found_shape : tuple[int, int]
        Found (seq_len, n_channels).
    """
    first_batch = next(iter(train_dataset.take(1)))
    data_tensor = extract_data_tensor_from_batch(first_batch)
    found_seq_len = int(data_tensor.shape[1])
    found_n_channels = int(data_tensor.shape[2])
    matches = (found_seq_len, found_n_channels) == (
        expected_seq_len,
        expected_n_channels,
    )
    return matches, (found_seq_len, found_n_channels)


def rebuild_tfrecord_dataset_from_token_files(
    tokenized_data_dir: str,
    tfrecord_dir: str,
    sequence_length: int,
    validation_split: float = 0.1,
    n_jobs: int = 16,
    session_label_name: str = "session_id",
) -> None:
    """
    Rebuild TFRecord datasets from already-tokenized per-session .npy files.
    """
    tokenized_data_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
    if len(tokenized_data_files) == 0:
        raise FileNotFoundError(
            f"No tokenized files found in {tokenized_data_dir} (expected x_*.npy)."
        )

    tokenized_data = Data(tokenized_data_files, n_jobs=n_jobs)
    tokenized_data.add_session_labels(
        session_label_name,
        np.arange(tokenized_data.n_sessions),
        "categorical",
    )
    tokenized_data.save_tfrecord_dataset(
        tfrecord_dir=tfrecord_dir,
        sequence_length=sequence_length,
        overwrite=True,
        validation_split=validation_split,
    )
