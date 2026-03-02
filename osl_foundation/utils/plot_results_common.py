from glob import glob
import pickle

import numpy as np

from osl_foundation import load_model


def load_pickled_series(generator_dir: str, prefix: str):
    """Load a sorted list of pickled arrays/objects by file prefix."""
    files = sorted(glob(f"{generator_dir}/{prefix}_*.pkl"))
    values = []
    for filepath in files:
        with open(filepath, "rb") as file_obj:
            values.append(pickle.load(file_obj))
    return values


def load_generator_checkpoint_first(generator_dir: str):
    """Load model preferring latest checkpoint, fallback to model.weights.h5."""
    try:
        return load_model(generator_dir, checkpoint="latest")
    except Exception as e:
        print(
            "Falling back to model.weights.h5 load because latest checkpoint restore failed: "
            f"{e}"
        )
        return load_model(generator_dir)


def load_and_standardize_original_data(
    data_dir: str,
    n_sessions: int,
    n_samples_per_session: int,
):
    """Load x*.npy files, z-score per session, and trim to generated length."""
    data_files = sorted(glob(f"{data_dir}/x*.npy"))[:n_sessions]
    original_data = []
    for filepath in data_files:
        dat = np.load(filepath)
        dat_std = dat.std()
        if dat_std > 0:
            dat = (dat - dat.mean()) / dat_std
        else:
            dat = dat - dat.mean()
        original_data.append(dat[:n_samples_per_session, :])
    return original_data
