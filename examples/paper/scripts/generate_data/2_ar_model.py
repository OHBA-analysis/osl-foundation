import os
import pickle

import numpy as np
from tqdm.auto import trange


def generate_data(
    ar_models: list, n_samples: int, n_sessions: int, noise: float
) -> np.ndarray:
    """Generate data using autoregressive models.

    Parameters
    ----------
    ar_models : list
        List of autoregressive models for each channel.
    n_samples : int
        Number of samples to generate for each session.
    n_sessions : int
        Number of sessions to generate data for.
    noise : float
        Standard deviation of the noise to add to the generated data.

    Returns
    -------
    generated_data : np.ndarray
        Generated data with shape (n_sessions, n_samples, n_channels).
    """
    n_channels = len(ar_models)
    prompt = np.random.normal(size=(n_channels, n_sessions, 80))

    generated_data = np.empty((n_channels, n_sessions, n_samples + 80))
    generated_data[:, :, :80] = prompt

    for i in trange(n_channels, desc="Generating data"):
        model = ar_models[i]
        for j in range(80, 80 + n_samples):
            model_input = generated_data[i, :, j - 80 : j]
            predicted = model.predict(model_input) + noise[i] * np.random.normal(
                size=n_sessions
            )
            generated_data[i, :, j] = predicted
    return np.transpose(generated_data[:, :, 80:], (1, 2, 0))


if __name__ == "__main__":
    ar_model_dir = "../../models/ar_model"
    with open(f"{ar_model_dir}/models.pkl", "rb") as f:
        models = pickle.load(f)

    se = np.load(f"{ar_model_dir}/accuracy.npz")["se"]
    generated_data = generate_data(models, n_samples=15000, n_sessions=612, noise=se)

    # Save the generated data
    save_dir = "../../data/generated_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/ar_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(generated_data, f)
