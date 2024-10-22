from glob import glob
import os

import numpy as np

from osl_dynamics.inference import tf_ops

from osl_foundation import create_model
from osl_foundation.simulation.bursts import Bursts
from osl_foundation.data import Data

tf_ops.gpu_growth()

# Simulation parameters
simulation_config = {
    "n_groups": 3,
    "n_subjects_per_group": 20,
    "true_freqs": np.array([3.0, 6.0, 10.0, 20.0]),
    "sampling_frequency": 100,
    "snr": 3.0,
    "data_dir": "sim_data",
}
n_modes = simulation_config["true_freqs"].shape[0]
n_channels = 12
channel_activity = np.zeros((n_modes, n_channels))
channel_activity[:2] = 1
channel_activity[:, : n_channels // 3] = 1
channel_activity[2, n_channels // 3 : 2 * n_channels // 3] = 1
channel_activity[3, 2 * n_channels // 3 :] = 1
simulation_config["channel_activity"] = channel_activity

# 5 minutes of data
simulation_config["n_samples"] = 5 * 60 * simulation_config["sampling_frequency"]

bursts = Bursts(**simulation_config)

# Simulate and Save data
bursts.simulate_and_save()

# Plot summary of the simulated data
bursts.plot_data()

# Load the simulated data
data = Data(sorted(glob(f"{simulation_config['data_dir']}/*.npy")), use_tfrecord=True)

# Standardize the data
data.standardize()

# # Concatenate the channels
# data.concatenate_channels()

# Model and training configuration
config = f"""
    model_config:
        name: osl_tokenizer
        sequence_length: 256
        n_channels: {n_channels}
        n_tokens: 128
        token_dim: 10
        rnn_n_units: 128
    training_config:
        optimizer:
            learning_rate: 0.005
        batch_size: 32
        n_epochs: 40
        temperature_annealing:
            n_stages: 40
        lr_decay: 0.1
        multi_gpu: True
"""

# Build model
model = create_model(config)

# Print summary of model
model.summary()

# Fit model
model.fit(data)

# Save model
results_dir = "results/figures"
os.makedirs(results_dir, exist_ok=True)
model.save("results")

# Plot percentage of explained variance
pve = model.get_pve(data)
print(f"Percentage of Explained Variance: {pve.mean():.2f}% ({pve.std():.2f}%)")
model.plot_pve(data, plot_dir=results_dir)

# Plot token counts
model.plot_token_counts(data, plot_dir=results_dir)

# Plot stimulus response of token kernels
model.plot_token_response(data, plot_dir=results_dir)

# Plot signals reconstructed from tokenized data
model.plot_fitted_signal(sess_id=0, plot_dir=results_dir)

# Clean up data directory
data.delete_dir()
