from glob import glob
import os

from osl_dynamics.inference import tf_ops

from osl_foundation import create_model
from osl_foundation.simulation.bursts import Bursts
from osl_foundation.data import Data

tf_ops.gpu_growth()

# Simulation parameters
simulation_config = {
    "n_groups": 1,
    "n_subjects_per_group": 1,
    "n_channels_per_mode": 1,
    "n_modes": 2,
    "n_samples": 512000,
    "sampling_frequency": 100,
    "snr": 3.0,
    "data_dir": "sim_data",
}

bursts = Bursts(**simulation_config)

# Simulate and Save data
bursts.simulate_and_save()

# Plot summary of the simulated data
bursts.plot_data()

# Load the simulated data
data = Data(sorted(glob(f"{simulation_config['data_dir']}/*.npy")), use_tfrecord=True)

# Standardize the data
data.standardize()

# Concatenate the channels
data.concatenate_channels()

# Model and training configuration
config = """
    model_config:
        name: osl_tokenizer
        n_tokens: 128
        token_dim: 32
        rnn_n_units: 128
    training_config:
        optimizer:
            learning_rate: 0.005
        batch_size: 64
        n_epochs: 40
        temperature_annealing:
            n_stages: 40
        lr_decay: 0.1
"""
# Build model
model = create_model(config)

# Print summary of model
model.summary()

# Fit model
model.fit(data)

# Save model
os.makedirs("results", exist_ok=True)
model.save("results")

# Percentage of Explained Variance
pve = model.get_pve(data)
print(f"Percentage of Explained Variance: {pve.mean():.2f}% ({pve.std():.2f}%)")

# Clean up data directory
data.delete_dir()
