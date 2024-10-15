from glob import glob
import os

import numpy as np

from osl_dynamics.inference import tf_ops

from osl_foundation import create_model, load_model
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
data = Data(
    sorted(glob(f"{simulation_config['data_dir']}/*.npy")),
    store_dir="data_tmp",
    use_tfrecord=True,
    n_jobs=8,
)

# Standardize the data
data.standardize()

train_tokenizer = False
# Model and training configuration
if train_tokenizer:
    config = f"""
        model_config:
            name: osl_tokenizer
            sequence_length: 256
            n_channels: {n_channels}
            n_tokens: 128
            token_dim: 32
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
    tokenizer = create_model(config)

    # Print summary of model
    tokenizer.summary()

    # Fit model
    tokenizer.fit(data)

    # Save model
    os.makedirs("model/tokenizer", exist_ok=True)
    tokenizer.save("model/tokenizer")
else:
    # Load model
    tokenizer = load_model("model/tokenizer")

# Percentage of Explained Variance
pve = tokenizer.get_pve(data)
print(f"Percentage of Explained Variance: {pve.mean():.2f}% ({pve.std():.2f}%)")


# Tokenize the data
tokenized_data = Data(
    tokenizer.tokenize_data(data)[0],
    store_dir="tokenized_data_tmp",
    use_tfrecord=True,
    n_jobs=8,
)

train_generator = True
if train_generator:
    generator_config = f"""
        model_config:
            name: ephys_gpt
            sequence_length: 256
            n_channels: {tokenized_data.n_channels}
            input_parameters:
                embedding_dim: 256
                n_tokens: 128
            decoder_parameters:
                n_layers: 2
                n_heads: 4
                model_dim: 256
                latent_sequence_length: 128
                n_patches: 64
                patch_length: 4
                unpatched_length: 4
                channel_attention_dropout: 0.0
                within_channel_attention_dropout: 0.0
                feed_forward_dim: 64
                feed_forward_activation: relu
                dropout: 0.0
            loss_parameters:
                loss_sequence_length: 64
        training_config:
            optimizer:
                learning_rate: 0.001
            batch_size: 64
            n_epochs: 20
            lr_decay: 0.1
            # multi_gpu: True
    """
    generator = create_model(generator_config)
    generator.summary()

    # Split data into train and validation sets
    train_data, val_data = tokenized_data.tfrecord_dataset(
        sequence_length=generator.config.model_config.sequence_length,
        batch_size=generator.config.training_config.batch_size,
        validation_split=0.2,
    )
    generator.fit(train_data, validation_data=val_data)
    os.makedirs("model/ephys_gpt", exist_ok=True)
    generator.save("model/ephys_gpt")
else:
    generator = load_model("model/ephys_gpt")

# TODO: Generate data

# Clean up directories
data.delete_dir()
tokenized_data.delete_dir()
