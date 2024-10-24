from glob import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import create_model, load_model
from osl_foundation.simulation.bursts import Bursts

tf_ops.gpu_growth()

# Results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

simulate_data = True
train_tokenizer = True
train_generator = True

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

if simulate_data:
    # Simulate and Save data
    bursts.simulate_and_save()

# Plot summary of the simulated data
bursts.plot_data(plot_dir=simulation_config["data_dir"])

# Load the simulated data
data = Data(
    sorted(glob(f"{simulation_config['data_dir']}/*.npy")),
    store_dir="data_tmp",
    use_tfrecord=True,
    n_jobs=8,
)

# Standardize the data
data.standardize()

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

    # Plot results of tokenizer
    tokenizer.refactor_vocab(data)
    tokenizer.plot_pve(data=data, plot_dir=results_dir)
    tokenizer.plot_token_counts(plot_dir=results_dir)
    tokenizer.plot_token_response(plot_dir=results_dir)
    tokenizer.plot_fitted_signal(simulation_config["data_dir"], plot_dir=results_dir)


if train_generator:
    generator_config = f"""
        model_config:
            name: ephys_gpt
            tokenizer_path: model/tokenizer
            sequence_length: 256
            n_channels: {n_channels}
            input_parameters:
                embedding_dim: 256
                n_tokens: 128
            decoder_parameters:
                n_layers: 4
                n_heads: 8
                model_dim: 256
                latent_sequence_length: 128
                n_patches: 64
                patch_length: 4
                unpatched_length: 4
                channel_attention_dropout: 0.0
                within_channel_attention_dropout: 0.0
                feed_forward_dim: 64
                feed_forward_activation: selu
                dropout: 0.0
                norm_type: group
                n_groups: 8
            loss_parameters:
                loss_sequence_length: 64
        training_config:
            optimizer:
                learning_rate: 0.0001
            batch_size: 32
            n_epochs: 30
            lr_decay: 0.1
    """
    generator = create_model(generator_config)
    generator.summary()

    generator.fit(
        data,
        validation_split=0.1,
        use_tfrecord=True,
        n_jobs=8,
        step_size=generator.config.model_config.sequence_length // 4,
    )
    os.makedirs("model/ephys_gpt", exist_ok=True)
    generator.save("model/ephys_gpt")
else:
    generator = load_model("model/ephys_gpt")

# Plot history
generator.plot_history(plot_dir=results_dir, keyword="loss")
generator.plot_history(plot_dir=results_dir, keyword="accuracy")

# Generate data
generated_data = np.concatenate(
    generator.generate_data(n_samples=2048, method="top_p", p=0.8, batch_size=64),
    axis=0,
)

# Plot PSD of generated data
fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
for i in range(n_channels):
    axes[i].psd(generated_data[:, i], Fs=100, NFFT=1024, color="black")
    axes[i].set_ylabel(f"Channel {i}")
fig.tight_layout()
fig.savefig(f"{results_dir}/psd_generated_data.png")
plt.close(fig)

# Clean up directories
data.delete_dir()
