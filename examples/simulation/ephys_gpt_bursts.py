from glob import glob
import os

from osl_dynamics.inference import tf_ops

from osl_foundation import get_config, create_model, load_model
from osl_foundation.simulation.bursts import Bursts
from osl_foundation.data import Data

tf_ops.gpu_growth()

# # Simulation parameters
simulation_config = {
    "n_groups": 2,
    "n_subjects_per_group": 10,
    "n_channels_per_mode": 5,
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
data = Data(
    sorted(glob(f"{simulation_config['data_dir']}/*.npy")), store_dir="sim_data"
)

# Standardize the data
data.standardize()

# Concatenate the channels
data.concatenate_channels()

train_tokenizer = True
# Model and training configuration
if train_tokenizer:
    configuration = """
        model_config:
            name: osl_tokenizer
            n_tokens: 128
            token_dim: 32
            rnn_n_units: 128
        training_config:
            optimizer:
                learning_rate: 0.005
            batch_size: 256
            n_epochs: 40
            temperature_annealing:
                n_stages: 40
            lr_decay: 0.1
    """
    config = get_config(configuration)

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
tokenized_data = Data(tokenizer.tokenize_data(data)[0], store_dir="tokenized_data")

train_generator = True
if train_generator:
    configuration = f"""
        model_config:
            name: ephys_gpt
            sequence_length: 256
            input_parameters:
                embedding_dim: 256
                n_tokens: 128
                n_channels: {tokenized_data.n_channels}
            decoder_parameters:
                n_layers: 1
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
    """
    generator_config = get_config(configuration)
    generator = create_model(generator_config)
    generator.summary()

    # Split data into train and validation sets
    train_data, val_data = tokenized_data.dataset(
        sequence_length=generator_config.model_config.sequence_length,
        batch_size=generator_config.training_config.batch_size,
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
