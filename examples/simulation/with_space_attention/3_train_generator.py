import os
from glob import glob

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import create_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = "sim_data"
plot_dir = "plots/generator"
tokenizer_dir = "models/tokenizer"
generator_dir = "models/generator"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(generator_dir, exist_ok=True)

# ---------- Load data ---------- #
data = Data(sorted(glob(f"{data_dir}/*.npy")))

# Standardize the data
data.standardize()

# ---------- Build generator ---------- #
generator_config = f"""
    model_config:
        name: ephys_gpt
        tokenizer_path: {tokenizer_dir}
        sequence_length: 80
        n_channels: {data.n_channels}
        input_parameters:
            embedding_dim: 100
            token_embedding_dim: 100
            pos_embedding_dim: 100
            channel_embedding_dim: 20
        decoder_parameters:
            n_layers: 4
            n_heads: 4
            model_dim: 100
            latent_sequence_length: 40
            n_patches: 20
            patch_length: 4
            unpatched_length: 12
            channel_attention_dropout: 0.5
            within_channel_attention_dropout: 0.5
            feed_forward_dim: 100
            dropout: 0.2
        loss_parameters:
            loss_sequence_length: 8
    training_config:
        optimizer:
            learning_rate: 0.00002
        batch_size: 8
        n_epochs: 40
        save_best:
            filepath: {generator_dir}/weights.h5
            monitor: val_loss
            save_best_only: true
            save_weights_only: true
            verbose: 1
"""

generator = create_model(generator_config)
generator.save_config(generator_dir)  # TODO: integrate this into create_model
generator.summary()

# ---------- Fit generator ---------- #
generator.fit(data, validation_split=0.1, use_tfrecord=True)

# ---------- Plot training curves ---------- #
generator.plot_history(plot_dir=plot_dir)

# Clean up directories
data.delete_dir()
