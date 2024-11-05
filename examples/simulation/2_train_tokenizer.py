from glob import glob
import os

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import create_model

tf_ops.gpu_growth()

data_dir = "sim_data"
plot_dir = "plots/tokenizer"
tokenizer_dir = "models/tokenizer"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

# Load data
data = Data(sorted(glob(f"{data_dir}/*.npy")), use_tfrecord=True)

# Standardize the data
data.standardize()

# Model and training configuration
config = f"""
    model_config:
        name: osl_tokenizer
        sequence_length: 256
        n_channels: {data.n_channels}
        n_tokens: 128
        token_dim: 32
        rnn_n_units: 128
    training_config:
        optimizer:
            learning_rate: 0.01
        batch_size: 16
        n_epochs: 40
        temperature_annealing:
            n_stages: 40
            n_annealing_epochs: 40
        lr_decay: 0.1
"""

# Build model
model = create_model(config)

# Print summary of model
model.summary()

# Fit model
model.fit(data)

# Save model
model.save(tokenizer_dir)

# Refactor tokens
model.refactor_vocab(data)

# Plot percentage of explained variance
model.plot_pve(data=data, plot_dir=plot_dir)

# Plot token counts
model.plot_token_counts(plot_dir=plot_dir)

# Plot stimulus response of token kernels
model.plot_token_response(plot_dir=plot_dir)

# Plot signals reconstructed from tokenized data
model.plot_fitted_signal(data_dir, plot_dir=plot_dir)

# Clean up data directory
data.delete_dir()
