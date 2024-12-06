from glob import glob
import os

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import create_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = "sim_data"
plot_dir = "plots/tokenizer"
tokenizer_dir = "models/tokenizer"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

# ---------- Load data ---------- #
data = Data(sorted(glob(f"{data_dir}/*.npy")), use_tfrecord=True)

# Standardize the data
data.standardize()

# ---------- Build tokenizer ---------- #
config = f"""
    model_config:
        name: osl_tokenizer
        sequence_length: 200
        n_channels: {data.n_channels}
        n_tokens: 128
        token_dim: 10
        rnn_n_units: 128
    training_config:
        optimizer:
            learning_rate: 0.0001
        batch_size: 32
        n_epochs: 40
        temperature_annealing:
            n_stages: 40
            n_annealing_epochs: 40
            end_temperature: 0.0
"""

# Build model
model = create_model(config)

# Print summary of model
model.summary()

# ---------- Fit tokenizer ---------- #
model.fit(data)

# ---------- Save tokenizer ---------- #
model.save(tokenizer_dir)


# ---------- summary plots ---------- #

# Plot percentage of explained variance
model.plot_pve(data=data, plot_dir=plot_dir)

# Plot token counts
model.plot_token_counts(plot_dir=plot_dir)

# Plot stimulus response of token kernels
model.plot_token_response(plot_dir=plot_dir)

# Plot signals reconstructed from tokenized data (for one session)
model.plot_fitted_signal(
    data_path=f"{data_dir}/x_00.npy",
    ground_truth_path=f"{data_dir}/ground_truth/true_signal_00.npy",
    plot_dir=plot_dir,
)

# Clean up data directory
data.delete_dir()
