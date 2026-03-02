from glob import glob
import os

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import create_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large"
plot_dir = f"{data_dir}/plots/"
tokenizer_dir = f"{data_dir}/tokenizer/"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

# ---------- Load data ---------- #
data_files = sorted(glob(f"{data_dir}/*.npy"))

data = Data(
    data_files,
    n_jobs=8,
    picks="misc",
    use_tfrecord=True,
    reject_by_annotation="omit",
    sampling_frequency=250,
)

methods = {
    "filter": {"low_freq": 5, "high_freq": 100, "use_raw": True},
    "standardize": {},
}
data.prepare(methods)

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
    data_path=f"{data_dir}",
    ground_truth_path=None,
    plot_dir=plot_dir,
)

# Clean up data directory
data.delete_dir()
