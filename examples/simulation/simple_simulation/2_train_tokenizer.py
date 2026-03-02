from glob import glob
import os
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data


dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

# python /Users/woolrich/dev/projects/osl-foundation/examples/simulation/simple_simulation/2_train_tokenizer.py

from osl_foundation import create_model
from osl_foundation.utils import plotting
from osl_foundation.models import load_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = f"{dev_dir}/results/osl-foundation/simple_simulation"
plot_dir = f"{data_dir}/plots/"
tokenizer_dir = f"{data_dir}/tokenizer/"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

# ---------- Load data ---------- #
data = Data(sorted(glob(f"{data_dir}/*.npy")), 
            sampling_frequency=100)

methods = {
    "filter": {"low_freq": 5, "high_freq": 40, "use_raw": True},
    "standardize": {},
}
data.prepare(methods)

if True:
    # ---------- Build tokenizer ---------- #
    config = f"""
        model_config:
            name: osl_tokenizer
            sequence_length: 200
            n_channels: {data.n_channels}
            n_tokens: 100
            token_dim: 10
            rnn_n_units: 100
            token_kernel_padding: causal
            token_use_bias: false
            token_groups: 10
        training_config:
            optimizer:
                learning_rate: 0.001
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
else:
    tokenizer = load_model(tokenizer_dir)

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

history = pickle.load(open(f"{tokenizer_dir}/history.pkl", "rb"))
plotting.plot_history(history, plot_dir=plot_dir)

# Clean up data directory
data.delete_dir()