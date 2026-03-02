from glob import glob
import os
import numpy as np
import pickle

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops

dev_dir = "/well/woolrich/users/vxw496"
dev_dir = "/Users/woolrich/dev"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import create_model
from osl_foundation.utils import plotting

tf_ops.gpu_growth()

# ---------- Directories ---------- #

# input dirs

data_dir = "/well/woolrich/projects/camcan/spring23/src"
data_dir = "/Users/woolrich/dev/results/raw_data/camcan/spring23/"

# output dirs

results_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23"
tokenizer_dir = f"{results_dir}/tokenizer"
plot_dir = f"{tokenizer_dir}/plots"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---- Load data ---- #

data_files_full = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))

print(f"len(data_files_full)={len(data_files_full)}")

nsessions2use = 3
data_files = list(np.random.choice(data_files_full, nsessions2use, replace=False))

print(f"len(data_files)={len(data_files)}")

print(data_files)
print(data_files_full[:nsessions2use])

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
        n_tokens: 200
        token_dim: 10
        rnn_n_units: 128
        token_kernel_padding: causal
        token_use_bias: false
        token_groups: {100//4}
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

tokenizer = create_model(config)
tokenizer.summary()
tokenizer.fit(data)
tokenizer.save(tokenizer_dir)

# Plot percentage of explained variance
tokenizer.plot_pve(data=data, plot_dir=plot_dir)

# Plot token counts
tokenizer.plot_token_counts(plot_dir=plot_dir)

# Plot stimulus response of token kernels
tokenizer.plot_token_response(plot_dir=plot_dir)

# Plot signals reconstructed from tokenized data (for one session)
tokenizer.plot_fitted_signal(
    data_path=data.time_series()[0],
    plot_dir=plot_dir,
)

history = pickle.load(open(f"{tokenizer_dir}/history.pkl", "rb"))
plotting.plot_history(history, plot_dir=plot_dir)

# Clean up data directory
data.delete_dir()

print(f"plot_dir={plot_dir}")

# scp -r vxw496@cluster1.bmrc.ox.ac.uk:/well/woolrich/users/vxw496/results/osl-foundation/camcan_spring23/tokenizer/plots/ ~/Downloads/