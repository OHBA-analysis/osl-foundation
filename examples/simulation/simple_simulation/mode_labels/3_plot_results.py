import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pickle

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops

from osl_foundation import load_model
from osl_foundation.utils import plotting

tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = "../sim_data"
generator_dir = "models/generator"
plot_dir = "plots/generated_data"
os.makedirs(plot_dir, exist_ok=True)

# ---------- Load data ---------- #
data_files = sorted(glob(f"{data_dir}/*.npy"))
data = Data(data_files)
data.standardize()

# ---------- Load generator ---------- #
generator = load_model(generator_dir, checkpoint="latest")

# ---------- Reconstructed data from the tokenizer ---------- #
tokens = generator.tokenizer.tokenize_data(data)
reconstructed_data = generator.tokenizer.reconstruct_data(tokens)

# ---------- Load generated data ---------- #
generated_data = pickle.load(open(f"{generator_dir}/generated_data.pkl", "rb"))

# ---------- Plot results ---------- #

# Compare AEC of original, reconstructed and generated data
plotting.plot_aec(
    data_files,
    reconstructed_data,
    generated_data[0],
    generated_data[1],
    generated_data[2],
    generated_data[3],
    window_size=20,
    sampling_frequency=100,
    titles=[
        "Original",
        "Reconstructed",
        "Generated (real mtc)",
        "Generated (all 0)",
        "Generated (all 1)",
        "Generated (interleaved)",
    ],
    filename=f"{plot_dir}/aec.png",
)

# Plot Time frequency content of original, reconstructed and generated data
original_data = np.concatenate(data.time_series())
reconstructed_data = np.concatenate(reconstructed_data)

n_samples = 2000

plotting.plot_time_frequency(
    original_data,
    reconstructed_data,
    generated_data[0],
    generated_data[1],
    generated_data[2],
    generated_data[3],
    sampling_frequency=100,
    n_samples=n_samples,  # 60 seconds
    titles=[
        "Original",
        "Reconstructed",
        "Generated (real mtc)",
        "Generated (all 0)",
        "Generated (all 1)",
        "Generated (interleaved)",
    ],
    filename=f"{plot_dir}/time_frequency.png",
)

data.delete_dir()
