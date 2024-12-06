import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import load_model
from osl_foundation.utils import plotting

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = "sim_data"
generator_dir = "models/generator"
plot_dir = "plots/generated_data"
os.makedirs(plot_dir, exist_ok=True)

# ---------- Load data ---------- #
data_files = sorted(glob(f"{data_dir}/*.npy"))
data = Data(data_files)
data.standardize()

# ---------- Load generator ---------- #
generator = load_model(generator_dir)

# ---------- Reconstructed data from the tokenizer ---------- #
tokens = generator.tokenizer.tokenize_data(data)
reconstructed_data = generator.tokenizer.reconstruct_data(tokens)

# ---------- Generate data using the generator ---------- #
generated_data = generator.generate_data(
    n_samples=2048,
    method="top_k",
    k=int(generator.config.model_config.n_tokens * 0.8),
    batch_size=len(data_files),
)

# ---------- Plot results ---------- #

# Compare AEC of original, reconstructed and generated data
plotting.plot_aec(
    data_files,
    reconstructed_data,
    generated_data,
    window_size=100,
    sampling_frequency=100,
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/aec.png",
)

# Plot Time frequency content of original, reconstructed and generated data
original_data = np.concatenate(data.time_series())
reconstructed_data = np.concatenate(reconstructed_data)
generated_data = np.concatenate(generated_data)

plotting.plot_time_frequency(
    original_data,
    reconstructed_data,
    generated_data,
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/time_frequency.png",
)

# Plot summary of the original, reconstructed and generated data

# Channel 0
fig = plt.figure(figsize=(10, 10))
ax_1 = [fig.add_subplot(3, 3, i) for i in range(1, 4)]
ax_2 = [fig.add_subplot(3, 3, i) for i in range(4, 7)]
ax_3 = fig.add_subplot(3, 1, 3)

plotting.plot_time_series_summary(
    original_data[:, 0],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[0], ax_2[0], ax_3],
    color="black",
    label="Original",
)
plotting.plot_time_series_summary(
    reconstructed_data[:, 0],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[1], ax_2[1], ax_3],
    color="green",
    label="Reconstructed",
)
plotting.plot_time_series_summary(
    generated_data[:, 0],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[2], ax_2[2], ax_3],
    color="red",
    label="Generated",
)
fig.suptitle("Time series summary (channel 0)")
fig.legend(
    loc="lower center",
    ncol=3,
    fontsize=14,
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True,
    shadow=True,
)
fig.tight_layout()
fig.savefig(f"{plot_dir}/summary_0.png")
plt.close(fig)

# Channel 2
fig = plt.figure(figsize=(10, 10))
ax_1 = [fig.add_subplot(3, 3, i) for i in range(1, 4)]
ax_2 = [fig.add_subplot(3, 3, i) for i in range(4, 7)]
ax_3 = fig.add_subplot(3, 1, 3)

plotting.plot_time_series_summary(
    original_data[:, 2],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[0], ax_2[0], ax_3],
    color="black",
    label="Original",
)
plotting.plot_time_series_summary(
    reconstructed_data[:, 2],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[1], ax_2[1], ax_3],
    color="green",
    label="Reconstructed",
)
plotting.plot_time_series_summary(
    generated_data[:, 2],
    sampling_frequency=100,
    n_samples=6000,  # 60 seconds
    axes=[ax_1[2], ax_2[2], ax_3],
    color="red",
    label="Generated",
)
fig.suptitle("Time series summary (channel 2)")
fig.legend(
    loc="lower center",
    ncol=3,
    fontsize=14,
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True,
    shadow=True,
)
fig.tight_layout()
fig.savefig(f"{plot_dir}/summary_2.png")
plt.close(fig)

# Plot PSD of generated data
n_channels = generator.config.model_config.n_channels
fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
for i in range(n_channels):
    axes[i].psd(generated_data[:, i], Fs=100, NFFT=1024, color="black")
    axes[i].set_ylabel(f"Channel {i}")
fig.tight_layout()
fig.savefig(f"{plot_dir}/psd_generated_data.png")
plt.close(fig)
