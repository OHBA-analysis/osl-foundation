import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting as osld_plotting

from osl_foundation import load_model
from osl_foundation.utils import plotting as oslf_plotting

# Set GPU memory growth
tf_ops.gpu_growth()

generate_data = True

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
generator = load_model(generator_dir, checkpoint="latest")

# ---------- Reconstructed data from the tokenizer ---------- #
tokens = generator.tokenizer.tokenize_data(data)
reconstructed_data = generator.tokenizer.reconstruct_data(tokens)

if generate_data:
    # ---------- Generate data using the generator ---------- #
    generated_data = generator.generate_data(
        n_samples=2048,
        top_k=int(generator.config.model_config.n_tokens * 0.8),
        batch_size=len(data_files),
        extra_labels={"session_id": np.arange(len(data_files))},
    )
    pickle.dump(generated_data, open(f"{generator_dir}/generated_data.pkl", "wb"))
else:
    generated_data = pickle.load(open(f"{generator_dir}/generated_data.pkl", "rb"))


# ---------- Plot results ---------- #

# Compare AEC of original, reconstructed and generated data
oslf_plotting.plot_aec(
    data_files,
    reconstructed_data,
    generated_data,
    window_size=100,
    sampling_frequency=100,
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/aec.png",
)

# Plot PSD of generated data
n_channels = generator.config.model_config.n_channels
fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
for i in range(n_channels):
    axes[i].psd(
        np.concatenate(generated_data[:30])[:, i], Fs=100, NFFT=1024, color="red"
    )
    axes[i].psd(
        np.concatenate(generated_data[30:])[:, i], Fs=100, NFFT=1024, color="green"
    )
    axes[i].set_ylabel(f"Channel {i}")
fig.tight_layout()
fig.savefig(f"{plot_dir}/psd_generated_data.png")
plt.close(fig)

# Plot embeddings
embeddings = generator.get_embeddings()
position_embeddings = embeddings["position"]
channel_embeddings = embeddings["channel"]
session_embeddings = embeddings["session_id"]

# PCA
pca = PCA(n_components=2)
position_pca = pca.fit_transform(position_embeddings)
channel_pca = pca.fit_transform(channel_embeddings)
session_pca = pca.fit_transform(session_embeddings)

osld_plotting.plot_scatter(
    [channel_pca[:4, 0], channel_pca[4:8, 0], channel_pca[8:, 0]],
    [channel_pca[:4, 1], channel_pca[4:8, 1], channel_pca[8:, 1]],
    labels=["Mode 0", "Mode 1", "Mode 2"],
    title="Channel embeddings",
    filename=f"{plot_dir}/channel_embeddings.png",
)
osld_plotting.plot_scatter(
    [session_pca[:30, 0], session_pca[30:, 0]],
    [session_pca[:30, 1], session_pca[30:, 1]],
    labels=["Group 0", "Group 1"],
    title="Session embeddings",
    filename=f"{plot_dir}/session_embeddings.png",
)

# Clean up directories
data.delete_dir()
