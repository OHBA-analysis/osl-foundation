import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import load_model

tf_ops.gpu_growth()

data_dir = "sim_data"
generator_dir = "models/generator"
plot_dir = "plots/generated_data"
os.makedirs(plot_dir, exist_ok=True)

data = Data(sorted(glob(f"{data_dir}/*.npy")), use_tfrecord=True)

generator = load_model(generator_dir, from_checkpoint=True)

# Generate data
prompt = np.array(
    [ts[: generator.config.model_config.sequence_length] for ts in data.time_series()]
)
generated_data = np.concatenate(
    generator.generate_data(
        n_samples=2048,
        method="top_p",
        p=0.8,
        prompt=prompt,
        batch_size=prompt.shape[0],
    ),
    axis=0,
)

n_channels = generated_data.shape[1]

# Plot PSD of generated data
fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
for i in range(n_channels):
    axes[i].psd(generated_data[:, i], Fs=100, NFFT=1024, color="black")
    axes[i].set_ylabel(f"Channel {i}")
fig.tight_layout()
fig.savefig(f"{plot_dir}/psd_generated_data.png")
plt.close(fig)

# Clean up directories
data.delete_dir()
