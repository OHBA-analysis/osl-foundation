import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm.auto import trange

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.analysis import static
from osl_dynamics.utils import plotting as osld_plotting

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import load_model
from osl_foundation.utils import plotting

tf_ops.gpu_growth()

sampling_frequency = 100
subs2plot = None

# ---------- Directories ---------- #
data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large"
generator_dir = f"{data_dir}/generator"
tokenized_data_dir = f"{data_dir}/tokenized_data"

plot_dir = f"{data_dir}/plots/"

os.makedirs(plot_dir, exist_ok=True)

# ---------- Load generated data ---------- #
generated_data_files = sorted(glob(f"{generator_dir}/generated_data_*.pkl"))
generated_data = [pickle.load(open(f, "rb")) for f in generated_data_files] 

lyapunov_data_files = sorted(glob(f"{generator_dir}/lyapunov_data_*.pkl"))
lyapunov_data = [pickle.load(open(f, "rb")) for f in lyapunov_data_files]

if subs2plot is None:
    subs2plot = list(range(len(generated_data)))

generated_data = [generated_data[s] for s in subs2plot]
lyapunov_data = [lyapunov_data[s] for s in subs2plot]

# ---------- Plot lyapunov data ---------- #
lyapunov_data = np.array(lyapunov_data)  # shape (n_sessions, n_timepoints)
if lyapunov_data.ndim == 1:
    lyapunov_data = np.expand_dims(lyapunov_data, axis=0)
plt.figure()
plt.plot(lyapunov_data[0, :3000], color="gray", alpha=0.5)
plt.xlabel("Timepoints")
plt.ylabel("Lyapunov exponent")
plt.tight_layout()
plt.savefig(f"{plot_dir}/lyapunov.png")
plt.close()

# plot histogram of lyapunov exponents
plt.figure()
plt.hist(lyapunov_data.flatten(), bins=100, color="gray")
plt.xlabel("Lyapunov exponent")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{plot_dir}/lyapunov_hist.png")
plt.close() 

# ---------- Load original data ---------- #
data_files = sorted(glob(f"{data_dir}/x*.npy"))
data_files = data_files[:len(generated_data)]

original_data = []
for ss in range(len(data_files)):
    dat = np.load(data_files[ss])
    dat = (dat - dat.mean()) / dat.std()
    dat = dat[:generated_data[ss].shape[0],:]
    original_data.append(dat)

# ---------- Load tokenized data ---------- #
tokenized_data_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
tokenized_data_files = tokenized_data_files[:len(generated_data)]
tokenized_data = [np.load(f) for f in tokenized_data_files]
tokenized_data = [f[:generated_data[ss].shape[0],:] for ss, f in enumerate(tokenized_data)]

# ---------- Get reconstructed data from original data using the tokenizer ---------- #
generator = load_model(generator_dir)
reconstructed_data = generator.tokenizer.reconstruct_data(tokenized_data)
reconstructed_data = [reconstructed_data[s] for s in subs2plot]

# ---------- Plot subject level PSDs ---------- #

def _plot_psd(f, psd, name, plot_dir):
    
    max_subj = np.minimum(len(psd), 50)
    if len(psd.shape) == 2:
        psd = np.expand_dims(psd, axis=0)
    nrows = int(np.ceil(max_subj/5))

    n_subjects, n_channels, _ = psd.shape
    os.makedirs(f"{plot_dir}/psd", exist_ok=True)
    for start in trange(0, n_subjects, max_subj, desc="Plotting PSD"):
        fig, axes = plt.subplots(nrows, 5, figsize=(40, 6*nrows))
        for i, ax in enumerate(axes.flatten()):
            if start + i >= n_subjects:
                break
            osld_plotting.plot_line(
                [f] * n_channels,
                psd[start + i],
                ax=ax,
            )
            ax.set_title(f"Subject {start + i}", fontsize=40)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/psd/{name}_{start}-{start + max_subj - 1}.png")
        plt.close(fig)

    psd_mean = np.mean(psd, axis=1)
    psd_std = np.std(psd, axis=1)

    for start in trange(0, n_subjects, max_subj, desc="Plotting PSD"):
        fig, axes = plt.subplots(nrows, 5, figsize=(40, 6*nrows))
        for i, ax in enumerate(axes.flatten()):
            if start + i >= n_subjects:
                break
            osld_plotting.plot_line(
                [f],
                [psd_mean[start + i]],
                errors=[
                    [psd_mean[start + i] - psd_std[start + i]],
                        [psd_mean[start + i] + psd_std[start + i]],
                    ],
                    ax=ax,
                )
            
            ax.set_title(f"Subject {start + i}", fontsize=40)
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/psd/{name}_mean_{start}-{start + max_subj - 1}.png")
            plt.close(fig)


for data, name in zip([generated_data, reconstructed_data], ["generated", "reconstructed"]):
    
    f, psd = static.welch_spectra(
        data=data,
        sampling_frequency=sampling_frequency,
        n_jobs=8,
        frequency_range=[1, 45],
    )

    _plot_psd(f, psd, name, plot_dir)

# ---------- Group averaged plots ---------- #

# Compare static correlation netmat of TDE of original, reconstructed and generated data
tde_corr = plotting.plot_tde_corr(
    original_data,
    reconstructed_data,
    generated_data,
    n_embeddings=15,
    n_pca_components=None,  
    sampling_frequency=sampling_frequency,
    frequency_range=[1, 40],
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/tde_corr.png",
)

np.save(f"{plot_dir}/tde_corr_orig.npy", tde_corr[0])

# Compare AEC of original, reconstructed and generated data
plotting.plot_aec(
    original_data,
    reconstructed_data,
    generated_data,
    window_size=500,
    sampling_frequency=sampling_frequency,
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/aec.png",
)

# Plot Time frequency content of original, reconstructed and generated data
n_samples = 60*sampling_frequency  # 60 seconds

original_data = np.concatenate(original_data)
reconstructed_data = np.concatenate(reconstructed_data)
generated_data = np.concatenate(generated_data)

plotting.plot_time_frequency(
    original_data,
    reconstructed_data,
    generated_data,
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
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
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[0], ax_2[0], ax_3],
    color="black",
    label="Original",
)
plotting.plot_time_series_summary(
    reconstructed_data[:, 0],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[1], ax_2[1], ax_3],
    color="green",
    label="Reconstructed",
)
plotting.plot_time_series_summary(
    generated_data[:, 0],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
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



# Plot summary of the original, reconstructed and generated data

# Channel 0
fig = plt.figure(figsize=(10, 10))
ax_1 = [fig.add_subplot(3, 2, i) for i in range(1, 3)]
ax_2 = [fig.add_subplot(3, 2, i) for i in range(3, 5)]
ax_3 = fig.add_subplot(3, 1, 5)

plotting.plot_time_series_summary(
    reconstructed_data[:, 0],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[0], ax_2[0], ax_3],
    color="green",
    label="Reconstructed",
)
plotting.plot_time_series_summary(
    generated_data[:, 0],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[1], ax_2[1], ax_3],
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
fig.savefig(f"{plot_dir}/summary_0b.png")
plt.close(fig)

# ---------- Plot history ---------- #

history = pickle.load(open(f"{generator_dir}/history.pkl", "rb"))

plotting.plot_history(history, plot_dir=plot_dir)

# ---------- Finish ---------- #

print(plot_dir)

