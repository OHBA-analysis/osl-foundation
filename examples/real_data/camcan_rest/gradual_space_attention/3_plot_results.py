import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import trange

from osl_dynamics.analysis import static, power, connectivity
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting

data_dir = "models/generator"
plot_dir = "plots/static"
os.makedirs(plot_dir, exist_ok=True)

with open(f"{data_dir}/generated_data_p99.pkl", "rb") as f:
    data = pickle.load(f)

f, psd = static.welch_spectra(
    data=data,
    sampling_frequency=250,
    n_jobs=8,
    frequency_range=[1, 45],
)

n_subjects, n_channels, n_freqs = psd.shape
os.makedirs(f"{plot_dir}/psd", exist_ok=True)
for start in trange(0, n_subjects, 50, desc="Plotting PSD"):
    fig, axes = plt.subplots(10, 5, figsize=(40, 60))
    for i, ax in enumerate(axes.flatten()):
        if start + i >= n_subjects:
            break
        plotting.plot_line(
            [f] * n_channels,
            psd[start + i],
            ax=ax,
        )
        ax.set_title(f"Subject {start + i}", fontsize=40)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/psd/{start}-{start + 50 - 1}.png")
    plt.close(fig)

psd_mean = np.mean(psd, axis=1)
psd_std = np.std(psd, axis=1)

for start in trange(0, n_subjects, 50, desc="Plotting PSD"):
    fig, axes = plt.subplots(10, 5, figsize=(40, 60))
    for i, ax in enumerate(axes.flatten()):
        if start + i >= n_subjects:
            break
        plotting.plot_line(
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
    fig.savefig(f"{plot_dir}/psd/mean_{start}-{start + 50 - 1}.png")
    plt.close(fig)

frequency_bands = {
    "delta": [1, 4],
    "theta": [4, 7],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 50],
}

p = {}
for band, (low, high) in frequency_bands.items():
    p[band] = power.variance_from_spectra(f, psd, frequency_range=[low, high])

group_p = {}
for band in frequency_bands.keys():
    group_p[band] = np.mean(p[band], axis=0)

power_maps = np.array(list(group_p.values()))

os.makedirs(f"{plot_dir}/maps", exist_ok=True)
power.save(
    power_maps,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
    show_plots=False,
    combined=True,
    titles=list(group_p.keys()),
    filename=f"{plot_dir}/maps/group_power.png",
)
power.save(
    power_maps - np.mean(power_maps, axis=1, keepdims=True),
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
    show_plots=False,
    combined=True,
    titles=list(group_p.keys()),
    filename=f"{plot_dir}/maps/group_power_demeaned.png",
)

data = Data(data, n_jobs=8, sampling_frequency=250)
ae_ts = {}
for band, (low, high) in frequency_bands.items():
    methods = {
        "filter": {"low_freq": low, "high_freq": high, "use_raw": True},
        "amplitude_envelope": {},
        "standardize": {},
        "moving_average": {"n_window": 25},
    }
    data.prepare(methods)
    ae_ts[band] = data.time_series()

aec = {band: static.functional_connectivity(ts) for band, ts in ae_ts.items()}
aec_no_diag = {band: aec[band] - np.eye(aec[band].shape[1]) for band in aec.keys()}
aec_mean = {band: np.mean(aec[band], axis=0) for band in aec.keys()}
connectivity_maps = np.array(list(aec_mean.values()))
thres_connectivity_maps = connectivity.threshold(connectivity_maps, percentile=95)

connectivity.save(
    connectivity_maps,
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    combined=True,
    titles=list(aec_mean.keys()),
    filename=f"{plot_dir}/maps/group_connectivity.png",
)
connectivity.save(
    thres_connectivity_maps,
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    combined=True,
    titles=list(aec_mean.keys()),
    filename=f"{plot_dir}/maps/thres_group_connectivity.png",
)

data.delete_dir()
