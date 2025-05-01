import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import trange

from osl_dynamics.analysis import static, power, connectivity
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting


def plot_psd(f, psd, plot_dir):
    """Plot the power spectral density (PSD) of generated data.

    Parameters
    ----------
    f : array
        Frequencies corresponding to the PSD.
    psd : array
        Power spectral density of the generated data.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    n_subjects, n_channels, n_freqs = psd.shape
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
        fig.savefig(f"{plot_dir}/psd_{start}-{start + 50 - 1}.png")
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
        fig.savefig(f"{plot_dir}/psd_mean_{start}-{start + 50- 1}.png")
        plt.close(fig)


def plot_power_maps(f, psd, plot_dir):
    """Plot the power maps of generated data.

    Parameters
    ----------
    f : array
        Frequencies corresponding to the PSD.
    psd : array
        Power spectral density of the generated data.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    frequency_bands = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 12],
        "beta": [12, 30],
        "gamma": [30, 45],
    }

    p = {}
    for band, (low, high) in frequency_bands.items():
        p[band] = power.variance_from_spectra(f, psd, frequency_range=[low, high])

    group_p = {}
    for band in frequency_bands.keys():
        group_p[band] = np.mean(p[band], axis=0)

    power_maps = np.array(list(group_p.values()))
    power.save(
        power_maps - np.mean(power_maps, axis=1, keepdims=True),
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"symmetric_cbar": True},
        combined=True,
        titles=list(group_p.keys()),
        filename=f"{plot_dir}/power.png",
    )


def plot_connectivity(data, plot_dir):
    """Plot the connectivity of generated data.

    Parameters
    ----------
    data : np.ndarray
        Generated data.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    data = Data(data, n_jobs=8, sampling_frequency=250)
    frequency_bands = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 12],
        "beta": [12, 30],
        "gamma": [30, 45],
    }

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
    aec_mean = {band: np.mean(aec[band], axis=0) for band in aec.keys()}
    connectivity_maps = np.array(list(aec_mean.values()))
    thres_connectivity_maps = connectivity.threshold(connectivity_maps, percentile=95)

    connectivity.save(
        connectivity_maps,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        combined=True,
        titles=list(aec_mean.keys()),
        filename=f"{plot_dir}/aec.png",
    )
    connectivity.save(
        thres_connectivity_maps,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        combined=True,
        titles=list(aec_mean.keys()),
        filename=f"{plot_dir}/thres_aec.png",
    )


if __name__ == "__main__":
    plot_dir = "results/plots/summary"

    # Load generated data
    with open(f"results/generated_data.pkl", "rb") as f:
        generated_data = pickle.load(f)

    f, psd = static.welch_spectra(
        data=generated_data,
        sampling_frequency=250,
        n_jobs=8,
        frequency_range=[1, 45],
    )

    # Plot PSD
    plot_psd(f, psd, plot_dir)

    # Plot power maps
    plot_power_maps(f, psd, plot_dir)

    # Plot connectivity
    plot_connectivity(generated_data, plot_dir)
