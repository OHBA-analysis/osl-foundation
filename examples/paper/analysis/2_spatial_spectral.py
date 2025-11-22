import os
from glob import glob
from typing import List, Tuple

import numpy as np
import pickle
import matplotlib.pyplot as plt

from osl_dynamics.analysis import static, power
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting


def get_psd() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the power spectral density (PSD) of real and generated data.

    Returns
    -------
    f : array
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    psd_real : array
        Power spectral density of the real data.
        Shape is (n_sessions, n_channels, n_frequencies).
    psd_gen : array
        Power spectral density of the generated data from MEG-GPT.
        Shape is (n_sessions, n_channels, n_frequencies).
    psd_gen_ar : array
        Power spectral density of the generated data from the AR model.
        Shape is (n_sessions, n_channels, n_frequencies).
    """
    real_data_dir = "/well/woolrich/projects/camcan/spring23/src"
    data_files = sorted(glob(f"{real_data_dir}/*/sflip_parc-raw.fif"))
    real_data = Data(
        data_files,
        n_jobs=8,
        picks="misc",
        reject_by_annotation="omit",
    )
    real_data = [d[:15000] for d in real_data.time_series()]

    generated_data_dir = "../../data/generated_data"
    with open(f"{generated_data_dir}/meg-gpt.pkl", "rb") as f:
        generated_data = pickle.load(f)

    with open(f"{generated_data_dir}/ar_model.pkl", "rb") as f:
        generated_data_ar = pickle.load(f)

    f, psd_real = static.welch_spectra(
        data=real_data,
        sampling_frequency=250,
        frequency_range=[1, 45],
        n_jobs=12,
    )
    _, psd_gen = static.welch_spectra(
        data=generated_data,
        sampling_frequency=250,
        frequency_range=[1, 45],
        n_jobs=12,
    )
    _, psd_gen_ar = static.welch_spectra(
        data=list(generated_data_ar),
        sampling_frequency=250,
        frequency_range=[1, 45],
        n_jobs=12,
    )
    return f, psd_real, psd_gen, psd_gen_ar


def plot_psd(
    *psd: List[np.ndarray], f: np.ndarray, plot_dir: str, titles: List[str]
) -> None:
    """Plot the power spectral density (PSD) of generated data.

    Parameters
    ----------
    psd : List[array]
        Power spectral density of the generated data.
        Each element has shape (n_sessions, n_channels, n_frequencies).
    f : array
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    plot_dir : str
        Directory to save the plots.
    titles : List[str]
        Titles for the plots, corresponding to each dataset.
    """
    os.makedirs(plot_dir, exist_ok=True)

    n_psds = len(psd)
    n_channels = psd[0].shape[1]

    psd_mean = [np.mean(p, axis=0) for p in psd]
    psd_std = [np.std(p, axis=0) for p in psd]

    fig, axes = plt.subplots(1, n_psds, figsize=(6 * n_psds, 6), sharey=True)
    for i, ax in enumerate(axes):
        plotting.plot_line(
            [f] * n_channels,
            psd_mean[i],
            ax=ax,
            y_range=[None, 0.18],
        )
        ax.set_title(titles[i], fontsize=20)
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
    axes[0].set_ylabel("PSD (a.u.)", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/psd_mean.png")
    plt.close(fig)

    psd_mean = [np.mean(p, axis=(0, 1)) for p in psd]
    psd_std = [np.std(p, axis=(0, 1)) for p in psd]
    fig, axes = plt.subplots(1, n_psds, figsize=(6 * n_psds, 6), sharey=True)
    for i, ax in enumerate(axes):
        plotting.plot_line(
            [f],
            [psd_mean[i]],
            errors=[[psd_mean[i] - psd_std[i]], [psd_mean[i] + psd_std[i]]],
            ax=ax,
            y_range=[None, 0.12],
        )
        ax.set_title(titles[i], fontsize=20)
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
    axes[0].set_ylabel("PSD (a.u.)", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/psd_mean_overall.png")
    plt.close(fig)


def plot_power_maps(*psd: List[np.ndarray], f: np.ndarray, plot_dir: str) -> None:
    """Plot the power maps of generated data.

    Parameters
    ----------
    psd : List[array]
        Power spectral density of the generated data.
        Each element has shape (n_sessions, n_channels, n_frequencies).
    f : array
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    frequency_bands = {
        "Delta (1-4 Hz)": [1, 4],
        "Theta (4-8 Hz)": [4, 8],
        "Alpha (8-12 Hz)": [8, 12],
        "Beta (12-30 Hz)": [12, 30],
        "Gamma (30-45 Hz)": [30, 45],
    }

    p = [{} for _ in range(len(psd))]
    for band, (low, high) in frequency_bands.items():
        for i, psd_i in enumerate(psd):
            p[i][band] = power.variance_from_spectra(
                f, psd_i, frequency_range=[low, high]
            )
    group_p = [{} for _ in range(len(psd))]
    for band in frequency_bands.keys():
        for i in range(len(psd)):
            group_p[i][band] = np.mean(p[i][band], axis=0)

    power_maps = []
    for i in range(len(psd)):
        power_maps += list(group_p[i].values())
    power_maps = np.array(power_maps)
    power_maps -= np.mean(power_maps, axis=1, keepdims=True)

    power.save(
        power_maps,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"symmetric_cbar": True},
        combined=True,
        n_rows=len(psd),
        filename=f"{plot_dir}/power.png",
    )


if __name__ == "__main__":
    """
    Plot summary of generated data.
    - Power spectral density (PSD)
    - Static power maps for different frequency bands
    """
    plot_dir = "../../plots/2_spatial_spectral"
    os.makedirs(plot_dir, exist_ok=True)

    # Get PSD
    f, psd_real, psd_gen, psd_gen_ar = get_psd()

    # Plot PSD
    plot_psd(
        psd_real,
        psd_gen,
        psd_gen_ar,
        f=f,
        plot_dir=plot_dir,
        titles=["Real", "MEG-GPT", "AR"],
    )

    # Plot power maps
    plot_power_maps(
        psd_real,
        psd_gen,
        psd_gen_ar,
        f=f,
        plot_dir=plot_dir,
    )
