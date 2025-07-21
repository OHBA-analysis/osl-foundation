import os
from glob import glob
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model

tf_ops.gpu_growth()


def calculate_pve(
    camcan_files: List[str],
    wh_files: List[str],
    tokenizer_dir: str,
    save_dir: str,
    overwrite: bool = False,
) -> None:
    """
    Calculate the Percentage of Variance Explained (PVE) for Cam-CAN and Wakeman-Henson datasets.

    Parameters
    ----------
    camcan_files : List[str]
        List of file paths for Cam-CAN dataset.
    wh_files : List[str]
        List of file paths for Wakeman-Henson dataset.
    tokenizer_dir : str
        Directory where the tokenizer model is saved.
    save_dir : str
        Directory where the PVE results will be saved.
    overwrite : bool, optional
        If True, overwrite existing PVE files. Default is False.
    """
    if (
        not overwrite
        and os.path.exists(f"{save_dir}/camcan_pve.npy")
        and os.path.exists(f"{save_dir}/wh_pve.npy")
    ):
        print("PVE files already exist. Use overwrite=True to recalculate.")
        return

    tokenizer = load_model(tokenizer_dir)

    camcan_data = Data(
        camcan_files,
        sampling_frequency=250,
        picks="misc",
        reject_by_annotation="omit",
        n_jobs=12,
    )
    wh_data = Data(
        wh_files,
        sampling_frequency=250,
        picks="misc",
        reject_by_annotation="omit",
        n_jobs=12,
    )
    camcan_data.standardize()
    wh_data.standardize()

    camcan_pve = tokenizer.get_pve(camcan_data)
    wh_pve = tokenizer.get_pve(wh_data)

    np.save(f"{save_dir}/camcan_pve.npy", camcan_pve)
    np.save(f"{save_dir}/wh_pve.npy", wh_pve)


def plot_fitted_signal(
    *data: List[np.ndarray],
    filename: str = None,
    axes: List[plt.axes] = None,
    y_labels: List[str] = None,
) -> Union[None, Tuple[plt.Figure, List[plt.axes]]]:
    """
    Plot the original and fitted signals for multiple datasets.

    Parameters
    ----------
    data : List[np.ndarray]
        List of data time series to plot.
        Each element has shape (n_samples, n_channels).
    filename : str, optional
        Filename to save the plot.
    axes : List[plt.axes], optional
        List of axes to plot on. If None, new axes will be created.
    y_labels : List[str], optional
        List of y-axis labels for each subplot.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    axes : List[plt.axes]
        List of axes objects for the subplots.
    """

    if y_labels is None:
        y_labels = [""] * len(data)

    assert len(data) == len(y_labels), "Number of data and y_labels must match"

    create_fig = axes is None
    n_figures = len(data)
    if create_fig:
        fig, axes = plt.subplots(n_figures, 1, figsize=(20, n_figures * 5), sharex=True)

    n_samples = len(data[0][0])
    sampling_frequency = 250
    timestamps = np.arange(n_samples) / sampling_frequency

    for i, (original_data, fitted_data) in enumerate(data):
        axes[i].plot(timestamps, original_data, label="Original Data", color="blue")
        axes[i].plot(timestamps, fitted_data, label="Fitted Data", color="red")
        axes[i].set_ylabel(y_labels[i], fontsize=20)
        axes[i].grid()

    axes[0].legend(fontsize=20)
    axes[-1].set_xlabel("Seconds", fontsize=20)
    axes[-1].tick_params(axis="x", labelsize=20)

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    elif create_fig:
        return fig, axes


def plot_boxplot(
    train: np.ndarray,
    test_camcan: np.ndarray,
    test_wh: np.ndarray,
    filename: str = None,
) -> Union[None, Tuple[plt.Figure, plt.axes]]:
    """
    Plot a boxplot of the Percentage of Variance Explained (PVE) for training and test datasets.

    Parameters
    ----------
    train : np.ndarray
        Array of PVE values for the training dataset.
    test_camcan : np.ndarray
        Array of PVE values for the Cam-CAN test dataset.
    test_wh : np.ndarray
        Array of PVE values for the Wakeman-Henson test dataset.
    filename : str, optional
        Filename to save the plot.

    Returns
    -------
    fig : plt.Figure
        Figure object containing the boxplot.
    ax : plt.axes
        Axes object for the boxplot.
    """
    data = {
        "data": ["Cam-CAN Train (N=50)"] * len(train)
        + ["Cam-CAN Test (N=562)"] * len(test_camcan)
        + ["Wakeman-Henson (N=114)"] * len(test_wh),
        "PVE": np.concatenate([train, test_camcan, test_wh]),
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="data",
        y="PVE",
        data=df,
        ax=ax,
        hue="data",
        palette=["blue", "orange", "green"],
    )
    ax.set_title("Percentage of Variance Explained (PVE)")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        return fig, ax


def plot_results(
    camcan_train_files: List[str],
    camcan_test_files: List[str],
    wh_files: List[str],
    channels: Union[int, List[int]],
    tokenizer_dir: str,
    results_dir: str,
    plot_dir: str,
    n_samples: int = 300,
) -> None:
    """
    Plot the results for tokenizer.

    Parameters
    ----------
    camcan_train_files : List[str]
        List of file paths for Cam-CAN training dataset.
    camcan_test_files : List[str]
        List of file paths for Cam-CAN test dataset.
    wh_files : List[str]
        List of file paths for Wakeman-Henson dataset.
    channels : Union[int, List[int]]
        Channel indices to plot.
    tokenizer_dir : str
        Directory where the tokenizer model is saved.
    results_dir : str
        Directory where the results will be saved.
    plot_dir : str
        Directory where the plots will be saved.
    n_samples : int, optional
        Number of samples to plot from each dataset. Default is 300.
    """
    if not isinstance(channels, list):
        channels = [channels]

    if not isinstance(camcan_train_files, list):
        camcan_train_files = [camcan_train_files]
    if not isinstance(camcan_test_files, list):
        camcan_test_files = [camcan_test_files]
    if not isinstance(wh_files, list):
        wh_files = [wh_files]

    tokenizer = load_model(tokenizer_dir)

    files = camcan_train_files + camcan_test_files + wh_files
    titles = (
        ["Cam-CAN Train"] * len(camcan_train_files)
        + ["Cam-CAN Test"] * len(camcan_test_files)
        + ["Wakeman-Henson"] * len(wh_files)
    )
    data = Data(
        files,
        picks="misc",
        reject_by_annotation="omit",
    )
    data.standardize()

    tokenized_data = tokenizer._tokenize_data(data)
    fitted_data = tokenizer._reconstruct_data(tokenized_data)
    original_data = data.time_series()

    fig, axes = plt.subplots(
        len(files),
        len(channels),
        figsize=(20 * len(channels), 5 * len(files)),
        sharex=True,
        sharey=True,
    )
    for i, c in enumerate(channels):
        data_to_plot = []
        for j in range(len(files)):
            data_to_plot.append(
                (
                    original_data[j][:n_samples, c],
                    fitted_data[j][:n_samples, c],
                )
            )
        ax = axes[:, i] if len(channels) > 1 else axes
        plot_fitted_signal(
            *data_to_plot,
            axes=ax,
            y_labels=titles if i == 0 else None,
        )
        ax[0].set_title(f"Channel {c}", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/fitted_signals.png")
    plt.close(fig)

    camcan_pve = np.load(f"{results_dir}/camcan_pve.npy")
    wh_pve = np.load(f"{results_dir}/wh_pve.npy")

    plot_boxplot(
        camcan_pve[:50],
        camcan_pve[50:],
        wh_pve,
        filename=f"{plot_dir}/boxplot_pve.png",
    )


if __name__ == "__main__":
    tokenizer_dir = "../../models/tokenizer"

    camcan_data_dir = "/well/woolrich/projects/camcan/spring23/src"
    wh_data_dir = "/well/woolrich/projects/wakeman_henson/summer23/src"
    camcan_data_paths = sorted(glob(f"{camcan_data_dir}/*/sflip_parc-raw.fif"))
    wh_data_paths = sorted(glob(f"{wh_data_dir}/sub*_run*/sflip_parc-raw.fif"))

    results_dir = "../../results/pve"
    os.makedirs(results_dir, exist_ok=True)

    # Calculate PVE
    calculate_pve(
        camcan_data_paths,
        wh_data_paths,
        tokenizer_dir=tokenizer_dir,
        save_dir=results_dir,
    )

    # Plot results
    plot_dir = "../../plots/1_tokeniser_reconstructs"
    os.makedirs(plot_dir, exist_ok=True)
    plot_results(
        camcan_data_paths[0],
        camcan_data_paths[100],
        wh_data_paths[0],
        channels=[0],
        tokenizer_dir=tokenizer_dir,
        results_dir=results_dir,
        plot_dir=plot_dir,
    )
