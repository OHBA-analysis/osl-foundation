from typing import List, Tuple, Union
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import signal

from osl_dynamics.data import Data
from osl_dynamics.analysis import static, spectral


def plot_time_series_summary(
    time_series: np.ndarray,
    sampling_frequency: int,
    w: float = 6.0,
    n_samples: int = None,
    axes: List[plt.Axes] = None,
    filename: str = None,
    color: str = "black",
    label: str = None,
):
    """
    Plot a summary of the time series, including the time series itself,
    the time-frequency content, and the power spectral density.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data of shape (n_samples,).
    sampling_frequency : int
        Sampling frequency in Hz.
    w : float, optional
        Omega0 parameter for the Morlet wavelet. Default is 6.0.
    n_samples : int, optional
        Number of samples to plot. Default is None, which will plot all samples.
    axes : list, optional
        List of matplotlib axes to plot on. If not provided, a new figure will
        be created.
    filename : str, optional
        If provided, the figure will be saved to this filename.
    color : str, optional
        Color of the time series plot. Default is 'black'.
    label : str, optional
        Label for the time series plot. Default is None.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure. Only returned if :code:`filename=None`.
    axes : List[plt.Axes]
        List of matplotlib axes. Only returned if :code:`filename=None`.
    """
    n_samples = n_samples or len(time_series)

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    else:
        fig = axes[0].get_figure()

    timestamps, freqs, cwtm = spectral.wavelet(
        time_series[-n_samples:], sampling_frequency=sampling_frequency
    )

    # Plot the time series
    axes[0].plot(timestamps, time_series[-n_samples:], color=color)
    axes[0].set_xlabel("Time (s)")

    # Plot the time frequency content
    axes[1].pcolormesh(timestamps, freqs, cwtm, shading="gouraud", cmap="viridis")
    axes[1].set_xlabel("Time (s)")

    # Plot the PSD
    f, Pxx = signal.welch(
        time_series[-n_samples:], fs=sampling_frequency, nperseg=2 * sampling_frequency
    )
    axes[2].plot(f, Pxx, color=color, label=label)
    axes[2].set_xlabel("Frequency (Hz)")

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig, axes


def plot_aec(
    *inputs,
    window_size: int,
    sampling_frequency: int,
    frequency_range: list = None,
    axes: List[plt.Axes] = None,
    titles: List[str] = None,
    cmap: str = "viridis",
    filename: str = None,
    cbar: bool = True,
) -> Union[Tuple[plt.Figure, List[plt.Axes]], None]:
    """
    Plot the amplitude envelope correlation (AEC) of the inputs.

    Parameters
    ----------
    inputs : list
        List of data inputs. Each input can be
        - A path to a directory containing :code:`.npy` files. Each
          :code:`.npy` file should be a subject or session.
        - A list of paths to :code:`.npy`, :code:`.mat` or :code:`.fif` files.
          Each file should be a subject or session. If a :code:`.fif` file is
          passed is must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
        - A numpy array. The array will be treated as continuous data from the
          same subject.
        - A list of numpy arrays. Each numpy array should be the data for a
          subject or session.
    window_size : int
        Window size in milliseconds.
    frequency_range : list, optional
        List of low and high frequencies for the filter.
    sampling_frequency : int
        Sampling frequency in Hz.
    axes : list, optional
        List of matplotlib axes to plot on. If not provided, a new figure will
        be created.
    titles : list, optional
        List of titles for each input. If not provided, the titles will be
        empty.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    filename : str, optional
        If provided, the figure will be saved to this filename.
    cbar : bool, optional
        Whether to show the colorbar. Default is True.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure. Only returned if :code:`filename=None`.
    axes : List[plt.Axes]
        List of matplotlib axes. Only returned if :code:`filename=None`.
    """

    # Validation
    if titles is None:
        titles = ["" for _ in range(len(inputs))]
    if len(inputs) != len(titles):
        raise ValueError("The number of titles must match the number of data inputs.")

    # Get n_window - i.e. number of tpts
    # Passed in window_size is in msecs
    window_size = window_size / 1000 # in secs
    n_window = int(window_size * sampling_frequency) # in num tpts

    # Make sure window_size is odd
    if n_window % 2 == 0:
        n_window += 1

    frequency_range = frequency_range or [None, None]

    methods = {
        "filter": {"low_freq": frequency_range[0], "high_freq": frequency_range[1]},
        "amplitude_envelope": {},
        "moving_average": {"n_window": n_window}, 
    }

    def _get_aec(data: Data) -> np.ndarray:
        data.prepare(methods)
        ts = data.time_series()
        aec = static.functional_connectivity(ts)
        if len(aec.shape) == 3:
            aec = np.mean(aec, axis=0)
        return aec - np.eye(aec.shape[0])

    if axes is None:
        fig, axes = plt.subplots(1, len(inputs), figsize=(5 * (len(inputs) + 1), 5))
    else:
        fig = axes[0].get_figure()

    aec = []
    for d in inputs:
        aec.append(_get_aec(Data(d, sampling_frequency=sampling_frequency)))

    vmin = np.min(aec)
    vmax = np.max(aec)

    for i in range(len(aec)):
        sns.heatmap(aec[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar)
        axes[i].set_title(titles[i])

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig, axes


def plot_time_frequency(
    *inputs: List[np.ndarray],
    sampling_frequency: int,
    w: float = 6.0,
    n_samples: int = None,
    titles: List[str] = None,
    cmap: str = "viridis",
    filename: str = None,
) -> Union[Tuple[plt.Figure, List[plt.Axes]], None]:
    """
    Plot the time-frequency content of the inputs.

    Parameters
    ----------
    inputs : List[np.ndarray]
        List of data inputs, where each input is a numpy array of shape
        (n_samples, n_channels).
    sampling_frequency : int
        Sampling frequency in Hz.
    w : float, optional
        Omega0 parameter for the Morlet wavelet. Default is 6.0.
    n_samples : int, optional
        Number of samples to plot. Default is None, which will plot all samples.
    titles : List[str], optional
        List of titles for each input. If not provided, the titles will be empty.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    filename : str, optional
        If provided, the figure will be saved to this filename.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure. Only returned if :code:`filename=None`.
    axes : List[plt.Axes]
        List of matplotlib axes. Only returned if :code:`filename=None`.
    """
    # Validation
    if len(inputs) != len(titles):
        raise ValueError("The number of titles must match the number of data inputs.")

    n_channels = inputs[0].shape[1]
    for i in range(1, len(inputs)):
        if inputs[i].shape[1] != n_channels:
            raise ValueError("All inputs must have the same number of channels.")

    fig, axes = plt.subplots(
        n_channels, len(inputs), figsize=(5 * len(inputs), 5 * n_channels)
    )

    for i in range(n_channels):
        for j, data in enumerate(inputs):
            n_samples_to_plot = n_samples or data.shape[0]
            timestamps, freqs, cwtm = spectral.wavelet(
                data[-n_samples_to_plot:, i],
                sampling_frequency=sampling_frequency,
            )
            axes[i, j].pcolormesh(timestamps, freqs, cwtm, cmap=cmap, shading="gouraud")

            # Set the title
            if i == 0:
                axes[i, j].set_title(titles[j])

            # Set the y-axis label
            if j == 0:
                axes[i, j].set_ylabel(f"Channel {i}")

            # Set the x-axis label
            if i == n_channels - 1:
                axes[i, j].set_xlabel("Time (s)")

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig, axes


def plot_history(
    history: dict,
    plot_dir: str = None,
    keywords: List[str] = None,
    epoch_range: tuple = None,
) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
    """Plot the training history.

    Parameters
    ----------
    history : dict
        History dictionary
    plot_dir : str, optional
        Directory to save the plot.
    keywords : List[str], optional
        List of keywords to filter the history. If None, all history
        will be plotted.
    epoch_range : tuple, optional
        Range of epochs to plot.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : List[plt.Axes]
        Axes object.
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    keywords = keywords or list(history.keys())

    if epoch_range is not None:
        epoch_start = epoch_range[0]
        epoch_end = epoch_range[1]
    else:
        epoch_start = 1
        epoch_end = len(history["loss"])

    epoch_indices = np.arange(epoch_start, epoch_end + 1)

    for key in history.keys():
        if key not in keywords:
            continue
        if "loss" in key:
            if "val_loss"==key:
                col = "b--"
            elif "loss"==key:
                col = "b."
            elif "val" in key:
                col = "g--"
            else:
                col = "g."
            ax1.plot(
                epoch_indices,
                history[key][epoch_start - 1 : epoch_end],
                col,
                label=key,
            )
        elif "top" in key:
            ax2.plot(
                epoch_indices,
                history[key][epoch_start - 1 : epoch_end],
                "r--" if "val" in key else "r.",
                label=key,
            )
        else:
            continue
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Set x-axis limits
    ax1.set_xlim((epoch_start - 1, epoch_end + 1))

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/history.png")
        plt.close(fig)
    else:
        return fig, [ax1, ax2]


def plot_pve(
    pve: np.ndarray, plot_dir: str = None
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """Plot the proportion of variance explained (PVE).

    Parameters
    ----------
    pve : np.ndarray
        Proportion of variance explained.
    plot_dir : str, optional
        Directory to save the plot.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """
    # Plot a histogram of PVEs
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.hist(pve, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("PVE (%)")
    ax.set_ylabel("Number of Sessions")
    ax.set_title("Percentage of Variance Explained (Avg: {:.2f}%)".format(pve.mean()))
    plt.tight_layout()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/pve_histogram.png")
        plt.close(fig)
    else:
        return fig, ax
    
def plot_tde_corr(
    *inputs,
    axes: List[plt.Axes] = None,
    titles: List[str] = None,
    n_embeddings: int = 15,
    n_pca_components: int = None,
    frequency_range: list = None,
    sampling_frequency: int,
    cmap: str = "viridis",
    filename: str = None,
    cbar: bool = True,
) -> Union[Tuple[plt.Figure, List[plt.Axes]], None]:
    """
    Plot the TDE correlations of the inputs.

    Parameters
    ----------
    inputs : list
        List of data inputs. Each input can be
        - A path to a directory containing :code:`.npy` files. Each
          :code:`.npy` file should be a subject or session.
        - A list of paths to :code:`.npy`, :code:`.mat` or :code:`.fif` files.
          Each file should be a subject or session. If a :code:`.fif` file is
          passed is must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
        - A numpy array. The array will be treated as continuous data from the
          same subject.
        - A list of numpy arrays. Each numpy array should be the data for a
          subject or session.
    axes : list, optional
        List of matplotlib axes to plot on. If not provided, a new figure will
        be created.
    titles : list, optional
        List of titles for each input. If not provided, the titles will be
        empty.
    n_embeddings : int, optional
        Number of embeddings to use for TDE. Default is 15.
    n_pca_components : int, optional
        Number of PCA components to use after TDE. Default is None.
    frequency_range: list = None,
        List of low and high frequencies (Hz) for the bandpass filter used prior 
        to TDE
    sampling_frequency : int
        Sampling frequency in Hz.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    filename : str, optional
        If provided, the figure will be saved to this filename.
    cbar : bool, optional
        Whether to show the colorbar. Default is True.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure. Only returned if :code:`filename=None`.
    axes : List[plt.Axes]
        List of matplotlib axes. Only returned if :code:`filename=None`.
    """

    # Validation
    if titles is None:
        titles = ["" for _ in range(len(inputs))]
    if len(inputs) != len(titles):
        raise ValueError("The number of titles must match the number of data inputs.")

    if n_pca_components is None:
        methods = {
            "filter": {"low_freq": frequency_range[0], "high_freq": frequency_range[1]},
            "tde": {"n_embeddings": n_embeddings},
            "standardize": {},
        }
    else:
        methods = {
            "filter": {"low_freq": frequency_range[0], "high_freq": frequency_range[1]},
            "tde_pca": {"n_embeddings": n_embeddings, "n_pca_components": n_pca_components},
            "standardize": {},
        }

    def _get_tde_corr(data: Data) -> np.ndarray:
        data.prepare(methods)
        ts = data.time_series()
        tde_corr = static.functional_connectivity(ts)
        if len(tde_corr.shape) == 3:
            tde_corr = np.mean(tde_corr, axis=0)        
        return tde_corr

    if axes is None:
        fig, axes = plt.subplots(1, len(inputs), figsize=(5 * (len(inputs) + 1), 5))
    else:
        fig = axes[0].get_figure()

    tde_corr4plot = []
    tde_corr = []
    for d in inputs:
        n_channels = Data(d).n_channels

        tcorr = _get_tde_corr(Data(d, sampling_frequency=sampling_frequency)) # n_embeddings*n_channels x n_embeddings*n_channels

        tde_corr.append(tcorr)

        # zero out (n_embeddings x n_embeddings) diagonal blocks
        tcorr4plot = tcorr.copy()
        for i in range(n_channels):
            tcorr4plot[i*n_embeddings:(i+1)*n_embeddings, i*n_embeddings:(i+1)*n_embeddings] = 0

        tde_corr4plot.append(tcorr4plot)

    vmin = np.min(tde_corr4plot)
    vmax = np.max(tde_corr4plot)

    for d, i in zip(inputs, range(len(tde_corr4plot))):

        n_channels = Data(d).n_channels

        sns.heatmap(tde_corr4plot[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar)
        axes[i].set_title(titles[i])

        # only have a label for the start of each channel block 
        axes[i].set_xticks(np.linspace(0, (n_channels-1)*n_embeddings, n_channels))
        axes[i].set_xticklabels(np.arange(0, n_channels))  
        axes[i].set_yticks(np.linspace(0, (n_channels-1)*n_embeddings, n_channels))
        axes[i].set_yticklabels(np.arange(0, n_channels)) 

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

        return tde_corr

    else:
        return tde_corr, fig, axes
