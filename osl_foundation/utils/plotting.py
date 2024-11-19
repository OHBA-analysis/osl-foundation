from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.analysis import static


def plot_aec(
    *inputs,
    window_size: int,
    sampling_frequency: int,
    axes: List[plt.Axes] = None,
    titles: List[str] = None,
    cmap: str = "viridis",
    filename: str = None,
) -> Union[Tuple[plt.Figure, List[plt.Axes]], None]:

    # Validation
    if len(inputs) != len(titles):
        raise ValueError("The number of titles must match the number of data inputs.")

    # Get the window size in number of samples
    window_size = int(window_size * sampling_frequency / 1000)

    # Make sure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    methods = {
        "filter": {"low_freq": 1, "high_freq": 45},
        "amplitude_envelope": {},
        "moving_average": {"n_window": window_size},
    }

    def _get_aec(data: Data) -> np.ndarray:
        data.prepare(methods)
        ts = data.time_series(concatenate=True)
        aec = static.functional_connectivity(ts)
        return aec

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
        sns.heatmap(aec[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(titles[i])

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
    else:
        return fig, axes
