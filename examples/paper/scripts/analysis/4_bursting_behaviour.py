from sys import argv

import os
from glob import glob
from typing import Tuple

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from osl_dynamics import run_pipeline
from osl_dynamics.inference import modes
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, spectral, statistics


def plot_channel_location(channel: int, plot_dir: str) -> None:
    """
    Plot the location of a specific channel on the brain.

    Parameters
    ----------
    channel : int
        The index of the channel to plot.
    plot_dir : str
        Directory to save the plot.
    """
    os.makedirs(plot_dir, exist_ok=True)

    power_map = np.zeros(52)
    power_map[channel] = 1
    power.save(
        power_map,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{plot_dir}/channel_location.png",
    )


def prepare_data(channel: int) -> Tuple[Data, Data, Data]:
    """Prepare data for bursting detection.

    Parameters
    ----------
    channel : int
        The index of the channel to prepare data for.

    Returns
    -------
    real_data : Data
        Real data for the specified channel.
    gen_data : Data
        Generated data from MEG-GPT for the specified channel.
    ar_gen_data : Data
        Generated data from the autoregressive model for the specified channel.
    """

    generated_data_dir = "../../data/generated_data"
    with open(f"{generated_data_dir}/meg-gpt.pkl", "rb") as f:
        gen_data = pickle.load(f)

    with open(f"{generated_data_dir}/ar_model.pkl", "rb") as f:
        ar_gen_data = pickle.load(f)

    n_samples = gen_data[0].shape[0]

    # Select the channel of interest
    gen_data = [d[:, [channel]] for d in gen_data]
    ar_gen_data = [d[:, [channel]] for d in ar_gen_data]

    # Load real data
    real_data_dir = "/well/woolrich/projects/camcan/spring23/src"
    data_files = sorted(glob(f"{real_data_dir}/*/sflip_parc-raw.fif"))
    real_data = Data(
        data_files,
        n_jobs=8,
        picks="misc",
        reject_by_annotation="omit",
    )
    real_data = [d[:n_samples, [channel]] for d in real_data.time_series()]

    def _prepare_data(data):
        methods = {
            "tde": {"n_embeddings": 21},
            "standardize": {},
        }

        data = Data(data, sampling_frequency=250, n_jobs=8)
        data.downsample(100)
        data = Data(data.time_series(), sampling_frequency=100, n_jobs=8)
        data.prepare(methods)
        return data

    real_data = _prepare_data(real_data)
    gen_data = _prepare_data(gen_data)
    ar_gen_data = _prepare_data(ar_gen_data)

    return real_data, gen_data, ar_gen_data


def get_summary_stats(data: Data, output_dir: str) -> None:
    """
    Get summary statistics for bursting behaviour.

    Parameters
    ----------
    data : Data
        Data object containing the time series.
    output_dir : str
        Directory to save the summary statistics.
    """
    inf_params_dir = f"{output_dir}/inf_params"
    summary_stats_dir = f"{output_dir}/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    stc = modes.argmax_time_courses(alp)

    lt = modes.mean_lifetimes(stc, data.sampling_frequency)
    intv = modes.mean_intervals(stc, data.sampling_frequency)
    bc = modes.switching_rates(stc, data.sampling_frequency)
    np.save(f"{summary_stats_dir}/lt.npy", lt)
    np.save(f"{summary_stats_dir}/intv.npy", intv)
    np.save(f"{summary_stats_dir}/bc.npy", bc)


def get_order(generated_output_dir: str) -> np.ndarray:
    """
    Get the order of states based on the covariance matrices.

    Parameters
    ----------
    generated_output_dir : str
        Directory containing the generated output.

    Returns
    -------
    order : np.ndarray
        The order of states based on the covariance matrices.
        Shape is (n_states,).
    """
    generated_inf_params_dir = f"{generated_output_dir}/inf_params"
    real_inf_params_dir = f"{os.path.dirname(generated_output_dir)}/real/inf_params"
    generated_covs = np.load(f"{generated_inf_params_dir}/covs.npy")
    real_covs = np.load(f"{real_inf_params_dir}/covs.npy")
    order = modes.match_covariances(real_covs, generated_covs, return_order=True)[1]
    return order


def plot_psd(results_dir: str, plot_dir: str) -> None:
    """
    Plot the state specific power spectral density (PSD)

    Parameters
    ----------
    results_dir : str
        Directory containing the results of the bursting detection.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    real_psd = np.load(f"{results_dir}/real/spectra/psd.npy")
    gen_psd = np.load(f"{results_dir}/meg-gpt/spectra/psd.npy")
    gen_ar_psd = np.load(f"{results_dir}/ar_model/spectra/psd.npy")
    f = np.load(f"{results_dir}/real/spectra/f.npy")

    gen_psd = gen_psd[:, get_order(f"{results_dir}/meg-gpt")]
    gen_ar_psd = gen_ar_psd[:, get_order(f"{results_dir}/ar_model")]

    real_psd = np.mean(real_psd, axis=2)
    gen_psd = np.mean(gen_psd, axis=2)
    gen_ar_psd = np.mean(gen_ar_psd, axis=2)

    real_psd_mean = np.mean(real_psd, axis=0)
    real_psd_std = np.std(real_psd, axis=0)
    gen_psd_mean = np.mean(gen_psd, axis=0)
    gen_psd_std = np.std(gen_psd, axis=0)
    gen_ar_psd_mean = np.mean(gen_ar_psd, axis=0)
    gen_ar_psd_std = np.std(gen_ar_psd, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    plotting.plot_line(
        [f] * 3,
        real_psd_mean,
        errors=[list(real_psd_mean - real_psd_std), list(real_psd_mean + real_psd_std)],
        labels=[f"State {i + 1}" for i in range(3)],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        ax=axes[0],
        title="Real Data",
    )
    plotting.plot_line(
        [f] * 3,
        gen_psd_mean,
        errors=[list(gen_psd_mean - gen_psd_std), list(gen_psd_mean + gen_psd_std)],
        x_label="Frequency (Hz)",
        x_range=[f[0], f[-1]],
        ax=axes[1],
        title="MEG-GPT",
    )
    plotting.plot_line(
        [f] * 3,
        gen_ar_psd_mean,
        errors=[
            list(gen_ar_psd_mean - gen_ar_psd_std),
            list(gen_ar_psd_mean + gen_ar_psd_std),
        ],
        x_label="Frequency (Hz)",
        x_range=[f[0], f[-1]],
        ax=axes[2],
        title="AR",
    )
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/state_psd.png")
    plt.close(fig)


def plot_wavelet(
    real_data: Data, gen_data: Data, ar_gen_data: Data, plot_dir: str
) -> None:
    """
    Plot the wavelet transform.

    Parameters
    ----------
    real_data : Data
        Real data.
    gen_data : Data
        Generated data from MEG-GPT.
    ar_gen_data : Data
        Generated data from the autoregressive model.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    vmin = float("inf")
    vmax = float("-inf")
    for data in [real_data, gen_data, ar_gen_data]:
        x = data.time_series(prepared=False)
        x = x[0][:, 0]
        t, f, wt = spectral.wavelet(
            data=x,
            sampling_frequency=250,
            w=5,
            standardize=True,
            time_range=[0, 60],
            frequency_range=None,
        )
        vmin = min(vmin, np.min(wt))
        vmax = max(vmax, np.max(wt))

    def _plot_wavelet(data, ax, vmin=None, vmax=None, add_colorbar=True):
        """Plot wavelet transform of the training data."""
        x = data.time_series(prepared=False)  # raw data (before time-delay embedding)
        x = x[0][:, 0]  # first subject
        plotting.plot_wavelet(
            x,
            ax=ax,
            sampling_frequency=data.sampling_frequency,
            time_range=[0, 60],
            plot_kwargs={"vmin": vmin, "vmax": vmax},
            add_colorbar=False,
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=20)
        ax.tick_params(axis="y", labelsize=15)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    _plot_wavelet(real_data, axes[0], vmin=vmin, vmax=vmax)
    _plot_wavelet(gen_data, axes[1], add_colorbar=False, vmin=vmin, vmax=vmax)
    _plot_wavelet(ar_gen_data, axes[2], add_colorbar=False, vmin=vmin, vmax=vmax)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Time (s)", fontsize=20)

    axes[0].set_title("Real Data", fontsize=20)
    axes[1].set_title("MEG-GPT", fontsize=20)
    axes[2].set_title("AR", fontsize=20)

    axes[2].tick_params(axis="x", labelsize=15)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/wavelet.png")
    plt.close(fig)


def plot_summary_stats(results_dir, plot_dir) -> None:
    """
    Plot summary statistics for bursting behaviour.

    Parameters
    ----------
    results_dir : str
        Directory containing the results of the bursting detection.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    real_bc = np.load(f"{results_dir}/real/summary_stats/bc.npy")
    real_intv = np.load(f"{results_dir}/real/summary_stats/intv.npy")
    real_lt = np.load(f"{results_dir}/real/summary_stats/lt.npy")
    real_features = np.transpose([real_bc, real_intv, real_lt], (1, 0, 2))

    gen_bc = np.load(f"{results_dir}/meg-gpt/summary_stats/bc.npy")
    gen_intv = np.load(f"{results_dir}/meg-gpt/summary_stats/intv.npy")
    gen_lt = np.load(f"{results_dir}/meg-gpt/summary_stats/lt.npy")
    gen_features = np.transpose([gen_bc, gen_intv, gen_lt], (1, 0, 2))
    gen_features = gen_features[:, :, get_order(f"{results_dir}/meg-gpt")]

    gen_ar_bc = np.load(f"{results_dir}/ar_model/summary_stats/bc.npy")
    gen_ar_intv = np.load(f"{results_dir}/ar_model/summary_stats/intv.npy")
    gen_ar_lt = np.load(f"{results_dir}/ar_model/summary_stats/lt.npy")
    gen_ar_features = np.transpose([gen_ar_bc, gen_ar_intv, gen_ar_lt], (1, 0, 2))
    gen_ar_features = gen_ar_features[:, :, get_order(f"{results_dir}/ar_model")]

    gen_sum_diff = np.abs(gen_features - real_features)
    gen_ar_sum_diff = np.abs(gen_ar_features - real_features)

    n_sessions = gen_features.shape[0]
    _, p = statistics.group_diff_max_stat_perm(
        np.concatenate([gen_sum_diff, gen_ar_sum_diff], axis=0),
        assignments=np.array([0] * n_sessions + [1] * n_sessions),
        n_perm=1000,
        metric="tstats",
    )

    summary_stat_names = [
        "Bursting Count (Hz)",
        "Mean Interval (s)",
        "Mean Lifetime (s)",
    ]

    def _plot_summary_stats(
        name,
        real_summary_stats,
        gen_summary_stats,
        gen_ar_summary_stats,
        pvalues,
        ax,
        clip_quantile=0.99,
    ):
        n_sessions, n_states = real_summary_stats.shape
        ss_dict = {name: [], "State": [], "Group": []}
        for i in range(n_sessions):
            for j in range(n_states):
                ss_dict[name].append(gen_summary_stats[i, j])
                ss_dict["State"].append(j + 1)
                ss_dict["Group"].append("MEG-GPT")

                ss_dict[name].append(real_summary_stats[i, j])
                ss_dict["State"].append(j + 1)
                ss_dict["Group"].append("Real Data")

                ss_dict[name].append(gen_ar_summary_stats[i, j])
                ss_dict["State"].append(j + 1)
                ss_dict["Group"].append("AR")

        ss_df = pd.DataFrame(ss_dict)

        # Clip outliers to avoid extreme violins
        lower = ss_df[name].quantile(1 - clip_quantile)
        upper = ss_df[name].quantile(clip_quantile)
        ss_df[name] = ss_df[name].clip(lower, upper)

        sns.violinplot(
            data=ss_df,
            x="State",
            y=name,
            hue="Group",
            width=0.6,
            dodge=True,
            cut=0,
            ax=ax,
        )

        # Larger x labels
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylabel(name, fontsize=20)

        # Get the positions of the violins
        n_states = ss_df["State"].nunique()
        n_hues = ss_df["Group"].nunique()

        violin_positions = []
        for artist in ax.findobj(mpl.collections.PolyCollection):
            path = artist.get_paths()[0].vertices
            x = np.mean(path[:, 0])
            violin_positions.append(x)
        violin_positions = np.array(violin_positions).reshape(n_states, n_hues)

        # Get y max for bracket height
        y_max = ss_df[name].max()
        # y_max = 1.2 * np.quantile(ss_df[name], 0.95)
        y_min = ss_df[name].min()
        y_bracket = y_max + 0.05 * (y_max - y_min)
        y_bracket_step = 0.025 * (y_max - y_min)

        for i in range(n_states):
            # Between first and second hue
            x1, x2 = violin_positions[i, 0], violin_positions[i, 1]
            ax.plot(
                [x1, x1, x2, x2],
                [
                    y_bracket,
                    y_bracket + y_bracket_step,
                    y_bracket + y_bracket_step,
                    y_bracket,
                ],
                c="k",
                lw=1.5,
            )

            # Between second and third hue
            x1, x2 = violin_positions[i, 1], violin_positions[i, 2]
            ax.plot(
                [x1, x1, x2, x2],
                [
                    y_bracket + y_bracket_step * 2,
                    y_bracket + y_bracket_step * 3,
                    y_bracket + y_bracket_step * 3,
                    y_bracket + y_bracket_step * 2,
                ],
                c="k",
                lw=1.5,
            )

            # Between first and third hue (add this bracket above the previous two)
            x1, x2 = (violin_positions[i, 0] + violin_positions[i, 1]) / 2, (
                violin_positions[i, 1] + violin_positions[i, 2]
            ) / 2
            ax.plot(
                [x1, x1, x2, x2],
                [
                    y_bracket + y_bracket_step * 4,
                    y_bracket + y_bracket_step * 5,
                    y_bracket + y_bracket_step * 5,
                    y_bracket + y_bracket_step * 4,
                ],
                c="k",
                lw=1.5,
            )

            if pvalues[i] <= 0.001:
                sig = "***"
            elif pvalues[i] <= 0.01:
                sig = "**"
            elif pvalues[i] <= 0.05:
                sig = "*"
            else:
                sig = "n.s."

            ax.text(
                (x1 + x2) / 2,
                y_bracket + y_bracket_step * 5 + 0.01,
                sig,
                ha="center",
                va="bottom",
                color="k",
                fontsize=15,
                fontweight="bold",
            )
            ax.set_ylim(0, 1.2 * (y_bracket + y_bracket_step * 5))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for i, name in enumerate(summary_stat_names):
        _plot_summary_stats(
            name,
            real_features[:, i, :],
            gen_features[:, i, :],
            gen_ar_features[:, i, :],
            p[i],
            axes[i],
        )

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("State", fontsize=20)
    axes[1].legend_.remove()
    axes[2].legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend_.remove()
        axes[0].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            fontsize=15,
        )
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/bursting_summary_stats.png")
    plt.close(fig)


if __name__ == "__main__":
    if len(argv) != 2:
        print(f"Please pass the channel id, e.g. python {argv[0]} 0")
        exit()

    channel = int(argv[1])
    results_dir = f"../../results/bursting_detection/channel_{channel}"
    plot_dir = f"../../plots/4_bursting_detection/channel_{channel}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot the channel location
    plot_channel_location(channel, plot_dir)

    # Prepare the data
    real_data, gen_data, ar_gen_data = prepare_data(channel)

    # Run the bursting detection HMM
    config = """
        train_hmm:
            config_kwargs:
                n_states: 3
                sequence_length: 200
                batch_size: 128
                learn_means: False
                learn_covariances: True
        multitaper_spectra:
            kwargs:
                frequency_range: [1, 45]
        get_summary_stats: {}
    """
    for data, name in zip(
        [real_data, gen_data, ar_gen_data],
        ["real", "meg-gpt", "ar_model"],
    ):
        run_pipeline(
            config,
            output_dir=f"{results_dir}/{name}",
            data=data,
            extra_funcs=[get_summary_stats],
        )

    # Plot the power spectral density (PSD)
    plot_psd(results_dir, plot_dir)

    # Plot the wavelet transform
    plot_wavelet(real_data, gen_data, ar_gen_data, plot_dir)

    # Plot the summary statistics
    plot_summary_stats(results_dir, plot_dir)
