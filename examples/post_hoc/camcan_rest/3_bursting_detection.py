import os
from glob import glob

import pickle
import numpy as np

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power

tf_ops.gpu_growth()


def plot_channel_location(channel, plot_dir):
    """
    Plot the location of a specific channel on the brain.

    Parameters
    ----------
    channel : int
        The index of the channel to plot.
    plot_dir : str
        Directory to save the plot.
    """
    power_map = np.zeros(52)
    power_map[channel] = 1
    power.save(
        power_map,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{plot_dir}/channel_location.png",
    )


def plot_wavelet(data, output_dir):
    """Plot wavelet transform of the training data."""

    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    x = data.time_series(prepared=False)  # raw data (before time-delay embedding)
    x = x[0][:, 0]  # first subject
    plotting.plot_wavelet(
        x,
        sampling_frequency=data.sampling_frequency,
        time_range=[0, 60],
        filename=f"{plots_dir}/wavelet.png",
    )


def get_order(generated_output_dir):
    """Get the order of states based on the covariance matrices."""
    generated_inf_params_dir = f"{generated_output_dir}/inf_params"
    real_inf_params_dir = f"{os.path.dirname(generated_output_dir)}/real/inf_params"

    generated_covs = np.load(f"{generated_inf_params_dir}/covs.npy")
    real_covs = np.load(f"{real_inf_params_dir}/covs.npy")

    order = modes.match_covariances(real_covs, generated_covs, return_order=True)[1]
    return order


def plot_amplitude_envelopes_and_alpha(data, output_dir, n_samples, is_generated=False):
    """Plot amplitude envelopes and inferred state probabilities."""

    inf_params_dir = f"{output_dir}/inf_params"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot state probability time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    alp = np.concatenate(alp)

    if is_generated:
        order = get_order(output_dir)
        alp = alp[:, order]

    plotting.plot_alpha(
        alp,
        n_samples=n_samples,
        sampling_frequency=data.sampling_frequency,
        fig_kwargs={"figsize": (12, 3)},
        filename=f"{plots_dir}/alpha.png",
    )

    def get_amp_env(low_freq, high_freq, n_embeddings=21, sequence_length=200):
        data.prepare(
            {
                "filter": {
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "use_raw": True,
                },
                "amplitude_envelope": {},
                "standardize": {},
            }
        )
        x = data.trim_time_series(sequence_length, n_embeddings, concatenate=True)
        return x[:, 0]

    # Get amplitude envelope data for different frequency bands
    x_beta = get_amp_env(13, 30)
    x_alpha = get_amp_env(7, 13)
    x_delta_theta = get_amp_env(1, 7)
    x = np.array([x_beta, x_alpha, x_delta_theta]).T

    # Calculate the correlation of each amplitude envelope with the state probabilities
    corr = np.corrcoef(x, alp, rowvar=False)[:3, 3:]
    fig, ax = plotting.plot_matrices(corr)
    ax[0][0].set_xticklabels([0, 1, 2, 3])
    ax[0][0].set_yticklabels(["", "Beta", "Alpha", "Delta/Theta"])
    fig.savefig(f"{plots_dir}/alp_amp_env_corr.png")

    # Plot amplitude envelopes
    t = np.arange(n_samples) / data.sampling_frequency
    x_beta = x_beta[:n_samples]
    x_alpha = x_alpha[:n_samples]
    x_delta_theta = x_delta_theta[:n_samples]
    plotting.plot_line(
        [t],
        [x_beta],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_beta.png",
    )
    plotting.plot_line(
        [t],
        [x_alpha],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_alpha.png",
    )
    plotting.plot_line(
        [t],
        [x_delta_theta],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_delta_theta.png",
    )


def plot_state_psds(data, output_dir, is_generated=False):
    """Plot power spectral density (PSD) of each state."""

    spectra_dir = f"{output_dir}/spectra"

    f = np.load(f"{spectra_dir}/f.npy")
    psd = np.load(f"{spectra_dir}/psd.npy")
    psd = np.mean(psd, axis=(0, 2))  # average over arrays and channels
    if is_generated:
        order = get_order(output_dir)
        psd = psd[order]
    n_states = psd.shape[0]

    from osl_dynamics.utils import plotting

    plotting.plot_line(
        [f] * n_states,
        psd,
        labels=[f"State {i + 1}" for i in range(n_states)],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=f"{spectra_dir}/psds.png",
    )


def plot_tde_covariances(data, output_dir, is_generated=False):
    """Plot covariance matrices of the time-delay embedding (TDE) data."""

    inf_params_dir = f"{output_dir}/inf_params"

    covs = np.load(f"{inf_params_dir}/covs.npy")
    if is_generated:
        order = get_order(output_dir)
        covs = covs[order]

    if hasattr(data, "pca_components"):
        if data.pca_components is not None:
            from osl_dynamics.analysis import modes

            covs = modes.reverse_pca(covs, data.pca_components)

    from osl_dynamics.utils import plotting

    plotting.plot_matrices(covs, filename=f"{inf_params_dir}/tde_covs.png")


def plot_burst_summary_stats(
    data, output_dir, sampling_frequency=None, is_generated=False
):
    """Plot summary statistics of bursting activity."""
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    from osl_dynamics.inference import modes

    # Load state time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    if is_generated:
        order = get_order(output_dir)
        alp = [a[:, order] for a in alp]

    stc = modes.argmax_time_courses(alp)

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate summary stats
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    bc = modes.switching_rates(stc, sampling_frequency)
    amp = modes.mean_amplitudes(stc, data)

    # Save summary stats
    np.save(f"{summary_stats_dir}/lt.npy", lt)
    np.save(f"{summary_stats_dir}/intv.npy", intv)
    np.save(f"{summary_stats_dir}/bc.npy", bc)
    np.save(f"{summary_stats_dir}/amp.npy", amp)

    from osl_dynamics.utils import plotting

    # Plot
    n_states = lt.shape[1]
    plotting.plot_violin(
        lt.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/lt.png",
    )
    plotting.plot_violin(
        intv.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
    )
    plotting.plot_violin(
        bc.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Burst Count (Hz)",
        filename=f"{summary_stats_dir}/bc.png",
    )
    plotting.plot_violin(
        amp.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Amplitude (a.u.)",
        filename=f"{summary_stats_dir}/amp.png",
    )


def prepare_data(channel):
    """Prepare data for bursting detection."""

    # Load generated data
    with open("results/generated_data.pkl", "rb") as f:
        generated_data = pickle.load(f)

    # Load generated data from untrained generator
    with open("results/generated_data_untrained.pkl", "rb") as f:
        generated_data_untrained = pickle.load(f)
    n_samples = generated_data[0].shape[0]

    # Select the channel of interest
    generated_data = [d[:, [channel]] for d in generated_data]
    generated_data_untrained = [d[:, [channel]] for d in generated_data_untrained]

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
    generated_data = _prepare_data(generated_data)
    generated_data_untrained = _prepare_data(generated_data_untrained)

    return real_data, generated_data, generated_data_untrained


if __name__ == "__main__":
    # Set up channel of interest and output directory
    # 0 - visual
    # 4 - motor
    # 22 - prefrontal

    channel = 4
    results_dir = f"results/bursting_detection/channel_{channel}"
    os.makedirs(results_dir, exist_ok=True)

    plot_channel_location(channel, results_dir)

    real_data, generated_data, generated_data_untrained = prepare_data(channel)

    real_config = """
        plot_wavelet: {}
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
        plot_amplitude_envelopes_and_alpha:
            n_samples: 6000
        plot_state_psds: {}
        plot_tde_covariances: {}
        plot_burst_summary_stats: {}
    """

    generated_config = """
        plot_wavelet: {}
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
        plot_amplitude_envelopes_and_alpha:
            n_samples: 6000
            is_generated: True
        plot_state_psds: {is_generated: True}
        plot_tde_covariances: {is_generated: True}
        plot_burst_summary_stats: {is_generated: True}
    """

    run_pipeline(
        real_config,
        output_dir=f"{results_dir}/real",
        data=real_data,
        extra_funcs=[
            plot_wavelet,
            plot_amplitude_envelopes_and_alpha,
            plot_state_psds,
            plot_tde_covariances,
            plot_burst_summary_stats,
        ],
    )
    run_pipeline(
        generated_config,
        output_dir=f"{results_dir}/generated",
        data=generated_data,
        extra_funcs=[
            plot_wavelet,
            plot_amplitude_envelopes_and_alpha,
            plot_state_psds,
            plot_tde_covariances,
            plot_burst_summary_stats,
        ],
    )
    run_pipeline(
        generated_config,
        output_dir=f"{results_dir}/generated_untrained",
        data=generated_data_untrained,
        extra_funcs=[
            plot_wavelet,
            plot_amplitude_envelopes_and_alpha,
            plot_state_psds,
            plot_tde_covariances,
            plot_burst_summary_stats,
        ],
    )
