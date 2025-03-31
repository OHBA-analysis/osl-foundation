import os
from glob import glob

import pickle
import numpy as np

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting

tf_ops.select_gpu(1)
tf_ops.gpu_growth()

channel_of_interest = 4

real_data_dir = "/well/woolrich/projects/camcan/spring23/src"
data_files = sorted(glob(f"{real_data_dir}/*/sflip_parc-raw.fif"))
real_data = Data(
    data_files,
    n_jobs=8,
    picks="misc",
    reject_by_annotation="omit",
)
real_data = [d[:15000, [channel_of_interest]] for d in real_data.time_series()]

generator_dir = "models/generator"
with open(f"{generator_dir}/generated_data.pkl", "rb") as f:
    generated_data = pickle.load(f)
generated_data = [d[:, [channel_of_interest]] for d in generated_data]

real_data = Data(real_data, sampling_frequency=250, n_jobs=8)
generated_data = Data(generated_data, sampling_frequency=250, n_jobs=8)

real_data.downsample(100)
generated_data.downsample(100)

real_data = Data(real_data.time_series(), sampling_frequency=100, n_jobs=8)
generated_data = Data(generated_data.time_series(), sampling_frequency=100, n_jobs=8)

methods = {
    "tde": {"n_embeddings": 21},
    "standardize": {},
}
real_data.prepare(methods)
generated_data.prepare(methods)


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


def plot_amplitude_envelopes_and_alpha(data, output_dir, n_samples):
    """Plot amplitude envelopes and inferred state probabilities."""

    inf_params_dir = f"{output_dir}/inf_params"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    order = [1, 2, 0]

    # Plot state probability time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    alp = np.concatenate(alp)[:, order]
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


config = """
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

# Run analysis
run_pipeline(
    config,
    output_dir=f"burst_results/real_data",
    data=real_data,
    extra_funcs=[plot_wavelet, plot_amplitude_envelopes_and_alpha],
)
run_pipeline(
    config,
    output_dir=f"burst_results/generated_data",
    data=generated_data,
    extra_funcs=[plot_wavelet, plot_amplitude_envelopes_and_alpha],
)
