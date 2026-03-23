import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm.auto import trange

# python /Users/woolrich/dev/projects/osl-foundation/examples/simulation/simple_simulation/new_space_attention/3_plot_results.py 

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.analysis import static
from osl_dynamics.utils import plotting as osld_plotting

from osl_foundation.utils import plotting
from osl_foundation.utils.plot_results_common import (
    load_pickled_series,
    load_generator_checkpoint_first,
    load_and_standardize_original_data,
)

tf_ops.gpu_growth()

DISABLE_LYAPUNOV = True

sampling_frequency = 100
welch_n_jobs = 1  # Avoid multiprocessing spawn issues in top-level script execution.
analysis_window_samples = 6000  # Match time_frequency.png and summary*.png windows.
diagnostic_window_samples = 1000  # Shorter window for readable Lyapunov diagnostic panels.
diagnostic_max_points = 1200  # Downsample dense traces for plotting only.

# ---------- Directories ---------- #

data_dir = f"{dev_dir}/results/osl-foundation/simple_simulation"
generator_dir = f"{dev_dir}/results/osl-foundation/simple_simulation/generator"
plot_dir = f"{data_dir}/plots/"

os.makedirs(plot_dir, exist_ok=True)

# ---------- Load data ---------- #

generated_data = load_pickled_series(generator_dir, "generated_data")
if DISABLE_LYAPUNOV:
    print("DISABLE_LYAPUNOV=True: skipping Lyapunov and spectral diagnostics plots.")
    lyapunov_data = []
    lyapunov_diag_data = []
else:
    lyapunov_data = load_pickled_series(generator_dir, "lyapunov_data")
    lyapunov_diag_data = load_pickled_series(generator_dir, "lyapunov_diagnostics")

def _first_diag_setting(diag_data, key, default=None):
    values = [
        diag.get(key)
        for diag in diag_data
        if isinstance(diag, dict) and key in diag
    ]
    values = [value for value in values if value is not None]
    if not values:
        return default
    first = values[0]
    if any(value != first for value in values[1:]):
        print(f"Warning: {key} differs across sessions; using first value: {first}.")
    return first

if not DISABLE_LYAPUNOV and lyapunov_diag_data:
    sampling_frequency = int(
        _first_diag_setting(lyapunov_diag_data, "sampling_rate", sampling_frequency)
    )

# ---------- Plot lyapunov data ---------- #

if not DISABLE_LYAPUNOV and lyapunov_data:
    lyapunov_data = np.array(lyapunov_data)  # shape (n_sessions, n_timepoints)
    plt.figure()
    plt.plot(lyapunov_data[0, :2000], color="gray", alpha=0.5)
    plt.xlabel("Timepoints")
    plt.ylabel("Lyapunov exponent")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/lyapunov.png")
    plt.close()

    # plot histogram of lyapunov exponents
    # print maximum lyapunov exponent
    print("Maximum lyapunov exponent: ", np.max(lyapunov_data))
    plt.figure()
    plt.hist(lyapunov_data.flatten(), bins=50, color="gray")
    plt.xlabel("Lyapunov exponent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/lyapunov_hist.png")
    plt.close() 

if not DISABLE_LYAPUNOV and lyapunov_diag_data:
    diag_plot_dir = f"{plot_dir}/lyapunov_diagnostics"
    os.makedirs(diag_plot_dir, exist_ok=True)

    final_invalid_rates = []
    initial_invalid_rates = []
    mean_attempts = []
    final_margins = []

    for i, diag in enumerate(lyapunov_diag_data):
        margin = float(diag.get("lyapunov_margin", 0.0))
        margin_used = np.asarray(diag.get("margin_used", []), dtype=np.float32)
        invalid_mask = np.asarray(diag["invalid_mask"])
        attempts_used = np.asarray(diag["attempts_used"])
        spectral_peakiness = np.asarray(diag.get("spectral_peakiness", []))
        spectral_penalty_value = np.asarray(diag.get("spectral_penalty_value", []))
        spectral_penalty_weight = np.asarray(diag.get("spectral_penalty_weight", []))
        spectral_penalty_active = np.asarray(diag.get("spectral_penalty_active", []))
        spectral_settings = diag.get("spectral_penalty_settings", {}) or {}
        spectral_penalty_threshold = diag.get(
            "spectral_penalty_threshold",
            spectral_settings.get("spectral_peakiness_threshold", None),
        )
        spectral_penalty_cap = diag.get(
            "spectral_penalty_cap",
            spectral_settings.get("spectral_penalty_cap", None),
        )
        spectral_penalty_normalize = diag.get(
            "spectral_penalty_normalize",
            spectral_settings.get("spectral_penalty_normalize", None),
        )
        spectral_penalty_calibrate_steps = diag.get(
            "spectral_penalty_calibrate_steps",
            spectral_settings.get("spectral_penalty_calibrate_steps", None),
        )
        spectral_min_freq = spectral_settings.get("spectral_min_freq", None)
        spectral_max_freq = spectral_settings.get("spectral_max_freq", None)
        spectral_penalty_weight_setting = spectral_settings.get(
            "spectral_penalty_weight", None
        )
        burst_window_s = spectral_settings.get("burst_window_s", None)
        max_burst_s = spectral_settings.get("max_burst_s", None)
        weight_used_value = None
        if spectral_penalty_weight.size > 0:
            weight_valid = spectral_penalty_weight[
                np.isfinite(spectral_penalty_weight)
            ]
            if weight_valid.size > 0:
                weight_used_value = float(weight_valid[-1])
        window = min(
            diagnostic_window_samples,
            lyapunov_data.shape[1],
            attempts_used.shape[0],
            invalid_mask.shape[0],
            margin_used.shape[0] if margin_used.size > 0 else diagnostic_window_samples,
        )
        plot_stride = max(1, window // diagnostic_max_points)
        plot_idx = np.arange(0, window, plot_stride)

        final_invalid_rates.append(float(diag["final_invalid_rate"]))
        initial_invalid_rates.append(float(diag["initial_invalid_rate"]))
        mean_attempts.append(float(diag["mean_attempts"]))
        final_margin_value = diag.get(
            "final_margin_per_session", diag.get("final_margin", margin)
        )
        final_margin_arr = np.asarray(final_margin_value, dtype=np.float32).reshape(-1)
        if final_margin_arr.size == 0:
            final_margins.append(margin)
        elif final_margin_arr.size == 1:
            final_margins.append(float(final_margin_arr[0]))
        else:
            final_margins.append(float(final_margin_arr[min(i, final_margin_arr.size - 1)]))

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        axes[0].plot(
            plot_idx,
            lyapunov_data[i, plot_idx],
            color="gray",
            alpha=0.45,
            linewidth=0.8,
            label="lyapunov_fn",
        )
        axes[0].scatter(
            plot_idx,
            lyapunov_data[i, plot_idx],
            s=8,
            color="gray",
            alpha=0.25,
            label="all points",
        )
        if margin_used.size > 0:
            axes[0].plot(
                plot_idx,
                margin_used[plot_idx],
                color="red",
                linestyle="--",
                alpha=0.8,
                label="adaptive_margin",
            )
        else:
            axes[0].axhline(margin, color="red", linestyle="--", label="margin")
        if np.any(invalid_mask):
            invalid_idx = np.where(invalid_mask[:window])[0]
            invalid_plot_idx = invalid_idx[::plot_stride]
            axes[0].scatter(
                invalid_plot_idx,
                lyapunov_data[i, invalid_plot_idx],
                s=8,
                color="red",
                alpha=0.7,
                label="invalid",
            )
        axes[0].set_ylabel("Lyapunov")
        axes[0].legend(loc="upper right")
        axes[0].set_title(
            f"Session {i}: final_invalid_rate={diag['final_invalid_rate']:.4f}, "
            f"mean_attempts={diag['mean_attempts']:.3f}, "
            f"rollout_h={diag.get('lyapunov_rollout_horizon', 1)}, "
            f"adaptive_rollout={diag.get('lyapunov_adaptive_rollout', False)} "
                f"(first {window} samples shown, stride={plot_stride})"
        )

        axes[1].plot(plot_idx, attempts_used[plot_idx], color="steelblue", linewidth=1.0)
        axes[1].set_ylabel("Attempts Used")
        axes[1].set_xlabel("Timepoints")
        axes[1].set_ylim([0.9, diag["max_attempts"] + 0.1])
        axes[1].axhline(diag["max_attempts"], color="red", linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(f"{diag_plot_dir}/session_{i}_lyapunov_diagnostics.png")
        plt.close(fig)

        if spectral_peakiness.size > 0:
            print("Plotting spectral diagnostics...")

            spec_window = min(
                diagnostic_window_samples,
                spectral_peakiness.shape[0],
                spectral_penalty_value.shape[0] if spectral_penalty_value.size > 0 else diagnostic_window_samples,
            )
            spec_stride = max(1, spec_window // diagnostic_max_points)
            spec_idx = np.arange(0, spec_window, spec_stride)

            fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
            axes[0].plot(
                spec_idx,
                spectral_peakiness[spec_idx],
                color="tab:purple",
                linewidth=1.0,
                label="peakiness",
            )
            if spectral_penalty_threshold is not None:
                axes[0].axhline(
                    spectral_penalty_threshold,
                    color="black",
                    linestyle="--",
                    alpha=0.7,
                    label="threshold",
                )
            if spectral_penalty_cap is not None and spectral_penalty_normalize:
                axes[0].axhline(
                    spectral_penalty_threshold * (1 + spectral_penalty_cap),
                    color="gray",
                    linestyle=":",
                    alpha=0.6,
                    label="cap (normalized)",
                )
            axes[0].set_ylabel("Peakiness")
            axes[0].legend(loc="upper right")
            settings_chunks = []
            if spectral_min_freq is not None or spectral_max_freq is not None:
                settings_chunks.append(
                    f"band={spectral_min_freq}-{spectral_max_freq}Hz"
                )
            if burst_window_s is not None and max_burst_s is not None:
                settings_chunks.append(
                    f"burst_window={burst_window_s}s, max_burst={max_burst_s}s"
                )
            if spectral_penalty_weight_setting is not None:
                settings_chunks.append(
                    f"weight_setting={spectral_penalty_weight_setting}"
                )
            if weight_used_value is not None:
                settings_chunks.append(f"weight_used={weight_used_value:.3g}")
            settings_suffix = (
                "; ".join(settings_chunks) if settings_chunks else "settings: n/a"
            )
            axes[0].set_title(
                f"Session {i}: spectral penalty diagnostics "
                f"(calibrate_steps={spectral_penalty_calibrate_steps})\n"
                f"{settings_suffix}"
            )

            if spectral_penalty_weight.size > 0:
                axes[1].plot(
                    spec_idx,
                    spectral_penalty_weight[spec_idx],
                    color="tab:blue",
                    linewidth=1.0,
                    label="weight used",
                )
            axes[1].set_ylabel("Weight")
            axes[1].legend(loc="upper right")

            if spectral_penalty_value.size > 0:
                axes[2].plot(
                    spec_idx,
                    spectral_penalty_value[spec_idx],
                    color="tab:red",
                    linewidth=1.0,
                    label="penalty value",
                )
            if spectral_penalty_active.size > 0:
                active_idx = np.where(spectral_penalty_active[:spec_window])[0][
                    ::spec_stride
                ]
                axes[2].scatter(
                    active_idx,
                    np.zeros_like(active_idx, dtype=float),
                    s=10,
                    color="black",
                    alpha=0.5,
                    label="gate active",
                )
            axes[2].set_ylabel("Penalty")
            axes[2].set_xlabel("Timepoints")
            axes[2].legend(loc="upper right")

            fig.tight_layout()
            fig.savefig(f"{diag_plot_dir}/session_{i}_spectral_penalty.png")
            plt.close(fig)

    x = np.arange(len(final_invalid_rates))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - width / 2, initial_invalid_rates, width=width, color="silver", label="Initial Invalid Rate")
    ax1.bar(x + width / 2, final_invalid_rates, width=width, color="dimgray", label="Final Invalid Rate")
    ax1.set_xlabel("Session")
    ax1.set_ylabel("Invalid Rate")
    ax1.set_xticks(x)
    ax1.set_title("Lyapunov Invalid Rates by Session")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{diag_plot_dir}/lyapunov_invalid_rates_by_session.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, mean_attempts, color="steelblue")
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean Attempts Used")
    ax.set_xticks(x)
    ax.set_title("Lyapunov Sampling Attempts by Session")
    fig.tight_layout()
    fig.savefig(f"{diag_plot_dir}/lyapunov_attempts_by_session.png")
    plt.close(fig)

    if len(final_margins) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x, final_margins, color="indianred")
        ax.set_xlabel("Session")
        ax.set_ylabel("Final Margin")
        ax.set_xticks(x)
        ax.set_title("Final Lyapunov Margin by Session")
        fig.tight_layout()
        fig.savefig(f"{diag_plot_dir}/lyapunov_final_margin_by_session.png")
        plt.close(fig)

# ---------- Load original data ---------- #
original_data = load_and_standardize_original_data(
    data_dir=data_dir,
    n_sessions=len(generated_data),
    n_samples_per_session=generated_data[0].shape[0],
)

# ---------- Get reconstructed data from original data using the tokenizer ---------- #

generator = load_generator_checkpoint_first(generator_dir)
tokenized_data = generator.tokenizer.tokenize_data(Data(original_data))
reconstructed_data = generator.tokenizer.reconstruct_data(tokenized_data)

if False:

    # ---------- Check ephys reconstruction works ---------- #
    refact_dat = np.expand_dims(tokenized_data[0], axis=0) # shape (1, T, C)
    y_recon = generator.reconstruct_data_from_tokens(refact_dat)  # shape (1, T, C)   
    y_recon = np.squeeze(y_recon)  # shape (T, C)

    plt.figure()
    plt.plot(y_recon[:1000, 0])
    plt.figure()
    plt.plot(original_data[0][:1000, 0])
    plt.figure()
    plt.plot(reconstructed_data[0][:1000, 0])
    plt.show()

# ---------- Plot subject level PSDs ---------- #

def _plot_psd(f, psd, name, plot_dir):

    print("Plotting PSDs...")

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
        n_jobs=welch_n_jobs,
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
    window_size=1000,
    sampling_frequency=sampling_frequency,
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/aec.png",
)

# Plot Time frequency content of original, reconstructed and generated data
n_samples = analysis_window_samples  # 60 seconds

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

# Channel 2
fig = plt.figure(figsize=(10, 10))
ax_1 = [fig.add_subplot(3, 3, i) for i in range(1, 4)]
ax_2 = [fig.add_subplot(3, 3, i) for i in range(4, 7)]
ax_3 = fig.add_subplot(3, 1, 3)

plotting.plot_time_series_summary(
    original_data[:, 2],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[0], ax_2[0], ax_3],
    color="black",
    label="Original",
)
plotting.plot_time_series_summary(
    reconstructed_data[:, 2],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[1], ax_2[1], ax_3],
    color="green",
    label="Reconstructed",
)
plotting.plot_time_series_summary(
    generated_data[:, 2],
    sampling_frequency=sampling_frequency,
    n_samples=n_samples,
    axes=[ax_1[2], ax_2[2], ax_3],
    color="red",
    label="Generated",
)
fig.suptitle("Time series summary (channel 2)")
fig.legend(
    loc="lower center",
    ncol=3,
    fontsize=14,
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True,
    shadow=True,
)
fig.tight_layout()
fig.savefig(f"{plot_dir}/summary_2.png")
plt.close(fig)

# Plot PSD of generated data
n_channels = generator.config.model_config.n_channels
fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))
for i in range(n_channels):
    axes[i].psd(generated_data[:, i], Fs=sampling_frequency, NFFT=1024, color="black")
    axes[i].set_ylabel(f"Channel {i}")
fig.tight_layout()
fig.savefig(f"{plot_dir}/psd_generated_data.png")
plt.close(fig)

# ---------- Plot history ---------- #

with open(f"{generator_dir}/history.pkl", "rb") as file_obj:
    history = pickle.load(file_obj)

plotting.plot_history(history, plot_dir=plot_dir)

if "lyapunov_loss" in history and "cross_entropy_loss" in history:
    lyap = np.asarray(history["lyapunov_loss"], dtype=float)
    ce = np.asarray(history["cross_entropy_loss"], dtype=float)
    ratio = lyap / np.maximum(ce, 1e-12)
    epochs = np.arange(1, len(ratio) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, ratio, color="tab:red", label="lyapunov/ce ratio")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Ratio")
    ax.set_title("Lyapunov Loss Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_loss_ratio.png")
    plt.close(fig)

if "lyapunov_beta" in history:
    beta = np.asarray(history["lyapunov_beta"], dtype=float)
    epochs = np.arange(1, len(beta) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, beta, color="tab:purple", label="lyapunov_beta")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Beta")
    ax.set_title("Adaptive Lyapunov Beta")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_beta.png")
    plt.close(fig)

if "lyapunov_v0_ratio" in history:
    v0_ratio = np.asarray(history["lyapunov_v0_ratio"], dtype=float)
    epochs = np.arange(1, len(v0_ratio) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, v0_ratio, color="tab:orange", label="mu*V0/core")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.set_title("Lyapunov V0-to-Core Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_v0_ratio.png")
    plt.close(fig)

if "lyapunov_mu" in history:
    mu = np.asarray(history["lyapunov_mu"], dtype=float)
    epochs = np.arange(1, len(mu) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, mu, color="tab:brown", label="lyapunov_mu")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mu")
    ax.set_title("Adaptive Lyapunov Mu")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_mu.png")
    plt.close(fig)

if (
    "lyapunov_collapse_loss" in history
    or "lyapunov_v_var" in history
    or "lyapunov_v_mean" in history
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    if "lyapunov_collapse_loss" in history:
        collapse = np.asarray(history["lyapunov_collapse_loss"], dtype=float)
        axes[0].plot(
            np.arange(1, len(collapse) + 1),
            collapse,
            color="tab:red",
            label="lyapunov_collapse_loss",
        )
        axes[0].set_yscale("log")
        axes[0].legend()
    axes[0].set_ylabel("Collapse Loss")
    axes[0].grid(alpha=0.3)

    if "lyapunov_v_var" in history:
        v_var = np.asarray(history["lyapunov_v_var"], dtype=float)
        axes[1].plot(
            np.arange(1, len(v_var) + 1),
            v_var,
            color="tab:blue",
            label="lyapunov_v_var",
        )
        axes[1].axhline(
            generator.config.model_config.lyapunov_collapse_target_var,
            color="tab:blue",
            linestyle="--",
            alpha=0.7,
            label="target_var",
        )
        axes[1].set_yscale("log")
        axes[1].legend()
    axes[1].set_ylabel("V Variance")
    axes[1].grid(alpha=0.3)

    if "lyapunov_v_mean" in history:
        v_mean = np.asarray(history["lyapunov_v_mean"], dtype=float)
        axes[2].plot(
            np.arange(1, len(v_mean) + 1),
            v_mean,
            color="tab:green",
            label="lyapunov_v_mean",
        )
        axes[2].axhline(
            generator.config.model_config.lyapunov_collapse_target_mean,
            color="tab:green",
            linestyle="--",
            alpha=0.7,
            label="target_mean",
        )
        axes[2].set_yscale("log")
        axes[2].legend()
    axes[2].set_ylabel("V Mean")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Lyapunov Collapse Diagnostics")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_collapse_diagnostics.png")
    plt.close(fig)

if "lyapunov_collapse_ratio" in history:
    ratio = np.asarray(history["lyapunov_collapse_ratio"], dtype=float)
    epochs = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, ratio, color="tab:cyan", label="collapse/core")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.set_title("Lyapunov Collapse-to-Core Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_collapse_ratio.png")
    plt.close(fig)
elif "lyapunov_collapse_loss" in history and "lyapunov_core_loss" in history:
    collapse = np.asarray(history["lyapunov_collapse_loss"], dtype=float)
    core = np.asarray(history["lyapunov_core_loss"], dtype=float)
    ratio = collapse / np.maximum(core, 1e-12)
    epochs = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, ratio, color="tab:cyan", label="collapse/core")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.set_title("Lyapunov Collapse-to-Core Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lyapunov_collapse_ratio.png")
    plt.close(fig)

# ---------- Finish ---------- #

print(plot_dir)
