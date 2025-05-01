import os
from glob import glob

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


from osl_dynamics.data import Data, processing
from osl_dynamics.analysis import static
from osl_dynamics.utils import plotting as osld_plotting


def get_features(
    load=False,
    feature_type="tde",
    save_dir="results/subject_variability",
):
    def _tde_features(x):
        tde_dimension = 21
        n_channels = x[0].shape[1]
        tde_x = [processing.time_embed(d, tde_dimension) for d in x]
        tde_cov = static.functional_connectivity(tde_x, conn_type="cov")
        m, n = np.tril_indices(tde_dimension, k=-1)

        features = []
        for i in range(len(tde_cov)):
            feature = []
            for j in range(n_channels):
                block = tde_cov[i][j * tde_dimension : (j + 1) * tde_dimension][
                    :, j * tde_dimension : (j + 1) * tde_dimension
                ]
                feature.extend(block[m, n])
            features.append(np.array(feature))
        features = np.array(features)
        return features

    def _spectral_features(x):
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=250,
            n_jobs=12,
            frequency_range=[1, 45],
        )
        return psd.mean(axis=1)

    def _spatial_features(x):
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=250,
            n_jobs=12,
            frequency_range=[1, 45],
        )
        return psd.mean(axis=2)

    def _spatial_spectral_features(x):
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=250,
            n_jobs=12,
            frequency_range=[1, 45],
        )
        return psd.reshape((psd.shape[0], -1))

    FEATURES = {
        "tde": _tde_features,
        "spectral": _spectral_features,
        "spatial": _spatial_features,
        "spatial_spectral": _spatial_spectral_features,
    }
    if not load:
        os.makedirs(save_dir, exist_ok=True)

        # Load original data
        data_dir = "/well/woolrich/projects/camcan/spring23/src"
        data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
        real_data = Data(
            data_files,
            picks="misc",
            reject_by_annotation="omit",
            n_jobs=12,
            sampling_frequency=250,
        )
        real_data.standardize()
        real_data = real_data.time_series()

        # Load generated data
        with open("results/generated_data.pkl", "rb") as f:
            generated_data = pickle.load(f)

        # Trim the original data to match the length of the generated data
        for i, (d1, d2) in enumerate(zip(real_data, generated_data)):
            real_data[i] = d1[: d2.shape[0]]

        real_features = FEATURES[feature_type](real_data)
        generated_features = FEATURES[feature_type](generated_data)

        # Save the covariance matrices
        np.save(f"{save_dir}/real_features.npy", real_features)
        np.save(f"{save_dir}/generated_features.npy", generated_features)
    else:
        # Load the covariance matrices
        real_features = np.load(f"{save_dir}/real_features.npy")
        generated_features = np.load(f"{save_dir}/generated_features.npy")

    return real_features, generated_features


def get_pairwise_distance(
    real_features,
    generated_features,
    metrics="correlation",
    load=False,
    save_dir="results/subject_variability",
):
    if not isinstance(metrics, list):
        metrics = [metrics]

    pdist_list = []
    if not load:
        # Concatenate the flattened covariance matrices
        concatenated_features = np.concatenate(
            (real_features, generated_features), axis=0
        )

        # Calculate the pairwise distance
        for metric in metrics:
            pdist_ = squareform(pdist(concatenated_features, metric=metric))
            # Save the pairwise distance
            np.save(f"{save_dir}/{metric}_pdist.npy", pdist_)
            pdist_list.append(pdist_)
    else:
        # Load the pairwise distance
        for metric in metrics:
            pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")
            pdist_list.append(pdist_)

    return pdist_list


def plot_pairwise_distance(metrics, save_dir):
    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")

        n_subjects = pdist_.shape[0] // 2
        vmax = np.max(pdist_)
        vmin = np.min(pdist_)

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(30, 10))
        gs = GridSpec(
            1, 4, width_ratios=[1, 1, 1, 0.05]
        )  # Allocate space for the colorbar

        # First heatmap
        ax0 = fig.add_subplot(gs[0])
        sns.heatmap(
            pdist_[:n_subjects, :n_subjects],
            cmap="coolwarm",
            cbar=False,
            ax=ax0,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        ax0.set_title("Original vs Original")
        ax0.set_xlabel("Subjects")
        ax0.set_ylabel("Subjects")

        # Second heatmap
        ax1 = fig.add_subplot(gs[1])
        sns.heatmap(
            pdist_[n_subjects:, n_subjects:],
            cmap="coolwarm",
            cbar=False,
            ax=ax1,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        ax1.set_title("Generated vs Generated")
        ax1.set_xlabel("Subjects")
        ax1.set_ylabel("Subjects")

        # Third heatmap
        ax2 = fig.add_subplot(gs[2])
        sns.heatmap(
            pdist_[:n_subjects, n_subjects:],
            cmap="coolwarm",
            cbar=True,
            cbar_ax=fig.add_subplot(gs[3]),  # Add colorbar to the last column
            ax=ax2,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        ax2.set_title("Original vs Generated")
        ax2.set_xlabel("Generated Subjects")
        ax2.set_ylabel("Original Subjects")

        fig.suptitle(f"Pairwise Distance ({metric})", fontsize=16)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/{metric}_pdist.png")
        plt.close(fig)


def get_accuracy(mat, top_k=1):
    n_subjects = mat.shape[0] // 2

    # Only get the top right block
    mat = mat[:n_subjects, n_subjects:]

    count = 0
    for i in range(n_subjects):
        sorted_column = np.sort(mat[:, i])
        if mat[i, i] <= sorted_column[top_k - 1]:
            count += 1
    return count / n_subjects


def plot_accuracy_curve(metrics, save_dir):
    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")
        n_subjects = pdist_.shape[0] // 2
        accuracies = [get_accuracy(pdist_, k) for k in range(1, n_subjects + 1)]
        osld_plotting.plot_line(
            [range(1, n_subjects + 1), range(1, n_subjects + 1)],
            [np.arange(1, n_subjects + 1) / n_subjects, accuracies],
            labels=["Random", "Ephys-GPT"],
            x_label="Top K",
            y_label="Accuracy",
            title=f"Accuracy Curve ({metric})",
            filename=f"{save_dir}/{metric}_accuracy_curve.png",
        )


def plot_consistency_score(metrics, save_dir, n_permutations=1000):
    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")
        n_subjects = pdist_.shape[0] // 2

        original_pdist = pdist_[:n_subjects, :n_subjects]
        generated_pdist = pdist_[n_subjects:, n_subjects:]

        m, n = np.tril_indices(n_subjects, k=-1)
        original_pdist_flat = original_pdist[m, n]
        generated_pdist_flat = generated_pdist[m, n]

        consistency = np.corrcoef(original_pdist_flat, generated_pdist_flat)[0, 1]

        # permutation test
        null = []
        for _ in range(n_permutations):
            permuted_pdist = np.random.permutation(original_pdist_flat)
            null.append(np.corrcoef(permuted_pdist, generated_pdist_flat)[0, 1])
        null = np.array(null)
        p_value = 1 - np.mean(consistency > null)

        fig, ax = osld_plotting.plot_hist(
            [null],
            [50],
            labels=["Null Distribution"],
        )
        ax.axvline(consistency, color="red", linestyle="--", label="Observed")
        ax.set_xlim(
            -np.abs(consistency) - 0.05,
            np.abs(consistency) + 0.05,
        )
        ax.set_title(
            f"{metric} Consistency Score: {consistency:.3f} (p-value: {p_value:.3f})"
        )
        fig.tight_layout()
        fig.savefig(f"{save_dir}/{metric}_consistency_score.png")
        plt.close(fig)


if __name__ == "__main__":
    save_dir = "results/plots/subject_variability"
    os.makedirs(save_dir, exist_ok=True)

    load = False
    # feature_types = ["tde", "spectral", "spatial", "spatial_spectral"]
    feature_types = ["tde"]
    metrics = ["correlation", "cosine", "euclidean"]

    # Get the covariance matrices
    for feature_type in feature_types:
        real_features, generated_features = get_features(
            load=load,
            feature_type=feature_type,
            save_dir=f"{save_dir}/{feature_type}",
        )

        # Get the pairwise distance
        pdist_list = get_pairwise_distance(
            real_features,
            generated_features,
            metrics=metrics,
            load=load,
            save_dir=f"{save_dir}/{feature_type}",
        )
        # Plot the pairwise distance
        plot_pairwise_distance(
            metrics,
            save_dir=f"{save_dir}/{feature_type}",
        )

        # Plot the accuracy curve
        plot_accuracy_curve(
            metrics,
            save_dir=f"{save_dir}/{feature_type}",
        )

        # Get the consistency score
        plot_consistency_score(
            metrics, save_dir=f"{save_dir}/{feature_type}", n_permutations=1000
        )
