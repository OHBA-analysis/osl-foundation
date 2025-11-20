import os
import pickle
from glob import glob
from typing import List, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from osl_dynamics.data import Data, processing
from osl_dynamics.analysis import static
from osl_dynamics.utils import plotting as osld_plotting


def get_demographics() -> pd.DataFrame:
    """Get the demographics.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the demographics of the participants.
    """

    df = pd.DataFrame(columns=["data_path", "participant_id"])
    df["data_path"] = sorted(
        glob("/well/woolrich/projects/camcan/winter23/src/sub*/sflip_parc.npy")
    )
    df["participant_id"] = df["data_path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    demo_df = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")
    df = df.merge(demo_df, on="participant_id")

    categories = [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60-66",
        "66-72",
        "72-78",
        "78-84",
        "84+",
    ]
    # Bin the age
    df["age_range"] = pd.cut(
        df["age"],
        bins=[18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 100],
        labels=categories,
        right=False,
    )
    df["age_range"] = pd.Categorical(
        df["age_range"],
        categories=categories,
        ordered=True,
    )
    return df


def get_psd() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the power spectral density (PSD) of real and generated data.

    Returns
    -------
    f : np.ndarray
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    psd_real : np.ndarray
        Power spectral density of the real data.
        Shape is (n_sessions, n_channels, n_frequencies).
    psd_gen : np.ndarray
        Power spectral density of the generated data from MEG-GPT.
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

    f, psd_real = static.welch_spectra(
        data=real_data,
        sampling_frequency=250,
        frequency_range=[1, 45],
    )
    _, psd_gen = static.welch_spectra(
        data=generated_data,
        sampling_frequency=250,
        frequency_range=[1, 45],
    )

    return f, psd_real, psd_gen


def plot_age_effect(plot_dir: str) -> None:
    """Plot the age effect on the power spectral density (PSD) of real and generated data.

    Parameters
    ----------
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    f, psd_real, psd_gen = get_psd()
    df = get_demographics()
    categories = df["age_range"].cat.categories

    psd_real_mean = psd_real.mean(axis=1)
    psd_gen_mean = psd_gen.mean(axis=1)

    psd_real_dict = {}
    psd_gen_dict = {}
    for cat in categories:
        psd_real_dict[cat] = np.mean(psd_real_mean[df["age_range"] == cat], axis=0)
        psd_gen_dict[cat] = np.mean(psd_gen_mean[df["age_range"] == cat], axis=0)

    sb_dict = {"age_range": [], "frequency": [], "psd_real": [], "psd_gen": []}
    for cat in categories:
        sb_dict["age_range"].extend([cat] * len(f))
        sb_dict["frequency"].extend(f)
        sb_dict["psd_real"].extend(psd_real_dict[cat])
        sb_dict["psd_gen"].extend(psd_gen_dict[cat])

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
    sns.lineplot(
        data=sb_dict,
        x="frequency",
        y="psd_real",
        hue="age_range",
        palette="coolwarm",
        legend=False,
        ax=axes[0],
    )
    sns.lineplot(
        data=sb_dict,
        x="frequency",
        y="psd_gen",
        hue="age_range",
        palette="coolwarm",
        legend="full",
        ax=axes[1],
    )
    axes[0].set_ylabel("")
    axes[0].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/age_effect.png")
    plt.close(fig)


def get_features(
    feature_type: str,
    save_dir: str,
    load: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the features for real and generated data.

    Parameters
    ----------
    feature_type : str
        Type of features to extract. Options are:
        - "tde": Time-delay embedding features.
        - "spectral": Spectral features.
        - "spatial": Spatial features
        - "spatial_spectral": Spatial spectral features.
    save_dir : str
        Directory to save the features.
    load : bool, optional
        Whether to load the features from the saved files.
        If False, the features will be computed and saved.

    Returns
    -------
    real_features : np.ndarray
        Features extracted from the real data.
        Shape is (n_sessions, n_features).
    generated_features : np.ndarray
        Features extracted from the generated data.
        Shape is (n_sessions, n_features).
    """

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
        generated_data_dir = "../../data/generated_data"
        with open(f"{generated_data_dir}/meg-gpt.pkl", "rb") as f:
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
    real_features: np.ndarray,
    generated_features: np.ndarray,
    metrics: Union[str, List[str]] = "correlation",
    load: bool = False,
    save_dir: str = "results/subject_variability",
) -> List[np.ndarray]:
    """
    Get the pairwise distance between real and generated features.

    Parameters
    ----------
    real_features : np.ndarray
        Features extracted from the real data.
        Shape is (n_sessions, n_features).
    generated_features : np.ndarray
        Features extracted from the generated data.
        Shape is (n_sessions, n_features).
    metrics : Union[str, List[str]], optional
        Metric(s) to use for calculating the pairwise distance.
        Can be a single metric or a list of metrics.
    load : bool, optional
        Whether to load the pairwise distance from the saved files.
        If False, the pairwise distance will be computed and saved.
    save_dir : str, optional
        Directory to save the pairwise distance files.

    Returns
    -------
    pdist_list : List[np.ndarray]
        List of pairwise distance matrices for each metric.
        Each matrix has shape (n_sessions * 2, n_sessions * 2).
    """
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


def plot_pairwise_distance(
    metrics: Union[str, List[str]],
    save_dir: str,
    plot_dir: str,
    cmap: str = "coolwarm",
    subjects_to_plot: int = 300,
    clip_percentile: float = 1.0,
) -> None:
    """
    Plot the pairwise distance between real and generated features.

    Parameters
    ----------
    metrics : Union[str, List[str]]
        Metric(s) to use for calculating the pairwise distance.
    save_dir : str
        Directory where the pairwise distance files are saved.
    plot_dir : str
        Directory to save the plots.
    cmap : str, optional
        Colormap to use for the heatmaps. Default is "coolwarm".
    subjects_to_plot : int, optional
        Number of subjects to plot in the heatmaps.
        Default is 300.
    clip_percentile : float, optional
        Percentile to clip the pairwise distance values.
        Values above this percentile will be clipped.
        Default is 1.0 (no clipping).
    """
    os.makedirs(plot_dir, exist_ok=True)

    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")

        # Clip the values to the 99th percentile
        pdist_ = np.clip(
            pdist_,
            a_min=None,
            a_max=np.quantile(pdist_, clip_percentile),
        )

        n_subjects = pdist_.shape[0] // 2

        p_dist_real = pdist_[:subjects_to_plot, :subjects_to_plot]
        p_dist_gen = pdist_[
            n_subjects : n_subjects + subjects_to_plot,
            n_subjects : n_subjects + subjects_to_plot,
        ]
        p_dist_combined = pdist_[
            :subjects_to_plot, n_subjects : n_subjects + subjects_to_plot
        ]
        vmax = np.max([p_dist_real, p_dist_gen, p_dist_combined])
        vmin = np.min([p_dist_real, p_dist_gen, p_dist_combined])

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(30, 10))
        gs = GridSpec(
            1, 4, width_ratios=[1, 1, 1, 0.05]
        )  # Allocate space for the colorbar

        # First heatmap
        ax0 = fig.add_subplot(gs[0])
        sns.heatmap(
            p_dist_real,
            cmap=cmap,
            cbar=False,
            ax=ax0,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        ax0.set_title("Real Data vs Real Data", fontsize=25, weight="bold")
        ax0.set_xlabel("Real Data Subjects", fontsize=20, weight="bold")
        ax0.set_ylabel("Real Data Subjects", fontsize=20, weight="bold")

        # Second heatmap
        ax1 = fig.add_subplot(gs[1])
        sns.heatmap(
            p_dist_gen,
            cmap=cmap,
            cbar=False,
            ax=ax1,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        ax1.set_title("MEG-GPT vs MEG-GPT", fontsize=25, weight="bold")
        ax1.set_xlabel("MEG-GPT Subjects", fontsize=20, weight="bold")
        ax1.set_ylabel("MEG-GPT Subjects", fontsize=20, weight="bold")

        # Third heatmap
        ax2 = fig.add_subplot(gs[2])
        sns.heatmap(
            p_dist_combined,
            cmap=cmap,
            cbar=True,
            cbar_ax=fig.add_subplot(gs[3]),  # Add colorbar to the last column
            ax=ax2,
            vmin=vmin,
            vmax=vmax,
            square=True,
        )
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        ax2.set_title("Real Data vs MEG-GPT", fontsize=25, weight="bold")
        ax2.set_xlabel("MEG-GPT Subjects", fontsize=20, weight="bold")
        ax2.set_ylabel("Real Data Subjects", fontsize=20, weight="bold")

        ax0.tick_params(labelsize=15)
        ax1.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/{metric}_pairwise_distance.png")
        plt.close(fig)


def get_accuracy(mat: np.ndarray, top_k: int = 1) -> float:
    """
    Calculate the accuracy of the subject fingerprints.
    The accuracy is defined as the proportion of subjects whose fingerprint
    is within the top k most similar fingerprints.

    Parameters
    ----------
    mat : np.ndarray
        Pairwise distance matrix of shape (n_subjects * 2, n_subjects * 2).
    top_k : int, optional
        Top k value for accuracy.

    Returns
    -------
    accuracy : float
        Accuracy of the classification.
    """
    n_subjects = mat.shape[0] // 2

    # Only get the top right block
    mat = mat[:n_subjects, n_subjects:]

    count = 0
    for i in range(n_subjects):
        sorted_column = np.sort(mat[:, i])
        if mat[i, i] <= sorted_column[top_k - 1]:
            count += 1
    return count / n_subjects


def plot_accuracy_curve(
    metrics: Union[str, List[str]], save_dir: str, plot_dir: str
) -> None:
    """
    Plot the accuracy curve for the subject fingerprints.

    Parameters
    ----------
    metrics : Union[str, List[str]]
        Metric(s) to use for calculating the pairwise distance.
    save_dir : str
        Directory where the pairwise distance files are saved.
    plot_dir : str
        Directory to save the plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        pdist_ = np.load(f"{save_dir}/{metric}_pdist.npy")
        n_subjects = pdist_.shape[0] // 2
        accuracies = [get_accuracy(pdist_, k) for k in range(1, n_subjects + 1)]
        fig, ax = plt.subplots(figsize=(8, 4))
        osld_plotting.plot_line(
            [range(1, n_subjects + 1), range(1, n_subjects + 1)],
            [np.arange(1, n_subjects + 1) / n_subjects, accuracies],
            labels=["Random", "MEG-GPT"],
            ax=ax,
        )
        fig.savefig(f"{plot_dir}/{metric}_accuracy_curve.png")
        plt.close(fig)


def get_consistency_score(mat: np.ndarray) -> float:
    """
    Calculate the consistency score between two pairwise distance matrices.

    Parameters
    ----------
    mat : np.ndarray
        Pairwise distance matrix of shape (n_subjects * 2, n_subjects * 2).

    Returns
    -------
    consistency : float
        Consistency score.
    """
    n_subjects = mat.shape[0] // 2
    real_pdist = mat[:n_subjects, :n_subjects]
    generated_pdist = mat[n_subjects:, n_subjects:]

    m, n = np.tril_indices(n_subjects, k=-1)
    real_pdist_flat = real_pdist[m, n]
    generated_pdist_flat = generated_pdist[m, n]
    consistency = np.corrcoef(real_pdist_flat, generated_pdist_flat)[0, 1]
    return consistency


def plot_consistency_score(
    metrics: Union[str, List[str]],
    save_dir: str,
    plot_dir: str,
    n_permutations: int = 1000,
) -> None:
    """
    Plot the consistency score for the subject fingerprints.

    Parameters
    ----------
    metrics : Union[str, List[str]]
        Metric(s) to use for calculating the pairwise distance.
    save_dir : str
        Directory where the pairwise distance files are saved.
    plot_dir : str
        Directory to save the plots.
    n_permutations : int, optional
        Number of permutations to perform for the null distribution.
    """
    os.makedirs(plot_dir, exist_ok=True)

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

        fig, ax = plt.subplots(figsize=(8, 4))
        osld_plotting.plot_hist(
            [null],
            [50],
            labels=["Null Distribution"],
            ax=ax,
        )
        ax.axvline(consistency, color="red", linestyle="--", label="Observed")
        ax.set_xlim(
            -np.abs(consistency) - 0.05,
            np.abs(consistency) + 0.05,
        )
        fig.savefig(f"{plot_dir}/{metric}_consistency_score.png")
        plt.close(fig)


def get_summary_table(
    feature_types: Union[str, List[str]],
    metric: str,
    save_dir: str,
) -> pd.DataFrame:
    if not isinstance(feature_types, list):
        feature_types = [feature_types]

    df = pd.DataFrame(
        columns=[
            "feature_type",
            "top_1_accuracy",
            "top_5_accuracy",
            "consistency_score",
        ]
    )
    for feature_type in feature_types:
        pdist_ = np.load(f"{save_dir}/{feature_type}/{metric}_pdist.npy")
        top_1_accuracy = get_accuracy(pdist_, top_k=1)
        top_5_accuracy = get_accuracy(pdist_, top_k=5)
        consistency_score = get_consistency_score(pdist_)

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "feature_type": feature_type,
                        "top_1_accuracy": top_1_accuracy,
                        "top_5_accuracy": top_5_accuracy,
                        "consistency_score": consistency_score,
                    },
                ),
            ]
        )

    df = df.reset_index(drop=True)
    df.to_csv(f"{save_dir}/summary_table.csv", index=False)
    return df


if __name__ == "__main__":
    save_dir = "../../results/subject_fingerprints"
    plot_dir = "../../plots/3_subject_fingerprints"
    os.makedirs(save_dir, exist_ok=True)

    plot_age_effect(plot_dir)

    load = False
    feature_types = [
        "spatial",
        "spectral",
        "spatial_spectral",
        "tde",
    ]
    metrics = ["correlation"]

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
            plot_dir=f"{plot_dir}/{feature_type}",
            cmap="viridis",
            subjects_to_plot=200,
            clip_percentile=0.99,
        )

        plot_accuracy_curve(
            metrics,
            save_dir=f"{save_dir}/{feature_type}",
            plot_dir=f"{plot_dir}/{feature_type}",
        )

        # Get the consistency score
        plot_consistency_score(
            metrics,
            save_dir=f"{save_dir}/{feature_type}",
            plot_dir=f"{plot_dir}/{feature_type}",
            n_permutations=1000,
        )

    # Get the summary table
    summary_df = get_summary_table(
        feature_types=feature_types,
        metric="correlation",
        save_dir=save_dir,
    )
    print(summary_df)
