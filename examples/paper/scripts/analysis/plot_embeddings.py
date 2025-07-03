import os
from glob import glob
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import nibabel as nib
import nilearn.plotting as nip
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.analysis import power

from osl_foundation import load_model

tf_ops.gpu_growth()


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


def get_mapped_embeddings(
    train: bool, ephys_gpt_dir: str, save_dir: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Get the mapped embeddings.

    Parameters
    ----------
    train : bool
        Whether to train the embeddings or load them.
    ephys_gpt_dir : str
        Directory containing the Ephys-GPT model.
    save_dir : str
        Directory to save the embeddings.

    Returns
    -------
    pca_embeddings : Dict[str, np.ndarray]
        Dictionary containing the PCA embeddings.
    tsne_embeddings : Dict[str, np.ndarray]
        Dictionary containing the t-SNE embeddings.
    umap_embeddings : Dict[str, np.ndarray]
        Dictionary containing the UMAP embeddings.
    """
    ephys_gpt = load_model(ephys_gpt_dir, checkpoint="latest")
    embeddings = ephys_gpt.get_embeddings()

    standardize = lambda x: (x - x.mean()) / x.std()
    for k, v in embeddings.items():
        embeddings[k] = standardize(v)

    if train:
        pca_embeddings, tsne_embeddings, umap_embeddings = {}, {}, {}
        for k, v in embeddings.items():
            pca_embeddings[k] = PCA(n_components=2).fit_transform(v)
            tsne_embeddings[k] = TSNE(n_components=2).fit_transform(v)
            umap_embeddings[k] = umap.UMAP(n_components=2).fit_transform(v)

        with open(f"{save_dir}/pca_embeddings.pkl", "wb") as f:
            pickle.dump(pca_embeddings, f)
        with open(f"{save_dir}/tsne_embeddings.pkl", "wb") as f:
            pickle.dump(tsne_embeddings, f)
        with open(f"{save_dir}/umap_embeddings.pkl", "wb") as f:
            pickle.dump(umap_embeddings, f)
    else:
        with open(f"{save_dir}/pca_embeddings.pkl", "rb") as f:
            pca_embeddings = pickle.load(f)
        with open(f"{save_dir}/tsne_embeddings.pkl", "rb") as f:
            tsne_embeddings = pickle.load(f)
        with open(f"{save_dir}/umap_embeddings.pkl", "rb") as f:
            umap_embeddings = pickle.load(f)

    return pca_embeddings, tsne_embeddings, umap_embeddings


def plot_embeddings(
    pca_embeddings: Dict[str, np.ndarray],
    tsne_embeddings: Dict[str, np.ndarray],
    umap_embeddings: Dict[str, np.ndarray],
    df: pd.DataFrame,
    plot_dir: str,
) -> None:
    """
    Plot the embeddings using PCA, t-SNE, and UMAP.

    Parameters
    ----------
    pca_embeddings : Dict[str, np.ndarray]
        Dictionary containing the PCA embeddings.
    tsne_embeddings : Dict[str, np.ndarray]
        Dictionary containing the t-SNE embeddings.
    umap_embeddings : Dict[str, np.ndarray]
        Dictionary containing the UMAP embeddings.
    df : pd.DataFrame
        DataFrame containing the demographics of the participants.
    plot_dir : str
        Directory to save the plots.
    """
    visual_ind = [0, 1, 2, 3, 26, 27, 28, 29]
    motor_ind = [4, 5, 6, 7, 8, 30, 31, 32, 33, 34]
    parietal_ind = [15, 16, 17, 18, 19, 41, 42, 43, 44, 45]
    frontal_ind = [22, 23, 24, 25, 48, 49, 50, 51]
    temporal_ind = [10, 11, 12, 13, 14, 36, 37, 38, 39, 40]
    insular_ind = [9, 35]
    pcc_ind = [20, 46]
    acc_ind = [21, 47]

    channel_colors = sns.color_palette("tab10", 8)

    def get_hue(k, v):
        if k == "session_id":
            return df["age_range"]
        if k == "channel":
            hue = np.array([""] * v.shape[0], dtype=object)
            hue[visual_ind] = "Visual"
            hue[motor_ind] = "Motor"
            hue[parietal_ind] = "Parietal"
            hue[frontal_ind] = "Frontal"
            hue[temporal_ind] = "Temporal"
            hue[insular_ind] = "Insular"
            hue[pcc_ind] = "PCC"
            hue[acc_ind] = "ACC"
            return hue
        return range(v.shape[0])

    def get_legend_title(key):
        if key == "session_id":
            return "Age range"
        if key == "channel":
            return "Lobe"
        if key == "position":
            return "position"
        return "Token"

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    for i, (k, v) in enumerate(pca_embeddings.items()):
        sns.scatterplot(
            x=v[:, 0],
            y=v[:, 1],
            hue=get_hue(k, v),
            palette="coolwarm" if k != "channel" else channel_colors,
            ax=axes[i, 0],
        )
    for i, (k, v) in enumerate(tsne_embeddings.items()):
        sns.scatterplot(
            x=v[:, 0],
            y=v[:, 1],
            hue=get_hue(k, v),
            palette="coolwarm" if k != "channel" else channel_colors,
            ax=axes[i, 1],
            legend=False,  # No legend for this column
        )
    for i, (k, v) in enumerate(umap_embeddings.items()):
        sns.scatterplot(
            x=v[:, 0],
            y=v[:, 1],
            hue=get_hue(k, v),
            palette="coolwarm" if k != "channel" else channel_colors,
            ax=axes[i, 2],
            legend=False,  # No legend for this column
        )

    for i, k in enumerate(pca_embeddings.keys()):
        handles, labels = axes[i, 0].get_legend_handles_labels()
        if handles:  # Only add legend if there are handles
            axes[i, 0].legend_.remove()
            legend = axes[i, 2].legend(
                handles, labels, loc="center left", bbox_to_anchor=(1, 0.5)
            )
            legend.set_title(get_legend_title(k))

    axes[0, 0].set_title("PCA")
    axes[0, 1].set_title("t-SNE")
    axes[0, 2].set_title("UMAP")

    axes[0, 0].set_ylabel("Token")
    axes[1, 0].set_ylabel("Position")
    axes[2, 0].set_ylabel("Channel")
    axes[3, 0].set_ylabel("Session")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/embeddings.png")
    plt.close(fig)


def plot_regions_power_map(plot_dir: str) -> None:
    """
    Plot the power map of the regions in the Glasser 52 parcellation.

    Parameters
    ----------
    plot_dir : str
        Directory to save the plot.
    """
    visual_ind = [0, 1, 2, 3, 26, 27, 28, 29]
    motor_ind = [4, 5, 6, 7, 8, 30, 31, 32, 33, 34]
    parietal_ind = [15, 16, 17, 18, 19, 41, 42, 43, 44, 45]
    frontal_ind = [22, 23, 24, 25, 48, 49, 50, 51]
    temporal_ind = [10, 11, 12, 13, 14, 36, 37, 38, 39, 40]
    insular_ind = [9, 35]
    pcc_ind = [20, 46]
    acc_ind = [21, 47]

    power_map = np.empty(52)
    power_map[visual_ind] = 0
    power_map[motor_ind] = 1
    power_map[insular_ind] = 2
    power_map[temporal_ind] = 3
    power_map[parietal_ind] = 4
    power_map[pcc_ind] = 5
    power_map[acc_ind] = 6
    power_map[frontal_ind] = 7

    power.save(
        power_map,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "symmetric_cbar": False,
            "cmap": ListedColormap(sns.color_palette("tab10", 8)),
        },
        filename=f"{plot_dir}/regions_power_map.png",
    )


def plot_regions_map(plot_dir: str) -> None:
    """
    Plot the regions map of the Glasser 52 parcellation.

    Parameters
    ----------
    plot_dir : str
        Directory to save the plot.
    """
    visual_ind = [0, 1, 2, 3, 26, 27, 28, 29]
    motor_ind = [4, 5, 6, 7, 8, 30, 31, 32, 33, 34]
    insular_ind = [9, 35]
    temporal_ind = [10, 11, 12, 13, 14, 36, 37, 38, 39, 40]
    parietal_ind = [15, 16, 17, 18, 19, 41, 42, 43, 44, 45]
    pcc_ind = [20, 46]
    acc_ind = [21, 47]
    prefrontal_ind = [22, 23, 24, 25, 48, 49, 50, 51]
    overall_list = [
        visual_ind,
        motor_ind,
        insular_ind,
        temporal_ind,
        parietal_ind,
        pcc_ind,
        acc_ind,
        prefrontal_ind,
    ]
    mapping = {}
    for i, l in enumerate(overall_list):
        for j in l:
            mapping[j] = i + 1

    parcellation = nib.load(
        "/well/woolrich/users/tjo747/osl-dynamics/osl_dynamics/files/parcellation/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"
    )
    parcellation_data = parcellation.get_fdata()
    mask = np.zeros(parcellation_data.shape[:-1])
    for i in range(52):
        inds = np.where(parcellation_data[:, :, :, i] == 1)
        mask[inds] = mapping[i]
    img = nib.Nifti1Image(mask, parcellation.affine)
    new_cmap = ListedColormap(sns.color_palette("tab10", 8))
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([0.025, 0.05, 0.8, 0.9])
    ax2 = fig.add_axes([0.875, 0.05, 0.1, 0.9])
    nip.plot_roi(img, cmap=new_cmap, display_mode="mosaic", axes=ax1)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # one more than number of color bins
    norm = BoundaryNorm(bounds, new_cmap.N)
    cb = ColorbarBase(
        ax2,
        cmap=new_cmap,
        norm=norm,
        boundaries=bounds,
        ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        orientation="vertical",
    )
    labels = [
        "Visual",
        "Motor",
        "Insular",
        "Temporal",
        "Parietal",
        "PCC",
        "ACC",
        "Prefrontal",
    ]
    cb.ax.set_yticklabels(labels)

    plt.savefig(f"{plot_dir}/regions_map.png")


if __name__ == "__main__":
    plot_dir = "results/plots/embeddings"
    os.makedirs(plot_dir, exist_ok=True)
    ephys_gpt_dir = "/well/woolrich/projects/foundation_models/ephys-gpt/sequence_length_80/without_channel_attention/model"
    train = True

    df = get_demographics()
    pca_embeddings, tsne_embeddings, umap_embeddings = get_mapped_embeddings(
        train, ephys_gpt_dir, "results"
    )
    plot_embeddings(pca_embeddings, tsne_embeddings, umap_embeddings, df, plot_dir)
    plot_regions_power_map(plot_dir)
    plot_regions_map(plot_dir)
