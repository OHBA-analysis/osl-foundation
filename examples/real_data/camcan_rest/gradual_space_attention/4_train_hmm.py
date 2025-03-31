from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics import run_pipeline

import pickle

tf_ops.gpu_growth()

data_dir = "models/generator"
with open(f"{data_dir}/generated_data.pkl", "rb") as f:
    data = pickle.load(f)

data = Data(
    data,
    n_jobs=8,
    sampling_frequency=250,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    use_tfrecord=True,
)

methods = {
    "filter": {"low_freq": 1, "high_freq": 45, "use_raw": True},
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
    "standardize": {},
}
data.prepare(methods)

config = """
    train_hmm:
        config_kwargs:
            sequence_length: 200
            n_states: 6
            learn_means: False
            learn_covariances: True
            batch_size: 32
            n_epochs: 20
            learning_rate: 0.005
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 16
        nnmf_components: 2
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        mask_file: MNI152_T1_8mm_brain.nii.gz
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
            combined: True
        conn_save_kwargs:
            combined: True
    plot_hmm_network_summary_stats: {}
"""

run_pipeline(
    config,
    output_dir="hmm_results",
    data=data,
)
