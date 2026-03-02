import os
from glob import glob

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse

from osl_dynamics.analysis import static, power, connectivity
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting as osld_plotting

dev_dir = "/Users/woolrich/dev"
dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import load_model
from osl_foundation.utils import plotting as oslf_plotting

def plot_results(args):
    uname = args.uname

    sampling_frequency=250
    subs2plot = [6, 7]
    subs2plot = None

    # ---------- Directories ---------- #

    tokenized_data_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/tokenizer/tokenized_data"

    generator_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/generator_{uname}"

    plot_dir = f"{generator_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_dir = f"{plot_dir}/static"
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- Load generated data ---------- #
    generated_data_files = sorted(glob(f"{generator_dir}/generated_data_*.pkl"))
    generated_data = [pickle.load(open(f, "rb")) for f in generated_data_files] 

    lyapunov_data_files = sorted(glob(f"{generator_dir}/lyapunov_data_*.pkl"))
    lyapunov_data = [pickle.load(open(f, "rb")) for f in lyapunov_data_files]

    if subs2plot is None:
        subs2plot = list(range(len(generated_data)))
        
    generated_data = [generated_data[s] for s in subs2plot]
    lyapunov_data = [lyapunov_data[s] for s in subs2plot]

    # ---------- Plot lyapunov data ---------- #
    lyapunov_data = np.array(lyapunov_data)  # shape (n_sessions, n_timepoints)
    plt.figure()
    plt.plot(lyapunov_data[0, :3000], color="gray", alpha=0.5)
    plt.xlabel("Timepoints")
    plt.ylabel("Lyapunov exponent")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/lyapunov.png")
    plt.close()

    # plot histogram of lyapunov exponents
    plt.figure()
    plt.hist(lyapunov_data.flatten(), bins=100, color="gray")
    plt.xlabel("Lyapunov exponent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/lyapunov_hist.png")
    plt.close() 

    # ---------- Load tokenized data ---------- #
    tokenized_data_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
    tokenized_data_files = tokenized_data_files[:len(generated_data)]
    tokenized_data = [np.load(f) for f in tokenized_data_files]
    tokenized_data = [f[:generated_data[ss].shape[0],:] for ss, f in enumerate(tokenized_data)]

    # ---------- Get reconstructed data from original data using the tokenizer ---------- #
    generator = load_model(generator_dir)
    reconstructed_data = generator.tokenizer.reconstruct_data(tokenized_data)
    reconstructed_data = [reconstructed_data[s] for s in subs2plot]

    # ---------- FC ---------- #
    print(f"reconstructed_data={reconstructed_data[0].shape}")
    print(f"generated_data={generated_data[0].shape}")

    # Compare static correlation netmat of TDE of original, reconstructed and generated data
    tde_corr = oslf_plotting.plot_tde_corr(
        reconstructed_data,
        generated_data,
        n_embeddings=15,
        n_pca_components=None,
        frequency_range=[1, 40],
        sampling_frequency=sampling_frequency,    
        titles=["Reconstructed", "Generated"],
        filename=f"{plot_dir}/tde_corr.png",
    )

    np.save(f"{plot_dir}/tde_corr_orig.npy", tde_corr[0])

    # Compare AEC of original, reconstructed and generated data
    oslf_plotting.plot_aec(
        reconstructed_data,
        generated_data,
        window_size=1000,
        frequency_range=[1, 40], 
        sampling_frequency=sampling_frequency,
        titles=["Reconstructed", "Generated"],
        filename=f"{plot_dir}/aec.png",
    )

    # ---------- funcs ---------- #

    def _plot_psd(f, psd, name, plot_dir):
        
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

    def _plot_power_maps(f, psd, name, plot_dir):

        if len(psd.shape) == 2:
            psd = np.expand_dims(psd, axis=0)
            
        frequency_bands = {
            "delta": [1, 4],
            "theta": [4, 7],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 50],
        }    

        p = {}
        for band, (low, high) in frequency_bands.items():
            p[band] = power.variance_from_spectra(f, psd, frequency_range=[low, high])

        group_p = {}
        for band in frequency_bands.keys():
            group_p[band] = np.mean(p[band], axis=0)

        power_maps = np.array(list(group_p.values()))

        os.makedirs(f"{plot_dir}/maps", exist_ok=True)
        power.save(
            power_maps,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={"symmetric_cbar": True},
            combined=True,
            titles=list(group_p.keys()),
            filename=f"{plot_dir}/maps/{name}_group_power.png",
        )
        power.save(
            power_maps - np.mean(power_maps, axis=1, keepdims=True),
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={"symmetric_cbar": True},
            combined=True,
            titles=list(group_p.keys()),
            filename=f"{plot_dir}/maps/{name}_group_power_demeaned.png",
        )

    def _plot_aec_connectomes(data, name, plot_dir):

        frequency_bands = {
            "delta": [1, 4],
            "theta": [4, 7],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 50],
        }    

        data = Data(data, n_jobs=8, sampling_frequency=250)
        ae_ts = {}
        for band, (low, high) in frequency_bands.items():
            methods = {
                "filter": {"low_freq": low, "high_freq": high, "use_raw": True},
                "amplitude_envelope": {},
                "standardize": {},
                "moving_average": {"n_window": 25},
            }
            data.prepare(methods)
            ae_ts[band] = data.time_series()

        aec = {band: static.functional_connectivity(ts) for band, ts in ae_ts.items()}
        aec_mean = {}
        for band in aec.keys():
            if len(aec[band].shape) == 3:
                aec_mean[band] = np.mean(aec[band], axis=0)
            else:
                aec_mean[band] = aec[band]
            
                connectivity_maps = np.array(list(aec_mean.values()))
        thres_connectivity_maps = connectivity.threshold(connectivity_maps, percentile=95)

        connectivity.save(
            connectivity_maps,
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            combined=True,
            titles=list(aec_mean.keys()),
            filename=f"{plot_dir}/maps/{name}_group_connectome.png",
        )
        connectivity.save(
            thres_connectivity_maps,
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            combined=True,
            titles=list(aec_mean.keys()),
            filename=f"{plot_dir}/maps/{name}_thres_group_connectome.png",
        )

    # ------------- Do plots for different data

    for data, name in zip([generated_data, reconstructed_data], ["generated", "reconstructed"]):
        
        f, psd = static.welch_spectra(
            data=data,
            sampling_frequency=250,
            n_jobs=8,
            frequency_range=[1, 45],
        )

        _plot_psd(f, psd, name, plot_dir)

        _plot_power_maps(f, psd, name, plot_dir)

        _plot_aec_connectomes(data, name, plot_dir)


    # ---------- Plot history ---------- #

    history = pickle.load(open(f"{generator_dir}/history.pkl", "rb"))

    oslf_plotting.plot_history(history, plot_dir=plot_dir)

    # ---------- Plot summary of the reconstructed and generated data ---------- #

    n_samples = 60*sampling_frequency  # 60 seconds

    reconstructed_data = np.concatenate(reconstructed_data)
    generated_data = np.concatenate(generated_data)

    chans2plot = [0, 10, 30, 40, 50]
    for c2plot in chans2plot:

        fig = plt.figure(figsize=(10, 10))
        ax_1 = [fig.add_subplot(3, 2, i) for i in range(1, 3)]
        ax_2 = [fig.add_subplot(3, 2, i) for i in range(3, 5)]
        ax_3 = fig.add_subplot(3, 1, 3)

        oslf_plotting.plot_time_series_summary(
            reconstructed_data[:, 0],
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            axes=[ax_1[0], ax_2[0], ax_3],
            color="green",
            label="Reconstructed",
        )
        oslf_plotting.plot_time_series_summary(
            generated_data[:, 0],
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            axes=[ax_1[1], ax_2[1], ax_3],
            color="red",
            label="Generated",
        )
        fig.suptitle(f"Time series summary (channel {c2plot})")
        fig.legend(
            loc="lower center",
            ncol=3,
            fontsize=14,
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
        )
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary_{c2plot}.png")
        plt.close(fig)

    print(f"Plot dir is:")
    print(f"{plot_dir}")


####


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uname",
        type=str,
        help="Unique name for this model",
    )
    args = parser.parse_args()

    print("Arguments:")
    print(f"  uname: {args.uname}")

    plot_results(args)

if __name__ == '__main__':
    main()

