import os
from glob import glob

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import trange

from osl_dynamics.data import Data
from osl_dynamics.utils import plotting as osld_plotting
from osl_dynamics.analysis import static, power, connectivity

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import load_model
from osl_foundation.utils import plotting as oslf_plotting

subs2plot = [0, 1, 4, 5]
sampling_frequency = 250
start = 100
finish = 5100
seq_len = 120
do_chan_att = True

# ---------- Directories ---------- #

tokenized_data_dir = "" #f"{dev_dir}/results/osl-foundation/camcan_spring23/tokenizer/tokenized_data"

uname = 'cTrue_l120'
generator_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/generator_{uname}"
generator_dir = f"/Users/woolrich/Desktop/generator_{uname}"
plot_dir = f"{generator_dir}/plots4"
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

# ---------- Plot generated data ---------- #

chans2plot = [0, 10, 30, 40, 50]

nsubs = len(subs2plot)
nsubs = np.minimum(nsubs, 6)
fig = plt.figure(figsize=(10, 8))

for ii in range(nsubs):
    for cc in range(len(chans2plot)):
        plt.subplot(nsubs, len(chans2plot)+1, ii*(len(chans2plot)+1)+cc+1)
        if cc == 0:
            plt.ylabel(f"Subj {ii}", fontsize=8)
        plt.plot(generated_data[ii][start:finish, cc])

        if ii == 0:
            plt.title(f"Channel {chans2plot[cc]}", fontsize=8)

    # Plot lyapunov
    plt.subplot(nsubs, len(chans2plot)+1, ii*(len(chans2plot)+1)+len(chans2plot)+1)
    plt.plot(lyapunov_data[ii][start:finish], color="red")
    if ii == 0:
        plt.title("Lyapunov exponent", fontsize=8)
plt.show()
plt.savefig(f"{plot_dir}/lyapunov_n_gendata.png")

# Now plot ffts

fig = plt.figure(figsize=(10, 8))
for ii in range(nsubs):
    for cc in range(len(chans2plot)):
        plt.subplot(nsubs, len(chans2plot), cc*nsubs+ii+1)
        f, psd = static.welch_spectra(
            data=generated_data[ii][start:finish, cc][:, np.newaxis],
            sampling_frequency=sampling_frequency,
            frequency_range=[1, 45],
        )
        plt.plot(f, psd)
        plt.xlim([1, 50])   

plt.show()

# ---------- Plot lyapunov data ---------- #

plt.figure()
plt.plot(lyapunov_data[0], color="gray", alpha=0.5)
plt.xlabel("Timepoints")
plt.ylabel("Lyapunov exponent")
plt.tight_layout()
plt.savefig(f"{plot_dir}/lyapunov.png")
plt.close()

# ---------- Load tokenized data ---------- #

tokenized_data_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
if len(tokenized_data_files) > 0:
    tokenized_data_files = tokenized_data_files[:len(generated_data)]
    tokenized_data = [np.load(f) for f in tokenized_data_files]
    tokenized_data = [f[:generated_data[ss].shape[0],:] for ss, f in enumerate(tokenized_data)]

    # ---------- Get reconstructed data from original data using the tokenizer ---------- #
    generator = load_model(generator_dir)
    reconstructed_data = generator.tokenizer.reconstruct_data(tokenized_data)
    reconstructed_data = [reconstructed_data[s] for s in subs2plot]
    is_recon_data = True
else:
    reconstructed_data = generated_data
    is_recon_data = False

# ---------- FC ---------- #

print(f"generated_data={generated_data[0].shape}")

# Compare static correlation netmat of TDE of original, reconstructed and generated data
tde_corr = oslf_plotting.plot_tde_corr(
    reconstructed_data,
    generated_data,
    n_embeddings=15,
    n_pca_components=None,
    frequency_range=[1, 40],
    sampling_frequency=sampling_frequency,    
    titles=["Generated", "Generated"],
    filename=f"{plot_dir}/tde_corr.png",
)

np.save(f"{plot_dir}/tde_corr_orig.npy", tde_corr[0])

# Compare AEC of original, reconstructed and generated data
oslf_plotting.plot_aec(
    reconstructed_data,
    generated_data,
    window_size=200,
    frequency_range=[1, 40], 
    sampling_frequency=sampling_frequency,
    titles=["Generated", "Generated"],
    filename=f"{plot_dir}/aec.png",
)

# ---------- funcs ---------- #

def _plot_psd(psd, name, plot_dir):
    
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
        if len(p[band].shape) == 2:
            group_p[band] = np.mean(p[band], axis=0)
        else:
            group_p[band] = p[band]

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

if is_recon_data:
    tmp = zip([generated_data, reconstructed_data], ["generated", "reconstructed"])
else:
    tmp = zip([generated_data], ["generated"])

for data, name in tmp:
    f, psd = static.welch_spectra(
        data=data,
        sampling_frequency=250,
        n_jobs=8,
        frequency_range=[1, 45],
    )

    _plot_psd(psd, name, plot_dir)

    
    _plot_power_maps(f, psd, name, plot_dir)

    _plot_aec_connectomes(data, name, plot_dir)


# ---------- Plot summary of the reconstructed and generated data ---------- #

n_samples = 60*sampling_frequency  # 60 seconds

reconstructed_data_cat = np.concatenate(reconstructed_data)
generated_data_cat = np.concatenate(generated_data)

chans2plot = [0, 10, 30, 40, 50]
for c2plot in chans2plot:

    fig = plt.figure(figsize=(10, 10))
    ax_1 = [fig.add_subplot(3, 2, i) for i in range(1, 3)]
    ax_2 = [fig.add_subplot(3, 2, i) for i in range(3, 5)]
    ax_3 = fig.add_subplot(3, 1, 3)

    if is_recon_data:
        oslf_plotting.plot_time_series_summary(
            reconstructed_data_cat[:, c2plot],
            sampling_frequency=sampling_frequency,
            n_samples=n_samples,
            axes=[ax_1[0], ax_2[0], ax_3],
            color="green",
            label="Reconstructed",
        )

    oslf_plotting.plot_time_series_summary(
        generated_data_cat[:, c2plot],
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

