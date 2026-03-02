import logging
import os
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import seaborn as sns

from osl_dynamics.analysis import static
from osl_dynamics.simulation import HMM
from osl_dynamics.data import Data

_logger = logging.getLogger("osl-foundation")

class TDEBursts:
    """Simulate data from bursting TDE-based generative model

    Parameters
    ----------
    
    true_tde_covs : list[np.ndarray] or list[str]
        List of n_modes items, each item must be a 
        CE x CE covariance matrix.
        where C is n_channels and E is n_embeddings
    n_subjects : int, optional
        Number of subjects. Default: 10.
    n_samples : int, optional
        Number of samples per subject per channel. Default: 1000.
    n_embeddings: int, optional
        Number of embeddings
    sampling_frequency : int, optional
        Sampling frequency in Hz. Default: 100.
    stay_prob : float, optional
        Probability of staying in the same state in the HMM. Default: 0.98.
    data_dir : str, optional
        Directory to save simulated data. Default: "sim_data".

    """

    def __init__(
        self,
        true_tde_covs,
        n_subjects: int = 10,
        n_samples: int = 1000,
        n_embeddings: int = 1,
        sampling_frequency: int = 100,
        stay_prob: float = 0.98,
        data_dir: str = None,
        rho = 0.1,
    ):
        
        self.n_subjects = n_subjects
        self.n_samples = n_samples
        self.n_embeddings = n_embeddings
        self.sampling_frequency = sampling_frequency
        self.stay_prob = stay_prob
        self.data_dir = data_dir or "sim_data"
        self.rho = rho

        if isinstance(true_tde_covs, str):
            self.true_tde_covs = [ 
                np.load(true_tde_cov) 
                for true_tde_cov in true_tde_covs
            ]

        else:
            self.true_tde_covs = true_tde_covs


        self.n_channels = self.true_tde_covs[0].shape[0]//self.n_embeddings
        self.n_modes = len(self.true_tde_covs)

    def _gen_data_from_tde_cov(self,
                               tde_cov):
            
        # Assumes tde_cov is CE x CE np.array 
        # where
        # C is n_channels
        # E is n_embeddings
        # Ordering of rows/columns in tde_cov
        # is assumed to correspond to tde_cov 
        # being made up of blocks of ExE matrices.

        n_embeddings = self.n_embeddings
        n_tpts2gen = self.n_samples
        rho=self.rho

        n_channels = tde_cov.shape[0]//n_embeddings

        # tde_cov is CE x CE
        tde_cov = tde_cov.reshape(n_channels, n_embeddings, n_channels, n_embeddings)
        tde_cov = np.transpose(tde_cov, [1, 0, 3, 2])
        tde_cov = tde_cov.reshape(n_embeddings*n_channels, n_embeddings*n_channels)

        # See "Conditional Distributions" here:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        # (x_1 conditional on x_2) equivalent to  (x_t conditional on x_tmo)

        Sig22 = tde_cov[:-n_channels, :-n_channels]
        Sig11 = tde_cov[-n_channels:, -n_channels:]
        Sig12 = tde_cov[-n_channels:, :-n_channels]
        Sig21 = tde_cov[:-n_channels, -n_channels:]

        x_2 = np.random.multivariate_normal(np.zeros(tde_cov.shape[0]), tde_cov, size = 1) # 1 x CE
        x_2 = x_2[:,:-n_channels].T # C*(E-1) x 1

        gen_data = np.zeros((n_tpts2gen, n_channels))
        
        invSig22 = np.linalg.pinv(Sig22 + np.eye(Sig22.shape[0])*rho)
        Sig = (Sig11 - Sig12@invSig22@Sig21) + np.eye(Sig11.shape[0])*0.001
        proj = Sig12@invSig22

        for tt in range(n_tpts2gen):
            mu = proj@x_2
            x_1 = np.expand_dims(np.random.multivariate_normal(mu.flatten(), Sig), axis=1)
            gen_data[tt,:] = x_1.flatten()
            x_2 = np.concatenate([x_2[n_channels:], x_1])

        # standardise
        gen_data = (gen_data - np.mean(gen_data, axis=0))/np.std(gen_data, axis=0)  
    
        return gen_data, tde_cov

    def _simulate_bursts(self):
        """Simulate bursts using HMM.

        Returns
        -------
        data : np.ndarray
            Shape: (n_subjects, n_samples, n_channels).
        mode_timecourses : np.ndarray
            Shape: (n_modes, n_samples).
        """

        hmm = HMM(
            trans_prob="uniform",
            stay_prob=self.stay_prob,
            n_states=self.n_modes,
        )
    
        # Build signal
        data = np.zeros((self.n_subjects, self.n_samples, self.n_channels))
        mode_timecourses = np.zeros((self.n_subjects, self.n_samples, self.n_modes))
        for k in tqdm(range(self.n_subjects)):
            mode_timecourses[k, :,:] = hmm.generate_states(self.n_samples)

            for i in range(self.n_modes):
                activity, _ = self._gen_data_from_tde_cov(self.true_tde_covs[i])

                data[k, :, :] += np.expand_dims(mode_timecourses[k, :, i], axis=1) * activity

        return data, mode_timecourses

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate data with bursts and noise.

        Returns
        -------
        data : np.ndarray
            Data with noise added. Shape: (n_subjects, n_samples, n_channels).
        mode_timecourses : np.ndarray
            Shape: (n_modes, n_samples).
        """

        data, mode_timecourses = self._simulate_bursts()

        return data, mode_timecourses

    def simulate_and_save(self) -> None:
        """Simulate data and save to disk.

        Data is saved in self.data_dir in the following format:
        - x_{i}.npy: Data for subject i. Shape: (n_samples, n_channels).
        - ground_truth/mode_tcs.npy: Mode timecourses. Shape: (n_modes, n_samples).
        """
        os.makedirs(f"{self.data_dir}/ground_truth", exist_ok=True)

        data, mode_timecourses = self.simulate()
        for i in range(self.n_subjects):
            np.save(f"{self.data_dir}/x_{i:0{len(str(self.n_subjects))}d}.npy", data[i])
            np.save(f"{self.data_dir}/ground_truth/mode_tcs_subj{i:0{len(str(self.n_subjects))}d}.npy", mode_timecourses[i])

    def plot_data(
        self, plot_dir: str = None, channels_to_plot: Union[list, np.ndarray] = None
    ) -> None:
        """Plot a summary of the simulated data. Plots include:
        - Mode timecourses
        - Simulated data
        - Power spectral density

        Parameters
        ----------
        plot_dir : str, optional
            Directory to save plots. Default: self.data_dir.
        channels_to_plot : list, optional
            List of channels to plot. Default: all channels.
        """

        sub1 = 0
        sub2 = self.n_subjects - 1

        n_samples = min(400, self.n_samples)
        index_from = 200
        index_to = index_from + 400
        timestamps = np.arange(index_to - index_from) / self.sampling_frequency

        data_files = [
            f"{self.data_dir}/x_{i:0{len(str(self.n_subjects))}d}.npy"
            for i in range(self.n_subjects)
        ]

        mode_timecourse_files = [
            f"{self.data_dir}/ground_truth/mode_tcs_subj{i:0{len(str(self.n_subjects))}d}.npy"
            for i in range(self.n_subjects)
        ]

        mode_timecourses_sub1 = np.load(mode_timecourse_files[sub1])

        # Plot mode timecourses
        nplts = self.n_modes if self.n_modes > 1 else 2

        fig, axes = plt.subplots(nplts, 1, figsize=(10, 5))
        for i in range(self.n_modes):
            axes[i].plot(timestamps, mode_timecourses_sub1[index_from:index_to, i])
            axes[i].set_ylim([-0.1, 1.1])
            axes[i].set_ylabel(f"Mode {i}")
            # remove xticks for all but bottom plot
            if i < self.n_modes - 1:
                axes[i].set_xticks([])
            else:
                axes[i].set_xlabel("Time (s)")
            axes[i].set_yticks([0, 1])

        fig.tight_layout()
        fig.suptitle("Mode timecourses")
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/mode_tcs.png")
            plt.close(fig)

        # Plot data
        data_sub1 = np.load(data_files[sub1])
        data_sub2 = np.load(data_files[sub2])
        n_channels = data_sub1.shape[1]

        channels_to_plot = channels_to_plot or np.arange(n_channels)
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(25, 5 * len(channels_to_plot)))
        for ind, i in enumerate(channels_to_plot):
            axes[ind].plot(
                timestamps,
                data_sub1[index_from:index_to, i],
                "r",
                label="Subject 1" if i == 0 else None,
            )

            if False:
                axes[ind].plot(
                    timestamps,
                    data_sub2[index_from:index_to, i],
                    "g",
                    label="Subject 2" if i == 0 else None,
                )

            axes[ind].set_ylabel(f"Channel {i}")

        fig.legend()
        fig.tight_layout()
        fig.suptitle("Simulated data")
        plt.xlabel("Time (s)")
        if plot_dir is not None:
            fig.savefig(f"{plot_dir}/data_chan.png")
            plt.close(fig)

        # Plot PSD
        fig, axes = plt.subplots(
            len(channels_to_plot), 1, figsize=(15, 5 * len(channels_to_plot))
        )
        for ind, i in enumerate(channels_to_plot):
            axes[ind].psd(
                data_sub1[:, i],
                Fs=self.sampling_frequency,
                NFFT=1024,
                color="r",
                label="Subject 1" if i == 0 else None,
            )
            axes[ind].psd(
                data_sub2[:, i],
                Fs=self.sampling_frequency,
                NFFT=1024,
                color="g",
                label="Subject 2" if i == 0 else None,
            )
            axes[ind].set_ylabel(f"Channel {i}")

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.legend()
        axes[0].set_title("PSD")
        fig.tight_layout()
        fig.suptitle("Power spectral density")
        if plot_dir is not None:
            fig.savefig(f"{plot_dir}/psd.png")
            plt.close(fig)

        def _get_tde_corr(data: Data) -> np.ndarray:

            methods = {
                "tde": {"n_embeddings": self.n_embeddings},
                "standardize": {},
            }
            
            data.prepare(methods)
            ts = data.time_series()
            tde_corr = static.functional_connectivity(ts)

            return tde_corr

        # Plot tde_covs generated for subj1 vs truth
        fig, axs = plt.subplots(2, np.max([self.n_modes, 2]))
        
        for i in range(self.n_modes):
            dat = data_sub1[mode_timecourses_sub1[:, i] == 1, :]
            tde_corr_gen = _get_tde_corr(Data(dat))

            # zero out (n_embeddings x n_embeddings) diagonal blocks
            tcorr4plot = self.true_tde_covs[i].copy()
            for kk in range(self.n_channels):
                tcorr4plot[kk*self.n_embeddings:(kk+1)*self.n_embeddings, kk*self.n_embeddings:(kk+1)*self.n_embeddings] = 0

            vmin = np.min(tcorr4plot)
            vmax = np.max(tcorr4plot)

            sns.heatmap(tcorr4plot, ax=axs[0, i], cmap='viridis', vmin=vmin, vmax=vmax, cbar=True)
            axs[0, i].set_title(f"Mode{i} True")

            tcorr4plot = tde_corr_gen.copy()
            for kk in range(self.n_channels):
                tcorr4plot[kk*self.n_embeddings:(kk+1)*self.n_embeddings, kk*self.n_embeddings:(kk+1)*self.n_embeddings] = 0

            vmin = np.min(tcorr4plot)
            vmax = np.max(tcorr4plot)

            sns.heatmap(tcorr4plot, ax=axs[1, i], cmap='viridis', vmin=vmin, vmax=vmax, cbar=True)
            axs[1, i].set_title(f"Mode{i} Gen")

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/true_tde_covs.png")

        _logger.info(f"Plots saved to {plot_dir}")


