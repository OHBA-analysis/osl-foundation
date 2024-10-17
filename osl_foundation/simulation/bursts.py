import logging
import os
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.simulation import HMM

_logger = logging.getLogger("osl-foundation")


class Bursts:
    """Simulate data with bursts.

    Parameters
    ----------
    n_groups : int, optional
        Number of groups of subjects. Default: 1.
    n_subjects_per_group : int, optional
        Number of subjects per group. Default: 1.
    n_samples : int, optional
        Number of samples per subject per channel. Default: 1000.
    n_channels_per_mode : int, optional
        Number of channels per frequency mode. Default: 8.
    n_modes : int, optional
        Number of frequency modes. Default: 4.
    true_freqs : np.ndarray, optional
        True frequencies for each mode and subject. Default: None.
        Shape: (n_subjects, n_modes).
        If provided, :code:`n_modes` is ignored.
    group_freq_shift : float, optional
        Frequency shift between groups in Hz. Default: 0.5.
    channel_activity : np.ndarray, optional
        Channel activity for each mode. Default: None.
        Shape: (n_modes, n_channels).
        If provided, :code:`n_channels_per_mode` is ignored.
    sampling_frequency : int, optional
        Sampling frequency in Hz. Default: 100.
    snr : float, optional
        Signal-to-noise ratio. Default: 4.
    data_dir : str, optional
        Directory to save simulated data. Default: "sim_data".
    """

    def __init__(
        self,
        n_groups: int = 1,
        n_subjects_per_group: int = 1,
        n_samples: int = 1000,
        n_channels_per_mode: int = 8,
        n_modes: int = 4,
        true_freqs: np.ndarray = None,
        group_freq_shift: float = 0.5,
        channel_activity: np.ndarray = None,
        sampling_frequency: int = 100,
        snr: float = 4,
        data_dir: str = None,
    ):
        self.n_groups = n_groups
        self.n_subjects_per_group = n_subjects_per_group
        self.n_subjects = n_groups * n_subjects_per_group
        self.n_samples = n_samples
        self.n_channels_per_mode = n_channels_per_mode
        self.n_modes = n_modes
        self.true_freqs = self._get_true_freqs(true_freqs, group_freq_shift)
        self.channel_activity = self._get_channel_activity(channel_activity)
        self.n_channels = self.channel_activity.shape[1]
        self.sampling_frequency = sampling_frequency
        self.snr = snr
        self.data_dir = data_dir or "sim_data"

    def _get_true_freqs(self, true_freqs, group_freq_shift):
        """Get true frequencies for each mode and subject.

        Returns
        -------
        true_freqs : np.ndarray
            Shape: (n_subjects, n_modes).
        """
        if self.n_modes > 0:
            if true_freqs is None:
                true_freqs = np.linspace(8, 25, self.n_modes)

            true_freqs = np.tile(true_freqs, (self.n_subjects, 1))
            # Shape: (n_subjects, n_modes)

            for i in range(self.n_groups):
                for j in range(self.n_subjects_per_group):
                    true_freqs[i * self.n_subjects_per_group + j] += (
                        i - (self.n_groups - 1) / 2
                    ) * group_freq_shift
        else:
            true_freqs = None

        return true_freqs

    def _get_channel_activity(self, channel_activity):
        """Get channel activity for each mode.

        Returns
        -------
        channel_activity : np.ndarray
            Shape: (n_modes, n_channels).
        """
        if self.n_modes > 0:
            if channel_activity is None:
                channel_activity = np.tile(
                    np.eye(self.n_modes), (self.n_channels_per_mode)
                )
            # Shape: (n_modes, n_channels)
            else:
                self.n_channels_per_mode = None
        else:
            channel_activity = None

        return channel_activity

    def _simulate_bursts(self):
        """Simulate bursts using HMM.

        Returns
        -------
        true_signal : np.ndarray
            Shape: (n_subjects, n_samples, n_channels).
        mode_timecourses : np.ndarray
            Shape: (n_modes, n_samples).
        """
        n_channels = self.channel_activity.shape[1]

        timestamps = np.arange(self.n_samples) / self.sampling_frequency
        hmm = HMM(
            trans_prob="uniform",
            stay_prob=0.98,
            n_states=self.n_modes + 1,
        )

        # mode timecourses and activity timecourses
        mode_timecourses = []
        sinusoidal_activity = np.zeros(
            (self.n_modes, self.n_subjects, n_channels, self.n_samples)
        )
        phase_diff = np.linspace(0, 0.25, n_channels)
        for i in range(self.n_modes):
            mode_timecourses.append(hmm.generate_states(self.n_samples)[:, 0])
            for j in range(n_channels):
                # Each chan has a unique phase
                # phase = (0.5*np.sin(2 * np.pi * 0.005 * ts)+np.random.uniform())*2 * np.pi
                phase = (
                    (0.5 * np.sin(2 * np.pi * 0.005 * timestamps) + phase_diff[j])
                    * 2
                    * np.pi
                )
                for k in range(self.n_subjects):
                    sinusoidal_activity[i, k, j] = np.sin(
                        2 * np.pi * self.true_freqs[k, i] * timestamps + phase
                    )

        mode_timecourses = np.array(mode_timecourses)

        # Build signal
        true_signal = np.zeros((self.n_subjects, self.n_samples, n_channels))
        for i in range(self.n_subjects):
            for j in range(self.n_modes):
                for k in range(n_channels):
                    if self.channel_activity[j, k] == 1:
                        true_signal[
                            i, mode_timecourses[j] == 1, k
                        ] += sinusoidal_activity[j, i, k, mode_timecourses[j] == 1]

        return true_signal, mode_timecourses

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate data with bursts and noise.

        Returns
        -------
        data : np.ndarray
            Data with noise added. Shape: (n_subjects, n_samples, n_channels).
        true_signal : np.ndarray
            True signal without noise. Shape: (n_subjects, n_samples, n_channels).
        mode_timecourses : np.ndarray
            Shape: (n_modes, n_samples).
        """
        if self.n_modes > 0:
            true_signal, mode_timecourses = self._simulate_bursts()
        else:
            true_signal = np.zeros(
                (self.n_subjects, self.n_samples, self.n_channels_per_mode)
            )
            mode_timecourses = [np.zeros(self.n_samples)]

        # Add noise
        noise_std = 1 / self.snr
        data = true_signal + np.random.normal(0, noise_std, true_signal.shape)

        return data, true_signal, mode_timecourses

    def simulate_and_save(self) -> None:
        """Simulate data and save to disk.

        Data is saved in self.data_dir in the following format:
        - x_{i}.npy: Data for subject i. Shape: (n_samples, n_channels).
        - ground_truth/true_signal_{i}.npy: True signal for subject i. Shape: (n_samples, n_channels).
        - ground_truth/mode_tcs.npy: Mode timecourses. Shape: (n_modes, n_samples).
        """
        os.makedirs(f"{self.data_dir}/ground_truth", exist_ok=True)

        data, true_signal, mode_timecourses = self.simulate()
        np.save(f"{self.data_dir}/ground_truth/mode_tcs.npy", mode_timecourses)
        for i in range(self.n_subjects):
            np.save(f"{self.data_dir}/x_{i}.npy", data[i])
            np.save(f"{self.data_dir}/ground_truth/true_signal_{i}.npy", true_signal[i])

    def plot_data(
        self, plot_dir: str = None, channels_to_plot: Union[list, np.ndarray] = None
    ) -> None:
        """Plot a summary of the simulated data. Plots include:
        - Mode timecourses
        - Simulated data
        - Power spectral density
        - Channel activity
        - True frequencies

        Parameters
        ----------
        plot_dir : str, optional
            Directory to save plots. Default: self.data_dir.
        channels_to_plot : list, optional
            List of channels to plot. Default: all channels.
        """
        data_files = [f"{self.data_dir}/x_{i}.npy" for i in range(self.n_subjects)]
        plot_dir = plot_dir or self.data_dir
        os.makedirs(plot_dir, exist_ok=True)
        mode_timecourses = np.load(f"{self.data_dir}/ground_truth/mode_tcs.npy")

        # Plot mode timecourses
        n_samples = min(2000, self.n_samples)
        timestamps = np.arange(n_samples) / self.sampling_frequency
        fig, axes = plt.subplots(self.n_modes, 1, figsize=(10, 5))
        for i in range(self.n_modes):
            axes[i].plot(timestamps, mode_timecourses[i][:n_samples])
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
        fig.savefig(f"{plot_dir}/mode_tcs.png")

        # Plot data
        sub1 = 0
        sub2 = self.n_subjects - 1
        n_samples = min(400, self.n_samples)
        index_from = 200
        index_to = index_from + 300

        true_signal_sub1 = np.load(
            f"{self.data_dir}/ground_truth/true_signal_{sub1}.npy"
        )
        true_signal_sub2 = np.load(
            f"{self.data_dir}/ground_truth/true_signal_{sub2}.npy"
        )
        data_sub1 = np.load(data_files[sub1])
        data_sub2 = np.load(data_files[sub2])
        n_channels = data_sub1.shape[1]

        timestamps = np.arange(index_to - index_from) / self.sampling_frequency
        channels_to_plot = channels_to_plot or np.arange(n_channels)
        fig, axes = plt.subplots(
            len(channels_to_plot), 1, figsize=(15, 5 * len(channels_to_plot))
        )
        for i in channels_to_plot:
            axes[i].plot(
                timestamps,
                data_sub1[index_from:index_to, i],
                "r",
                label="Subject 1" if i == 0 else None,
            )
            axes[i].plot(
                timestamps,
                data_sub2[index_from:index_to, i],
                "g",
                label="Subject 2" if i == 0 else None,
            )
            axes[i].plot(
                timestamps,
                true_signal_sub1[index_from:index_to, i],
                "r--",
                label="True signal 1" if i == 0 else None,
            )
            axes[i].plot(
                timestamps,
                true_signal_sub2[index_from:index_to, i],
                "g--",
                label="True signal 2" if i == 0 else None,
            )
            axes[i].set_ylabel(f"Channel {i}")

        fig.legend()
        fig.tight_layout()
        fig.suptitle("Simulated data")
        plt.xlabel("Time (s)")

        plt.yticks([])
        fig.savefig(f"{plot_dir}/data_chan.png")

        # Plot PSD
        fig, axes = plt.subplots(
            len(channels_to_plot), 1, figsize=(15, 5 * len(channels_to_plot))
        )
        for i in channels_to_plot:
            axes[i].psd(
                data_sub1[:, i],
                Fs=self.sampling_frequency,
                NFFT=1024,
                color="r",
                label="Subject 1" if i == 0 else None,
            )
            axes[i].psd(
                data_sub2[:, i],
                Fs=self.sampling_frequency,
                NFFT=1024,
                color="g",
                label="Subject 2" if i == 0 else None,
            )
            axes[i].set_ylabel(f"Channel {i}")

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.legend()
        axes[0].set_title("PSD")
        fig.tight_layout()
        fig.suptitle("Power spectral density")
        fig.savefig(f"{plot_dir}/psd.png")

        # Plot channel activity
        fig, axes = plt.subplots(figsize=(15, 5))
        mat = axes.imshow(self.channel_activity)
        axes.set_xlabel("Channels")
        axes.set_ylabel("Modes")
        axes.set_yticks(np.arange(self.n_modes))
        fig.colorbar(mat, ax=axes)
        fig.tight_layout()
        fig.suptitle("Channel activity (0 = inactive, 1 = active)")
        fig.savefig(f"{plot_dir}/channel_activity.png")

        # Plot true frequencies
        fig, axes = plt.subplots(figsize=(15, 5))
        mat = axes.imshow(self.true_freqs.T)
        axes.set_xlabel("Subjects")
        axes.set_ylabel("Modes")
        axes.set_yticks(np.arange(self.n_modes))
        fig.tight_layout()
        fig.suptitle("True frequencies")
        fig.colorbar(mat, ax=axes)
        fig.savefig(f"{plot_dir}/true_freqs.png")

        _logger.info(f"Plots saved to {plot_dir}")
