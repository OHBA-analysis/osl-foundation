import os
import numpy as np
from osl_foundation.simulation.bursts import Bursts

# ---------- Directories ---------- #
data_dir = "sim_data"
plot_dir = "plots/sim_data"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---------- Simulation configuration ---------- #
simulation_config = {
    "n_groups": 2,
    "n_subjects_per_group": 30,
    "true_freqs": np.array([3.0, 10.0, 17.0]),
    "sampling_frequency": 100,
    "snr": 4.0,
    "data_dir": data_dir,
    "group_freq_shift": 1.0,
}
n_modes = simulation_config["true_freqs"].shape[0]
n_channels = 12

# Channel activity
# channels 0, 1, 2, 3 are active for mode 0
# channels 4, 5, 6, 7 are active for mode 1
# channels 8, 9, 10, 11 are active for mode 2
channel_activity = np.zeros((n_modes, n_channels))
channel_activity[0, : n_channels // 3] = 1
channel_activity[1, n_channels // 3 : 2 * n_channels // 3] = 1
channel_activity[2, 2 * n_channels // 3 :] = 1
simulation_config["channel_activity"] = channel_activity

# 5 minutes of data
simulation_config["n_samples"] = 5 * 60 * simulation_config["sampling_frequency"]

bursts = Bursts(**simulation_config)

# ---------- Simulate and Save data ---------- #
bursts.simulate_and_save()

# ---------- Plot summary of the simulated data ---------- #
bursts.plot_data(plot_dir=plot_dir)
