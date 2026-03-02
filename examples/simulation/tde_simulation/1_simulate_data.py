import os
import numpy as np

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation.simulation.tde_bursts import TDEBursts

# ---------- Directories ---------- #

data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large"
plot_dir = f"{data_dir}/plots/"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
   
# ---------- Simulation configuration ---------- #

tde_dir = f"{dev_dir}/projects/osl-foundation/examples/simulation/tde_simulation"
tde_cov = np.load(f"{tde_dir}/tde_corr_orig_large.npy")

tde_covs = [tde_cov, 0.5*tde_cov + 0.5*np.eye(tde_cov[0].shape[0])]

n_embeddings = 15
n_channels = tde_cov[0].shape[0]//n_embeddings

print(f"n_channels: {n_channels}")
print(f"n_embeddings: {n_embeddings}")

simulation_config = {
    "true_tde_covs": tde_covs,
    "n_subjects": 10,
    "n_embeddings": n_embeddings,
    "sampling_frequency": 250,
    "stay_prob": 0.98,
    "data_dir": data_dir
}

# 5 minutes of data
simulation_config["n_samples"] = 5 * 60 * simulation_config["sampling_frequency"]

bursts = TDEBursts(**simulation_config, rho=0.000000001)

# ---------- Simulate and Save data ---------- #
bursts.simulate_and_save()

# ---------- Plot summary of the simulated data ---------- #
bursts.plot_data(plot_dir=plot_dir, channels_to_plot=[10, 20, 30, 40])
