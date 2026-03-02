import pickle
import os

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation.utils import plotting

data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation"
generator_dir = f"{data_dir}/generator"
plot_dir = f"{data_dir}/plots"

os.makedirs(plot_dir, exist_ok=True)

history = pickle.load(open(f"{generator_dir}/history.pkl", "rb"))

plotting.plot_history(history, plot_dir=plot_dir)