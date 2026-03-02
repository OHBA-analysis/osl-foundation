import math
import numpy as np
import pickle
from tqdm.auto import tqdm
import pickle
import os

from osl_dynamics.utils.misc import set_random_seed
from osl_dynamics.inference import tf_ops

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import load_model

# Set GPU memory growth
tf_ops.gpu_growth()

set_random_seed(42)

# ---------- Directories ---------- #
generator_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large/generator"

# ---------- Load generator ---------- #
generator = load_model(generator_dir)

# ---------- Generate data using the generator ---------- #

n_sessions = 2
session_labels = np.arange(n_sessions)
lyapunov_margin = 0.0001

sample_frequency = 100
for label in tqdm(session_labels, desc="Generating data", total=len(session_labels)):
    generated_data, lyapunov_data = generator.generate_data(
        n_samples=2*60*sample_frequency,  
        top_p=0.99,
        extra_labels={"session_id": np.ones((1,))*label},
        lyapunov_margin=lyapunov_margin,
    )

    pickle.dump(generated_data[0], open(f"{generator_dir}/generated_data_{label}.pkl", "wb"))
    pickle.dump(lyapunov_data[0], open(f"{generator_dir}/lyapunov_data_{label}.pkl", "wb"))    
