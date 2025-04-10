import math
import numpy as np
from tqdm.auto import tqdm
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.utils.misc import set_random_seed

from osl_foundation import load_model

tf_ops.gpu_growth()

set_random_seed(42)

generator_dir = f"models/generator"
generator = load_model(generator_dir, checkpoint="latest")

n_sessions = 612
batch_size = 32
session_labels = np.array_split(
    np.arange(n_sessions), math.ceil(n_sessions / batch_size)
)
generated_data = []
for labels in tqdm(session_labels, desc="Generating data", total=len(session_labels)):
    gen_data = generator.generate_data(
        n_samples=15000,  # 60 seconds
        top_p=0.99,
        batch_size=len(labels),
        extra_labels={"session_id": labels},
    )
    generated_data.extend(gen_data)

with open(f"{generator_dir}/generated_data.pkl", "wb") as f:
    pickle.dump(generated_data, f)
