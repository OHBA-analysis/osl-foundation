import numpy as np
import pickle

from osl_dynamics.inference import tf_ops

from osl_foundation import load_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
generator_dir = "models/generator"

# ---------- Load generator ---------- #
generator = load_model(generator_dir, from_checkpoint=True)

# ---------- Generate data using the generator ---------- #
generated_data = generator.generate_data(
    n_samples=2048,
    top_p=0.95,
    batch_size=20,
)
pickle.dump(generated_data, open(f"{generator_dir}/generated_data.pkl", "wb"))
