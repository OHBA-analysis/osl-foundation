import os
from glob import glob

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import load_model
from osl_foundation.utils import plotting

tf_ops.gpu_growth()

data_dir = "sim_data"
plot_dir = "plots/generated_data"
generator_dir = "models/generator"
os.makedirs(plot_dir, exist_ok=True)

data_files = sorted(glob(f"{data_dir}/*.npy"))

generator = load_model(generator_dir)

tokens = generator.tokenizer.tokenize_data(Data(data_files))
reconstructed_data = generator.tokenizer.reconstruct_data(tokens)

generated_data = generator.generate_data(
    n_samples=1024,
    method="top_p",
    p=0.8,
    batch_size=len(data_files),
)

plotting.plot_aec(
    data_files,
    reconstructed_data,
    generated_data,
    window_size=100,
    sampling_frequency=100,
    titles=["Original", "Reconstructed", "Generated"],
    filename=f"{plot_dir}/aec.png",
)
