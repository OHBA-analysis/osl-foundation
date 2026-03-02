import os
from glob import glob

import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

from osl_foundation import load_model

tf_ops.gpu_growth()

seq_len = 100

# ---------- Directories ---------- #
data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large"
tokenizer_dir = f"{data_dir}/tokenizer"

tokenized_data_dir = f"{data_dir}/tokenized_data"
tokenized_data_tf_dir = f"{data_dir}/tokenized_data_tfrecords"
os.makedirs(tokenized_data_dir, exist_ok=True)
os.makedirs(tokenized_data_tf_dir, exist_ok=True)

# ---------- Load data ---------- #
data_files = sorted(glob(f"{data_dir}/*.npy"))

data = Data(
    data_files,
    n_jobs=8,
    picks="misc",
    use_tfrecord=True,
    reject_by_annotation="omit",
    sampling_frequency=250,
)

methods = {
    "filter": {"low_freq": 5, "high_freq": 100, "use_raw": True},
    "standardize": {},
}
data.prepare(methods)

# ---------- Load tokenizer and tokenize data ---------- #

tokenizer = load_model(tokenizer_dir)
tokenized_data = tokenizer.tokenize_data(data)

for i, token_data in enumerate(tokenized_data):
    np.save(
        f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
        token_data,
    )

data.delete_dir()
