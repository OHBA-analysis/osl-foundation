import os
from glob import glob

import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

dev_dir = "/well/woolrich/users/vxw496"
dev_dir = "/Users/woolrich/dev"

os.chdir(f"{dev_dir}/projects/osl-foundation")

from osl_foundation import load_model

tf_ops.gpu_growth()

seq_len = 80
do_tokenize = True

# ---------- Directories ---------- #

# input dirs


data_dir = "/well/woolrich/projects/camcan/spring23/src"
data_dir = "/Users/woolrich/dev/results/raw_data/camcan/spring23/"

results_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23"
tokenizer_dir = f"{results_dir}/tokenizer"

# ---------- Tokenise ---------- #

tokenized_data_dir = f"{tokenizer_dir}/tokenized_data"
os.makedirs(tokenized_data_dir, exist_ok=True)

if do_tokenize:

    # ---- Load data ---- #

    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
    data = Data(
        data_files,
        n_jobs=16,
        picks="misc",
        use_tfrecord=True,
        reject_by_annotation="omit",
    )
    data.standardize()

    tokenizer = load_model(tokenizer_dir)
    tokenized_data = tokenizer.tokenize_data(data)

    for i, token_data in enumerate(tokenized_data):
        np.save(
            f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
            token_data,
        )
    data.delete_dir()

