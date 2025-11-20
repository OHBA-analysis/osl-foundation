"""
This script is for tokenizing the continuous parcel time courses of Cam-CAN
with the trained tokenizer.
"""

import os
from glob import glob

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model

tf_ops.gpu_growth()

# ---------- Load data ---------- #
data_dir = "/well/woolrich/projects/camcan/spring23/src"  # Adjust this path as needed
data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
data = Data(
    data_files,
    n_jobs=16,
    picks="misc",
    reject_by_annotation="omit",
)
data.standardize()

# ---------- Load tokenizer ---------- #
tokenizer_dir = "../../models/tokenizer"
tokenizer = load_model(tokenizer_dir)

# ---------- Tokenise data ---------- #
tokenized_data = tokenizer.tokenize_data(data)

# ---------- Save tokenised data ---------- #
tokenized_data_dir = "../../data/tokenized_data"
os.makedirs(tokenized_data_dir, exist_ok=True)

for i, token_data in enumerate(tokenized_data):
    np.save(
        f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
        token_data,
    )
