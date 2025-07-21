import os
from glob import glob

import numpy as np
import pandas as pd

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data
from osl_dynamics.inference import modes

from osl_foundation import load_model

tf_ops.gpu_growth()

tokenizer_dir = "/well/woolrich/projects/foundation_models/tokenizers/base/model"
tokenized_data_dir = "../../data/wh_tokenized_data"
os.makedirs(tokenized_data_dir, exist_ok=True)

data_dir = "/well/woolrich/projects/wakeman_henson/summer23/src"
data_paths = sorted(glob(f"{data_dir}/sub*_run*/sflip_parc-raw.fif"))
data = Data(
    data_paths,
    sampling_frequency=250,
    picks="misc",
    reject_by_annotation="omit",
    n_jobs=12,
)
data.standardize()

tokenizer = load_model(tokenizer_dir)
tokenized_data = tokenizer.tokenize_data(data)

tokenized_data = Data(tokenized_data, n_jobs=16).time_series()

for t, file in zip(tokenized_data, data_paths):
    session_id = file.split("/")[-2]
    raw = modes.convert_to_mne_raw(t, file)
    raw.save(f"{tokenized_data_dir}/{session_id}_raw.fif", overwrite=True)
