import os
from glob import glob

import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation.models.tokenizers import load_tokenizer

tf_ops.gpu_growth()

data_dir = "sim_data"
tokenizer_dir = "models/tokenizer"
tokenized_data_dir = "tokenized_data"
tokenized_data_tf_dir = "tokenized_data_tfrecords"
os.makedirs(tokenized_data_dir, exist_ok=True)
os.makedirs(tokenized_data_tf_dir, exist_ok=True)

# Load data
data_files = sorted(glob(f"{data_dir}/*.npy"))
data = Data(data_files)
data.standardize()

tokenizer = load_tokenizer(tokenizer_dir)
tokenized_data = tokenizer.tokenize_data(data)

for i, token_data in enumerate(tokenized_data):
    np.save(
        f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
        token_data,
    )

tokenized_data = Data(tokenized_data)
tokenized_data.save_tfrecord_dataset(
    tfrecord_dir=tokenized_data_tf_dir,
    sequence_length=81,
    overwrite=True,
    validation_split=0.1,
)

data.delete_dir()
