"""
This script is for preparing the training data for Ephys-GPT.
Session ID is added as an extra input and TFRecords are saved for training.
"""

import os

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_foundation.config import get_config

tf_ops.gpu_growth()

# ---------- Load tokenised data ---------- #
tokenized_data_dir = "../../data/tokenized_data"
tokenized_data = Data(tokenized_data_dir, n_jobs=16)
tokenized_data.add_session_labels(
    "session_id", np.arange(tokenized_data.n_sessions), "categorical"
)

# ---------- Save as TFRecord dataset ---------- #
meg_gpt_dir = "../../models/meg-gpt"
meg_gpt_config = get_config(f"{meg_gpt_dir}/config.yml")
sequence_length = meg_gpt_config.model_config.sequence_length

tfrecord_dir = "../../data/tfrecords"
os.makedirs(tfrecord_dir, exist_ok=True)
tokenized_data.save_tfrecord_dataset(
    tfrecord_dir=tfrecord_dir,
    sequence_length=sequence_length + 1,
    overwrite=True,
    validation_split=0.1,
)
