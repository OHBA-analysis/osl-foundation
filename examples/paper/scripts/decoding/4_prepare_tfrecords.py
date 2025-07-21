from glob import glob
import os

import numpy as np

from osl_dynamics.data import Data
from osl_foundation.config import get_config

tokenized_files = sorted(glob("../../data/wh_tokenized_data/*.fif"))
tokenized_data = Data(
    tokenized_files, n_jobs=16, picks="misc", reject_by_annotation="omit"
)

tokenized_data.add_session_labels(
    "session_id", np.arange(tokenized_data.n_sessions), "categorical"
)

ephys_gpt_dir = "../../models/ephys-gpt"
ephys_gpt_config = get_config(f"{ephys_gpt_dir}/config.yml")
sequence_length = ephys_gpt_config.model_config.sequence_length

tfrecords_dir = "../../data/wh_tfrecords"
os.makedirs(tfrecords_dir, exist_ok=True)
tokenized_data.save_tfrecord_dataset(
    tfrecord_dir=tfrecords_dir,
    sequence_length=80 + 1,
    overwrite=True,
)
