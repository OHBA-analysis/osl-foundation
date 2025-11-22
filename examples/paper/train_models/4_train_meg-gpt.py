"""
This script is for training MEG-GPT on the tokenized Cam-CAN
resting-state dataset.
"""

from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.inference import tf_ops
from osl_foundation import create_model

tf_ops.gpu_growth()

# ---------- Initialise MEG-GPT ---------- #
meg_gpt_dir = "../../models/meg-gpt"
meg_gpt = create_model(f"{meg_gpt_dir}/config.yml")
meg_gpt.summary()

# ---------- Load data ---------- #
tfrecord_dir = "../../data/tfrecords"
batch_size = meg_gpt.config.training_config.batch_size
train_data, val_data = load_tfrecord_dataset(
    tfrecord_dir,
    batch_size=batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
)

# ---------- Train MEG-GPT ---------- #
meg_gpt.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)
