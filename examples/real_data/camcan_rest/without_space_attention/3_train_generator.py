import os

import pickle
import tensorflow as tf

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset

from osl_foundation import create_model

tf_ops.gpu_growth()

data_dir = "tokenized_data_tfrecords"
generator_dir = "models/generator"
checkpoint_dir = f"{generator_dir}/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

generator = create_model(f"{generator_dir}/config.yml")
checkpoint = tf.train.Checkpoint(
    model=generator.model, optimizer=generator.model.optimizer
)

# Load previous model if exists
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_path:
    checkpoint.restore(checkpoint_path).expect_partial()

# Load data
train_data, val_data = load_tfrecord_dataset(
    data_dir,
    batch_size=generator.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
)

generator.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)

# Update and save history
if os.path.exists(f"{generator_dir}/history.pkl"):
    with open(f"{generator_dir}/history.pkl", "rb") as f:
        history = pickle.load(f)
else:
    history = {}

for k, v in generator.history.items():
    if k in history:
        history[k].extend(v)
    else:
        history[k] = v

with open(f"{generator_dir}/history.pkl", "wb") as f:
    pickle.dump(history, f)

# Save model
checkpoint.save(file_prefix=f"{checkpoint_dir}/ckpt")
