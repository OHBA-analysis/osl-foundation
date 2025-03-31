import os

import pickle
import tensorflow as tf
import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset

from osl_foundation import create_model
from osl_foundation.config import get_config
from osl_foundation.inference.callbacks import SpaceAttentionAnnealingCallback

tf_ops.gpu_growth()

data_dir = "../tokenized_data_tfrecords"
generator_dir = "models/generator"

config = get_config(f"{generator_dir}/config.yml")


def get_epoch_number():
    checkpoint_path = tf.train.latest_checkpoint(f"{generator_dir}/checkpoints")
    if checkpoint_path:
        return int(checkpoint_path.split("-")[-1])
    return 1


def get_space_attention_dropout(epoch):
    start_epoch = 15
    end_epoch = 25
    start_dropout = 1.0
    end_dropout = 0.0
    n_stages = end_epoch - start_epoch
    dropout_rates = np.linspace(start_dropout, end_dropout, n_stages)

    stage = epoch - start_epoch + 1
    if stage < 0:
        stage = 0
    if stage >= n_stages:
        stage = n_stages - 1

    return dropout_rates[stage]


config.training_config.callbacks.append(
    SpaceAttentionAnnealingCallback(
        start_epoch=1,
        end_epoch=2,
        start_dropout_rate=get_space_attention_dropout(get_epoch_number()),
        end_dropout_rate=get_space_attention_dropout(get_epoch_number()),
    )
)

generator = create_model(config)
generator.summary()

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
