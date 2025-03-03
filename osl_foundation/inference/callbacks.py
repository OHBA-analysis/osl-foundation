import os
from pathlib import Path
import logging

import numpy as np
import pickle
import tensorflow as tf

from osl_foundation.utils.misc import update_history

_logger = logging.getLogger("osl-foundation")


class TemperatureAnnealingCallback(tf.keras.callbacks.Callback):
    """Callback to anneal the temperature of the token weights layer.

    Parameters
    ----------
    n_stages : int
        Number of stages for the annealing.
    n_epochs : int
        Total number of epochs.
    start_temperature : float, optional
        Starting temperature for the annealing.
    end_temperature : float, optional
        Ending temperature for the annealing.
    """

    def __init__(
        self,
        n_stages: int,
        n_epochs: int,
        start_temperature: float = 1.0,
        end_temperature: float = 1e-3,
    ):
        self.n_stages = n_stages
        self.n_epochs = n_epochs
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.temperatures = np.linspace(start_temperature, end_temperature, n_stages)
        self.n_epochs_per_stage = n_epochs // n_stages

    def on_epoch_begin(self, epoch, logs=None):
        stage = epoch // self.n_epochs_per_stage
        if stage >= self.n_stages:
            stage = self.n_stages - 1
        temperature = self.temperatures[stage]

        token_weights_layer = self.model.get_layer("token_weights")
        token_weights_layer.temperature.assign(temperature)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        token_weights_layer = self.model.get_layer("token_weights")
        logs["temperature"] = float(token_weights_layer.temperature.numpy())


class CheckpointCallback(tf.keras.callbacks.Callback):
    """Callback to create checkpoints during training.

    Parameters
    ----------
    save_freq : int
        Frequency (in epochs) at which to save the model.
    checkpoint_dir : str
        Directory in which to save the checkpoints.
    """

    def __init__(
        self, save_freq: int, checkpoint_dir: str, strategy: tf.distribute.Strategy
    ):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = f"{checkpoint_dir}/ckpt"
        self.checkpoint = None
        self.history_dir = str(Path(checkpoint_dir).parent)
        self.strategy = strategy

    def on_train_begin(self, logs=None):
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.model.optimizer
        )
        checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        if checkpoint_path:
            _logger.info(f"Restoring from {checkpoint_path}")
            with self.strategy.scope():
                self.checkpoint.restore(checkpoint_path).expect_partial()
        else:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            _logger.info(f"Saving checkpoint to {self.checkpoint_prefix}")
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def on_train_end(self, logs=None):
        if os.path.exists(f"{self.history_dir}/history.pkl"):
            with open(f"{self.history_dir}/history.pkl", "rb") as f:
                history = pickle.load(f)
        else:
            history = None

        history = update_history(history, self.model.history.history)
        _logger.info(f"Saving history to {self.history_dir}/history.pkl")
        with open(f"{self.history_dir}/history.pkl", "wb") as f:
            pickle.dump(history, f)


class SpaceAttentionAnnealingCallback(tf.keras.callbacks.Callback):
    """Callback to anneal the dropout rate of the space attention layers.

    Parameters
    ----------
    start_epoch : int
        Epoch at which to start annealing the dropout rate.
    end_epoch : int
        Epoch at which to end annealing the dropout rate.
    start_dropout_rate : float
        Starting dropout rate for the annealing.
    end_dropout_rate : float
        Ending dropout rate for the annealing.
    """

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        start_dropout_rate: float,
        end_dropout_rate: float,
    ):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_dropout_rate = start_dropout_rate
        self.end_dropout_rate = end_dropout_rate
        self.n_stages = end_epoch - start_epoch
        self.dropout_rates = np.linspace(
            start_dropout_rate, end_dropout_rate, self.n_stages
        )

    def on_train_begin(self, logs=None):
        decoder_layer = self.model.get_layer("decoder")
        attention_layers = decoder_layer.attention_layers

        for layer in attention_layers:
            passta_layer = layer.passta_layer
            passta_layer.channel_attention_dropout.assign(self.start_dropout_rate)

    def on_epoch_begin(self, epoch, logs=None):
        stage = epoch - self.start_epoch + 1
        if stage < 0:
            stage = 0
        if stage >= self.n_stages:
            stage = self.n_stages - 1
        dropout_rate = self.dropout_rates[stage]

        decoder_layer = self.model.get_layer("decoder")
        attention_layers = decoder_layer.attention_layers

        for layer in attention_layers:
            passta_layer = layer.passta_layer
            passta_layer.channel_attention_dropout.assign(dropout_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        decoder_layer = self.model.get_layer("decoder")
        attention_layers = decoder_layer.attention_layers
        logs["space_attention_dropout"] = attention_layers[
            0
        ].passta_layer.channel_attention_dropout.numpy()
