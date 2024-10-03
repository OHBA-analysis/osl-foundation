import numpy as np
import tensorflow as tf


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
        start_temperature: float = 0.0,
        end_temperature: float = 1.0,
    ):
        self.n_stages = n_stages
        self.n_epochs = n_epochs
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.temperatures = np.linspace(start_temperature, end_temperature, n_stages)
        self.n_epochs_per_stage = n_epochs // n_stages

    def on_epoch_begin(self, epoch, logs=None):
        stage = epoch // self.n_epochs_per_stage
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

    def __init__(self, save_freq: int, checkpoint_dir: str):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = f"{checkpoint_dir}/ckpt"

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.model.optimizer
            )
        if (epoch + 1) % self.save_freq == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
