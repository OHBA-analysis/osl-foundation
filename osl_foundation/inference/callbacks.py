import os
from pathlib import Path
import logging

import numpy as np
import pickle
import tensorflow as tf
import keras

from osl_foundation.utils.misc import update_history

_logger = logging.getLogger("osl-foundation")


class TemperatureAnnealingCallback(keras.callbacks.Callback):
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


class CheckpointCallback(keras.callbacks.Callback):
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
        
        #checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        #if checkpoint_path:
            #_logger.info(f"Restoring from {checkpoint_path}")
            #with self.strategy.scope():
            #    self.checkpoint.restore(checkpoint_path).expect_partial()
        #else:
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


class SpaceAttentionAnnealingCallback(keras.callbacks.Callback):
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


class LyapunovBetaSchedulerCallback(keras.callbacks.Callback):
    """
    Adapt lyapunov beta to target a Lyapunov/cross-entropy loss ratio.

    Intuition:
    - ratio = lyapunov_loss / cross_entropy_loss
    - if ratio > target, decrease beta
    - if ratio < target, increase beta
    Updates are multiplicative in log-space and clipped to [min_beta, max_beta].
    """

    def __init__(
        self,
        target_ratio: float = 0.1,
        adaptation_rate: float = 0.1,
        min_beta: float = 1e-6,
        max_beta: float = 10.0,
        warmup_epochs: int = 0,
        ema_decay: float = 0.9,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.eps = eps
        self._ema_ce = None
        self._ema_lyap = None

    def on_train_begin(self, logs=None):
        layer = self.model.get_layer("lyapunov_loss")
        self._ema_ce = None
        self._ema_lyap = None
        _logger.info(
            "Lyapunov beta scheduler enabled: beta=%.6g target_ratio=%.4g",
            float(layer.beta.numpy()),
            self.target_ratio,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ce = logs.get("cross_entropy_loss")
        lyap = logs.get("lyapunov_loss")
        if ce is None or lyap is None:
            return

        ce = float(ce)
        lyap = float(lyap)
        if self._ema_ce is None:
            # Initialize EMA with first observed epoch values.
            self._ema_ce = ce
            self._ema_lyap = lyap
        else:
            # Smooth noisy epoch-to-epoch losses before computing the control ratio.
            self._ema_ce = self.ema_decay * self._ema_ce + (1 - self.ema_decay) * ce
            self._ema_lyap = self.ema_decay * self._ema_lyap + (1 - self.ema_decay) * lyap

        # Controlled ratio: how strong Lyapunov regularization is relative to CE loss.
        ratio = self._ema_lyap / max(self.eps, self._ema_ce)
        layer = self.model.get_layer("lyapunov_loss")
        beta = float(layer.beta.numpy())

        if epoch + 1 > self.warmup_epochs:
            # log-space controller: stable multiplicative updates around target ratio.
            log_error = np.log((ratio + self.eps) / (self.target_ratio + self.eps))
            # ratio > target -> log_error > 0 -> beta decreases.
            # ratio < target -> log_error < 0 -> beta increases.
            beta = beta * np.exp(-self.adaptation_rate * log_error)
            # Safety bounds prevent runaway growth/collapse.
            beta = float(np.clip(beta, self.min_beta, self.max_beta))
            layer.beta.assign(beta)

        logs["lyapunov_beta"] = beta
        logs["lyapunov_loss_ratio"] = ratio


class LyapunovMuSchedulerCallback(keras.callbacks.Callback):
    """
    Adapt lyapunov mu to target V0/core-loss ratio.

    Intuition:
    - ratio = (mu * V0_loss) / core_lyapunov_loss
    - if ratio > target, decrease mu
    - if ratio < target, increase mu
    Updates are multiplicative in log-space and clipped to [min_mu, max_mu].
    """

    def __init__(
        self,
        target_ratio: float = 0.1,
        adaptation_rate: float = 0.1,
        min_mu: float = 1e-6,
        max_mu: float = 100.0,
        warmup_epochs: int = 0,
        ema_decay: float = 0.9,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.eps = eps
        self._ema_core = None
        self._ema_v0 = None

    def on_train_begin(self, logs=None):
        layer = self.model.get_layer("lyapunov_loss")
        self._ema_core = None
        self._ema_v0 = None
        _logger.info(
            "Lyapunov mu scheduler enabled: mu=%.6g target_ratio=%.4g",
            float(layer.mu.numpy()),
            self.target_ratio,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        core = logs.get("lyapunov_core_loss")
        v0 = logs.get("lyapunov_v0_loss")
        if core is None or v0 is None:
            return

        core = float(core)
        v0 = float(v0)
        if self._ema_core is None:
            # Initialize EMA with first observed epoch values.
            self._ema_core = core
            self._ema_v0 = v0
        else:
            # Smooth noisy epoch-to-epoch losses before computing the control ratio.
            self._ema_core = self.ema_decay * self._ema_core + (1 - self.ema_decay) * core
            self._ema_v0 = self.ema_decay * self._ema_v0 + (1 - self.ema_decay) * v0

        layer = self.model.get_layer("lyapunov_loss")
        mu = float(layer.mu.numpy())
        # Controlled ratio: relative weight of the V(0) anchor term.
        ratio = (mu * self._ema_v0) / max(self.eps, self._ema_core)

        if epoch + 1 > self.warmup_epochs:
            # log-space controller: stable multiplicative updates around target ratio.
            log_error = np.log((ratio + self.eps) / (self.target_ratio + self.eps))
            # ratio > target -> mu decreases; ratio < target -> mu increases.
            mu = mu * np.exp(-self.adaptation_rate * log_error)
            # Safety bounds prevent runaway growth/collapse.
            mu = float(np.clip(mu, self.min_mu, self.max_mu))
            layer.mu.assign(mu)

        logs["lyapunov_mu"] = mu
        logs["lyapunov_v0_ratio"] = ratio
