from dataclasses import dataclass
from typing import List
from packaging import version
from abc import abstractmethod

import tensorflow as tf
import keras

if version.parse(tf.__version__) >= version.parse("2.16"):
    # Use public API
    get_strategy = tf.distribute.get_strategy
else:
    # Use private fall-backs for historical releases
    try:  # 2.13 - 2.15
        from tensorflow.python.distribute.distribute_lib import get_strategy
    except ImportError:  # < 2.13
        from tensorflow.python.distribute.distribution_strategy_context import (
            get_strategy,
        )


@dataclass
class BaseTrainingConfig:
    """Base class for training configuration."""

    optimizer: keras.optimizers.Optimizer = None
    batch_size: int = None
    n_epochs: int = None
    callbacks: List[keras.callbacks.Callback] = None
    strategy: tf.distribute.Strategy = None

    def validate(self) -> None:
        assert self.optimizer is not None, "optimizer must be set"
        assert self.batch_size is not None, "batch_size must be set"
        assert self.n_epochs is not None, "n_epochs must be set"

        self.callbacks = self.callbacks or []
        self.strategy = self.strategy or get_strategy()

    def set_optimizer(self, optimizer: keras.optimizers.Optimizer) -> None:
        self.optimizer = optimizer

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_n_epochs(self, n_epochs: int) -> None:
        self.n_epochs = n_epochs

    def set_callbacks(self, callbacks: List[keras.callbacks.Callback]) -> None:
        self.callbacks = callbacks

    def set_strategy(self, strategy: tf.distribute.Strategy) -> None:
        self.strategy = strategy

    def get_config(self) -> dict:
        return {
            "optimizer": self.optimizer.get_config(),
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "callbacks": [c.get_config() for c in self.callbacks],
            "strategy": self.strategy.get_config(),
        }


@dataclass
class BaseModelConfig:
    name: str = None
    sequence_length: int = None
    n_channels: int = None

    def validate(self):
        assert self.name is not None, "name must be set"
        assert self.sequence_length is not None, "sequence_length must be set"
        assert self.n_channels is not None, "n_channels must be set"

    def set_name(self, name: str):
        self.name = name

    def set_sequence_length(self, sequence_length: int):
        self.sequence_length = sequence_length

    def set_n_channels(self, n_channels: int):
        self.n_channels = n_channels

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "sequence_length": self.sequence_length,
            "n_channels": self.n_channels,
        }
