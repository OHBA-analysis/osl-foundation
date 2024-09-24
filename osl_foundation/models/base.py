from abc import abstractmethod
from contextlib import contextmanager

import tensorflow as tf
import numpy as np

from osl_dynamics.utils.misc import get_argument, replace_argument
from osl_foundation.config.base import BaseConfig


# Class for base model
class BaseModel:
    def __init__(self, config: BaseConfig):
        self._identifier = np.random.randint(100000)
        self.config = config
        self.model: tf.keras.Model = None
        with self.config.training_config.strategy.scope():
            self.build_model()
            self.compile()

    @abstractmethod
    def build_model(self) -> None:
        pass

    def compile(self) -> None:
        self.model.compile(
            optimizer=self.config.training_config.optimizer,
        )

    def fit(self, *args, **kwargs) -> tf.keras.callbacks.History:
        # Use the number of epochs in the config if it has not been passed
        if get_argument(self.model.fit, "epochs", args, kwargs) is None:
            args, kwargs = replace_argument(
                self.model.fit,
                "epochs",
                self.config.training_config.n_epochs,
                args,
                kwargs,
            )

        # Add callbacks
        args, kwargs = replace_argument(
            self.model.fit,
            "callbacks",
            self.config.training_config.callbacks,
            args,
            kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def load_weights(self, filepath: str) -> None:
        with self.config.training_config.strategy.scope():
            self.model.load_weights(filepath)

    def reset_weights(self, keep: list = None) -> None:
        pass

    def reset(self) -> None:
        self.reset_weights()
        self.compile()

    def make_dataset():
        pass

    def save_config(self):
        pass

    def save(self):
        pass

    @contextmanager
    def set_trainable(self):
        pass
