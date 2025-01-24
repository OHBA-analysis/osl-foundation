from abc import abstractmethod
from typing import Union, List, Tuple
import logging

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from osl_dynamics.data import Data
from osl_dynamics.data.tf import concatenate_datasets
from osl_dynamics.inference import initializers as osld_initializers
from osl_dynamics.utils.misc import get_argument, replace_argument

from osl_foundation.config import Config

_logger = logging.getLogger("osl-foundation")


# Class for base model
class BaseModel:
    """
    Base class for models.

    Parameters
    ----------
    config : Config
        Configuration object.
    """

    def __init__(self, config: Config):
        config.validate()
        self._identifier = np.random.randint(100000)
        self.config = config
        self.model: tf.keras.Model = None
        self.history: dict = None
        with self.config.training_config.strategy.scope():
            self.build_model()
            self.compile()

    @abstractmethod
    def build_model(self) -> None:
        """Abstract method to build the model"""
        pass

    def compile(self) -> None:
        """Compile the model with the optimizer within the strategy scope."""
        with self.config.training_config.strategy.scope():
            self.model.compile(
                optimizer=self.config.training_config.optimizer,
            )

    def make_dataset(
        self,
        inputs: Union[Data, str, np.ndarray],
        shuffle: bool = False,
        concatenate: bool = False,
        step_size: int = None,
        drop_last_batch: bool = False,
        validation_split: float = None,
        repeat_count: int = 1,
    ) -> Union[tf.data.Dataset, List[tf.data.Dataset]]:
        """
        Make a TensorFlow Dataset from an osl-dynamics Data object.

        Parameters
        ----------
        inputs : osl_dynamics.data.Data or str or np.ndarray
            Data object. If a :code:`str` or :np.ndarray: is passed this
            function will first convert it into a Data object.
        shuffle : bool, optional
            Should we shuffle the data?
        concatenate : bool, optional
            Should we return a single TensorFlow Dataset or a list of Datasets.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        validation_split : float, optional
            Fraction of the data to use for validation.
        repeat_count : int, optional
            Number of times to repeat the dataset. Default is 1.

        Returns
        -------
        dataset : tf.data.Dataset or List[tf.data.Dataset]
            TensorFlow Dataset (or list of Datasets) that can be used for
            training/evaluating.
        """
        if isinstance(inputs, str) or isinstance(inputs, np.ndarray):
            # str or numpy array -> Data object
            inputs = Data(inputs)

        if isinstance(inputs, Data):
            # Validation
            if (
                isinstance(
                    self.config.training_config.strategy, tf.distribute.MirroredStrategy
                )
                and not inputs.use_tfrecord
            ):
                _logger.warning(
                    "Using a multiple GPUs with a non-TFRecord dataset. "
                    + "This will result in poor performance. "
                    + "Consider using a TFRecord dataset with Data(..., use_tfrecord=True)."
                )

            sequence_length = self.config.model_config.sequence_length
            if self.config.model_config.name == "ephys_gpt":
                sequence_length += 1

            # Data object -> list of Dataset if concatenate=False
            # or Data object -> Dataset if concatenate=True
            if inputs.use_tfrecord:
                outputs = inputs.tfrecord_dataset(
                    sequence_length,
                    self.config.training_config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                    validation_split=validation_split,
                    repeat_count=repeat_count,
                    overwrite=True,
                )
            else:
                outputs = inputs.dataset(
                    sequence_length,
                    self.config.training_config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                    validation_split=validation_split,
                    repeat_count=repeat_count,
                )

        elif isinstance(inputs, tf.data.Dataset) and not concatenate:
            # Dataset -> list of Dataset if concatenate=False
            outputs = [inputs]

        elif isinstance(inputs, list) and concatenate:
            outputs = concatenate_datasets(inputs)

        else:
            outputs = inputs

        return outputs

    def fit(self, *args, **kwargs) -> None:
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        args : arguments
            Arguments for :code:`tf.keras.Model.fit()`.
        kwargs : keyword arguments, optional
            Keyword arguments for :code:`tf.keras.Model.fit()`.
        """
        # If step_per_epoch is passed, repeat the dataset indefinitely
        steps_per_epoch = get_argument(self.model.fit, "steps_per_epoch", args, kwargs)
        repeat_count = 1 if steps_per_epoch is None else -1

        # If a osl_dynamics.data.Data object has been passed for the x
        # arguments, replace it with a tensorflow dataset
        x = get_argument(self.model.fit, "x", args, kwargs)
        x = self.make_dataset(
            x,
            shuffle=True,
            concatenate=True,
            drop_last_batch=True,
            repeat_count=repeat_count,
        )
        args, kwargs = replace_argument(self.model.fit, "x", x, args, kwargs)

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

        history = self.model.fit(*args, **kwargs)
        self.history = history.history

    def load_weights(self, filepath: str) -> tf.keras.Model:
        """Load weights from a file.
        This is a wrapper of :code:`tf.keras.Model.load_weights()`.

        Parameters
        ----------
        filepath : str
            Path to the weights file.

        Returns
        -------
        model : tf.keras.Model
            Model with the loaded weights.
        """
        with self.config.training_config.strategy.scope():
            return self.model.load_weights(filepath)

    def reset_weights(self, keep: list = None) -> None:
        """Reset the weights of the model.

        Parameters
        ----------
        keep : list, optional
            List of names of layers to keep the weights of.
        """
        with self.config.training_config.strategy.scope():
            osld_initializers.reinitialize_model_weights(self.model, keep=keep)

    def reset(self) -> None:
        """Reset the model and compile it."""
        self.reset_weights()
        with self.config.training_config.strategy.scope():
            self.compile()

    def save_config(self, dirname: str) -> None:
        """
        Save the config to a directory.
        The config is saved as a YAML file as 'config.yml'.

        Parameters
        ----------
        dirname : str
            Directory to save the configuration.
        """
        self.config.save_config(dirname)

    def save(self, dirname: str) -> None:
        """
        Save the model config and weights to a directory.

        Parameters
        ----------
        dirname : str
            Directory to save the model.
        """

        # Save model config and history
        self.save_config(dirname)
        with open(f"{dirname}/history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        # Save model weights if the best model is not already saved
        saved_best = any(
            isinstance(cb, tf.keras.callbacks.ModelCheckpoint)
            for cb in self.config.training_config.callbacks
        )
        if not saved_best:
            self.model.save_weights(f"{dirname}/weights.h5")

    def plot_history(
        self, plot_dir: str = None, keywords: List[str] = None
    ) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
        """Plot the training history.

        Parameters
        ----------
        plot_dir : str, optional
            Directory to save the plot.
        keyword : List[str], optional
            List of keywords to filter the history. If None, all history
            will be plotted.

        Returns
        -------
        fig : plt.Figure
            Figure object.
        ax : List[plt.Axes]
            Axes object.
        """
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy")
        keywords = keywords or list(self.history.keys())
        for key in self.history.keys():
            if key not in keywords:
                continue
            if "loss" in key:
                ax1.plot(self.history[key], "b" if "val" in key else "r", label=key)
            elif "top" in key:
                ax2.plot(self.history[key], "b--" if "val" in key else "r--", label=key)
            else:
                continue
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        if plot_dir is not None:
            fig.savefig(f"{plot_dir}/history.png")
            plt.close(fig)
        else:
            return fig, [ax1, ax2]

    def summary(self, **kwargs) -> None:
        """Print a summary of the model."""
        self.model.summary(**kwargs)
