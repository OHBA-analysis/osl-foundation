from typing import Union
from dataclasses import dataclass
from packaging import version

import tensorflow as tf
import numpy as np
import yaml

if version.parse(tf.__version__) < version.parse("2.13"):
    from tensorflow.python.distribute.distribution_strategy_context import get_strategy
elif version.parse(tf.__version__) < version.parse("2.16"):
    from tensorflow.python.distribute.distribute_lib import get_strategy
else:
    raise ImportError(
        f"Unsupported TensorFlow version: {tf.__version__}. Please use <= 2.15."
    )

from osl_dynamics.config_api.pipeline import load_config

from osl_foundation.inference.callbacks import (
    TemperatureAnnealingCallback,
    CheckpointCallback,
    SpaceAttentionAnnealingCallback,
)
from osl_foundation.config.base import BaseTrainingConfig, BaseModelConfig
from osl_foundation.config.tokenizer_config import (
    OSLTokenizerModelConfig,
    MuTransformTokenizerModelConfig,
    StandardQuantileTokenizerModelConfig,
)
from osl_foundation.config.generator_config import EphysGPTModelConfig


@dataclass
class Config:
    """
    Config class for building and training models.

    Parameters
    ----------
    model_config : Union[BaseModelConfig, OSLTokenizerModelConfig]
        Model config object.
    training_config : BaseTrainingConfig
        Training config object.
    config_dict : dict
        Dictionary representation of the config.
    """

    model_config: Union[
        BaseModelConfig,
        OSLTokenizerModelConfig,
        EphysGPTModelConfig,
        MuTransformTokenizerModelConfig,
        StandardQuantileTokenizerModelConfig,
    ] = None
    training_config: BaseTrainingConfig = None
    config_dict: dict = None

    def validate(self) -> None:
        """Validate the config."""
        self.model_config.validate()
        if any(
            name in self.model_config.name
            for name in ["mu_transform", "standard_quantile"]
        ):
            # non-learnable tokenizer does not have a training config
            return
        self.training_config.validate()

    def set_model_config(self, model_config: dict) -> None:
        """Set the model config.

        Parameters
        ----------
        model_config : dict
            Dictionary representation of the model config.
        """
        self.model_config = get_model_config(model_config)

    def set_training_config(self, training_config: dict):
        """Set the training config.

        Parameters
        ----------
        training_config : dict
            Dictionary representation of the training config.
        """
        if not training_config:
            return
        self.training_config = get_training_config(training_config)

    def set_config_dict(self, config_dict: dict):
        """Set the config dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary representation of the config.
        """
        self.config_dict = config_dict

    def save_config(self, dirname: str):
        """Save the config to a directory.
        The config is saved as a YAML file as 'config.yml'.

        Parameters
        ----------
        dirname : str
            Directory to save the config.
        """
        with open(f"{dirname}/config.yml", "w") as f:
            yaml.dump(self.config_dict, f)

    def get_config(self) -> dict:
        return {
            "model_config": self.model_config.get_config(),
            "training_config": self.training_config.get_config(),
        }


def get_config(configuration: Union[dict, str] = None) -> Config:
    """Get a Config object from a dictionary or a YAML configuration file.

    Parameters
    ----------
    configuration : Union[dict, str]
        Path to yaml file, :code:`str` to convert to :code:`dict`,
        or :code:`dict` containing the config.

    Returns
    -------
    config : Config
        Config object.
    """

    config_dict = load_config(configuration)
    config = Config()
    config.set_model_config(config_dict["model_config"])
    config.set_training_config(config_dict.get("training_config", {}))
    config.set_config_dict(config_dict)
    config.validate()
    return config


def get_training_config(config: dict) -> BaseTrainingConfig:
    """
    Get a training config object from a dictionary.

    Parameters
    ----------
    config : dict
        Dictionary representation of the training config.

    Returns
    -------
    training_config : BaseTrainingConfig
        Training config object.
    """
    training_config = BaseTrainingConfig()

    # Set optimizer
    if "optimizer" in config:
        optimizer_name = config["optimizer"].get("name", "adam")
        learning_rate = config["optimizer"].get("learning_rate", 0.001)
        clip_norm = config["optimizer"].get("clip_norm", None)
        optimizer = tf.keras.optimizers.get(
            {
                "class_name": optimizer_name.lower(),
                "config": {"learning_rate": learning_rate, "clipnorm": clip_norm},
            }
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    training_config.set_optimizer(optimizer)

    # Set batch size
    training_config.set_batch_size(config.get("batch_size", 32))

    # Set number of epochs
    training_config.set_n_epochs(config.get("n_epochs", 10))

    # Set strategy
    if not config.get("multi_gpu", False):
        training_config.set_strategy(get_strategy())
    else:
        training_config.set_strategy(tf.distribute.MirroredStrategy())

    # Set callbacks
    callbacks = []
    if "temperature_annealing" in config:
        n_stages = config["temperature_annealing"]["n_stages"]
        start_temperature = config["temperature_annealing"].get(
            "start_temperature", 1.0
        )
        end_temperature = config["temperature_annealing"].get("end_temperature", 1e-3)
        n_annealing_epochs = config["temperature_annealing"].get(
            "n_annealing_epochs", training_config.n_epochs
        )
        callbacks.append(
            TemperatureAnnealingCallback(
                n_stages=n_stages,
                n_epochs=n_annealing_epochs,
                start_temperature=start_temperature,
                end_temperature=end_temperature,
            )
        )
    if "checkpoint" in config:
        save_freq = config["checkpoint"].get("save_freq", 1)
        save_dir = config["checkpoint"]["save_dir"]
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                checkpoint_dir=f"{save_dir}/checkpoints",
                strategy=training_config.strategy,
            )
        )

    if "lr_decay" in config:
        lr_decay = config["lr_decay"]
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: learning_rate * np.exp(-lr_decay * epoch)
            )
        )

    if "save_best" in config:
        save_best_kwargs = config["save_best"]
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(**save_best_kwargs))

    if "space_attention_annealing" in config:
        callbacks.append(
            SpaceAttentionAnnealingCallback(
                **config["space_attention_annealing"],
            )
        )

    training_config.set_callbacks(callbacks)

    return training_config


def get_model_config(
    config: dict,
) -> Union[
    BaseModelConfig,
    OSLTokenizerModelConfig,
    EphysGPTModelConfig,
    MuTransformTokenizerModelConfig,
    StandardQuantileTokenizerModelConfig,
]:
    """
    Get a model config object from a dictionary.

    Parameters
    ----------
    config : dict
        Dictionary representation of the model config.

    Returns
    -------
    model_config : Union[BaseModelConfig, OSLTokenizerModelConfig]
        Model config object.
    """
    MODEL_CONFIGS = {
        "osl_tokenizer": OSLTokenizerModelConfig,
        "ephys_gpt": EphysGPTModelConfig,
        "mu_transform_tokenizer": MuTransformTokenizerModelConfig,
        "standard_quantile_tokenizer": StandardQuantileTokenizerModelConfig,
    }

    name = config.get("name", None)
    if name is None:
        raise ValueError("config must contain a 'name' key")
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model config {name} not implemented. "
            f"Options are {', '.join(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[name]()
    if name == "osl_tokenizer" or name == "ephys_gpt":
        model_config.set_n_channels(config.get("n_channels", None))
        model_config.set_sequence_length(config.get("sequence_length", 256))
    model_config.set_config(config)

    return model_config
