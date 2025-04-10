from typing import Union

import pickle
import tensorflow as tf

from osl_foundation.config import Config, get_config

from osl_foundation.models.tokenizers import OSLTokenizer
from osl_foundation.models.ephys_gpt import EphysGPT

models = {
    "osl_tokenizer": OSLTokenizer,
    "ephys_gpt": EphysGPT,
}


def create_model(
    config: Union[Config, dict, str],
    save_dir: str = None,
    strategy: tf.distribute.Strategy = None,
):
    """Create a model based on the configuration.

    Parameters
    ----------
    config : Union[Config, dict, str]
        String, path to a configuration file, dictionary, or a Config object.
    save_dir : str, optional
        Path to a directory where the configuration will be saved.
        Defaults to None, in which case the configuration will not be saved.
    strategy : tf.distribute.Strategy, optional
        TensorFlow distribution strategy for distributed training.
        Defaults to None, in which case the config strategy will be used.

    Returns
    -------
    model
        Model object.
    """
    if not isinstance(config, Config):
        config = get_config(config)

    if save_dir:
        config.save_config(save_dir)

    if config.model_config.name not in models:
        raise ValueError(
            f"Model {config.model_config.name} not implemented. "
            f"Options are {', '.join(models.keys())}"
        )

    return models[config.model_config.name](config, strategy=strategy)


def load_model(
    model_dir: str,
    checkpoint: str = None,
    strategy: tf.distribute.Strategy = None,
):
    """Load a saved model from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.
    checkpoint : str, optional
        Path to the checkpoint file. If `latest`, the latest checkpoint will be used.
        Defaults to None, in which case the weights will be loaded from `weights.h5`.
    strategy : tf.distribute.Strategy, optional
        TensorFlow distribution strategy for distributed training.
        Defaults to None, in which case the config strategy will be used.

    Returns
    -------
    model
        Model object.
    """
    config = get_config(configuration=f"{model_dir}/config.yml")

    return models[config.model_config.name].load_model(
        model_dir,
        checkpoint,
        strategy=strategy,
    )
