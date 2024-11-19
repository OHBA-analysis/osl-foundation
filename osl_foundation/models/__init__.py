from typing import Union

import pickle
import tensorflow as tf

from osl_foundation.config import Config, get_config


def create_model(config: Union[Config, str]):
    """Create a model based on the configuration.

    Parameters
    ----------
    config : Union[Config, str]
        String, path to a configuration file, or a Config object.

    Returns
    -------
    model
        Model object.
    """
    if isinstance(config, str):
        config = get_config(config)

    if config.model_config.name == "osl_tokenizer":
        from osl_foundation.models.tokenizers import OSLTokenizer

        return OSLTokenizer(config)
    elif config.model_config.name == "ephys_gpt":
        from osl_foundation.models.ephys_gpt import EphysGPT

        return EphysGPT(config)
    else:
        raise ValueError(f"Model {config.model_config.name} not implemented.")


def load_model(model_dir: str, from_checkpoint: bool = False):
    """Load a saved model from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.
    from_checkpoint : bool, optional
        Whether to load the model from a checkpoint.

    Returns
    -------
    model
        Model object.
    """
    config = get_config(configuration=f"{model_dir}/config.yml")
    model = create_model(config)
    if from_checkpoint:
        checkpoint = tf.train.Checkpoint(
            model=model.model, optimizer=model.model.optimizer
        )
        checkpoint.restore(
            tf.train.latest_checkpoint(f"{model_dir}/checkpoints")
        ).expect_partial()
    else:
        model.load_weights(f"{model_dir}/weights.h5")

    try:
        with open(f"{model_dir}/history.pkl", "rb") as f:
            model.history = pickle.load(f)
    except FileNotFoundError:
        pass
    return model
