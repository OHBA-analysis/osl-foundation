from typing import Union

import pickle

from osl_foundation.config import Config, get_config


def create_model(config: Union[Config, str]):
    """Create a model based on the configuration.

    Parameters
    ----------
    config : Union[Config, str]
        String, path to a configuration file, or a Config object.

    Returns
    -------
    model : Union[OSLTokenizer]
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


def load_model(model_dir: str):
    """Load a saved model from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.

    Returns
    -------
    model : Union[OSLTokenizer]
        Model object.
    """
    config = get_config(configuration=f"{model_dir}/config.yml")
    model = create_model(config)
    # TODO: Restore model from checkpoint
    model.load_weights(f"{model_dir}/weights").expect_partial()
    with open(f"{model_dir}/history.pkl", "rb") as f:
        model.history = pickle.load(f)
    return model
