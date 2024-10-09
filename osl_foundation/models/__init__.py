from typing import Union
from osl_foundation.config import Config, get_config


def create_model(config: Config):
    """Create a model based on the configuration.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    model : Union[OSLTokenizer]
        Model object.
    """
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
    config = get_config(configuration_file=f"{model_dir}/config.yml")
    model = create_model(config)
    # TODO: Restore model from checkpoint
    model.load_weights(f"{model_dir}/weights").expect_partial()
    return model
