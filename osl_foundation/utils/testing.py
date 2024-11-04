import numpy as np
import tensorflow as tf
from typing import Tuple, Union, List
from osl_foundation.config import Config


def get_dataset_size(
    model_config: Config,
    dataset: Union[tf.data.Dataset, List[tf.data.Dataset]],
) -> List[Tuple[int, int]]:
    """Gets the size of the dataset for the model.

    Parameters
    ----------
    model_config : Config
        The model config object.
    dataset : tf.data.Dataset or List[tf.data.Dataset]
        The dataset(s) to get the size of.

    Returns
    -------
    dataset_size : List[Tuple[int, int]]
        The size of each datasets. Shape is (n_datasets, n_samples, n_channels).
    """

    if not isinstance(dataset, list):
        dataset = [dataset]

    # Get the size of the dataset
    n_channels = model_config.n_channels
    n_sequence_length = model_config.sequence_length

    dataset_size = []
    for ds in dataset:
        n_samples = n_sequence_length * np.sum(
            [d["data"].shape.as_list()[0] for d in ds]
        )
        dataset_size.append((n_samples, n_channels))

    return dataset_size


def create_random_tokens(n_tokens: int, n_samples: int, n_channels: int) -> np.ndarray:
    """
    Creates random token array. This is a wrapper around np.random.randint.

    Parameters
    ----------
    n_tokens : int
        Number of tokens to draw from.
    n_samples : int
        Number of samples to draw.
    n_channels : int
        Number of channels.

    Returns
    -------
    random_tokens : np.ndarray
        Random tokens drawn from the discrete uniform distribution.
        Shape is (n_samples, n_channels).
    """
    random_tokens = np.random.randint(n_tokens, size=(n_samples, n_channels))

    return random_tokens
