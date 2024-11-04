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


def create_random_tokens(
    min_val: int,
    max_val: int = None,
    size: Union[int, tuple] = (1,),
    concatenate: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Creates random token array for a given size. Wrapper around np.random.randint.

    Parameters
    ----------
    min_val : int
        The minimum value of the random tokens to sample.
    max_val : int, optional
        The maximum value of the random tokens to sample. Defaults to None, where
        tokens are sampled from [0, min_val).
    size : Union[int, tuple], optional
        Array size of the random tokens to create. By default, a single
        value is returned.
    concatenate : bool, optional
        Whether to concatenate the tokens over all sessions, by default False.

    Returns
    -------
    random_tokens : Union[np.ndarray, List[np.ndarray]]
        Random tokens drawn from the discrete uniform distribution.
    """

    # Standardise the size to a list of tuples
    if isinstance(size, int):
        size = [(size,)]
    if isinstance(size, tuple):
        size = [size]

    # Generate random tokens
    random_tokens = [
        np.random.randint(low=min_val, high=max_val, size=sz) for sz in size
    ]
    if concatenate:
        random_tokens = np.concatenate(random_tokens)

    return random_tokens
