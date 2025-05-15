import os
import math
import numpy as np
from tqdm.auto import tqdm
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.utils.misc import set_random_seed

from osl_foundation import load_model, create_model

tf_ops.gpu_growth()

set_random_seed(42)


def generate_data(
    generator_dir, n_sessions=612, batch_size=32, untrained=False, **kwargs
):
    """Generate data using the generator model.

    Parameters
    ----------
    generator_dir : str
        Directory containing the generator model.
    n_sessions : int, optional
        Number of sessions to generate, by default 612.
    batch_size : int, optional
        Batch size for data generation, by default 32.
    untrained : bool, optional
        Whether to use an untrained generator model, by default False.
    **kwargs : dict, optional
        Additional keyword arguments for generating data.

    Returns
    -------
    generated_data : List[np.ndarray]
        List of generated data arrays.
    """
    if untrained:
        generator = create_model(f"{generator_dir}/config.yml")
    else:
        generator = load_model(generator_dir, from_checkpoint=True)

    session_labels = np.array_split(
        np.arange(n_sessions), math.ceil(n_sessions / batch_size)
    )
    generated_data = []
    for labels in tqdm(
        session_labels, desc="Generating data", total=len(session_labels)
    ):
        gen_data = generator.generate_data(
            batch_size=len(labels),
            extra_labels={"session_id": labels},
            **kwargs,
        )
        generated_data.extend(gen_data)

    return generated_data


if __name__ == "__main__":
    """
    Generate data using a trained generator model and an untrained generator model.
    """
    generator_dir = "/well/woolrich/projects/foundation_models/ephys-gpt/sequence_length_80/without_channel_attention/model"

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # Generate data using the generator model
    generated_data = generate_data(
        generator_dir,
        n_sessions=612,
        batch_size=32,
        untrained=False,
        n_samples=15000,  # 60 seconds per session
        top_p=0.99,
    )
    save_path = f"{save_dir}/generated_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(generated_data, f)

    generated_data_untrained = generate_data(
        generator_dir,
        n_sessions=612,
        batch_size=32,
        untrained=True,
        n_samples=15000,
        top_p=0.99,
    )
    # Save the generated data to a file
    untrained_save_path = f"{save_dir}/generated_data_untrained.pkl"
    with open(untrained_save_path, "wb") as f:
        pickle.dump(generated_data_untrained, f)
