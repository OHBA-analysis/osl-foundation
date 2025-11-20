import os
import math
import pickle
from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from osl_dynamics.inference import tf_ops
from osl_foundation import load_model

tf_ops.gpu_growth()


def generate_data(
    meg_gpt_dir: str,
    sessions: Union[int, np.ndarray, List[int]] = 612,
    batch_size: int = 32,
    **kwargs,
) -> List[np.ndarray]:
    """Generate data using the MEG-GPT.

    Parameters
    ----------
    meg_gpt_dir : str
        Directory containing the meg-gpt model.
    sessions : int or np.ndarray or List[int]
        Sessions for which to generate data.
    batch_size : int, optional
        Batch size for data generation, by default 32.
    **kwargs : dict, optional
        Additional keyword arguments for generating data.

    Returns
    -------
    generated_data : List[np.ndarray]
        List of generated data arrays.
    """
    meg_gpt = load_model(meg_gpt_dir, checkpoint="latest")

    if isinstance(sessions, int):
        sessions = np.arange(sessions)
    sessions = np.array(sessions)

    session_labels = np.array_split(sessions, math.ceil(len(sessions) / batch_size))
    generated_data = []
    for labels in tqdm(
        session_labels, desc="Generating data", total=len(session_labels)
    ):
        gen_data = meg_gpt.generate_data(
            batch_size=len(labels),
            extra_labels={"session_id": labels},
            **kwargs,
        )
        generated_data.extend(gen_data)

    return generated_data


if __name__ == "__main__":
    """
    Generate data using a trained MEG-GPT model.
    """
    generator_dir = "../../models/meg-gpt"
    save_dir = "../../data/generated_data"
    os.makedirs(save_dir, exist_ok=True)

    # Generate data using the MEG-GPT foundation model
    generated_data = generate_data(
        generator_dir,
        sessions=612,
        batch_size=32,
        n_samples=15000,  # 60 seconds per session
        top_p=0.99,
    )
    save_path = f"{save_dir}/meg-gpt.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(generated_data, f)
