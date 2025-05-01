import pickle
import os
from osl_foundation.utils import plotting


def plot_history(generator_dir, plot_dir):
    """Plot the training history of the generator model.

    Parameters
    ----------
    generator_dir : str
        Directory containing the generator model.
    plot_dir : str
        Directory to save the plots.
    """

    history = pickle.load(open(f"{generator_dir}/history.pkl", "rb"))

    os.makedirs(plot_dir, exist_ok=True)
    plotting.plot_history(history, plot_dir=plot_dir)


if __name__ == "__main__":
    generator_dir = "/well/woolrich/projects/foundation_models/ephys-gpt/sequence_length_80/without_channel_attention/model"
    plot_dir = "results/plots"

    # Plot the training history of the generator model
    plot_history(generator_dir, plot_dir)
