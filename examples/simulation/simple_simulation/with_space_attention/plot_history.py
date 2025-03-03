import pickle
import os
from osl_foundation.utils import plotting

history = pickle.load(open("models/generator/history.pkl", "rb"))

os.makedirs("plots", exist_ok=True)

plotting.plot_history(history, plot_dir="plots")
