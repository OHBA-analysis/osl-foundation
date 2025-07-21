from glob import glob
import os
import numpy as np
import mne
import math
import pickle

from tqdm.auto import tqdm
import tensorflow as tf

from osl_dynamics.inference import tf_ops
from osl_foundation import load_model

tf_ops.gpu_growth()


def find_events(raw):
    new_event_ids = {"famous": 1, "unfamiliar": 2, "scrambled": 3, "button": 4}
    old_event_ids = {
        "famous": [5, 6, 7],
        "unfamiliar": [13, 14, 15],
        "scrambled": [17, 18, 19],
        "buttonp": [
            256,
            261,
            262,
            263,
            269,
            270,
            271,
            273,
            274,
            275,
            4096,
            4101,
            4102,
            4103,
            4109,
            4110,
            4111,
            4114,
            4114,
            4115,
            4352,
            4357,
            4359,
            4365,
            4369,
        ],
    }
    events = mne.find_events(raw, min_duration=0.005, verbose=False)
    for old_event_codes, new_event_codes in zip(
        old_event_ids.values(), new_event_ids.values()
    ):
        events = mne.merge_events(events, old_event_codes, new_event_codes)
    return events, new_event_ids


def get_trials_and_labels(fif_files, sequence_length):
    trials, trial_labels = [], []
    for file in fif_files:
        raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
        events, event_ids = find_events(raw)
        epochs = mne.Epochs(
            raw,
            events,
            event_ids,
            tmin=0.0,
            tmax=sequence_length / raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
            on_missing="ignore",
        )
        trials_, trial_labels_ = [], []
        for key in ["famous", "unfamiliar", "scrambled", "button"]:
            e = np.transpose(epochs[key].get_data(picks="misc"), (0, 2, 1))
            trials_.extend(e)
            trial_labels_.extend([key] * len(e))

        trials_ = np.array(trials_)
        trial_labels_ = np.array(trial_labels_)
        trials.append(trials_)
        trial_labels.append(trial_labels_)

    return trials, trial_labels


def get_session_features(generator, trials, batch_size):
    shift_token_layer = generator.model.get_layer("shift_token")
    input_embedding_layer = generator.model.get_layer("input_embedding")
    decoder_layer = generator.model.get_layer("decoder")

    features = []
    n_trials = len(trials)
    indices = np.array_split(np.arange(n_trials), math.ceil(n_trials / batch_size))

    @tf.function
    def _get_features(t):
        x, _ = shift_token_layer(t, training=False)
        x = input_embedding_layer([x, []], training=False)
        x = decoder_layer(x, training=False)
        return x

    for idx in tqdm(indices):
        batch_trials = trials[idx]

        # Get features
        x = _get_features(batch_trials)

        # Store features
        features.extend(list(x.numpy()))

    return np.array(features)


def get_features(generator, trials, batch_size):
    features = []
    for trials_ in trials:
        features.append(get_session_features(generator, trials_, batch_size))

    return features


tokenized_files = sorted(glob("../../data/wh_tokenized_data/*.fif"))

decoding_model = load_model("../../models/decoding_model", checkpoint="latest")
sequence_length = decoding_model.config.model_config.sequence_length

trials, labels = get_trials_and_labels(tokenized_files, sequence_length)
features = get_features(decoding_model, trials, batch_size=64)

data_dict = {}
for file, features_, labels_ in zip(tokenized_files, features, labels):
    session_id = file.split("/")[-1]
    data_dict[session_id] = (features_, labels_)

save_dir = "../../results/decoding"
os.makedirs(save_dir, exist_ok=True)

with open(f"{save_dir}/fine_tune_data_dict.pkl", "wb") as f:
    pickle.dump(data_dict, f)
