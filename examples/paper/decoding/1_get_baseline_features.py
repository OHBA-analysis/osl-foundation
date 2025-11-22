import mne
import numpy as np
from glob import glob
import pickle
import os


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


if __name__ == "__main__":

    data_dir = "/well/woolrich/projects/wakeman_henson/summer23/src"
    data_paths = sorted(glob(f"{data_dir}/sub*_run*/sflip_parc-raw.fif"))

    trials, labels = get_trials_and_labels(data_paths, sequence_length=80)

    data_dict = {}
    for file, trials_, labels_ in zip(data_paths, trials, labels):
        session_id = file.split("/")[-2]
        data_dict[session_id] = (trials_, labels_)

    save_dir = "../../results/decoding"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/baseline_data_dict.pkl", "wb") as f:
        pickle.dump(data_dict, f)
