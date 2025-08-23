import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import trange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    def _load_ephys_data(file_path):
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        X = list(
            map(
                lambda x: np.mean(x[0], axis=1).reshape(x[0].shape[0], -1),
                data_dict.values(),
            )
        )
        y = list(map(lambda x: x[1], data_dict.values()))
        return X, y, list(data_dict.keys())

    def _load_baseline_data():
        with open("../../results/decoding/baseline_data_dict.pkl", "rb") as f:
            data_dict = pickle.load(f)

        X = list(
            map(
                lambda x: x[0].reshape(x[0].shape[0], -1),
                data_dict.values(),
            )
        )
        y = list(map(lambda x: x[1], data_dict.values()))
        return X, y, list(data_dict.keys())

    b_data = _load_baseline_data()
    zs_data = _load_ephys_data("../../results/decoding/zero_shot_data_dict.pkl")
    ft_data = _load_ephys_data("../../results/decoding/fine_tune_data_dict.pkl")

    return b_data, zs_data, ft_data


def standardize(X_train, X_test):
    X_mean = np.mean(np.concatenate(X_train), axis=0, keepdims=True)
    X_std = np.std(np.concatenate(X_train), axis=0, keepdims=True)
    X_train = [(x - X_mean) / X_std for x in X_train]
    X_test = [(x - X_mean) / X_std for x in X_test]
    return X_train, X_test


def split_data(data_tuple, test_subject, test_run):
    train_Xs, train_ys, test_Xs, test_ys = [], [], [], []
    for x, y, session_id in zip(*data_tuple):
        subject_id = session_id.split("_")[0]
        run_id = session_id.split("_")[1]
        if subject_id == test_subject or run_id == test_run:
            test_Xs.append(x)
            test_ys.append(y)
        else:
            train_Xs.append(x)
            train_ys.append(y)
    train_Xs, test_Xs = standardize(train_Xs, test_Xs)
    return train_Xs, train_ys, test_Xs, test_ys


def get_accuracy(data_tuple, test_subject, test_run):
    train_Xs, train_ys, test_Xs, test_ys = split_data(
        data_tuple, f"sub{test_subject:02d}", f"run{test_run:02d}"
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(np.concatenate(train_Xs), np.concatenate(train_ys))
    y_preds = [clf.predict(x) for x in test_Xs]

    acc = np.array(
        [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(test_ys, y_preds)]
    )

    out_of_subject_acc = acc[test_subject - 1 : test_subject + 5]
    in_subject_acc = np.concatenate([acc[: test_subject - 1], acc[test_subject + 5 :]])
    return in_subject_acc, out_of_subject_acc


def plot_fine_tune_results(fine_tune_results_dict, plot_dir):
    plot_dict = {"In-or-out": [], "Model": [], "Accuracy": []}
    for acc, model in zip(
        [
            fine_tune_results_dict["acc_b"][0],
            fine_tune_results_dict["acc_zs"][0],
            fine_tune_results_dict["acc_ft"][0],
        ],
        ["Baseline", "Zero-shot", "Fine-tuned"],
    ):
        plot_dict["In-or-out"].extend(["Within subject"] * len(acc))
        plot_dict["Model"].extend([model] * len(acc))
        plot_dict["Accuracy"].extend(acc)

    for acc, model in zip(
        [
            fine_tune_results_dict["acc_b"][1],
            fine_tune_results_dict["acc_zs"][1],
            fine_tune_results_dict["acc_ft"][1],
        ],
        ["Baseline", "Zero-shot", "Fine-tuned"],
    ):
        plot_dict["In-or-out"].extend(["Out of subjects"] * len(acc))
        plot_dict["Model"].extend([model] * len(acc))
        plot_dict["Accuracy"].extend(acc)

    plot_df = pd.DataFrame(plot_dict)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(
        data=plot_df,
        x="In-or-out",
        y="Accuracy",
        hue="Model",
        ax=ax,
    )

    # Add chance level line
    ax.axhline(0.25, color="red", linestyle="--", linewidth=2, label="Chance level")

    # Increase font sizes for better visibility
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/fine_tune_results.png")
    plt.close(fig)

    print("decoding accuracies:\n")
    print(plot_df.groupby(["In-or-out", "Model"]).mean().reset_index())


def plot_all_subjects_results(all_subjects_results, plot_dir):
    acc_b_in = [np.mean(x["acc_b"][0]) for x in all_subjects_results]
    acc_zs_in = [np.mean(x["acc_zs"][0]) for x in all_subjects_results]

    acc_b_out = [np.mean(x["acc_b"][1]) for x in all_subjects_results]
    acc_zs_out = [np.mean(x["acc_zs"][1]) for x in all_subjects_results]

    plot_dict = {"In-or-out": [], "Model": [], "Accuracy": []}
    for acc, model in zip([acc_b_in, acc_zs_in], ["Baseline", "Zero-shot"]):
        plot_dict["In-or-out"].extend(["Within subject"] * len(acc))
        plot_dict["Model"].extend([model] * len(acc))
        plot_dict["Accuracy"].extend(acc)

    for acc, model in zip([acc_b_out, acc_zs_out], ["Baseline", "Zero-shot"]):
        plot_dict["In-or-out"].extend(["Out of subject"] * len(acc))
        plot_dict["Model"].extend([model] * len(acc))
        plot_dict["Accuracy"].extend(acc)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_dict,
        x="In-or-out",
        y="Accuracy",
        hue="Model",
        ax=ax,
    )
    fig.tight_layout()
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=14)
    ax.legend(fontsize=14)
    ax.axhline(0.25, color="red", linestyle="--", linewidth=2, label="Chance level")
    fig.savefig(f"{plot_dir}/all_subjects_results.png")
    plt.close(fig)


if __name__ == "__main__":
    results_dir = "../../results/decoding"
    plot_dir = "../../plots/decoding"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    overwrite_fine_tune_results = False
    overwrite_all_subjects_results = False

    if (
        not os.path.exists(f"{results_dir}/fine_tune_results.pkl")
        or overwrite_fine_tune_results
    ):
        b_data, zs_data, ft_data = load_data()
        fine_tune_results_dict = {}
        fine_tune_results_dict["acc_b"] = get_accuracy(
            b_data, test_subject=19, test_run=6
        )
        fine_tune_results_dict["acc_zs"] = get_accuracy(
            zs_data, test_subject=19, test_run=6
        )
        fine_tune_results_dict["acc_ft"] = get_accuracy(
            ft_data, test_subject=19, test_run=6
        )

        with open(f"{results_dir}/fine_tune_results.pkl", "wb") as f:
            pickle.dump(fine_tune_results_dict, f)
    else:
        with open(f"{results_dir}/fine_tune_results.pkl", "rb") as f:
            fine_tune_results_dict = pickle.load(f)

    plot_fine_tune_results(fine_tune_results_dict, plot_dir)

    if (
        not os.path.exists(f"{results_dir}/all_subjects_results.pkl")
        or overwrite_all_subjects_results
    ):
        b_data, zs_data, ft_data = load_data()
        all_subjects_results = []
        for subject in trange(1, 20, desc="Processing subjects"):
            results_dict = {}
            results_dict["acc_b"] = get_accuracy(
                b_data, test_subject=subject, test_run=6
            )
            results_dict["acc_zs"] = get_accuracy(
                zs_data, test_subject=subject, test_run=6
            )
            all_subjects_results.append(results_dict)

        with open(f"{results_dir}/all_subjects_results.pkl", "wb") as f:
            pickle.dump(all_subjects_results, f)
    else:
        with open(f"{results_dir}/all_subjects_results.pkl", "rb") as f:
            all_subjects_results = pickle.load(f)

    plot_all_subjects_results(all_subjects_results, plot_dir)
