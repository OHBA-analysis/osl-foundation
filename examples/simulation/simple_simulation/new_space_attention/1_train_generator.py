import os
import shutil
from glob import glob
import yaml

# Reduce TensorFlow C++ runtime log noise (e.g. benign OUT_OF_RANGE/info lines).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from osl_dynamics.inference import tf_ops
from osl_foundation import create_model, load_model
from osl_foundation.utils.training_data import (
    load_tfrecord_train_val_data,
    dataset_shape_matches,
    rebuild_tfrecord_dataset_from_token_files,
)

# python /Users/woolrich/dev/projects/osl-foundation/examples/simulation/simple_simulation/new_space_attention/1_train_generator.py
# rm -rf /Users/woolrich/dev/results/osl-foundation/simple_simulation/generator

dev_dir = "/Users/woolrich/dev"
# dev_dir = "/well/woolrich/users/vxw496"

# Set GPU memory growth
tf_ops.gpu_growth()

DISABLE_LYAPUNOV = True

# ---------- Directories ---------- #

data_dir = f"{dev_dir}/results/osl-foundation/simple_simulation/tokenized_data_tfrecords"
tokenized_data_dir = f"{dev_dir}/results/osl-foundation/simple_simulation/tokenized_data"
config_dir = f"{dev_dir}/projects/osl-foundation/examples/simulation/simple_simulation/new_space_attention/models/generator"
generator_dir = f"{dev_dir}/results/osl-foundation/simple_simulation/generator"
print(generator_dir)

os.makedirs(generator_dir, exist_ok=True)

# ---------- Build generator ---------- #

# copy config.yml from config_dir to generator_dir
src_config = f"{config_dir}/config.yml"
dst_config = f"{generator_dir}/config.yml"
shutil.copy2(src_config, dst_config)

with open(dst_config, "r") as f:
    config_dict = yaml.safe_load(f)

# Keep session label cardinality aligned with the actual tokenized sessions.
tokenized_session_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
updated = False
if tokenized_session_files:
    n_sessions = len(tokenized_session_files)
    print(f"Detected {n_sessions} tokenized sessions in {tokenized_data_dir}.")
    extra_labels = (
        config_dict.get("model_config", {})
        .get("input_parameters", {})
        .get("extra_labels", [])
    )
    for label_cfg in extra_labels:
        if label_cfg.get("name") == "session_id" and label_cfg.get("n_classes") != n_sessions:
            print(
                f"Updating session_id n_classes from {label_cfg.get('n_classes')} to {n_sessions} "
                "to match tokenized sessions."
            )
            label_cfg["n_classes"] = n_sessions
            updated = True

else:
    print(f"No tokenized session files found in {tokenized_data_dir}.")

if DISABLE_LYAPUNOV:
    loss_parameters = config_dict.setdefault("model_config", {}).setdefault(
        "loss_parameters", {}
    )
    loss_parameters["lyapunov_beta"] = 0.0
    loss_parameters["lyapunov_mu"] = 0.0
    loss_parameters["lyapunov_collapse_weight"] = 0.0
    training_config = config_dict.setdefault("training_config", {})
    removed_beta_scheduler = training_config.pop("lyapunov_beta_scheduler", None)
    removed_mu_scheduler = training_config.pop("lyapunov_mu_scheduler", None)
    updated = True
    print(
        "Lyapunov regularization disabled for training "
        "(beta=0, mu=0, collapse_weight=0, schedulers removed)."
    )
    if removed_beta_scheduler is None and removed_mu_scheduler is None:
        print("No Lyapunov scheduler blocks found in training config.")

if updated:
    with open(dst_config, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

checkpoint_files = glob(f"{generator_dir}/checkpoints/ckpt-*.index")
weights_path = f"{generator_dir}/model.weights.h5"
if checkpoint_files or os.path.exists(weights_path):
    # Default load_model behavior is checkpoint-first (latest), then weights.h5 fallback.
    generator = load_model(generator_dir)
else:
    print("No saved model/checkpoint found. Creating a fresh model.")
    generator = create_model(dst_config)

if DISABLE_LYAPUNOV:
    generator.lyapunov_loss_layer.beta.assign(0.0)
    generator.lyapunov_loss_layer.mu.assign(0.0)
    generator.lyapunov_loss_layer.collapse_weight.assign(0.0)
    print(
        "Applied runtime Lyapunov disable: "
        f"beta={float(generator.lyapunov_loss_layer.beta.numpy()):.6g}, "
        f"mu={float(generator.lyapunov_loss_layer.mu.numpy()):.6g}, "
        "collapse_weight="
        f"{float(generator.lyapunov_loss_layer.collapse_weight.numpy()):.6g}"
    )

generator.summary(expand_nested=True)

# ---------- Load data ---------- #

train_data, val_data = load_tfrecord_train_val_data(
    data_dir,
    batch_size=generator.config.training_config.batch_size,
)
expected_seq_len = generator.config.model_config.sequence_length + 1
expected_n_channels = generator.config.model_config.n_channels
shape_matches, found_shape = dataset_shape_matches(
    train_data,
    expected_seq_len=expected_seq_len,
    expected_n_channels=expected_n_channels,
)

if not shape_matches:
    print(
        "TFRecord shape mismatch detected. "
        f"Expected (batch, {expected_seq_len}, {expected_n_channels}), "
        f"found (batch, {found_shape[0]}, {found_shape[1]}). "
        "Rebuilding TFRecords from tokenized .npy files."
    )
    validation_split = generator.config.config_dict.get("training_config", {}).get(
        "validation_split", 0.1
    )
    print(
        f"Rebuilding TFRecords (sequence_length={expected_seq_len}, "
        f"validation_split={validation_split}) in {data_dir}."
    )
    rebuild_tfrecord_dataset_from_token_files(
        tokenized_data_dir=tokenized_data_dir,
        tfrecord_dir=data_dir,
        sequence_length=expected_seq_len,
        validation_split=validation_split,
    )
    train_data, val_data = load_tfrecord_train_val_data(
        data_dir,
        batch_size=generator.config.training_config.batch_size,
    )

# ---------- Fit generator ---------- #

generator.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)

generator.save_weights(generator_dir)
