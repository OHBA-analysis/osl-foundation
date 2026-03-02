import os
import argparse
import tensorflow as tf

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"
os.chdir(f"{dev_dir}/projects/osl-foundation")
from osl_foundation import create_model

tf_ops.gpu_growth()

seq_len = 120
uname = f'cTrue_l{seq_len}'

# ---------- Directories ---------- #

# rm -rf /well/woolrich/users/vxw496/results/osl-foundation/camcan_spring23/generator_cTrue_l120
# rm -rf /Users/woolrich/dev/results/osl-foundation/camcan_spring23/generator_cTrue_l120

data_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/tokenizer/tokenized_data_tfrecords_{seq_len}"
data_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/tokenizer/tokenized_data_tfrecords_{seq_len}"

config_dir = f"{dev_dir}/projects/osl-foundation/examples/real_data/camcan_rest/new_space_attention/models/generator"
generator_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/generator_{uname}"

print("Generator dir:")
print(generator_dir)

os.makedirs(generator_dir, exist_ok=True)

# ---------- Build generator ---------- 

# copy config.yml from config_dir to generator_dir
os.system(f"cp {config_dir}/config_{uname}.yml {generator_dir}/config.yml") 

generator = create_model(f"{generator_dir}/config.yml")

generator.summary(expand_nested=True)

# ---------- Load data ---------- #

# Load data
train_data, val_data = load_tfrecord_dataset(
    data_dir,
    batch_size=generator.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
)

methods = {
    "filter": {"low_freq": 4, "high_freq": 100, "use_raw": True},
    "standardize": {},
}
train_data.prepare(methods)
val_data.prepare(methods)

if True:
    generator.build_model()

    from osl_dynamics.data import Data
    from glob import glob
    import numpy as np

    # get next train_data
    # n_samples = generator.config.training_config.batch_size * seq_len + 1
    for dat in train_data.take(1):
        print("dat.keys():", dat.keys())
        print("Train data shape:", dat['data'].shape)
        dt = np.expand_dims(np.concatenate(dat['data'].numpy().astype('int32')), axis=0) # (n_samples, n_channels)
        detokenized_gen = generator.tokenizer.reconstruct_data(dt)

        for kk in range(dat['data'].shape[-1]):
            print(f"Channel {kk} mean: {tf.reduce_mean(detokenized_gen[:, kk])}, std: {tf.math.reduce_std(detokenized_gen[:, kk])}")
        break


    # original data
    n_samples = detokenized_gen.shape[0]
    original_data_dir = "/Users/woolrich/dev/results/raw_data/camcan/spring23/"
    original_data_files = sorted(glob(f"{original_data_dir}/*/sflip_parc-raw.fif"))
    original_data = Data(
        original_data_files,
        n_jobs=16,
        picks="misc",
        use_tfrecord=False,
        reject_by_annotation="omit",
    )
    original_data.standardize()
    original_data = original_data[0][:n_samples] # n_samples, n_channels

    # calc info on orig
    for kk in range(dat['data'].shape[-1]):
        print(f"Channel {kk} mean: {tf.reduce_mean(original_data[:, kk])}, std: {tf.math.reduce_std(original_data[:, kk])}")

    # tokenized original data
    tokenized_data = generator.tokenizer.tokenize_data(original_data)[0] # n_samples, n_channels    

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(detokenized_seq[:, 0])
    plt.show()


# ---------- Fit ---------- #

saved_model_path = f"{generator_dir}/model.weights.h5"

# see if save model path exists:
if os.path.exists(saved_model_path):
    print(f"Loading model from {saved_model_path}")
    generator.load_weights(saved_model_path)
else:
    print(f"No saved model found at {saved_model_path}")

generator.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)

generator.save_weights(generator_dir)
