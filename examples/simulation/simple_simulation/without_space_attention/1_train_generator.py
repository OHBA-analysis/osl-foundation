import os
from glob import glob
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset

from osl_foundation import create_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
data_dir = "../tokenized_data_tfrecords"
generator_dir = "models/generator"
os.makedirs(generator_dir, exist_ok=True)

# ---------- Build generator ---------- #
generator = create_model(f"{generator_dir}/config.yml")
generator.summary()
generator.save(generator_dir)

# ---------- Load data ---------- #
train_data, val_data = load_tfrecord_dataset(
    data_dir,
    batch_size=generator.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
)

# ---------- Fit generator ---------- #
generator.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)
generated_data = generator.generate_data(
    n_samples=2048,
    top_p=0.95,
    batch_size=20,
)
pickle.dump(generated_data, open(f"{generator_dir}/generated_data.pkl", "wb"))
