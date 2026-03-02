import os

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

os.chdir(f"{dev_dir}/projects/osl-foundation")
from osl_foundation import create_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #

# rm -rf /Users/woolrich/dev/results/osl-foundation/tde_simulation_large/generator/*
data_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large/tokenized_data_tfrecords"
config_dir = f"{dev_dir}/projects/osl-foundation/examples/simulation/tde_simulation/new_space_attention/models/generator"
generator_dir = f"{dev_dir}/results/osl-foundation/tde_simulation_large/generator"

os.makedirs(generator_dir, exist_ok=True)

# ---------- Build generator ---------- 

# copy config.yml from config_dir to generator_dir
os.system(f"cp {config_dir}/config_large.yml {generator_dir}/config.yml") 
generator = create_model(f"{generator_dir}/config.yml")

generator.summary(expand_nested=True)

# ---------- Load data ---------- #

train_data, val_data = load_tfrecord_dataset(
    data_dir,
    batch_size=generator.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
)

for dat in train_data.take(1):
    print("Data shape: ")
    print(dat['data'].shape)

# ---------- Fit generator ---------- #

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
