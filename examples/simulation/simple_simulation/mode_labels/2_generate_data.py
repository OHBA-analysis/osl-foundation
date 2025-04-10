import numpy as np
import pickle

from osl_dynamics.inference import tf_ops

from osl_foundation import load_model

# Set GPU memory growth
tf_ops.gpu_growth()

# ---------- Directories ---------- #
generator_dir = "models/generator"

# ---------- Load generator ---------- #
generator = load_model(generator_dir, checkpoint="latest")

# ---------- Generate data using the generator ---------- #
real_mtc = np.load("../sim_data/ground_truth/mode_tcs.npy").astype(np.int32)[:, :4000]

all_0_mtc = np.zeros_like(real_mtc)
all_1_mtc = np.ones_like(real_mtc)

# 100 samples of 0 and 100 samples of 1 interleaved
mtc_0 = ([0] * 100 + [1] * 100) * 20
mtc_1 = ([1] * 100 + [0] * 100) * 20

interleaved_mtc = np.array([mtc_0, mtc_1], dtype=np.int32)

extra_channels = {
    "mode_0": np.array([real_mtc[0], all_0_mtc[0], all_1_mtc[0], interleaved_mtc[0]]),
    "mode_1": np.array([real_mtc[1], all_0_mtc[1], all_1_mtc[1], interleaved_mtc[1]]),
}

# Add zeros to the start of the mode time courses
for k in extra_channels.keys():
    extra_channels[k] = np.concatenate(
        [
            np.zeros(
                (4, generator.config.model_config.sequence_length), dtype=np.int32
            ),
            extra_channels[k],
        ],
        axis=1,
    )

generated_data = generator.generate_data(
    n_samples=2048,
    top_p=0.95,
    batch_size=4,
    extra_channels=extra_channels,
)
pickle.dump(generated_data, open(f"{generator_dir}/generated_data.pkl", "wb"))
