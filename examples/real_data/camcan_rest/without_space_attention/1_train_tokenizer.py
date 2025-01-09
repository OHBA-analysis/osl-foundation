from glob import glob
import os

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops

from osl_foundation import create_model

tf_ops.gpu_growth()

data_dir = "/well/woolrich/projects/camcan/spring23/src"
plot_dir = "plots/tokenizer"
tokenizer_dir = "models/tokenizer"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))[:50]
data = Data(
    data_files,
    n_jobs=8,
    picks="misc",
    use_tfrecord=True,
    reject_by_annotation="omit",
)
data.standardize()

tokenizer = create_model(f"{tokenizer_dir}/config.yml")
tokenizer.summary()
tokenizer.fit(data)
tokenizer.save(tokenizer_dir)

# Plot percentage of explained variance
tokenizer.plot_pve(data=data, plot_dir=plot_dir)

# Plot token counts
tokenizer.plot_token_counts(plot_dir=plot_dir)

# Plot stimulus response of token kernels
tokenizer.plot_token_response(plot_dir=plot_dir)

# Plot signals reconstructed from tokenized data (for one session)
tokenizer.plot_fitted_signal(
    data_path=data.time_series()[0],
    plot_dir=plot_dir,
)

# Clean up data directory
data.delete_dir()
