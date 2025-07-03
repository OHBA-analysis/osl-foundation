"""
This script is for training the tokenizer on the first 50 subjects
in the Cam-CAN resting state dataset.
"""

from glob import glob

from osl_dynamics.data import Data
from osl_foundation import create_model

# ---------- Load data ---------- #
data_dir = "/well/woolrich/projects/camcan/spring23/src"  # Adjust this path as needed
# Use the first 50 subjects as training data
data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))[:50]

data = Data(
    data_files,
    n_jobs=12,
    picks="misc",
    use_tfrecord=True,
    reject_by_annotation="omit",
)
data.standardize()

# ---------- Train and save tokenizer ---------- #
tokenizer_dir = "../../models/tokenizer"
tokenizer = create_model(f"{tokenizer_dir}/config.yml")
tokenizer.summary()
tokenizer.fit(data)
tokenizer.save(tokenizer_dir)
