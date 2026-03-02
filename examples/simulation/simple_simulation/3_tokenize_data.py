import os
from glob import glob

import numpy as np

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

# python /Users/woolrich/dev/projects/osl-foundation/examples/simulation/simple_simulation/3_tokenize_data.py

from osl_foundation import load_model
import logging
_logger = logging.getLogger("osl-foundation")
    
tf_ops.gpu_growth()

# ---------- Directories ---------- #

data_dir = f"{dev_dir}/results/osl-foundation/simple_simulation"
tokenizer_dir = f"{data_dir}/tokenizer"

tokenized_data_dir = f"{data_dir}/tokenized_data"
tokenized_data_tf_dir = f"{data_dir}/tokenized_data_tfrecords"
os.makedirs(tokenized_data_dir, exist_ok=True)
os.makedirs(tokenized_data_tf_dir, exist_ok=True)

# Load data
data_files = sorted(glob(f"{data_dir}/*.npy"))
data = Data(data_files,
            
            sampling_frequency=100)
methods = {
    "filter": {"low_freq": 4, "high_freq": 40, "use_raw": True},
    "standardize": {},
}
data.prepare(methods)

if True:
    # ---------- Tokenise ---------- #

    tokenizer = load_model(tokenizer_dir)
    tokenized_data = tokenizer.tokenize_data(data)

    for i, token_data in enumerate(tokenized_data):
        np.save(
            f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
            token_data,
        )

    mtc = np.load(f"{data_dir}/ground_truth/mode_tcs.npy")

    tokenized_data = Data(tokenized_data)
    tokenized_data.add_session_labels(
        "session_id", np.arange(tokenized_data.n_sessions), "categorical"
    )
    #tokenized_data.add_extra_channel("mode_0", [mtc[0]] * tokenized_data.n_sessions)
    #tokenized_data.add_extra_channel("mode_1", [mtc[1]] * tokenized_data.n_sessions)
    tokenized_data.save_tfrecord_dataset(
        tfrecord_dir=tokenized_data_tf_dir,
        sequence_length=101,
        overwrite=True,
        validation_split=0.1,
    )

    data.delete_dir()

tokenizer = load_model(tokenizer_dir)
start = 0
end = data.n_sessions 

for i, dat in enumerate(data):
    if i >= start and i < end:
        _logger.info(f"Tokenizing session {i}")
        token_data = tokenizer.tokenize_data(dat)  
        np.save(
            f"{tokenized_data_dir}/x_{i:0{len(str(data.n_sessions))}}.npy",
            token_data[0],
        )

data.delete_dir()
