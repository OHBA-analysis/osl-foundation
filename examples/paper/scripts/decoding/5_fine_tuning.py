import numpy as np
from osl_dynamics.inference import tf_ops
from osl_dynamics.data import load_tfrecord_dataset
from osl_foundation import create_model

tf_ops.gpu_growth()

decoding_model = create_model("../../models/decoding_model/config.yml")
decoding_model.model.get_layer("decoder").trainable = True
decoding_model.model.get_layer("prediction_head").trainable = True
decoding_model.compile()
decoding_model.summary()

# Only sessions 1-5 of subjects 1-18 are used for training.
session_id = np.tile([1, 2, 3, 4, 5, 6], 19)
subject_id = np.repeat(np.arange(1, 20), 6)

train_mask = [False] * 6 * 19
for i in range(len(train_mask)):
    if session_id[i] == 6:
        continue

    if subject_id[i] == 19:
        continue

    train_mask[i] = True

train_data = load_tfrecord_dataset(
    "../../data/wh_tfrecords",
    batch_size=decoding_model.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
    keep=list(np.where(train_mask)[0]),
)
val_data = load_tfrecord_dataset(
    "../../data/wh_tfrecords",
    batch_size=decoding_model.config.training_config.batch_size,
    buffer_size=2000,
    drop_last_batch=True,
    concatenate=True,
    shuffle=False,
    keep=list(np.where(np.logical_not(train_mask))[0]),
)
decoding_model.fit(
    train_data,
    validation_data=val_data,
    tokenize=False,
)
