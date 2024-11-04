import os
from glob import glob

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data

from osl_foundation import create_model

tf_ops.gpu_growth()

data_dir = "sim_data"
plot_dir = "plots/generator"
tokenizer_dir = "models/tokenizer"
generator_dir = "models/generator"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(generator_dir, exist_ok=True)

# Load data
data = Data(sorted(glob(f"{data_dir}/*.npy")), use_tfrecord=True)

generator_config = f"""
    model_config:
        name: ephys_gpt
        tokenizer_path: {tokenizer_dir}
        sequence_length: 80
        n_channels: {data.n_channels}
        input_parameters:
            embedding_dim: 100
            n_tokens: 128
            token_embedding_dim: 100
            pos_embedding_dim: 100
            channel_embedding_dim: 100
        decoder_parameters:
            n_layers: 4
            n_heads: 4
            model_dim: 100
            latent_sequence_length: 40
            n_patches: 20
            patch_length: 4
            unpatched_length: 16
            channel_attention_dropout: 0.0
            within_channel_attention_dropout: 0.0
            feed_forward_dim: 100
            dropout: 0.2
            # norm_type: group
            # n_groups: 8
        loss_parameters:
            loss_sequence_length: 8
    training_config:
        optimizer:
            learning_rate: 0.0001
        batch_size: 32
        n_epochs: 40
        checkpoint:
            save_freq: 5
            checkpoint_dir: {generator_dir}
"""

generator = create_model(generator_config)
generator.summary()
generator.save_config(generator_dir)

generator.fit(
    data,
    validation_split=0.1,
    use_tfrecord=True,
    n_jobs=8,
    step_size=generator.config.model_config.sequence_length // 4,
)
generator.plot_history(plot_dir=plot_dir, keyword="loss")
generator.plot_history(plot_dir=plot_dir, keyword="accuracy")

# Clean up directories
data.delete_dir()
