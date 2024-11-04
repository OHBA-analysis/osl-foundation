import os

from osl_dynamics.inference import tf_ops

from osl_foundation import create_model

tf_ops.gpu_growth()

plot_dir = "plots/ephys_gpt"
os.makedirs(plot_dir, exist_ok=True)

generator_config = f"""
    model_config:
        name: ephys_gpt
        sequence_length: 256
        n_channels: 10
        input_parameters:
            embedding_dim: 256
            n_tokens: 8
        decoder_parameters:
            n_layers: 4
            n_heads: 8
            model_dim: 256
            latent_sequence_length: 128
            n_patches: 64
            patch_length: 4
            unpatched_length: 32
            channel_attention_dropout: 0.0
            within_channel_attention_dropout: 0.0
            feed_forward_dim: 256
        loss_parameters:
            loss_sequence_length: 64
    training_config:
        optimizer:
            learning_rate: 0.001
        batch_size: 16
        n_epochs: 20
"""

generator = create_model(generator_config)
generator.summary()

generator.fit_random_tokens(n_samples=256000, validation_split=0.1)
generator.plot_history(plot_dir=plot_dir, keyword="loss")
generator.plot_history(plot_dir=plot_dir, keyword="accuracy")
