import os

from osl_dynamics.inference import tf_ops

from osl_foundation import create_model

tf_ops.gpu_growth()

plot_dir = "plots/ephys_gpt"
os.makedirs(plot_dir, exist_ok=True)

generator_config = f"""
    model_config:
        name: ephys_gpt
        sequence_length: 80
        n_channels: 4
        input_parameters:
            embedding_dim: 100
            n_tokens: 32
            token_embedding_dim: 100
            pos_embedding_dim: 100
            channel_embedding_dim: 20
        decoder_parameters:
            n_layers: 4
            n_heads: 4
            model_dim: 100
            latent_sequence_length: 40
            n_patches: 20
            patch_length: 4
            unpatched_length: 12
            channel_attention_dropout: 0.0
            within_channel_attention_dropout: 0.0
            feed_forward_dim: 100
        loss_parameters:
            loss_sequence_length: 8
    training_config:
        optimizer:
            learning_rate: 0.0002
        batch_size: 8
        n_epochs: 40
"""

generator = create_model(generator_config)
generator.summary()

generator.fit_random_tokens(n_samples=256000, validation_split=0.1)
generator.plot_history(plot_dir=plot_dir)
