from typing import List

import tensorflow as tf
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.utils.misc import get_argument, replace_argument

from osl_foundation.models.base import BaseModel
from osl_foundation.models.tokenizers import OSLTokenizer, load_tokenizer
from osl_foundation.config.generator_config import Label
from osl_foundation.inference.layers import (
    IdentityLayer,
    MultiHeadPASSTALayer,
    NormalizationLayer,
)


class ShiftTokenLayer(tf.keras.layers.Layer):
    """
    This layer added a start of sequence token and removes the end of sequence token.

    Parameters
    ----------
    n_tokens : int
        Number of tokens in the vocabulary.
    """

    def __init__(self, n_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.concatenate_layer = tf.keras.layers.Concatenate(axis=1)

    def call(self, x):
        # Add start of sequence token
        x = self.concatenate_layer([tf.zeros_like(x[:, :1]) + self.n_tokens, x])

        # Remove end of sequence token
        return x[:, :-1]


class CrossEntropyLossLayer(tf.keras.layers.Layer):
    """
    Layer for calculating the cross-entropy loss.

    Parameters
    ----------
    loss_sequence_length : int
        Number of tokens to calculate the loss for.
    """

    def __init__(
        self,
        loss_sequence_length: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_sequence_length = loss_sequence_length
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, y_true, y_pred, **kwargs):
        # y_true.shape = (batch_size, sequence_length, n_channels)
        # y_pred.shape = (batch_size, latent_sequence_length, n_channels, n_tokens)

        # Only calculate loss for the last loss_sequence_length tokens
        y_true = y_true[:, -self.loss_sequence_length :]
        y_pred = y_pred[:, -self.loss_sequence_length :]

        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=True
            )
        )
        accuracy = self.accuracy(y_true, y_pred)
        self.add_loss(loss)
        self.add_metric(accuracy, name="accuracy")
        return tf.expand_dims(loss, -1), y_pred


class InputEmbeddingLayer(tf.keras.layers.Layer):
    """
    Layer for creating the input embeddings for the decoder.
    This returns the sum of the token, position, channel, and extra embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the embeddings.
    n_tokens : int
        Number of tokens in the vocabulary.
    sequence_length : int
        Length of the input sequence.
    n_channels : int
        Number of channels in the input data.
    token_embedding_dim : int, optional
        Dimension of the token embeddings. If None, it is set to embedding_dim.
    pos_embedding_dim : int, optional
        Dimension of the position embeddings. If None, it is set to embedding_dim.
    channel_embedding_dim : int, optional
        Dimension of the channel embeddings. If None, it is set to embedding_dim.
    extra_labels : List[Label], optional
        List of extra labels to add to the embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_tokens: int,
        sequence_length: int,
        n_channels: int,
        token_embedding_dim: int = None,
        pos_embedding_dim: int = None,
        channel_embedding_dim: int = None,
        extra_labels: List[Label] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.token_embedding_dim = token_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.extra_labels = extra_labels

        # ---------- Initialize layers ----------

        # The token embedding layer
        if token_embedding_dim is None:
            token_embedding_dim = embedding_dim

        self.token_embedding_layer = tf.keras.layers.Embedding(
            n_tokens + 1, token_embedding_dim
        )
        # Token embedding output layer
        if token_embedding_dim != embedding_dim:
            self.token_embedding_output_layer = tf.keras.layers.Dense(embedding_dim)
        else:
            self.token_embedding_output_layer = IdentityLayer()

        # The position embedding layer
        if pos_embedding_dim is None:
            pos_embedding_dim = embedding_dim

        self.position_embedding_layer = tf.keras.layers.Embedding(
            sequence_length, pos_embedding_dim
        )
        # Position embedding output layer
        if pos_embedding_dim != embedding_dim:
            self.position_embedding_output_layer = tf.keras.layers.Dense(embedding_dim)
        else:
            self.position_embedding_output_layer = IdentityLayer()

        # The channel embedding layer
        if channel_embedding_dim is None:
            channel_embedding_dim = embedding_dim

        self.channel_embedding_layer = tf.keras.layers.Embedding(
            n_channels, channel_embedding_dim
        )
        # Channel embedding output layer
        if channel_embedding_dim != embedding_dim:
            self.channel_embedding_output_layer = tf.keras.layers.Dense(embedding_dim)
        else:
            self.channel_embedding_output_layer = IdentityLayer()

        # The extra embedding layers
        self.extra_embedding_layers = []
        self.extra_embedding_output_layers = []
        for label in extra_labels:
            label_dim = label.label_dim or embedding_dim
            n_classes = label.n_classes
            self.extra_embedding_layers.append(
                tf.keras.layers.Embedding(n_classes, label_dim)
            )
            if label_dim != embedding_dim:
                self.extra_embedding_output_layers.append(
                    tf.keras.layers.Dense(embedding_dim)
                )
            else:
                self.extra_embedding_output_layers.append(IdentityLayer())

    def call(self, inputs, **kwargs):
        data, extra_labels = inputs
        # data.shape = (batch_size, sequence_length, n_channels)
        # extra_labels[0].shape = (batch_size, sequence_length, n_channels)

        # ---------- Token embeddings ---------- #
        x = data
        # x.shape = (batch_size, sequence_length, n_channels)

        embeddings = self.token_embedding_output_layer(self.token_embedding_layer(x))
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Position embeddings ---------- #
        positions = tf.constant(
            np.arange(self.sequence_length)[None, :, None], dtype=tf.int32
        )
        # positions.shape = (1, sequence_length, 1)

        embeddings += self.position_embedding_output_layer(
            self.position_embedding_layer(positions)
        )
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Channel embeddings ---------- #
        channels = tf.constant(
            np.arange(self.n_channels)[None, None, :], dtype=tf.int32
        )
        # channels.shape = (1, 1, n_channels)

        embeddings += self.channel_embedding_output_layer(
            self.channel_embedding_layer(channels)
        )
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Extra embeddings ---------- #
        for label, layer, output_layer in zip(
            extra_labels,
            self.extra_embedding_layers,
            self.extra_embedding_output_layers,
        ):
            embeddings += output_layer(layer(label))

        return embeddings


class DecoderLayer(tf.keras.layers.Layer):
    """
    Layer for the decoder of the EphysGPT model.

    Parameters
    ----------
    n_layers : int
        Number of layers in the decoder.
    n_heads : int
        Number of heads in the multi-head attention layers.
    model_dim : int
        Dimension of the model.
    embedding_dim : int
        Dimension of the embeddings.
    n_channels : int
        Number of channels in the input data.
    sequence_length : int
        Length of the input sequence.
    latent_sequence_length : int
        Length of the latent sequence.
    n_patches : int
        Number of patches in the sequence.
    patch_length : int
        Length of the patches.
    unpatched_length : int
        Length of the unpatched sequence.
    channel_attention_dropout : float
        Dropout rate for the channel attention layer.
    within_channel_attention_dropout : float
        Dropout rate for the within-channel attention layer.
    feed_forward_dim : int
        Dimension of the feed-forward layer.
    feed_forward_activation : str
        Activation function for the feed-forward layer.
    dropout : float
        Dropout rate for the feed-forward layer.
    norm_type : str
        Type of normalization layer to use.
    n_groups : int, optional
        Number of groups for group normalization.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        model_dim: int,
        embedding_dim: int,
        n_channels: int,
        sequence_length: int,
        latent_sequence_length: int,
        n_patches: int,
        patch_length: int,
        unpatched_length: int,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        feed_forward_dim: int,
        feed_forward_activation: str,
        dropout: float,
        norm_type: str,
        n_groups: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers

        # ---------- Initialize layers ----------

        # Multi-head attention layers (Special first layer)
        self.attention_layers = [
            MultiHeadPASSTALayer(
                n_heads,
                model_dim,
                embedding_dim,
                n_channels,
                sequence_length,
                latent_sequence_length,
                n_patches,
                patch_length,
                unpatched_length,
                channel_attention_dropout,
                within_channel_attention_dropout,
            )
        ]
        for _ in range(n_layers - 1):
            self.attention_layers.append(
                MultiHeadPASSTALayer(
                    n_heads,
                    model_dim,
                    embedding_dim,
                    n_channels,
                    latent_sequence_length,
                    latent_sequence_length,
                    latent_sequence_length // patch_length,
                    patch_length,
                    unpatched_length,
                    channel_attention_dropout,
                    within_channel_attention_dropout,
                )
            )
        # for _ in range(n_layers - 1):
        #     self.attention_layers.append(
        #         MultiHeadSSTALayer(
        #             n_heads,
        #             model_dim,
        #             embedding_dim,
        #             n_channels,
        #             latent_sequence_length,
        #             channel_attention_dropout,
        #             within_channel_attention_dropout,
        #         )
        #     )

        # Normalization layers

        # Normalization 1: after attention but before feed-forward layer
        self.normalization_layers_1 = [
            NormalizationLayer(norm_type, n_groups) for _ in range(n_layers)
        ]

        # Normalization 2: after feed-forward layer
        self.normalization_layers_2 = [
            NormalizationLayer(norm_type, n_groups) for _ in range(n_layers)
        ]

        # Feed-forward layers
        self.feed_forward_layers = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        feed_forward_dim, activation=feed_forward_activation
                    ),
                    tf.keras.layers.Dense(model_dim),
                    tf.keras.layers.Dropout(dropout),
                ]
            )
            for _ in range(n_layers)
        ]

    def call(self, inputs, training=None, **kwargs):
        # inputs.shape = (batch_size, sequence_length, n_channels, embedding_dim)
        x = inputs
        for i in range(self.n_layers):

            # Residual connection
            x_residual = x

            # Attention layer
            x = self.attention_layers[i](x, training=training)

            # Add to residual
            x = x + x_residual[:, -tf.shape(x)[1] :]

            # Normalization 1
            x = self.normalization_layers_1[i](x, training=training)

            # Residual connection
            x_residual = x

            # Feed-forward layer
            x = self.feed_forward_layers[i](x, training=training)

            # Add to residual
            x = x + x_residual

            # Normalization 2
            x = self.normalization_layers_2[i](x, training=training)

        # x.shape = (batch_size, latent_sequence_length, n_channels, model_dim)

        return x


class EphysGPT(BaseModel):
    """
    EphysGPT model for electrophysiology data.

    Parameters
    ----------
    config : Config
        Config object.
    """

    def build_model(self) -> None:
        self.tokenizer = self._load_tokenizer()
        self.model = self._build_model()

    def fit(
        self,
        *args,
        use_tfrecord: bool = False,
        n_jobs: int = 1,
        step_size: int = None,
        **kwargs,
    ) -> None:
        """
        First tokenizes the data and then fits the model.

        Parameters
        ----------
        *args : list
            Positional arguments to pass to the model's fit method.
        use_tfrecord : bool, optional
            Whether to use tfrecord for the tokenized data.
        n_jobs : int, optional
            Number of jobs to use for the tokenized data.
        step_size : int, optional
            Step size when creating the dataset.
        **kwargs : dict
            Keyword arguments to pass to the model's fit method.
        """
        x = get_argument(self.model.fit, "x", args, kwargs)

        # Tokenise the data and build Data object
        tokenized_x = self.tokenizer.tokenize_data(x)[0]
        tokenized_x = Data(
            tokenized_x,
            store_dir=f"{getattr(x, 'store_dir', 'tmp')}/tokenized",
            use_tfrecord=use_tfrecord,
            n_jobs=n_jobs,
        )

        validation_split = get_argument(
            self.model.fit, "validation_split", args, kwargs
        )
        dataset = self.make_dataset(
            tokenized_x,
            shuffle=True,
            concatenate=True,
            step_size=step_size,
            drop_last_batch=True,
            validation_split=validation_split,
        )
        if validation_split is None:
            args, kwargs = replace_argument(
                self.model.fit,
                "x",
                dataset,
                args,
                kwargs,
            )
        else:
            args, kwargs = replace_argument(
                self.model.fit,
                "x",
                dataset[0],
                args,
                kwargs,
            )
            args, kwargs = replace_argument(
                self.model.fit,
                "validation_data",
                dataset[1],
                args,
                kwargs,
            )

        super().fit(*args, **kwargs)

    def _load_tokenizer(self) -> OSLTokenizer:
        """Load a trained tokenizer."""
        tokenizer_path = self.config.model_config.tokenizer_path
        return load_tokenizer(tokenizer_path)

    def _build_model(self) -> tf.keras.Model:
        config = self.config.model_config

        # ---------- Inputs ---------- #
        true_token = tf.keras.layers.Input(
            shape=(config.sequence_length, config.n_channels),
            dtype=tf.int32,
            name="data",
        )

        extra_labels = []
        for label in config.extra_labels:
            extra_labels.append(
                tf.keras.layers.Input(
                    shape=(config.sequence_length, config.n_channels),
                    dtype=tf.int32,
                    name=label.name,
                )
            )

        # ---------- Initialize layers ---------- #
        shift_token_layer = ShiftTokenLayer(config.n_tokens, name="shift_token")
        input_embedding_layer = InputEmbeddingLayer(
            config.embedding_dim,
            config.n_tokens,
            config.sequence_length,
            config.n_channels,
            config.token_embedding_dim,
            config.pos_embedding_dim,
            config.channel_embedding_dim,
            config.extra_labels,
            name="input_embedding",
        )
        decoder_layer = DecoderLayer(
            config.n_layers,
            config.n_heads,
            config.model_dim,
            config.embedding_dim,
            config.n_channels,
            config.sequence_length,
            config.latent_sequence_length,
            config.n_patches,
            config.patch_length,
            config.unpatched_length,
            config.channel_attention_dropout,
            config.within_channel_attention_dropout,
            config.feed_forward_dim,
            config.feed_forward_activation,
            config.dropout,
            config.norm_type,
            config.n_groups,
            name="decoder",
        )
        prediction_head_layer = tf.keras.layers.Dense(
            config.n_tokens, name="prediction_head"
        )
        loss_layer = CrossEntropyLossLayer(
            config.loss_sequence_length,
            name="loss",
        )

        # ---------- Forward Pass ---------- #

        # Shift the tokens
        x = shift_token_layer(true_token)
        # x.shape = (batch_size, sequence_length, n_channels)

        # Get the input embeddings to the decoder
        x = input_embedding_layer([x, extra_labels])
        # x.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # Run the decoder
        x = decoder_layer(x)
        # x.shape = (batch_size, latent_sequence_length, n_channels, model_dim)

        # Get the prediction of the next token
        x = prediction_head_layer(x)
        # x.shape = (batch_size, latent_sequence_length, n_channels, n_tokens)

        # Calculate the loss
        loss, x_pred = loss_layer(true_token, x)

        # ---------- Model ---------- #
        return tf.keras.Model(
            inputs=[true_token] + extra_labels, outputs=[loss, x_pred], name="ephys_gpt"
        )
