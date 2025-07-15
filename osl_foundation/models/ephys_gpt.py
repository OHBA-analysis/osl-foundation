from typing import List, Union, Dict
import logging

import tensorflow as tf
import numpy as np
from tqdm.auto import trange

from osl_dynamics.data import Data
from osl_dynamics.utils.misc import get_argument, replace_argument

from osl_foundation.models.base import BaseModel
from osl_foundation.config import get_config
from osl_foundation.models.tokenizers import (
    OSLTokenizer,
    MuTransformTokenizer,
    StandardQuantileTokenizer,
)
from osl_foundation.config.generator_config import Label
from osl_foundation.inference.layers import (
    IdentityLayer,
    SinusoidalPositionalEncodingLayer,
    MultiHeadPASSTALayer,
    NormalizationLayer,
    PositionEmbedding,
    SessionChannelEmbeddingLayer,
)
from osl_foundation.utils.sampling import sample_from_logits
from osl_foundation.utils.testing import create_random_tokens

_logger = logging.getLogger("osl-foundation")


class ShiftTokenLayer(tf.keras.layers.Layer):
    """
    This layer added a start of sequence token and removes the end of sequence token.
    """

    def call(self, x):
        x_input = x[:, :-1]
        # x_input.shape = (batch_size, sequence_length, n_channels)

        y_true = x[:, 1:]
        # y_true.shape = (batch_size, sequence_length, n_channels)

        return x_input, y_true


class CrossEntropyLossLayer(tf.keras.layers.Layer):
    """
    Layer for calculating the cross-entropy loss.

    Parameters
    ----------
    loss_sequence_length : int
        Number of tokens to calculate the loss for.
    top_k : list, optional
        List of top k values to calculate the accuracy for.
        By default only top 1 accuracy is calculated.
    class_weights : dict, optional
        Weights of the classes for the cross-entropy loss.
    """

    def __init__(
        self,
        loss_sequence_length: int,
        top_k: list = None,
        class_weights: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_sequence_length = loss_sequence_length
        self.top_k = top_k or [1]
        self.class_weights = class_weights

    def _macro_f1(self, y_true, y_pred_logits) -> tf.Tensor:
        """
        Compute macro-averaged F1 for a binary (2-class) problem.

        y_true:  (...,)               int32  in {0,1}
        y_pred_logits: (..., 2)       float32 in raw logits
        """
        # Convert logits → predicted class id
        y_pred = tf.argmax(y_pred_logits, axis=-1, output_type=tf.int32)

        # Flatten so we don’t need to worry about time/channel dims
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Per-class true-positives, false-positives, false-negatives
        tp = []
        fp = []
        fn = []
        for c in (0, 1):
            y_true_c = tf.equal(y_true, c)
            y_pred_c = tf.equal(y_pred, c)

            tp.append(
                tf.reduce_sum(
                    tf.cast(tf.logical_and(y_true_c, y_pred_c), self.dtype)
                )
            )
            fp.append(
                tf.reduce_sum(
                    tf.cast(tf.logical_and(tf.logical_not(y_true_c), y_pred_c), self.dtype)
                )
            )
            fn.append(
                tf.reduce_sum(
                    tf.cast(tf.logical_and(y_true_c, tf.logical_not(y_pred_c)), self.dtype)
                )
            )

        tp = tf.stack(tp)
        fp = tf.stack(fp)
        fn = tf.stack(fn)

        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall    = tf.math.divide_no_nan(tp, tp + fn)
        f1        = tf.math.divide_no_nan(2 * precision * recall, precision + recall)

        # macro-average across the two classes
        return tf.reduce_mean(f1), precision, recall

    def call(self, inputs, training=None, **kwargs):
        y_pred, y_true = inputs
        # y_true.shape = (batch_size, sequence_length, n_channels)
        # y_pred.shape = (batch_size, latent_sequence_length, n_channels, n_tokens)

        # Only calculate loss for the last loss_sequence_length tokens
        y_pred = y_pred[:, -self.loss_sequence_length :]
        y_true = y_true[:, -self.loss_sequence_length :]
        # y_true.shape = (batch_size, loss_sequence_length, n_channels)
        # y_pred.shape = (batch_size, loss_sequence_length, n_channels, n_tokens)

        loss = tf.keras.metrics.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        # Apply class weights if provided
        if training and self.class_weights is not None:
            weights = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(y_pred.shape[-1])],
                dtype=loss.dtype,
            )
            weights = tf.gather(weights, y_true)
            loss = loss * weights  # shape: (batch_size, loss_sequence_length, n_channels)

        self.add_loss(loss)
        for k in self.top_k:
            accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                y_true, y_pred, k=k
            )
            self.add_metric(accuracy, name=f"top_{k}")

        if y_pred.shape[-1] == 2:  # only add when prediction_type == "binary"
            f1_macro, precision, recall = self._macro_f1(y_true, y_pred)
            self.add_metric(f1_macro, name="f1_macro")
            self.add_metric(precision[0], name="precision_c0")
            self.add_metric(precision[1], name="precision_c1")
            self.add_metric(recall[0], name="recall_c0")
            self.add_metric(recall[1], name="recall_c1")

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
    n_sessions : int
        Number of sessions or subjects in the input data.
    token_embedding_dim : int, optional
        Dimension of the token embeddings. If None, it is set to embedding_dim.
    pos_embedding_dim : int, optional
        Dimension of the position embeddings. If None, it is set to embedding_dim.
    pos_embedding_type : str, optional
        Type of the position embeddings. Defaults to "absolute", which uses learned
        absolute position embeddings.
    channel_embedding_dim : int, optional
        Dimension of the channel embeddings. If None, it is set to embedding_dim.
    channel_embedding_type : str, optional
        Type of the channel embeddings. Defaults to "group", which broadcasts the
        channel embeddings across the sequence length.
    prediction_type : str, optional
        Type of the prediction head. Defaults to "token", which predicts the next token.
    extra_labels : List[Label], optional
        List of extra labels to add to the embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_tokens: int,
        sequence_length: int,
        n_channels: int,
        n_sessions: int = None,
        token_embedding_dim: int = None,
        pos_embedding_dim: int = None,
        pos_embedding_type: str = "absolute",
        channel_embedding_dim: int = None,
        channel_embedding_type: str = "group",
        prediction_type: str = "token",
        extra_labels: List[Label] = None,
        pretrained_layer: tf.keras.layers.Layer = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.n_sessions = n_sessions
        self.token_embedding_dim = token_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.pos_embedding_type = pos_embedding_type
        self.channel_embedding_dim = channel_embedding_dim
        self.channel_embedding_type = channel_embedding_type
        self.prediction_type = prediction_type
        self.extra_labels = extra_labels
        self.use_pretrained_layer = pretrained_layer is not None

        self.extra_labels_name = [label.name for label in extra_labels]

        # ---------- Initialize layers ----------

        if pretrained_layer is None:
            # The token embedding layer
            self.token_embedding_layer = tf.keras.layers.Embedding(
                n_tokens, token_embedding_dim or embedding_dim
            )
            self.token_embedding_output_layer = (
                IdentityLayer()
                if token_embedding_dim is None
                else tf.keras.layers.Dense(embedding_dim)
            )

            # The position embedding layer
            if pos_embedding_type == "sinusoidal":
                self.position_embedding_layer = SinusoidalPositionalEncodingLayer(
                    sequence_length=sequence_length,
                )
                self.position_embedding_output_layer = IdentityLayer()
            elif pos_embedding_type == "absolute":
                self.position_embedding_layer = PositionEmbedding(
                    sequence_length=sequence_length, trainable=True
                )
                self.position_embedding_output_layer = (
                    IdentityLayer()
                    if pos_embedding_dim is None
                    else tf.keras.layers.Dense(embedding_dim)
                )

            # The channel embedding layer
            if channel_embedding_type == "group":
                self.channel_embedding_layer = PositionEmbedding(
                    sequence_length=n_channels, trainable=True
                )
            elif channel_embedding_type == "session":
                self.channel_embedding_layer = SessionChannelEmbeddingLayer(
                    session_dim=n_sessions,
                    channel_dim=n_channels,
                    trainable=True,
                )
            
            self.channel_embedding_output_layer = (
                IdentityLayer()
                if channel_embedding_dim is None
                else tf.keras.layers.Dense(embedding_dim)
            )
        else:
            self.token_embedding_layer = pretrained_layer.token_embedding_layer
            self.token_embedding_output_layer = (
                pretrained_layer.token_embedding_output_layer
            )
            self.position_embedding_layer = pretrained_layer.position_embedding_layer
            self.position_embedding_output_layer = (
                pretrained_layer.position_embedding_output_layer
            )
            self.channel_embedding_layer = pretrained_layer.channel_embedding_layer
            self.channel_embedding_output_layer = (
                pretrained_layer.channel_embedding_output_layer
            )

        # The extra embedding layers
        self.extra_embedding_layers = []
        self.extra_embedding_output_layers = []
        for label in extra_labels:
            n_classes = label.n_classes
            self.extra_embedding_layers.append(
                tf.keras.layers.Embedding(n_classes, label.label_dim or embedding_dim)
            )
            self.extra_embedding_output_layers.append(
                IdentityLayer()
                if label.label_dim is None
                else tf.keras.layers.Dense(embedding_dim)
            )

    def call(self, inputs, training=None, **kwargs):
        if self.use_pretrained_layer:
            training_1 = False
        else:
            training_1 = training

        data, extra_labels = inputs
        # data.shape = (batch_size, sequence_length, n_channels)
        # extra_labels[0].shape = (batch_size, sequence_length + 1)

        # ---------- Token embeddings ---------- #
        x = data
        # x.shape = (batch_size, sequence_length, n_channels)

        embeddings = self.token_embedding_output_layer(
            self.token_embedding_layer(x, training=training_1), training=training_1
        )
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Position embeddings ---------- #
        if self.pos_embedding_type in ["absolute", "sinusoidal"]:
            positions = tf.transpose(
                tf.zeros(tf.concat([tf.shape(x), [self.pos_embedding_dim]], axis=0)),
                perm=[0, 2, 1, 3],
            )
            # positions.shape = (batch_size, n_channels, sequence_length, pos_embedding_dim)

            embeddings += tf.transpose(
                self.position_embedding_output_layer(
                    self.position_embedding_layer(positions, training=training_1),
                    training=training_1,
                ),
                perm=[0, 2, 1, 3],
            )
            # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Channel embeddings ---------- #
        channels = tf.zeros(
            tf.concat([tf.shape(x), [self.channel_embedding_dim]], axis=0)
        )
        # channels.shape = (batch_size, sequence_length, n_channels, channel_embedding_dim)

        if self.channel_embedding_type == "group":
            embeddings += self.channel_embedding_output_layer(
                self.channel_embedding_layer(channels, training=training_1),
                training=training_1,
            )
        elif self.channel_embedding_type == "session":
            # Extract the subject/session label tensor from extra_labels
            label_dict = {
                name: tensor for name, tensor in zip(self.extra_labels_name, extra_labels)
            }
            session_labels = None
            if "session_id" in label_dict:
                session_labels = label_dict["session_id"]
            elif "subject_id" in label_dict:
                session_labels = label_dict["subject_id"]
            else:
                raise ValueError("Neither 'session_id' nor 'subject_id' found in extra_labels.")
            if self.prediction_type == "token":
                session_labels = session_labels[:, :-1]
            embeddings += self.channel_embedding_output_layer(
                self.channel_embedding_layer(channels, session_labels, training=training_1),
                training=training_1,
            )
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Extra embeddings ---------- #
        for label, layer, output_layer in zip(
            extra_labels,
            self.extra_embedding_layers,
            self.extra_embedding_output_layers,
        ):
            # label.shape = (batch_size, sequence_length + 1)

            if self.prediction_type == "token":
                label = tf.expand_dims(label[:, :-1], -1)
            elif self.prediction_type == "binary":
                label = tf.expand_dims(label, -1)
            # label.shape = (batch_size, sequence_length, 1)

            embeddings += output_layer(
                layer(label, training=training), training=training
            )

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
    pos_embedding_type : str
        Type of positional embedding to use.
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
    prediction_type : str, optional
        Type of the prediction head. Defaults to "token", which predicts the next token.
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
        pos_embedding_type: str,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        feed_forward_dim: int,
        feed_forward_activation: str,
        dropout: float,
        norm_type: str,
        n_groups: int = None,
        prediction_type: str = "token",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers

        # ---------- Initialize layers ----------

        # Input dropout layer
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout)
        self.input_projection_layer = (
            IdentityLayer()
            if model_dim == embedding_dim
            else tf.keras.layers.Dense(model_dim)
        )

        # Multi-head attention layers (Special first layer)
        self.attention_layers = [
            MultiHeadPASSTALayer(
                n_heads,
                model_dim,
                n_channels,
                sequence_length if i == 0 else latent_sequence_length,
                latent_sequence_length,
                n_patches if i == 0 else latent_sequence_length // patch_length,
                patch_length,
                unpatched_length,
                pos_embedding_type,
                channel_attention_dropout,
                within_channel_attention_dropout,
                prediction_type,
            )
            for i in range(n_layers)
        ]
        self.attention_dropout_layers = [
            tf.keras.layers.Dropout(dropout) for _ in range(n_layers)
        ]

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
        x = self.input_projection_layer(x)
        # x.shape = (batch_size, sequence_length, n_channels, model_dim)

        x = self.input_dropout_layer(x, training=training, **kwargs)

        for i in range(self.n_layers):
            # Input dropout

            # Residual connection
            x_residual = x

            # Attention layer
            x = self.attention_layers[i](x, training=training, **kwargs)
            x = self.attention_dropout_layers[i](x, training=training, **kwargs)

            # Add to residual
            x = x + x_residual[:, -tf.shape(x)[1] :]

            # Normalization 1
            x = self.normalization_layers_1[i](x, training=training, **kwargs)

            # Residual connection
            x_residual = x

            # Feed-forward layer
            x = self.feed_forward_layers[i](x, training=training, **kwargs)

            # Add to residual
            x = x + x_residual

            # Normalization 2
            x = self.normalization_layers_2[i](x, training=training, **kwargs)

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
        self.pretrained_model = self._load_pretrained_model()
        self.tokenizer = self._load_tokenizer()
        self.model = self._build_model()

    def fit(
        self,
        *args,
        tokenize: bool = True,
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
        tokenize: bool, optional
            Whether to tokenize data before fitting model.
        use_tfrecord : bool, optional
            Whether to use tfrecord for the tokenized data.
        n_jobs : int, optional
            Number of jobs to use for the tokenized data.
        step_size : int, optional
            Step size when creating the dataset.
        **kwargs : dict
            Keyword arguments to pass to the model's fit method.
        """
        if tokenize:
            x = get_argument(self.model.fit, "x", args, kwargs)

            # Tokenise the data and build Data object
            tokenized_x = self.tokenizer.tokenize_data(x)
            tokenized_x = Data(
                tokenized_x,
                store_dir=f"{getattr(x, 'store_dir', 'tmp')}/tokenized",
                use_tfrecord=use_tfrecord,
                n_jobs=n_jobs,
            )
            tokenized_x.session_labels = x.session_labels
        else:
            tokenized_x = get_argument(self.model.fit, "x", args, kwargs)

        validation_split = get_argument(
            self.model.fit, "validation_split", args, kwargs
        )

        # If step_per_epoch is passed, repeat the dataset indefinitely
        steps_per_epoch = get_argument(self.model.fit, "steps_per_epoch", args, kwargs)
        repeat_count = 1 if steps_per_epoch is None else -1

        dataset = self.make_dataset(
            tokenized_x,
            shuffle=True,
            concatenate=True,
            step_size=step_size,
            drop_last_batch=True,
            validation_split=validation_split,
            repeat_count=repeat_count,
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

    def fit_random_tokens(
        self,
        n_samples: int,
        validation_split: float,
        use_tfrecord: bool = False,
    ) -> None:
        """
        First generates random tokens and then fits the model on them.

        Parameters
        ----------
        n_samples : int
            Number of samples per sequence to generate.
        validation_split : float
            Fraction of the data to use for validation.
        use_tfrecord : bool, optional
            Whether to use tfrecord for the tokenized data.
        """
        config = self.config.model_config

        # This function is for testing purposes only
        # Raise error if extra labels are present
        if config.extra_labels:
            raise ValueError("Extra labels are not supported for fit_random_tokens().")

        # validation_split cannot be None
        if validation_split is None:
            raise ValueError("validation_split cannot be None.")

        # Generate random tokens
        random_tokens = create_random_tokens(
            config.n_tokens, n_samples, config.n_channels
        )

        # Build Data object
        data = Data(
            random_tokens, store_dir="tmp_random_tokens", use_tfrecord=use_tfrecord
        )
        dataset = self.make_dataset(
            data,
            shuffle=True,
            concatenate=True,
            drop_last_batch=True,
            validation_split=validation_split,
        )
        super().fit(dataset[0], validation_data=dataset[1])
        data.delete_dir()

    def _load_tokenizer(self) -> OSLTokenizer:
        """Load a trained tokenizer."""
        if self.pretrained_model is not None:
            tokenizer = self.pretrained_model.tokenizer
        else:
            TOKENIZERS = {
                "osl_tokenizer": OSLTokenizer,
                "mu_transform_tokenizer": MuTransformTokenizer,
                "standard_quantile_tokenizer": StandardQuantileTokenizer,
            }
            tokenizer_path = self.config.model_config.tokenizer_path
            if tokenizer_path is None:
                return None

            tokenizer_config = get_config(f"{tokenizer_path}/config.yml")
            tokenizer_name = tokenizer_config.model_config.name
            if tokenizer_name not in TOKENIZERS:
                raise ValueError(
                    f"Tokenizer {tokenizer_name} not supported."
                    + f"Supported tokenizers are: {list(TOKENIZERS.keys())}"
                )

            tokenizer = TOKENIZERS[tokenizer_name].load_model(
                tokenizer_path, strategy=self.config.training_config.strategy
            )
            _logger.info(f"Loaded tokenizer from {tokenizer_path}")

        if isinstance(tokenizer, OSLTokenizer):
            n_tokens = len(tokenizer.vocab["token_order"]) + 1
        else:
            n_tokens = tokenizer_config.model_config.n_tokens

        _logger.info(f"Setting n_tokens to {n_tokens}")
        self.config.model_config.n_tokens = n_tokens

        return tokenizer

    def _load_pretrained_model(self) -> tf.keras.Model:
        """Load a pretrained model."""
        pretrained_model_path = self.config.model_config.pretrained_model_path
        if pretrained_model_path is None:
            return

        pretrained_model_checkpoint = (
            self.config.model_config.pretrained_model_checkpoint
        )
        pretrained_model = EphysGPT.load_model(
            pretrained_model_path,
            pretrained_model_checkpoint,
            strategy=self.config.training_config.strategy,
        )
        _logger.info(f"Loaded pretrained model from {pretrained_model_path}")
        pretrained_model.model.trainable = False
        return pretrained_model

    @property
    def pretrained_layers(self) -> List[str]:
        if self.pretrained_model is None:
            return []
        return self.config.model_config.pretrained_layers

    def _get_input_embedding_layer(self) -> InputEmbeddingLayer:
        config = self.config.model_config
        if "input_embedding" in self.pretrained_layers:
            pretrained_layer = self.pretrained_model.model.get_layer("input_embedding")
        else:
            pretrained_layer = None

        return InputEmbeddingLayer(
            config.embedding_dim,
            config.n_tokens,
            config.sequence_length,
            config.n_channels,
            config.n_sessions,
            config.token_embedding_dim,
            config.pos_embedding_dim,
            config.pos_embedding_type,
            config.channel_embedding_dim,
            config.channel_embedding_type,
            config.prediction_type,
            config.extra_labels,
            pretrained_layer=pretrained_layer,
            name="input_embedding",
        )

    def _get_decoder_layer(self) -> DecoderLayer:
        config = self.config.model_config
        if "decoder" in self.pretrained_layers:
            return self.pretrained_model.model.get_layer("decoder")
        else:
            return DecoderLayer(
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
                config.pos_embedding_type,
                config.channel_attention_dropout,
                config.within_channel_attention_dropout,
                config.feed_forward_dim,
                config.feed_forward_activation,
                config.dropout,
                config.norm_type,
                config.n_groups,
                config.prediction_type,
                name="decoder",
            )

    def _get_prediction_head_layer(self) -> tf.keras.layers.Layer:
        config = self.config.model_config
        if "prediction_head" in self.pretrained_layers:
            return self.pretrained_model.model.get_layer("prediction_head")
        else:
            if config.prediction_type == "token":
                return tf.keras.layers.Dense(config.n_tokens, name="prediction_head")
            elif config.prediction_type == "binary":
                return tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        config.n_tokens, activation="gelu" # EDIT: may abstract activation out
                    ),
                    tf.keras.layers.Dense(2),
                ], name="prediction_head"
            )

    def _build_model(self) -> tf.keras.Model:
        config = self.config.model_config

        # ---------- Inputs ---------- #
        if config.prediction_type == "token":
            data_shape = (config.sequence_length + 1, config.n_channels)
        elif config.prediction_type == "binary":
            data_shape = (config.sequence_length, config.n_channels)
        
        data = tf.keras.layers.Input(
            shape=data_shape,
            dtype=tf.int32,
            name="data",
        )

        if config.prediction_type == "token":
            extra_label_shape = (config.sequence_length + 1,)
        elif config.prediction_type == "binary":
            extra_label_shape = (config.sequence_length,)
        
        extra_labels = []
        for label in config.extra_labels:
            extra_labels.append(
                tf.keras.layers.Input(
                    shape=extra_label_shape,
                    dtype=tf.int32,
                    name=label.name,
                )
            )

        # ---------- Initialize layers ---------- #
        shift_token_layer = ShiftTokenLayer(name="shift_token")
        input_embedding_layer = self._get_input_embedding_layer()
        decoder_layer = self._get_decoder_layer()
        prediction_head_layer = self._get_prediction_head_layer()
        loss_layer = CrossEntropyLossLayer(
            config.loss_sequence_length,
            config.top_k,
            class_weights={0: 2.3035579218113904, 1: 0.6386148994587955}, # EDIT: need to make it as an arugment
            name="loss",
        )

        # ---------- Forward Pass ---------- #

        if config.prediction_type == "token":
            # Shift the tokens
            x, y_true = shift_token_layer(data)
        elif config.prediction_type == "binary":
            x = data
            y_true = tf.keras.layers.Input(
                shape=(config.sequence_length, config.n_channels),
                dtype=tf.int32,
                name="y_true",
            )
        # x.shape = (batch_size, sequence_length, n_channels)
        # y_true.shape = (batch_size, sequence_length, n_channels)

        # Get the input embeddings to the decoder
        x = input_embedding_layer([x, extra_labels])
        # x.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # Run the decoder
        x = decoder_layer(x, training=not "decoder" in self.pretrained_layers)
        # x.shape = (batch_size, latent_sequence_length, n_channels, model_dim)

        # Get the prediction of the next token
        y_pred = prediction_head_layer(
            x, training=not "prediction_head" in self.pretrained_layers
        )
        # y_pred.shape = (batch_size, latent_sequence_length, n_channels, n_tokens)

        y_pred = tf.reduce_mean(y_pred, axis=2)

        # Calculate the loss
        loss, y_pred = loss_layer([y_pred, y_true])

        # ---------- Model ---------- #
        inputs = [data] + extra_labels
        if config.prediction_type == "binary":
            inputs.append(y_true)
        return tf.keras.Model(
            inputs=inputs, outputs=[loss, y_pred], name="ephys_gpt"
        )

    @tf.function
    def one_step_sample(
        self,
        inputs: list,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
    ) -> tf.Tensor:
        """
        Generate a single token using the model.

        Parameters
        ----------
        inputs : list
            List of inputs to the model.
        top_p : float, optional
            Top p proportion of values to keep.
        top_k : int, optional
            Top k number of values to keep.
        temperature : float, optional
            Temperature for sampling from the logits.

        Returns
        -------
        sampled_tokens : tf.Tensor
            The sampled tokens. Shape: (*batch_dims).
        """
        output = self.model(inputs, training=False)
        logits = output[1][:, -1] / temperature
        return sample_from_logits(logits, top_p, top_k)

    def generate_tokens(
        self,
        n_samples: int,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        batch_size: int = None,
        prompt: np.ndarray = None,
        extra_labels: Dict[str, np.ndarray] = None,
        extra_channels: Dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate tokens using the model.

        Parameters
        ----------
        n_samples : int
            Number of samples per sequence to generate.
        top_p : float, optional
            Top p proportion of values to keep.
        top_k : int, optional
            Top k number of values to keep.
        temperature : float, optional
            Temperature for sampling from the logits.
        batch_size : int, optional
            Batch size for generating the samples.
            If None, batch size in the config is used.
        prompt : str or np.ndarray, optional
            Prompt to start the generation.
            If None, a random sequence is used.
            If np.ndarray, the shape must be (sequence_length, n_channels)
            or (batch_size, sequence_length, n_channels).
        extra_labels : Dict[np.ndarray], optional
            Dictionary of extra labels.
            Keys are the names of the extra labels. Each value must have shape (batch_size,).
        extra_channels : Dict[np.ndarray], optional
            Dictionary of extra channels. Keys are the names of the extra channels.
            Each value must have shape (batch_size, > sequence_length + n_samples).

        Returns
        -------
        generated_tokens : np.ndarray
            Generated tokens. Shape is (batch_size, n_samples, n_channels).
        """
        batch_size = batch_size or self.config.training_config.batch_size
        n_channels = self.config.model_config.n_channels
        sequence_length = self.config.model_config.sequence_length

        # ---------- Helper functions ---------- #
        def _random_tokens() -> np.ndarray:
            _rng = np.random.default_rng()
            try:
                token_weights = self.tokenizer.vocab["total_token_counts"].astype(
                    np.float32
                )
            except AttributeError:
                token_weights = np.ones(
                    self.config.model_config.n_tokens, dtype=np.float32
                )
            token_weights /= np.sum(token_weights)

            tokens = (
                _rng.choice(
                    len(token_weights),
                    size=(batch_size, sequence_length, n_channels),
                    p=token_weights,
                )
                + 1
            )
            return tokens.astype(np.int32)

        # ---------- Validation ---------- #
        if prompt is None:
            prompt = _random_tokens()
        elif isinstance(prompt, np.ndarray):
            if prompt.shape != (batch_size, sequence_length, n_channels):
                if prompt.shape == (sequence_length, n_channels):
                    prompt = np.array([prompt] * batch_size)
                else:
                    raise ValueError(
                        f"Prompt must have shape (batch_size, sequence_length, n_channels) or (sequence_length, n_channels)."
                    )
        else:
            raise ValueError(f"Prompt must be a numpy array.")

        if extra_labels is None:
            extra_labels = {}
        for k in extra_labels.keys():
            if extra_labels[k].shape != (batch_size,):
                raise ValueError(f"Extra label {k} must have shape (batch_size,).")
            extra_labels[k] = np.broadcast_to(
                extra_labels[k][:, None], (batch_size, sequence_length + 1)
            )

        if extra_channels is None:
            extra_channels = {}
        for k in extra_channels.keys():
            if extra_channels[k].ndim != 2:
                raise ValueError(
                    f"Extra channel {k} must have shape (batch_size, n_samples)."
                )
            if extra_channels[k].shape[0] != batch_size:
                raise ValueError(
                    f"Extra channel {k} must have shape (batch_size, n_samples)."
                )
            if not extra_channels[k].shape[1] > sequence_length + n_samples:
                raise ValueError(
                    f"Extra channel {k} must have at least n_samples + sequence_length + 1 samples."
                )
            extra_channels[k] = extra_channels[k][:, : sequence_length + n_samples + 1]

        # ---------- Generate tokens ---------- #
        generated_tokens = np.zeros(
            (batch_size, sequence_length + n_samples, n_channels), dtype=np.int32
        )
        generated_tokens[:, :sequence_length] = prompt
        for i in trange(
            sequence_length, sequence_length + n_samples, desc="Generating tokens"
        ):
            model_inputs = {"data": generated_tokens[:, i - sequence_length : i + 1]}
            # model_input["data"].shape = (batch_size, sequence_length + 1, n_channels)

            # Add extra labels to the model inputs
            model_inputs.update(extra_labels)

            # Add extra channels to the model inputs
            for k in extra_channels.keys():
                model_inputs[k] = extra_channels[k][:, i - sequence_length : i + 1]

            # Prediction logits for the next token
            with self.model.distribute_strategy.scope():
                next_token = self.one_step_sample(
                    model_inputs,
                    top_p,
                    top_k,
                    temperature,
                )

            # Add the next token to the prompt
            generated_tokens[:, i] = next_token.numpy().astype(np.int32)

        return generated_tokens[:, sequence_length:]

    def generate_data(self, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate data using the model.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to pass to the generate_tokens method.

        Returns
        -------
        generated_data : np.ndarray or List[np.ndarray]
            Generated data. Shape of each array is (n_samples, n_channels).
        """
        tokens = self.generate_tokens(**kwargs)
        return self.tokenizer.reconstruct_data(list(tokens))

    def get_embeddings(self) -> dict:
        """
        Get the embeddings from the model.

        Returns
        -------
        embeddings : dict
            Dictionary of embeddings.
        """
        config = self.config.model_config
        input_embedding_layer = self.model.get_layer("input_embedding")
        embeddings = dict()
        embeddings["token"] = (
            input_embedding_layer.token_embedding_layer.embeddings.numpy()
        )
        if self.config.model_config.pos_embedding_type == "absolute":
            embeddings["position"] = (
                input_embedding_layer.position_embedding_layer.position_embeddings.numpy()
            )
        embeddings["channel"] = (
            input_embedding_layer.channel_embedding_layer.position_embeddings.numpy()
        )
        for i, label in enumerate(config.extra_labels):
            embeddings[label.name] = input_embedding_layer.extra_embedding_layers[
                i
            ].embeddings.numpy()

        return embeddings
