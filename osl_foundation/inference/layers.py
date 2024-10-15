from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class IdentityLayer(tf.keras.layers.Layer):
    """
    Identity layer.
    This layer directly returns the input tensor.
    """

    def call(self, inputs, **kwargs):
        return inputs


class MSELossLayer(tf.keras.layers.Layer):
    """
    Layer for computing the mean squared error loss.
    This is a wrapper around tf.keras.losses.MeanSquaredError.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))

    def call(self, y_true, y_pred, **kwargs):
        loss = self.loss_fn(y_true, y_pred)
        self.add_loss(loss)
        return tf.expand_dims(loss, -1)


class TokenWeightsLayer(tf.keras.layers.Layer):
    """
    Layer for computing token weights.

    Parameters
    ----------
    output_dim : int
        Dimension of the output.
    activation : str, optional
        Activation function to use.
    """

    def __init__(self, output_dim: int, activation: str = "linear", **kwargs):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation=activation)
        self.activation_layer = tf.keras.layers.Activation(activation)
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.temperature = tf.Variable(0.0, trainable=False)

    def call(self, inputs, **kwargs):
        ell = self.activation_layer(self.dense_layer(inputs))
        ell = self.norm_layer(ell)
        # Shape: (batch_size * n_channels, sequence_length, n_tokens)

        theta_weight = tf.nn.softmax(ell, axis=2)
        # Shape: (batch_size * n_channels, sequence_length, n_tokens)

        # Sample from gumbel softmax parameterized by ell
        dist = tfp.distributions.Gumbel(0, 1)
        theta_sample = tf.argmax(dist.sample() + ell, axis=2)
        theta_sample = tf.one_hot(theta_sample, self.output_dim)

        token_weight = (
            self.temperature * theta_weight + (1 - self.temperature) * theta_sample
        )
        # Shape: (batch_size * n_channels, sequence_length, n_tokens)

        return token_weight


class NormalizationLayer(tf.keras.layers.Layer):
    """
    Layer for performing normalization.

    Parameters
    ----------
    norm_type : str
        Type of normalization to perform.
        Options: "layer", "batch", "group".
    n_groups : int, optional
        Number of groups for group normalization.
        Required if norm_type is "group".
    """

    def __init__(self, norm_type: str = "layer", n_groups=None, **kwargs):
        super().__init__(**kwargs)
        if norm_type == "layer":
            self.norm_layer = tf.keras.layers.LayerNormalization()
        elif norm_type == "batch":
            self.norm_layer = tf.keras.layers.BatchNormalization()
        elif norm_type == "group":
            if n_groups is None:
                raise ValueError("n_groups must be specified for group normalization")
            self.norm_layer = tf.keras.layers.GroupNormalization(groups=n_groups)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def call(self, inputs, **kwargs):
        return self.norm_layer(inputs)


class TimeAttentionLayer(tf.keras.layers.Layer):
    """
    Layer for performing time attention.

    Parameters
    ----------
    key_dim : int
        Key dimension.
    """

    def __init__(self, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim

    def call(self, inputs, mask=None, **kwargs):
        # q (Query): (batch_size, n_heads, out_sequence_length, n_channels, key_dim)
        # k (Key): (batch_size, n_heads, in_sequence_length, n_channels, key_dim)
        # v (Value): (batch_size, n_heads, in_sequence_length, n_channels, key_dim)
        q, k, v = inputs

        # Transpose inputs for time attention

        q = tf.transpose(q, perm=(0, 1, 3, 2, 4))
        # q: (batch_size, n_heads, n_channels, out_sequence_length, key_dim)

        k = tf.transpose(k, perm=(0, 1, 3, 2, 4))
        # k: (batch_size, n_heads, n_channels, in_sequence_length, key_dim)

        v = tf.transpose(v, perm=(0, 1, 3, 2, 4))
        # v: (batch_size, n_heads, n_channels, in_sequence_length, key_dim)

        # Compute attention
        attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )
        # attention: (batch_size, n_heads, n_channels, out_sequence_length, in_sequence_length)

        # Apply mask
        if mask is not None:
            attention += -1e-9 * mask

        # Normalise attention with softmax
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention to value
        output = tf.matmul(attention, v)
        # output: (batch_size, n_heads, n_channels, out_sequence_length, key_dim)

        # Transpose output back to original shape
        output = tf.transpose(output, perm=(0, 1, 3, 2, 4))
        # output: (batch_size, n_heads, out_sequence_length, n_channels, key_dim)

        return output


class ChannelAttention(tf.keras.layers.Layer):
    """
    Layer for performing channel attention.

    Parameters
    ----------
    key_dim : int
        Key dimension.
    """

    def __init__(self, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim

    def call(self, inputs, mask=None, **kwargs):
        # q (Query): (batch_size, n_heads, sequence_length, n_channels, key_dim)
        # k (Key): (batch_size, n_heads, sequence_length, n_channels, key_dim)
        # v (Value): (batch_size, n_heads, sequence_length, n_channels, key_dim)
        q, k, v = inputs

        # Compute attention
        attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )
        # attention: (batch_size, n_heads, sequence_length, n_channels, n_channels)

        # Apply mask
        if mask is not None:
            attention += -1e-9 * mask

        # Normalise attention with softmax
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention to value
        output = tf.matmul(attention, v)
        # output: (batch_size, n_heads, sequence_length, n_channels, key_dim)

        return output


class DummyChannelAttentionLayer(tf.keras.layers.Layer):
    """
    Dummy layer for channel attention.
    This layer directly returns the value tensor.
    """

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs
        return v


class SSTALayer(tf.keras.layers.Layer):
    """
    The Separable Space-Time self-Attention (SSTA) layer.
    This layer performs space-time attention on the input tensor.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    latent_sequence_length : int
        Latent sequence length.
    key_dim : int
        Key dimension.
    channel_attention_dropout : float
        Dropout rate for channel attention.
        Values greater than 1.0 means no channel attention.
        Values less than 0.0 means no dropout.
    within_channel_attention_dropout : float
        Dropout rate for within-channel attention.
        Values greater than 1.0 means no within-channel attention.
        Values less than 0.0 means no dropout
    """

    def __init__(
        self,
        n_channels: int,
        latent_sequence_length: int,
        key_dim: int,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.latent_sequence_length = latent_sequence_length
        self.key_dim = key_dim

        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout

        # Time attention layer
        self.time_attention_layer = TimeAttentionLayer(key_dim)
        # Mask for time attention (This is fixed).
        self.time_attention_mask = self._compute_time_attention_mask()

        # Channel attention layer
        if self.channel_attention_dropout < 1.0:
            self.channel_attention_layer = ChannelAttention(key_dim)
        else:
            self.channel_attention_layer = DummyChannelAttentionLayer()

    def _compute_time_attention_mask(self) -> tf.Tensor:
        """
        Compute the causal mask for time attention.

        Returns
        -------
        mask : tf.Tensor
            Mask for time attention.
            Shape: (latent_sequence_length, latent_sequence_length).
        """
        mask = np.triu(
            np.ones((self.latent_sequence_length, self.latent_sequence_length)), k=1
        )
        mask = tf.constant(mask, dtype=tf.float32)

        return mask

    def _compute_channel_attention_mask(
        self, training: bool = False
    ) -> Union[tf.Tensor, None]:
        """
        Compute the mask for channel attention.

        Parameters
        ----------
        training : bool
            Whether the model is training.
            If False, no dropout is applied.

        Returns
        -------
        mask : tf.Tensor or None
            Mask for channel attention.
            Shape: (n_channels, n_channels).
            If None, full attention is applied.
        """
        if training:
            uniform_sampler = tfp.distributions.Uniform()

            # Sample whether to apply channel attention
            if uniform_sampler.sample() < self.channel_attention_dropout:
                # Does not apply channel attention and mask all off-diagonal elements
                mask = 1 - np.eye(self.n_channels)
                return tf.constant(mask, dtype=tf.float32)

            # Sample whether to apply within-channel attention
            elif uniform_sampler.sample() < self.within_channel_attention_dropout:
                # Does not apply within-channel attention and mask all diagonal elements
                mask = np.eye(self.n_channels)
                return tf.constant(mask, dtype=tf.float32)

            # Apply channel attention and does not mask any elements, return None
            else:
                mask = np.zeros((self.n_channels, self.n_channels))
                return tf.constant(mask, dtype=tf.float32)

        # If not training, return None
        # else:
        #     return None

    def call(self, inputs, training=None, **kwargs):
        # ---------- Unpack Inputs ---------- #

        # q (Query): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # k (Key): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # v (Value): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # c_q (Channel Query): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # c_k (Channel Key): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        q, k, v, c_q, c_k = inputs

        # ---------- Time Attention ---------- #
        output = self.time_attention_layer(
            [q, k, v], mask=self.time_attention_mask, training=training, **kwargs
        )

        # ---------- Channel Attention ---------- #

        # First sample channel attention mask
        if self.channel_attention_dropout < 1.0:
            channel_attention_mask = self._compute_channel_attention_mask(
                training=training
            )
        else:
            channel_attention_mask = None

        # Apply channel attention
        output = self.channel_attention_layer(
            [c_q, c_k, output], mask=channel_attention_mask, training=training, **kwargs
        )

        return output


class MultiHeadSSTALayer(tf.keras.layers.Layer):
    """
    The Multi-head SSTA layer.

    Parameters
    ----------
    n_heads : int
        Number of heads.
    model_dim : int
        Model dimension.
    embedding_dim : int
        Token dimension.
    n_channels : int
        Number of channels.
    latent_sequence_length : int
        Latent sequence length.
    channel_attention_dropout : float
        Dropout rate for channel attention.
        Values greater than 1.0 means no channel attention.
        Values less than 0.0 means no dropout.
    within_channel_attention_dropout : float
        Dropout rate for within-channel attention.
        Values greater than 1.0 means no within-channel attention.
        Values less than 0.0 means no dropout
    """

    def __init__(
        self,
        n_heads: int,
        model_dim: int,
        embedding_dim: int,
        n_channels: int,
        latent_sequence_length: int,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.key_dim = model_dim // n_heads
        self.n_channels = n_channels
        self.latent_sequence_length = latent_sequence_length
        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout

        # Input projections
        self.projection = tf.keras.layers.Dense(5 * self.model_dim)

        # SSTALayer for time and channel attention
        self.ssta_layer = SSTALayer(
            n_channels,
            latent_sequence_length,
            self.key_dim,
            channel_attention_dropout,
            within_channel_attention_dropout,
        )

        # Output projection
        self.output_projection = tf.keras.layers.Dense(self.model_dim)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                tf.shape(x)[1],
                self.n_channels,
                self.n_heads,
                self.key_dim,
            ),
        )
        # x.shape: (batch_size, latent_sequence_length, n_channels, n_heads, key_dim)

        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # x.shape: (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        return x

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        x = tf.transpose(x, perm=(0, 2, 3, 1, 4))
        # x.shape: (batch_size, latent_sequence_length, n_channels, n_heads, key_dim)

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                self.latent_sequence_length,
                self.n_channels,
                self.model_dim,
            ),
        )
        # x.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        return x

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        # x.shape: (batch_size, latent_sequence_length, n_channels, embedding_dim)

        # ---------- Project inputs to Q, K, V ---------- #
        q, k, v, c_q, c_k = tf.split(self.projection(x), 5, axis=-1)
        # q.shape: (batch_size, latent_sequence_length, n_channels, model_dim)
        # k.shape: (batch_size, latent_sequence_length, n_channels, model_dim)
        # v.shape: (batch_size, latent_sequence_length, n_channels, model_dim)
        # c_q.shape: (batch_size, latent_sequence_length, n_channels, model_dim)
        # c_k.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        # ---------- Split heads ---------- #

        q = self._split_heads(q)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        k = self._split_heads(k)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        v = self._split_heads(v)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        c_q = self._split_heads(c_q)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        c_k = self._split_heads(c_k)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        # ---------- SSTALayer ---------- #
        output = self.ssta_layer([q, k, v, c_q, c_k], training=training, **kwargs)

        # ---------- Combine heads ---------- #
        output = self._combine_heads(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        # ---------- Output projection ---------- #
        output = self.output_projection(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        return output


class PASSTALayer(SSTALayer):
    """
    The Perceiver AR Separable Space-Time self-Attention (PASSTA) layer.
    This layer performs space-time attention on the input tensor.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    latent_sequence_length : int
        Sequence length of latent space.
    n_patches : int
        Number of patches to attend to.
    patch_length : int
        Patch length.
    unpatched_length : int
        Number of unpatched elements to attend to.
    key_dim : int
        Key dimension.
    channel_attention_dropout : float
        Dropout rate for channel attention.
        Values greater than 1.0 means no channel attention.
        Values less than 0.0 means no dropout.
    within_channel_attention_dropout : float
        Dropout rate for within-channel attention.
        Values greater than 1.0 means no within-channel attention.
        Values less than 0.0 means no dropout.
    """

    def __init__(
        self,
        n_channels: int,
        latent_sequence_length: int,
        n_patches: int,
        patch_length: int,
        unpatched_length: int,
        key_dim: int,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        **kwargs,
    ):
        # TODO: Option to leave out unpatched_length
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length
        super().__init__(
            n_channels,
            latent_sequence_length,
            key_dim,
            channel_attention_dropout,
            within_channel_attention_dropout,
            **kwargs,
        )

    def _compute_time_attention_mask(self) -> tf.Tensor:
        """
        Compute the mask for time attention.

        Returns
        -------
        mask : tf.Tensor
            Mask for time attention.
            Shape: (latent_sequence_length, n_patches + unpatched_length).
        """

        patch_mask = np.zeros((self.latent_sequence_length, self.n_patches))
        for i in range(self.n_patches):
            m_indx = max(
                0,
                (i + 1 - self.n_patches) * self.patch_length
                + self.latent_sequence_length
                - 1,
            )
            patch_mask[:m_indx, i] = 1

        unpatched_mask = np.zeros((self.latent_sequence_length, self.unpatched_length))
        for i in range(self.unpatched_length):
            m_indx = self.latent_sequence_length - self.unpatched_length + i
            unpatched_mask[:m_indx, i] = 1

        mask = np.concatenate([patch_mask, unpatched_mask], axis=1)
        mask = tf.constant(mask, dtype=tf.float32)

        return mask

    def call(self, inputs, training=None, **kwargs):
        # ---------- Unpack Inputs ---------- #

        # q (Query): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # k (Key): (batch_size, n_heads, n_patches + unpatched_length, n_channels, key_dim)
        # v (Value): (batch_size, n_heads, n_patches + unpatched_length, n_channels, key_dim)
        # c_q (Channel Query): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        # c_k (Channel Key): (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)
        q, k, v, c_q, c_k = inputs

        # ---------- Time Attention ---------- #
        output = self.time_attention_layer(
            [q, k, v], mask=self.time_attention_mask, training=training, **kwargs
        )

        # ---------- Channel Attention ---------- #

        # First sample channel attention mask
        if self.channel_attention_dropout < 1.0:
            channel_attention_mask = self._compute_channel_attention_mask(
                training=training
            )
        else:
            channel_attention_mask = None

        # Apply channel attention
        output = self.channel_attention_layer(
            [c_q, c_k, output], mask=channel_attention_mask, training=training, **kwargs
        )

        return output


class MultiHeadPASSTALayer(tf.keras.layers.Layer):
    """
    The Multi-head PASSTA layer.

    Parameters
    ----------
    n_heads : int
        Number of heads.
    model_dim : int
        Model dimension.
    embedding_dim : int
        Token dimension.
    n_channels : int
        Number of channels.
    sequence_length : int
        Sequence length.
    latent_sequence_length : int
        Latent sequence length.
    n_patches : int
        Number of patches to attend to.
    patch_length : int
        Patch length.
    unpatched_length : int
        Number of unpatched elements to attend to.
    channel_attention_dropout : float
        Dropout rate for channel attention.
        Values greater than 1.0 means no channel attention.
        Values less than 0.0 means no dropout.
    within_channel_attention_dropout : float
        Dropout rate for within-channel attention.
        Values greater than 1.0 means no within-channel attention.
        Values less than 0.0 means no dropout
    """

    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.key_dim = model_dim // n_heads
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.latent_sequence_length = latent_sequence_length
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length
        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout

        # Patch projection
        self.patch_projection = tf.keras.layers.Dense(1)

        # Input projections
        self.time_patched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_unpatched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_query_projection = tf.keras.layers.Dense(self.model_dim)
        self.channel_projection = tf.keras.layers.Dense(2 * self.model_dim)

        # PASSTA layer for time and channel attention
        self.passta_layer = PASSTALayer(
            n_channels,
            latent_sequence_length,
            n_patches,
            patch_length,
            unpatched_length,
            self.key_dim,
            channel_attention_dropout,
            within_channel_attention_dropout,
        )

        # Output projection
        self.output_projection = tf.keras.layers.Dense(self.model_dim)

    def _patch_x(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, sequence_length, n_channels, embedding_dim)

        x = tf.transpose(x, perm=(0, 2, 3, 1))
        # x.shape: (batch_size, n_channels, embedding_dim, sequence_length)

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                self.n_channels,
                self.embedding_dim,
                self.n_patches,
                self.patch_length,
            ),
        )
        # x.shape: (batch_size, n_channels, embedding_dim, n_patches, patch_length)

        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # x.shape: (batch_size, n_patches, n_channels, embedding_dim, patch_length)

        return x

    def _perceiver_x(self, x: tf.Tensor) -> tf.Tensor:
        """Get the last latent_sequence_length elements of the sequence."""
        # x.shape: (batch_size, sequence_length, n_channels, embedding_dim)

        x = tf.slice(
            x,
            [0, self.sequence_length - self.latent_sequence_length, 0, 0],
            [-1, -1, -1, -1],
        )
        # x.shape: (batch_size, latent_sequence_length, n_channels, embedding_dim)

        return x

    def _unpatch_x(self, x: tf.Tensor) -> tf.Tensor:
        """Get the last unpatched_length elements of the sequence."""
        # x.shape: (batch_size, sequence_length, n_channels, embedding_dim)

        x = tf.slice(
            x,
            [0, self.sequence_length - self.unpatched_length, 0, 0],
            [-1, -1, -1, -1],
        )
        # x.shape: (batch_size, unpatched_length, n_channels, embedding_dim)

        return x

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, time_length, n_channels, model_dim)
        # Here time_length is either n_patches + unpatched_length or latent_sequence_length

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                tf.shape(x)[1],
                self.n_channels,
                self.n_heads,
                self.key_dim,
            ),
        )
        # x.shape: (batch_size, time_length, n_channels, n_heads, key_dim)

        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # x.shape: (batch_size, n_heads, time_length, n_channels, key_dim)

        return x

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        x = tf.transpose(x, perm=(0, 2, 3, 1, 4))
        # x.shape: (batch_size, latent_sequence_length, n_channels, n_heads, key_dim)

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                self.latent_sequence_length,
                self.n_channels,
                self.model_dim,
            ),
        )
        # x.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        return x

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        # x.shape: (batch_size, sequence_length, n_channels, embedding_dim)

        # ---------- Process inputs ---------- #
        # Input is processed into 3 parts:
        # 1. Patched input: patche_x
        # 2. Unpatched input: unpatched_x
        # 3. Perceiver input: perceiver_x

        patched_x = self._patch_x(x)
        # patched_x.shape: (batch_size, n_patches, n_channels, embedding_dim, patch_length)

        # Apply patch projection
        patched_x = tf.squeeze(self.patch_projection(patched_x), axis=-1)
        # patched_x.shape: (batch_size, n_patches, n_channels, embedding_dim)

        perceiver_x = self._perceiver_x(x)
        # perceiver_x.shape: (batch_size, latent_sequence_length, n_channels, embedding_dim)

        unpatched_x = self._unpatch_x(x)
        # unpatched_x.shape: (batch_size, unpatched_length, n_channels, embedding_dim)

        # ---------- Project inputs to Q, K, V ---------- #
        q = self.time_query_projection(perceiver_x)
        # q.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        k_patched, v_patched = tf.split(
            self.time_patched_projection(patched_x), 2, axis=-1
        )
        # k_patched.shape: (batch_size, n_patches, n_channels, model_dim)
        # v_patched.shape: (batch_size, n_patches, n_channels, model_dim)

        k_unpatched, v_unpatched = tf.split(
            self.time_unpatched_projection(unpatched_x), 2, axis=-1
        )
        # k_unpatched.shape: (batch_size, unpatched_length, n_channels, model_dim)
        # v_unpatched.shape: (batch_size, unpatched_length, n_channels, model_dim)

        # Concatenate k and v for time attention
        k = tf.concat([k_patched, k_unpatched], axis=1)
        v = tf.concat([v_patched, v_unpatched], axis=1)
        # k.shape: (batch_size, n_patches + unpatched_length, n_channels, model_dim)
        # v.shape: (batch_size, n_patches + unpatched_length, n_channels, model_dim)

        c_q, c_k = tf.split(self.channel_projection(perceiver_x), 2, axis=-1)
        # c_q.shape: (batch_size, latent_sequence_length, n_channels, model_dim)
        # c_k.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        # ---------- Split heads ---------- #

        q = self._split_heads(q)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        k = self._split_heads(k)
        # (batch_size, n_heads, n_patches + unpatched_length, n_channels, key_dim)

        v = self._split_heads(v)
        # (batch_size, n_heads, n_patches + unpatched_length, n_channels, key_dim)

        c_q = self._split_heads(c_q)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        c_k = self._split_heads(c_k)
        # (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        # ---------- PASSTA Layer ---------- #
        output = self.passta_layer([q, k, v, c_q, c_k], training=training, **kwargs)
        # output.shape: (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        # ---------- Combine heads ---------- #
        output = self._combine_heads(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        # ---------- Output projection ---------- #
        output = self.output_projection(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        return output
