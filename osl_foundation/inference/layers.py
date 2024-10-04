from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class MSELossLayer(tf.keras.layers.Layer):
    """
    Layer for computing the mean squared error loss.
    This is a wrapper around tf.keras.losses.MeanSquaredError.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred, **kwargs):
        loss = self.loss_fn(y_true, y_pred)
        self.add_loss(loss)
        return loss


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
        self.temperature = tf.Variable(1.0, trainable=False)

    def call(self, inputs, **kwargs):
        ell = self.activation_layer(self.dense_layer(inputs))
        ell = self.norm_layer(ell)
        # Shape: (batch_size, sequence_length, n_tokens)

        theta_weight = tf.nn.softmax(ell, axis=2)
        # Shape: (batch_size, sequence_length, n_tokens)

        # Sample from gumbel softmax parameterized by ell
        dist = tfp.distributions.Gumbel(0, 1)
        theta_sample = tf.argmax(dist.sample() + ell, axis=2)
        theta_sample = tf.one_hot(theta_sample, self.output_dim)

        token_weight = (
            self.temperature * theta_weight + (1 - self.temperature) * theta_sample
        )
        # Shape: (batch_size, sequence_length, n_tokens)

        return token_weight


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


class PASSTALayer(tf.keras.layers.Layer):
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
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.latent_sequence_length = latent_sequence_length
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length
        self.key_dim = key_dim

        # Time attention layer
        self.time_attention_layer = TimeAttentionLayer(key_dim)
        # Mask for time attention (This is fixed).
        self.time_attention_mask = self._compute_time_attention_mask()

        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout

        # Channel attention layer
        if self.channel_attention_dropout < 1.0:
            self.channel_attention_layer = ChannelAttention(key_dim)
        else:
            self.channel_attention_layer = DummyChannelAttentionLayer()

    def _compute_time_attention_mask(self) -> tf.Tensor[bool]:
        """
        Compute the mask for time attention.

        Returns
        -------
        mask : tf.Tensor[bool]
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
        mask = tf.constant(mask, dtype=tf.bool)

        return mask

    def _compute_channel_attention_mask(
        self, training: bool = False
    ) -> Union[tf.Tensor[bool], None]:
        """
        Compute the mask for channel attention.

        Parameters
        ----------
        training : bool
            Whether the model is training.
            If False, no dropout is applied.

        Returns
        -------
        mask : tf.Tensor[bool] or None
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
                return tf.constant(mask, dtype=tf.bool)

            # Sampler whether to apply within-channel attention
            if uniform_sampler.sample() < self.within_channel_attention_dropout:
                # Does not apply within-channel attention and mask all diagonal elements
                mask = np.eye(self.n_channels)
                return tf.constant(mask, dtype=tf.bool)

            # Apply channel attention and does not mask any elements, return None

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
