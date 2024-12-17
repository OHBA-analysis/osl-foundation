from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def rnn_layer(
    rnn_type: str, rnn_n_units: int, return_sequences: bool = True
) -> tf.keras.layers.Layer:
    """
    Create an RNN layer.

    Parameters
    ----------
    rnn_type : str
        Type of RNN layer. Options are 'gru', 'lstm'.
    rnn_n_units : int
        Number of units in the RNN layer.
    return_sequences : bool, optional
        Whether to return sequences.

    Returns
    -------
    rnn_layer : tf.keras.layers.Layer
        RNN layer.
    """
    if rnn_type == "gru":
        return tf.keras.layers.GRU(
            rnn_n_units, return_sequences=return_sequences, stateful=False
        )
    elif rnn_type == "lstm":
        return tf.keras.layers.LSTM(
            rnn_n_units, return_sequences=return_sequences, stateful=False
        )
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")


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


class SinusoidalPositionalEncodingLayer(tf.keras.layers.Layer):
    """
    Layer for generating sinusoidal positional encoding for position embeddings.
    Implemented as in "Attention is All You Need" (Vaswani et al., 2017), and partly
    adpated from the `huggingface/transformers` library and TensorFlow
    (https://www.tensorflow.org/text/tutorials/transformer).

    Parameters
    ----------
    sequence_length : int
        Length of the sequence.
    """

    def __init__(self, sequence_length: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def call(self, inputs, **kwargs):
        # Validation
        embedding_dim = tf.shape(inputs)[-1]
        tf.debugging.assert_equal(
            tf.math.floormod(embedding_dim, 2),
            tf.constant(0, dtype=tf.int32),
            message="embedding dimension must be even for positional encoding.",
        )

        # Precompute scaling factor
        embedding_dim_float = tf.cast(embedding_dim, tf.float32)
        denominator = tf.math.pow(
            10000.0,
            2 * (tf.range(embedding_dim_float) // 2) / embedding_dim_float,
        )

        # Get position indices
        position_indices = tf.range(self.sequence_length, dtype=tf.float32)[
            :, tf.newaxis
        ]
        # position_indices.shape = (sequence_length, 1)

        # Compute positional encoding
        angle_rads = position_indices / denominator
        # angle_rads.shape = (sequence_length, embedding_dim)
        pos_encoding = tf.TensorArray(dtype=tf.float32, size=embedding_dim)
        for i in range(embedding_dim):
            if i % 2 == 0:
                pos_encoding = pos_encoding.write(i, tf.sin(angle_rads[:, i]))
            else:
                pos_encoding = pos_encoding.write(i, tf.cos(angle_rads[:, i]))
        pos_encoding = tf.transpose(pos_encoding.stack())
        # pos_encoding.shape = (sequence_length, embedding_dim)

        # Reshape the positions to match shapes of other embeddings
        positions = tf.expand_dims(
            tf.expand_dims(pos_encoding, axis=0), axis=0
        )  # positions.shape = (1, 1, sequence_length, embedding_dim)
        return positions

class RotaryPositionEmbeddingLayer(tf.keras.layers.Layer):
    """
    Layer for generating Rotary Position Embedding (RoPE). Implemented as in 
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2022), and 
    adpated from the PyTorch and Keras 3 libraries (see below for the reference).

    References:
    - https://pytorch.org/torchtune/0.2/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
    - https://github.com/keras-team/keras-hub/blob/v0.17.0/keras_hub/src/layers/modeling/rotary_embedding.py

    Parameters
    ----------
    embedding_dim : int
        Dimension of the rotary position embedding. When using multi-head attention blocks,
        the dimension is usually `embedding_dim // n_heads`.
    max_sequence_length : int
        Maximum expected length of the sequence. If the model dimensions of the key and 
        query vectors differ, the sequence length should be the maximum of the two.
    base : int, optional
        The base value for computing the rotation angles.
    """

    def __init__(self, embedding_dim: int, max_sequence_length: int, base: int = 10000, **kwargs):
        super().__init__(**kwargs)

        # Validation
        if embedding_dim % 2 != 0:
            raise ValueError("embedding dimension must be even for positional encoding.")

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.base = base
        self.rotation_values = self._compute_rotation_matrix()

    def _compute_rotation_matrix(self) -> tf.Tensor:
        # Compute pre-defined theta angles
        theta = 1.0 / (
            self.base ** (tf.range(0, self.embedding_dim, 2, dtype=tf.float32)[:(self.embedding_dim // 2)] / self.embedding_dim)
        )
        
        # Create position indices
        seq_idx = tf.range(self.max_sequence_length, dtype=tf.float32)
        # NOTE: Having 0 as a starting index is okay, since we don't have to rotate the first position.
        
        # Compute outer product of position index and theta
        idx_theta = tf.einsum("i,j->ij", seq_idx, theta)
        # idx_theta.shape = (sequence_length, embedding_dim // 2)
        
        # Compute elements of the rotation matrix
        rotation_values = tf.stack([tf.cos(idx_theta), tf.sin(idx_theta)], axis=-1)
        # rotation_values.shape = (sequence_length, embedding_dim // 2, 2)

        return rotation_values
    
    def call(self, inputs, **kwargs):
        # Get input sequence length
        sequence_length = tf.shape(inputs)[3]
        # inputs.shape = (batch_size, n_heads, n_channels, sequence_length, key_dim)

        # Validation
        tf.debugging.assert_greater_equal(
            self.max_sequence_length,
            sequence_length,
            message="input sequence length should not exceed max_sequence_length.",
        )
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            self.embedding_dim,
            message="the model dimension of the input tensor must match the embedding dimension.",
        )

        # Reshape the rotation values to be compabitle with the input shape
        rot_vals = tf.reshape(
            self.rotation_values[:sequence_length],  
            (1, 1, 1, sequence_length, self.embedding_dim // 2, 2),
        )
        # NOTE: We clip the rotation values to the sequence length, in case sequence_length is 
        # less than max_sequence_length as the dimension for key and query differs.

        # Match the input shape to the rope shape
        rot_shape = tf.concat([tf.shape(inputs)[:-1], [self.embedding_dim // 2, 2]], axis=0)
        x_reshaped = tf.reshape(inputs, rot_shape)
        # x_reshaped.shape = (batch_size, n_heads, n_channels, sequence_length, key_dim // 2, 2)

        # Calculate the rotary position embedding
        rope = tf.stack([
            rot_vals[..., 0] * x_reshaped[..., 0]
            - rot_vals[..., 1] * x_reshaped[..., 1],
            rot_vals[..., 1] * x_reshaped[..., 0]
            + rot_vals[..., 0] * x_reshaped[..., 1],
        ], axis=-1)
        # rope.shape = (batch_size, n_heads, n_channels, sequence_length, key_dim // 2, 2)
        rope = tf.reshape(rope, tf.shape(inputs))
        # rope.shape = (batch_size, n_heads, n_channels, sequence_length, key_dim)
        
        return rope


class ALiBiPositionEmbeddingLayer(tf.keras.layers.Layer):
    """
    Layer for generating Attention with Linear Biases (ALiBi) position embedding. Implemented 
    based on "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    (Press et al., 2022) and adapted from the references below.

    References:
    - https://github.com/ofirpress/attention_with_linear_biases
    - https://nn.labml.ai/transformers/alibi/index.html

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    n_patches : int
        Number of patches to attend to.
    patch_length : int
        Patch length.
    unpatched_length : int
        Number of unpatched elements to attend to.
    """

    def __init__(
        self,
        n_heads: int,
        n_patches: int,
        patch_length: int,
        unpatched_length: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length

    def _get_slopes(self) -> tf.Tensor:
        # Compute the slopes for the linear biases (static, non-learned)
        h = tf.cast(self.n_heads, tf.float32)
        n = 2 ** tf.math.floor(tf.math.log(h) / tf.math.log(2.0))  # nearest power of 2
        ratio = tf.pow(2.0, -8.0 / n)
        m = tf.pow(ratio, tf.range(1, tf.cast(n, tf.int32) + 1, dtype=tf.float32))  # head-specific slopes

        # When n_heads is not a power of 2, then we add additional slopes
        if n < h:
            ratio_hat = tf.pow(2.0, -4.0 / n)
            m_hat = tf.pow(ratio_hat, tf.range(1, 1 + 2 * (h - n), 2, dtype=tf.float32))
            m = tf.concat([m, m_hat], axis=0)
        
        return m

    def _get_relative_bias_matrix(self, mask: tf.Tensor) -> tf.Tensor:
        # Get head-specific slopes
        m = self._get_slopes()

        # Get a binary masking matrix
        mask = 1 - mask
        cum_mask = tf.cumsum(mask, axis=0)  # cumulative sum of the mask
        # NOTE: In the mask matrix, columns are where you attend to and rows are the targets.

        # Compute the relative distance matrix for the patches
        patch_indices = tf.cast(tf.range(self.n_patches), dtype=tf.float32)
        avg_idx = ((self.patch_length * patch_indices + 1) + self.patch_length * (patch_indices + 1))  / 2.0
        distance_patched = -(cum_mask[:, :self.n_patches] + self.patch_length * (patch_indices + 1)) + avg_idx
        # NOTE: Considering a patch as a single receptive field, we calculate relative distance between the patch 
        # and the current sequence index (i.e., the distance between the center of the patch and the current index).
        # This amounts to the average distance between the current index and all the indices in the patch.

        # Compute the relative distance matrix for the unpatched elements
        distance_unpatched = -cum_mask[:, self.n_patches:self.n_patches + self.unpatched_length]

        # Concatenate two matrices
        distance = tf.concat([distance_patched, distance_unpatched], axis=1)
        # distance.shape = (latent_sequence_length, n_patches + unpatched_length)

        # Compute head-specific relative bias matrices
        masked_distance = distance * mask
        alibi_matrix = masked_distance[None, :, :] * m[:, None, None]
        # alibi_matrix.shape = (n_heads, latent_sequence_length, n_patches + unpatched_length)

        return alibi_matrix

    def call(self, inputs, mask: tf.Tensor, **kwargs):
        # Get the relative bias matrix
        alibi_matrix = self._get_relative_bias_matrix(mask)
        # alibi_matrix.shape = (n_heads, latent_sequence_length, n_patches + unpatched_length)

        # Reshape the relative bias matrix to match the attention matrix
        alibi_matrix = tf.reshape(alibi_matrix, (1, self.n_heads, 1, tf.shape(mask)[0], tf.shape(mask)[1]))

        # Add the relative bias matrix to the attention matrix
        attention_with_alibi = inputs + alibi_matrix
        # attention_with_alibi.shape = (batch_size, n_heads, n_channels, out_sequence_length, in_sequence_length)
        return attention_with_alibi


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

    def call(self, inputs, training=None, **kwargs):
        ell = self.activation_layer(self.dense_layer(inputs))
        ell = self.norm_layer(ell) / 0.1
        # Shape: (batch_size * n_channels, sequence_length, n_tokens)

        if training:
            # Sample from gumbel softmax parameterized by ell
            dist = tfp.distributions.Gumbel(0, 1)
            theta_sample = tf.argmax(dist.sample() + ell, axis=2)
            theta_sample = tf.one_hot(theta_sample, self.output_dim)

            #  Annealing
            theta_weight = tf.nn.softmax(ell, axis=2)
            # Shape: (batch_size * n_channels, sequence_length, n_tokens)

            token_weight = (
                self.temperature * theta_weight + (1 - self.temperature) * theta_sample
            )
            # Shape: (batch_size * n_channels, sequence_length, n_tokens)
        else:
            token_weight = tf.one_hot(tf.argmax(ell, axis=2), self.output_dim)
        return token_weight


class PositionEmbedding(tf.keras.layers.Layer):
    """
    Layer for learning position embeddings.

    Parameters
    ----------
    sequence_length : int
        Sequence length.
    initializer : str, optional
        Initializer for the position embeddings.
    """

    def __init__(
        self,
        sequence_length: int,
        initializer: str = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, start_index=0):
        inputs_shape = tf.shape(inputs)
        feature_length = inputs_shape[-1]
        sequence_length = inputs_shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, inputs_shape)


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
    n_heads : int
        Number of heads.
    latent_sequence_length : int
        Sequence length of latent space.
    key_dim : int
        Key dimension.
    n_patches : int
        Number of patches to attend to.
    patch_length : int
        Patch length.
    unpatched_length : int
        Number of unpatched elements to attend to.
    transform_attention : str, optional
        Type of attention transformation to apply.
    """

    def __init__(
        self,
        n_heads: int,
        latent_sequence_length : int,
        key_dim: int,
        n_patches: int,
        patch_length: int,
        unpatched_length: int,
        transform_attention: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length
        self.transform_attention = transform_attention

        # Create a position embedding layer
        if transform_attention == "rope":
            self.rotary_position_embedding_layer = RotaryPositionEmbeddingLayer(
                embedding_dim=self.key_dim,
                max_sequence_length=tf.cast(
                    tf.math.maximum(self.latent_sequence_length, self.n_patches + self.unpatched_length),
                    dtype=tf.int32,
                ),
            )
        if transform_attention == "alibi":
            self.alibi_position_embedding_layer = ALiBiPositionEmbeddingLayer(
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                patch_length=self.patch_length,
                unpatched_length=self.unpatched_length,
            )

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

        if self.transform_attention == "rope":
            # Apply rotary position embedding to the query and key vectors
            q = self.rotary_position_embedding_layer(q)
            k = self.rotary_position_embedding_layer(k)

        # Compute attention
        attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )
        # attention: (batch_size, n_heads, n_channels, out_sequence_length, in_sequence_length)

        if self.transform_attention == "alibi":
            # Apply ALiBi position embedding to the attention matrix
            attention = self.alibi_position_embedding_layer(attention, mask=mask)

        # Apply mask
        if mask is not None:
            attention += -1e9 * mask

        # Normalise attention with softmax
        attention = tf.nn.softmax(attention, axis=-1)

        if mask is not None:
            attention = attention * (1 - mask)

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
            attention += -1e9 * mask

        # Normalise attention with softmax
        attention = tf.nn.softmax(attention, axis=-1)

        if mask is not None:
            attention = attention * (1 - mask)

        # Apply attention to value
        output = tf.matmul(attention, v)
        # output: (batch_size, n_heads, sequence_length, n_channels, key_dim)

        return output


class PASSTALayer(tf.keras.layers.Layer):
    """
    The Perceiver AR Separable Space-Time self-Attention (PASSTA) layer.
    This layer performs space-time attention on the input tensor.

    Parameters
    ----------
    n_heads : int
        Number of heads.
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
    pos_embedding_type : str
        Type of positional embedding to use.
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
        n_heads: int,
        n_channels: int,
        latent_sequence_length: int,
        n_patches: int,
        patch_length: int,
        unpatched_length: int,
        key_dim: int,
        pos_embedding_type : str,
        channel_attention_dropout: float,
        within_channel_attention_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.n_channels = n_channels
        self.latent_sequence_length = latent_sequence_length
        self.key_dim = key_dim
        self.pos_embedding_type = pos_embedding_type
        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length

        # Time attention layer
        self.time_attention_layer = TimeAttentionLayer(
            n_heads,
            latent_sequence_length,
            key_dim,
            n_patches,
            patch_length,
            unpatched_length,
            pos_embedding_type,
        )
        # Mask for time attention (This is fixed).
        self.time_attention_mask = self._compute_time_attention_mask()

        # Channel attention layer
        self.channel_attention_layer = ChannelAttention(key_dim)

    def _compute_time_attention_mask(self) -> tf.Tensor:
        """
        Compute the mask for time attention.

        Returns
        -------
        mask : tf.Tensor
            Mask for time attention.
            Shape: (latent_sequence_length, n_patches + unpatched_length).
        """
        mask = np.zeros(
            (self.latent_sequence_length, self.n_patches + self.unpatched_length)
        )

        # Patch masking
        for i in range(self.n_patches):
            m_indx = max(
                0,
                (i + 1 - self.n_patches) * self.patch_length
                + self.latent_sequence_length
                - 1,
            )
            mask[:m_indx, i] = 1

        # Unpatched masking
        for i in range(self.n_patches, self.n_patches + self.unpatched_length):
            mask[
                : self.latent_sequence_length
                - (self.n_patches + self.unpatched_length)
                + i,
                i,
            ] = 1
        mask = tf.constant(mask, dtype=tf.float32)

        return mask

    def _compute_channel_attention_mask(self, training: bool) -> Union[tf.Tensor, None]:
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
        if self.channel_attention_dropout < 1.0:
            # We apply channel attention dropout only during training
            if not training:
                return tf.zeros((self.n_channels, self.n_channels), dtype=tf.float32)
            else:
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
                    return tf.zeros(
                        (self.n_channels, self.n_channels), dtype=tf.float32
                    )
        else:
            # If channel_attention_dropout >= 1.0, no channel attention is applied
            # Mask all off-diagonal elements
            mask = 1 - np.eye(self.n_channels)
            return tf.constant(mask, dtype=tf.float32)

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
        channel_attention_mask = self._compute_channel_attention_mask(training=training)

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
    pos_embedding_type : str
        Type of positional embedding to use.
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
        pos_embedding_type: str,
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
        self.pos_embedding_type = pos_embedding_type
        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout

        # Patch projection
        self.patch_projection = tf.keras.layers.Dense(n_heads)

        # Input projections
        self.time_patched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_unpatched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_query_projection = tf.keras.layers.Dense(self.model_dim)
        self.channel_projection = tf.keras.layers.Dense(2 * self.model_dim)

        # PASSTA layer for time and channel attention
        self.passta_layer = PASSTALayer(
            n_heads,
            n_channels,
            latent_sequence_length,
            n_patches,
            patch_length,
            unpatched_length,
            self.key_dim,
            pos_embedding_type,
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
        # 1. Patched input: patched_x
        # 2. Unpatched input: unpatched_x
        # 3. Perceiver input: perceiver_x

        patched_x = self._patch_x(x)
        # patched_x.shape: (batch_size, n_patches, n_channels, embedding_dim, patch_length)

        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches,
                self.n_channels,
                self.key_dim,
                self.n_heads,
                self.patch_length,
            ),
        )
        # patched_x.shape: (batch_size, n_patches, n_channels, key_dim, n_heads, patch_length)
        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches,
                self.n_channels,
                self.key_dim,
                self.n_heads * self.patch_length,
            ),
        )
        # patched_x.shape: (batch_size, n_patches, n_channels, key_dim, n_heads * patch_length)

        patched_x = self.patch_projection(patched_x)
        # patched_x.shape: (batch_size, n_patches, n_channels, key_dim, n_heads)

        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches,
                self.n_channels,
                self.embedding_dim,
            ),
        )
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
