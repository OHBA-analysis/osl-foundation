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

    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = tf.keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config

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

    def compute_output_shape(self, input_shape):
        return input_shape


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
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.latent_sequence_length = latent_sequence_length
        self.key_dim = key_dim
        self.channel_attention_dropout = channel_attention_dropout
        self.within_channel_attention_dropout = within_channel_attention_dropout
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.unpatched_length = unpatched_length

        # Time attention layer
        self.time_attention_layer = TimeAttentionLayer(key_dim)
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


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        SEQ_LEN_LATENT,
        NUM_PATCHES,
        PATCH_LEN,
        SEQ_LEN_UNPATCHED,
        chan_att,
        chanatt_dropout_rate,
        within_chanatt_dropout_rate,
        **kwargs,
    ):
        super(DotProductAttention, self).__init__(**kwargs)

        self.chan_att = chan_att

        # (1-chanatt_dropout_rate) is proportion of times that chan att is used when training
        # (1-within_chanatt_dropout_rate)*(1-chanatt_dropout_rate) is proportion of times that within chan att is used when training
        self.chanatt_dropout_rate = chanatt_dropout_rate
        self.within_chanatt_dropout_rate = within_chanatt_dropout_rate

        self.apply_chan_att = tf.Variable(True, trainable=False)
        self.apply_within_chan_att = tf.Variable(True, trainable=False)

        time_attention_mask = np.zeros(
            (SEQ_LEN_LATENT, NUM_PATCHES + SEQ_LEN_UNPATCHED)
        )  # output x input

        # patch masking
        for i in range(NUM_PATCHES):  # i is indexing the input
            mm = (1 - NUM_PATCHES + i) * PATCH_LEN + SEQ_LEN_LATENT - 1
            mm = max(0, mm)
            time_attention_mask[0:mm, i] = 1

        # unpatched (tpts) masking
        for i in range(
            NUM_PATCHES, NUM_PATCHES + SEQ_LEN_UNPATCHED
        ):  # i is indexing the input
            time_attention_mask[
                0 : SEQ_LEN_LATENT - (NUM_PATCHES + SEQ_LEN_UNPATCHED) + i, i
            ] = 1

        self.time_attention_mask = tf.constant(time_attention_mask, dtype=tf.float32)

    def call(
        self,
        inputs,
        training,
    ):
        queries, chan_queries, keys_for_timeatt, keys_for_chanatt, values = inputs
        # queries are expected to be (B, H, L_Q, C, K)
        # keys_for_timeatt are expected to be (B, H, P+L_u, C, K)
        # keys_for_chanatt are expected to be (B, H, L_Q, C, K)
        # values are expected to be (B, H, P+L_u, C, K)

        ### Prepare time attention

        # reshape queries to (B, H, C, L_Q, K) so that we can compute time_attention_scores
        queries_trans = tf.transpose(queries, perm=(0, 1, 3, 2, 4))
        # reshape  keys to (B, H, C, P+L_u, K) so that we can compute time_attention_scores
        keys_trans = tf.transpose(keys_for_timeatt, perm=(0, 1, 3, 2, 4))

        key_dims = tf.shape(keys_for_timeatt)[-1]
        time_attention_scores = tf.matmul(
            queries_trans, keys_trans, transpose_b=True
        ) / tf.math.sqrt(
            tf.cast(key_dims, tf.float32)
        )  # [B, H, C, L_Q, P+L_u]

        # Apply mask by setting masked values to be -1e9
        # so that they are masked out when applying softmax

        time_attention_scores += -1e9 * self.time_attention_mask
        time_attention_scores = tf.math.softmax(
            time_attention_scores
        )  # [B, H, C, L_Q, P+L_u] # softmax will be done on the last dimension
        time_attention_scores = time_attention_scores * (1 - self.time_attention_mask)

        ### Prepare channel attention

        # dropout for chan attention
        if self.chan_att:
            if not training:
                self.apply_chan_att.assign(True)
                self.apply_within_chan_att.assign(True)
            else:
                # each batch has a probability of applying channel attention
                # channel attention is (or is not) then applied to the entire batch (over all sequences)
                self.apply_chan_att.assign(
                    tf.random.uniform([1], 0, 1)[0] > self.chanatt_dropout_rate
                )

                if self.apply_chan_att:
                    # within chan att can only be turned off if chan att is on
                    self.apply_within_chan_att.assign(
                        tf.random.uniform([1], 0, 1)[0]
                        > self.within_chanatt_dropout_rate
                    )
                else:
                    self.apply_within_chan_att.assign(True)
        else:
            self.apply_chan_att.assign(False)
            self.apply_within_chan_att.assign(True)

        if self.apply_chan_att:

            # do channel attention
            key_dims = tf.shape(keys_for_chanatt)[-1]

            chan_attention_scores = tf.matmul(
                chan_queries, keys_for_chanatt, transpose_b=True
            ) / tf.math.sqrt(
                tf.cast(key_dims, tf.float32)
            )  # [B, H, L_Q, C, C]

            if self.apply_within_chan_att:
                # within channel attention
                chan_attention_scores = chan_attention_scores * (
                    1 - tf.eye(tf.shape(chan_attention_scores)[3])
                )

                # Values in mask are set to one to mask that channel out
                # Mask out any within channel attention

                chan_attention_mask = tf.eye(tf.shape(chan_attention_scores)[3])

                # Apply mask by setting masked values to be -1e9
                # so that they are masked out when applying softmax
                chan_attention_scores += -1e9 * chan_attention_mask
                chan_attention_scores = tf.math.softmax(chan_attention_scores)
                chan_attention_scores = chan_attention_scores * (
                    1 - chan_attention_mask
                )
            else:
                # across channel attention
                chan_attention_scores = tf.math.softmax(chan_attention_scores)

        else:
            NUM_CHANS = tf.shape(queries)[3]
            chan_attention_scores = tf.eye(NUM_CHANS)  # [C, C]
            chan_attention_scores = tf.expand_dims(
                tf.expand_dims(tf.expand_dims(chan_attention_scores, axis=0), axis=0),
                axis=0,
            )  # [1, 1, C, C]

        ### Apply the attention scores to the values to get the output

        # Computing the output of the scaled dot product attention,

        reshape_values = tf.transpose(
            values, perm=(0, 1, 3, 2, 4)
        )  # (B, H, C, P+L_u, K)
        reshape_output = tf.matmul(
            time_attention_scores, reshape_values
        )  # (B, H, C, L_Q, K)
        reshape_output2 = tf.transpose(
            reshape_output, perm=(0, 1, 3, 2, 4)
        )  # (B, H, L_Q, C, K)
        output = tf.matmul(chan_attention_scores, reshape_output2)  # (B, H, L_Q, C, K)

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
        self.patch_projection = tf.keras.layers.Dense(n_heads)

        # Input projections
        self.time_patched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_unpatched_projection = tf.keras.layers.Dense(2 * self.model_dim)
        self.time_query_projection = tf.keras.layers.Dense(self.model_dim)
        self.channel_projection = tf.keras.layers.Dense(2 * self.model_dim)

        # PASSTA layer for time and channel attention
        self.attention_layer = DotProductAttention(
            latent_sequence_length,
            n_patches,
            patch_length,
            unpatched_length,
            True,
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
        output = self.attention_layer([q, c_q, k, c_k, v], training=training, **kwargs)
        # output.shape: (batch_size, n_heads, latent_sequence_length, n_channels, key_dim)

        # ---------- Combine heads ---------- #
        output = self._combine_heads(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        # ---------- Output projection ---------- #
        output = self.output_projection(output)
        # output.shape: (batch_size, latent_sequence_length, n_channels, model_dim)

        return output


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        NUM_HEADS,
        EMBED_DIM,
        NUM_PATCHES,
        SEQ_LEN_LATENT,  # L_Q
        PATCH_LEN,
        SEQ_LEN_UNPATCHED,
        chan_att,
        chanatt_dropout_rate,
        within_chanatt_dropout_rate,
        **kwargs,
    ):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

        self.attention = DotProductAttention(
            SEQ_LEN_LATENT,
            NUM_PATCHES,
            PATCH_LEN,
            SEQ_LEN_UNPATCHED,
            chan_att,
            chanatt_dropout_rate,
            within_chanatt_dropout_rate,
        )  # Scaled dot product attention
        self.NUM_HEADS = NUM_HEADS  # Number of attention heads to use
        self.EMBED_DIM = (
            EMBED_DIM  # Dimensionality of the linearly projected queries and keys
        )
        self.NUM_PATCHES = NUM_PATCHES
        self.PATCH_LEN = PATCH_LEN
        self.SEQ_LEN_UNPATCHED = SEQ_LEN_UNPATCHED
        self.SEQ_LEN_LATENT = SEQ_LEN_LATENT

        self.W_q = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the queries
        self.W_chan_q = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the queries
        self.W_o = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the multi-head output
        self.W_k_p = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the patched keys
        self.W_v_p = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the patched values
        self.W_k_u = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the unpatched keys
        self.W_chan_k_u = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the unpatched keys for channel attention
        self.W_v_u = tf.keras.layers.Dense(
            EMBED_DIM
        )  # Learned projection matrix for the unpatched values

        self.W_patches = tf.keras.layers.Dense(
            NUM_HEADS
        )  # Learned projection matrix for the unpatched values to get the patch values

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # x is expected to be (B, L, C, E)
            # Tensor shape after reshaping and transposing: (B, H, L or P, C, K)
            x2 = tf.reshape(
                x, shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], heads, -1)
            )  # (B, L or P, C, H, K)
            x3 = tf.transpose(x2, perm=(0, 3, 1, 2, 4))  # (B, H, L or P, C, K)
        else:
            # Reverting the reshaping and transposing operations: (B, L or P, C, E)
            x2 = tf.transpose(x, perm=(0, 2, 3, 1, 4))
            x3 = tf.reshape(
                x2,
                shape=(
                    tf.shape(x2)[0],
                    tf.shape(x2)[1],
                    tf.shape(x2)[2],
                    self.EMBED_DIM,
                ),
            )
        return x3

    def call(self, inputs, training):

        # Inputs are expected to be (B, L, C, E)
        # Note that L will vary depending on the layer, due to perceiver AR

        # For perceiver AR
        lq_inputs = tf.slice(
            inputs,
            [0, tf.shape(inputs)[1] - self.SEQ_LEN_LATENT, 0, 0],
            [-1, -1, -1, -1],
        )  # (B, L_Q, C, E)

        # inputs are expected to be (B, L, C, E)
        # Do patching
        patched_inputs = tf.reshape(
            inputs,
            shape=(
                tf.shape(inputs)[0],
                self.NUM_PATCHES,
                self.PATCH_LEN,
                tf.shape(inputs)[2],
                self.EMBED_DIM,
            ),
        )  # (B, P, L_P, C, E)
        patched_tmp = tf.transpose(
            patched_inputs, perm=(0, 1, 3, 4, 2)
        )  # (B, P, C, E, L_P)

        # Ultimately we want to map from (B, P, C, E, L_P) to (B, P, C, E):
        # Option 1: (B, P, C, E, L_P) -> (B, P, C, E*L_P) -[Dense(E)]-> (B, P, C, E)   High numparams, low RAM
        # Option 2: (B, P, C, E, L_P) -[Dense(1)]-> (B, P, C, E, 1) -> (B, P, C, E)    Low numparams, high RAM
        # Option 3: (B, P, C, E, L_P) -> (B, P, C, EonK, K, L_P) -> (B, P, C, EonK, K*L_P) -[Dense(K)]-> (B, P, C, EonK, K) -> (B, P, C, E)    Med numparams, Med RAM
        #
        # Option 3 is used here, as a tradeoff between numparams and RAM, with K=H

        EonK = self.EMBED_DIM // self.NUM_HEADS
        patched_tmp = tf.reshape(
            patched_tmp,
            (
                tf.shape(patched_tmp)[0],
                tf.shape(patched_tmp)[1],
                tf.shape(patched_tmp)[2],
                EonK,
                self.NUM_HEADS,
                self.PATCH_LEN,
            ),
        )  # (B, P, C, EonK, K, Lp)
        patched_tmp = tf.reshape(
            patched_tmp,
            (
                tf.shape(patched_tmp)[0],
                tf.shape(patched_tmp)[1],
                tf.shape(patched_tmp)[2],
                EonK,
                self.NUM_HEADS * self.PATCH_LEN,
            ),
        )  # (B, P, C, EonK, K*Lp)
        patched_dat = self.W_patches(patched_tmp)  # (B, P, C, EonK, K)
        patched_dat = tf.reshape(
            patched_dat,
            (
                tf.shape(patched_dat)[0],
                tf.shape(patched_dat)[1],
                tf.shape(patched_dat)[2],
                self.EMBED_DIM,
            ),
        )  # (B, P, C, E)

        q_reshaped = self.reshape_tensor(
            self.W_q(lq_inputs), self.NUM_HEADS, True
        )  # (B, H, L_Q, C, K)
        chan_q_reshaped = self.reshape_tensor(
            self.W_chan_q(lq_inputs), self.NUM_HEADS, True
        )  # (B, H, L_Q, C, K)

        # time keys
        k_lq = self.W_k_u(lq_inputs)  # (B, L_Q, C, E)

        # chan keys
        chan_k_lq = self.W_chan_k_u(lq_inputs)  # (B, L_Q, C, E)

        # keys for time attention
        k_unpatched = tf.slice(
            k_lq,
            [0, tf.shape(k_lq)[1] - self.SEQ_LEN_UNPATCHED, 0, 0],
            [-1, -1, -1, -1],
        )  # (B, L_u, C, E)
        k_patched = self.W_k_p(patched_dat)  # (B, P, C, E)
        k_full = tf.concat([k_patched, k_unpatched], axis=1)  # (B, P+L_u, C, E)
        k_reshaped_for_timeatt = self.reshape_tensor(k_full, self.NUM_HEADS, True)
        # Resulting tensor shape: (B, H, P+L_u, C, K)
        # Note E = HK

        # keys for chan attention
        chan_k_reshaped = self.reshape_tensor(chan_k_lq, self.NUM_HEADS, True)
        # Resulting tensor shape: (B, H, L_Q, C, K)

        # values (for time attention)
        v_patched = self.W_v_p(patched_dat)  # (B, P, C, E)
        patched_dat = tf.slice(
            inputs,
            [0, tf.shape(inputs)[1] - self.SEQ_LEN_UNPATCHED, 0, 0],
            [-1, -1, -1, -1],
        )  # (B, L_u, C, E)
        v_unpatched = self.W_v_u(patched_dat)  # (B, L_u, C, E)
        v_full = tf.concat([v_patched, v_unpatched], axis=1)  # (B, P+L_u, C, E)
        v_reshaped = self.reshape_tensor(v_full, self.NUM_HEADS, True)
        # Resulting tensor shape: (B, H, P+L_u, C, K)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(
            [
                q_reshaped,
                chan_q_reshaped,
                k_reshaped_for_timeatt,
                chan_k_reshaped,
                v_reshaped,
            ],
            training,
        )
        # Resulting tensor shape: (B, H, L_Q, C, K)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.NUM_HEADS, False)
        # Resulting tensor shape: (B, L_Q, C, E)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (B, L_Q, C, E)
        return self.W_o(output)
