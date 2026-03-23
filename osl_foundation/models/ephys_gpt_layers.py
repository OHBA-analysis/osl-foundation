from typing import List

import keras
import numpy as np
import tensorflow as tf

from osl_foundation.config.generator_config import Label
from osl_foundation.inference.gasp_layers import MultiHeadGASPLayer
from osl_foundation.inference.layers import (
    IdentityLayer,
    NormalizationLayer,
    PositionEmbedding,
    SinusoidalPositionalEncodingLayer,
)

class GumbelSoftmaxLayer(keras.layers.Layer):
    def __init__(self, temperature=1.0, hard=True, **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, inputs):
        # inputs should be the logits [Batch, Seq, Vocab]
        logits = inputs
        
        # 1. Generate Gumbel Noise
        u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        gumbel_noise = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
        
        # 2. Add noise and apply temperature
        y = tf.nn.softmax((logits + gumbel_noise) / self.temperature)
        
        if self.hard:
            # 3. Straight-Through Estimator
            y_hard = tf.one_hot(tf.argmax(y, axis=-1), depth=tf.shape(logits)[-1])
            y = tf.stop_gradient(y_hard - y) + y
            
        return y
        
    def get_config(self):
        config = super(GumbelSoftmaxLayer, self).get_config()
        config.update({
            "temperature": self.temperature,
            "hard": self.hard
        })
        return config

class ReconstructDataLayer(keras.layers.Layer):
    """
    Layer for reconstructing data from tokens.
    """

    def __init__(self, refactored2original_projection, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.refactored2original_projection = refactored2original_projection
        self.tokenizer = tokenizer
        self._token_basis_time_distributed = keras.layers.TimeDistributed(
            self.tokenizer.model.get_layer("decoder").token_basis_layer
        )

    def call(self, inputs, **kwargs):
        tokens = inputs
        # tokens.shape = (batch_size, sequence_length, n_channels)

        tokens_one_hot = tf.one_hot(tokens, depth=self.refactored2original_projection.shape[0])  # shape (B, T, C, n_refactored_tokens) 
        orig_dat_prob = tf.tensordot(tokens_one_hot, self.refactored2original_projection, axes=[[3], [0]])  # shape (B, T, C, n_original_tokens)
        orig_dat_prob = tf.transpose(orig_dat_prob, (0, 2, 1, 3))  # shape (B, C, T, n_original_tokens)
        y_recon = self._token_basis_time_distributed(orig_dat_prob)  # shape (B, n_channels, l_out, n_refactored_tokens)
        y_recon = tf.transpose(y_recon, (0, 2, 1, 3))  # shape (B, T, C, n_original_tokens)
        y_recon = tf.reduce_sum(y_recon, axis=-1)  # shape (B, l_out, n_channels)

        return y_recon
        
class SampleTokenLayer(keras.layers.Layer):
    """
    Layer for sampling tokens from logits.
    """

    def call(self, input):
        
        y_pred = input
        # y_pred.shape = (batch_size, buffer_sequence_length, n_channels, n_refactored_tokens)

        # Sample from y_pred
        y_pred_sample = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        # y_pred_sample.shape = (batch_size * buffer_sequence_length * n_channels, n_refactored_tokens)
        y_pred_sample = tf.random.categorical(y_pred_sample, num_samples=1)
        # y_pred_sample.shape = (batch_size * buffer_sequence_length * n_channels, 1)
        y_pred_sample = tf.reshape(y_pred_sample, [-1, y_pred.shape[1], y_pred.shape[2]])
        # y_pred_sample.shape = (batch_size, buffer_sequence_length, n_channels)

        return y_pred_sample
    
class ShiftTokenLayer(keras.layers.Layer):

    """
    This layer shifts the input tokens forward by one time step, 
    e.g. to create the input and target tokens for the loss calculation.
    """

    def call(self, x):
        # x.shape = (batch_size, sequence_length+1, n_channels)   

        x_plus_one = x[:, :-1]
        # x_input.shape = (batch_size, sequence_length, n_channels)

        x = x[:, 1:]
        # y_true.shape = (batch_size, sequence_length, n_channels)

        return x_plus_one, x

class CrossEntropyLossLayer(keras.layers.Layer):
    """
    Layer for calculating the cross-entropy loss.

    Parameters
    ----------
    loss_sequence_length : int
        Number of tokens to calculate the loss for.
    top_k : list, optional
        List of top k values to calculate the accuracy for.
        By default only top 1 accuracy is calculated.
    """

    def __init__(
        self,
        loss_sequence_length: int,
        top_k: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_sequence_length = loss_sequence_length
        self.top_k = top_k or [1]

        for k in self.top_k:
            setattr(
                self,
                f"top_{k}",
                keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=f"top_{k}"),
            )


        self.cross_entropy_loss_tracker = keras.metrics.Mean(name="cross_entropy_loss")

    def call(self, inputs, **kwargs):
        y_pred, y_true = inputs
        
        # y_true.shape = (batch_size, sequence_length, n_channels)
        # y_pred.shape = (batch_size, latent_sequence_length, n_channels, n_tokens)

        # Only calculate loss for the last loss_sequence_length tokens
        y_pred_loss = y_pred[:, -self.loss_sequence_length :]
        y_true_loss = y_true[:, -self.loss_sequence_length :]
        # y_true.shape = (batch_size, loss_sequence_length, n_channels)
        # y_pred.shape = (batch_size, loss_sequence_length, n_channels, n_tokens)

        loss = keras.metrics.sparse_categorical_crossentropy(
            y_true_loss, y_pred_loss, from_logits=True
        )
        # loss.shape = (batch_size, loss_sequence_length, n_channels)

        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # sum over channels, mean over batch and sequence length
        # loss.shape = scalar

        self.cross_entropy_loss_tracker.update_state(loss)
        
        self.add_loss(loss)
        for k in self.top_k:
            getattr(self, f"top_{k}").update_state(y_true_loss, y_pred_loss)

        return tf.expand_dims(loss, -1), y_pred_loss


class LyapunovLossLayer(keras.layers.Layer):
    """
    Layer for calculating the Lyapunov regularization loss
    using a learned Lyapunov function.

    Parameters
    ----------
    loss_sequence_length : int
        Number of tokens to calculate the loss for.
    beta : float
        Weighting for the Lyapunov loss.
    mu : float
        Weighting for the V(0) anchoring term.
    collapse_weight : float
        Weight for anti-collapse regularization on V statistics.
    collapse_target_var : float
        Minimum target variance of V over batch/time.
    collapse_target_mean : float
        Minimum target mean of V over batch/time.
    dim : int
        Dimension of the Lyapunov function output.

    """

    def __init__(
        self,
        loss_sequence_length: int,
        beta: float = 1.0,
        mu: float = 10.0,
        collapse_weight: float = 0.0,
        collapse_target_var: float = 1e-4,
        collapse_target_mean: float = 1e-4,
        dim: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_sequence_length = loss_sequence_length
        self.beta = keras.Variable(
            beta, trainable=False, dtype=tf.float32, name="lyapunov_beta"
        )
        self.mu = keras.Variable(
            mu, trainable=False, dtype=tf.float32, name="lyapunov_mu"
        )
        self.collapse_weight = keras.Variable(
            collapse_weight,
            trainable=False,
            dtype=tf.float32,
            name="lyapunov_collapse_weight",
        )
        self.collapse_target_var = tf.constant(
            collapse_target_var, dtype=tf.float32
        )
        self.collapse_target_mean = tf.constant(
            collapse_target_mean, dtype=tf.float32
        )

        self.dim = dim

        # MLP for Lyapunov function
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(self.dim),
            ]
        )

        self.loss_tracker = keras.metrics.Mean(name="lyapunov_loss")        
        self.core_loss_tracker = keras.metrics.Mean(name="lyapunov_core_loss")
        self.v0_loss_tracker = keras.metrics.Mean(name="lyapunov_v0_loss")
        self.mu_tracker = keras.metrics.Mean(name="lyapunov_mu")
        self.collapse_loss_tracker = keras.metrics.Mean(name="lyapunov_collapse_loss")
        self.collapse_ratio_tracker = keras.metrics.Mean(name="lyapunov_collapse_ratio")
        self.v_var_tracker = keras.metrics.Mean(name="lyapunov_v_var")
        self.v_mean_tracker = keras.metrics.Mean(name="lyapunov_v_mean")

    def _compute_V(self, xx):
        # xx.shape = (batch_size, 1, n_channels, embedding_dim)
        xx = tf.reshape(xx, (tf.shape(xx)[0], tf.shape(xx)[1], tf.shape(xx)[2] * tf.shape(xx)[3]))
        # xx.shape = (batch_size, 1, n_channels * embedding_dim)

        V_t = self.mlp(xx)
        # V_t.shape = (batch_size, 1, dim)

        # squared norm over lyapunov dimension
        V_t = tf.reduce_sum(V_t**2, axis=-1)
        # V_t.shape = (batch_size, 1)

        return V_t
    
    def call(self, inputs, **kwargs):

        y_recon_plus_one, y_recon = inputs
        # y_recon shape = (batch_size, >=loss_sequence_length, n_channels, embedding_dim)
        # y_recon_plus_one shape = (batch_size, >=loss_sequence_length, n_channels, embedding_dim)

        # Only calculate loss for the last loss_sequence_length tokens
        y_recon_plus_one = y_recon_plus_one[:, -self.loss_sequence_length :, :, :]
        y_recon = y_recon[:, -self.loss_sequence_length :, :, :]
        # y_recon.shape = (batch_size, loss_sequence_length, n_channels, embedding_dim)

        # Lyapunov loss: V(t+1) - V(t) should be less than or equal to 0
        # Inputs are already temporally aligned before this layer.
        V_t_plus_one = self._compute_V(y_recon_plus_one)
        V_t = self._compute_V(y_recon)
        # loss.shape = (batch_size, loss_sequence_length)

        lyapunov_fn = tf.nn.relu(V_t_plus_one - V_t)
        # lyapunov_fn.shape = (batch_size, loss_sequence_length)

        core_loss = tf.reduce_mean(lyapunov_fn) # mean over batch and sequence length
        # core_loss.shape = scalar

        # Anchor Lyapunov scale by pushing V(0)->0, so the origin is a low-energy reference.
        V_0 = self._compute_V(tf.zeros_like(y_recon_plus_one))
        v0_loss = tf.reduce_mean(V_0)
        v_values = tf.concat([V_t, V_t_plus_one], axis=1)
        v_var = tf.math.reduce_variance(v_values)
        v_mean = tf.reduce_mean(v_values)
        # Normalized squared hinge keeps anti-collapse pressure scale-stable:
        # if targets are small, the penalty is still O(1) when collapsed.
        var_gap = tf.nn.relu(
            1.0 - v_var / tf.maximum(self.collapse_target_var, 1e-12)
        )
        mean_gap = tf.nn.relu(
            1.0 - v_mean / tf.maximum(self.collapse_target_mean, 1e-12)
        )
        collapse_loss = tf.square(var_gap) + tf.square(mean_gap)
        collapse_ratio = collapse_loss / tf.maximum(core_loss, 1e-12)
        loss = core_loss + self.mu * v0_loss + self.collapse_weight * collapse_loss
        loss *= self.beta

        lyapunov_disabled = tf.logical_and(
            tf.equal(self.beta, 0.0),
            tf.logical_and(
                tf.equal(self.mu, 0.0),
                tf.equal(self.collapse_weight, 0.0),
            ),
        )
        core_loss_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(core_loss), core_loss
        )
        v0_loss_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(v0_loss), v0_loss
        )
        collapse_loss_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(collapse_loss), collapse_loss
        )
        collapse_ratio_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(collapse_ratio), collapse_ratio
        )
        v_var_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(v_var), v_var
        )
        v_mean_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(v_mean), v_mean
        )
        mu_for_metrics = tf.where(
            lyapunov_disabled, tf.zeros_like(self.mu), self.mu
        )

        self.loss_tracker.update_state(loss) 
        self.core_loss_tracker.update_state(core_loss_for_metrics)
        self.v0_loss_tracker.update_state(v0_loss_for_metrics)
        self.mu_tracker.update_state(mu_for_metrics)
        self.collapse_loss_tracker.update_state(collapse_loss_for_metrics)
        self.collapse_ratio_tracker.update_state(collapse_ratio_for_metrics)
        self.v_var_tracker.update_state(v_var_for_metrics)
        self.v_mean_tracker.update_state(v_mean_for_metrics)

        self.add_loss(loss)

        return tf.expand_dims(loss, -1), lyapunov_fn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1), (input_shape[0][0], self.loss_sequence_length)

class InputEmbeddingLayer(keras.layers.Layer):
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
    pos_embedding_type : str, optional
        Type of the position embeddings. Defaults to "absolute", which uses learned
        absolute position embeddings.
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
        pos_embedding_type: str = "absolute",
        channel_embedding_dim: int = None,
        extra_labels: List[Label] = None,
        pretrained_layer: keras.layers.Layer = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.token_embedding_dim = token_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim or embedding_dim
        self.pos_embedding_type = pos_embedding_type
        self.channel_embedding_dim = channel_embedding_dim or embedding_dim
        self.extra_labels = extra_labels
        self.use_pretrained_layer = pretrained_layer is not None

        # ---------- Initialize layers ----------

        if pretrained_layer is None:
            # The token embedding layer
            self.token_embedding_layer = keras.layers.Embedding(
                n_tokens, token_embedding_dim or embedding_dim
            )
            self.token_embedding_output_layer = (
                IdentityLayer()
                if token_embedding_dim is None
                else keras.layers.Dense(embedding_dim)
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
                    else keras.layers.Dense(embedding_dim)
                )

            # The channel embedding layer
            self.channel_embedding_layer = PositionEmbedding(
                sequence_length=n_channels, trainable=True
            )
            self.channel_embedding_output_layer = (
                IdentityLayer()
                if channel_embedding_dim is None
                else keras.layers.Dense(embedding_dim)
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
                keras.layers.Embedding(n_classes, label.label_dim or embedding_dim)
            )
            self.extra_embedding_output_layers.append(
                IdentityLayer()
                if label.label_dim is None
                else keras.layers.Dense(embedding_dim)
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

        embeddings += self.channel_embedding_output_layer(
            self.channel_embedding_layer(channels, training=training_1),
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

            label = tf.expand_dims(label[:, :-1], -1)
            # label.shape = (batch_size, sequence_length, 1)

            embeddings += output_layer(
                layer(label, training=training), training=training
            )

        return embeddings

class DecoderLayer(keras.Model):
    def __init__(
            self,
            n_heads: int,
            model_dim: int,
            n_channels: int,
            n_patches_out: int,
            patch_length_out: int,
            n_patches_in: int,
            patch_length_in: int,        
            unpatched_length_in: int,
            l_unpatched_b: int,
            l_patched_b: int,
            do_chan_attention: bool,
            do_cross_attention: bool,
            chan_attention_mask: np.array,
            chan_att_chandim: int,
            feed_forward_dim: int,
            feed_forward_activation: str,
            dropout: float,
            norm_type: str,
            n_groups: int = None,
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.model_dim = model_dim
            self.do_chan_attention = do_chan_attention
            self.do_cross_attention = do_cross_attention
            self.seq_len_out = n_patches_out*patch_length_out

            # ---------- Chan att ----------

            if do_chan_attention:

                self.chan_attention_dropout_layer = keras.layers.Dropout(dropout, name = f"chan_attention_dropout_layer")
                
                # Normalization 1: before chan attention
                self.normalization_layer_chan1 = NormalizationLayer(norm_type, n_groups, name = f"normalization_chan1_layer")

                # Dense layer to project to chan_att_chandim
                # Shared across all layers and heads
                if chan_att_chandim is None:
                    self.chan_att_chandim_dense_layer = None
                else:
                    self.chan_att_chandim_dense_layer = keras.layers.Dense(
                        chan_att_chandim, name=f"chan_att_chandim"
                    )

                chan_att_modeldim = model_dim

                # Multi-head space attention layers
                self.gasp_chan_attention_layer = MultiHeadGASPLayer(
                        n_heads=n_heads,
                        model_dim=chan_att_modeldim,
                        l_other=n_patches_in*patch_length_in,
                        n_patches_out=n_channels,
                        patch_length_out=1,
                        n_patches_in=chan_att_chandim,
                        patch_length_in=1,
                        unpatched_length_in=0,
                        causal=False,
                        l_unpatched_b=None,
                        l_patched_b=None,
                        attention_mask=chan_attention_mask,
                        name = f"gasp_chan_attention_layer"
                    )
    
                # Normalization 2: before feed-forward layer
                self.normalization_layer_chan2 = NormalizationLayer(norm_type, n_groups, name = f"normalization_chan2_layer")

                # Feed-forward chan att layers
                self.feed_forward_layer_chan = keras.Sequential(
                        [
                            keras.layers.Dense(
                                feed_forward_dim, activation=feed_forward_activation
                            ),
                            keras.layers.Dense(model_dim),
                            keras.layers.Dropout(dropout),
                        ], 
                        name = f"feed_forward_chan_layer"
                    )
                    
            else:
                self.chan_attention_dropout_layer = None
                self.normalization_layer_chan1 = None 
                self.gasp_chan_attention_layer = None
                self.normalization_layer_chan2 = None
                self.feed_forward_layer_chan = None

            # ---------- Time attention layers ----------    

            self.time_attention_dropout_layer = keras.layers.Dropout(dropout, name = f"time_attention_dropout_layer")

            # Normalization 1: before time attention
            self.normalization_layer_time1 = NormalizationLayer(norm_type, n_groups, name = f"normalization_time1_layer")

            n_pat_out = n_patches_in if do_cross_attention else n_patches_out
            pat_length_out = patch_length_in if do_cross_attention else patch_length_out

            self.gasp_time_attention_layer = MultiHeadGASPLayer(
                        n_heads,
                        model_dim,
                        n_channels,
                        n_pat_out,
                        pat_length_out,
                        n_patches_in,
                        patch_length_in,
                        unpatched_length_in,
                        True,
                        l_unpatched_b,
                        l_patched_b,
                        None,
                        name = f"gasp_time_attention_layer"
                    )

            if do_chan_attention and do_cross_attention:

                # Multi-head space attention layers
                self.gasp_cross_attention_layer = MultiHeadGASPLayer(
                        n_heads,
                        model_dim,
                        n_channels,
                        n_patches_out,
                        patch_length_out,
                        n_patches_in,
                        patch_length_in,
                        0,
                        True,
                        l_unpatched_b,
                        l_patched_b,
                        None,
                        name = f"gasp_cross_attention_layer"
                    )
                        
            else:
                self.gasp_cross_attention_layer = None

            # Normalization 2: before feed-forward layer
            self.normalization_layer_time2 = NormalizationLayer(norm_type, n_groups, name = f"normalization_time2_layer")

            # Feed-forward time att layers
            self.feed_forward_layer_time = keras.Sequential(
                        [
                            keras.layers.Dense(
                                feed_forward_dim, activation=feed_forward_activation
                            ),
                            keras.layers.Dense(model_dim),
                            keras.layers.Dropout(dropout),
                        ],
                        name = f"feed_forward_time_layer"
                    )

    def build(self, input_shape):

        ongoing_shape = input_shape # (batch_size, sequence_length, n_channels, model_dim)

        # Build chan attention
        if self.do_chan_attention:
            
            # transpose
            ongoing_shape =  list(np.array(ongoing_shape)[(0, 2, 1, 3),]) 
            # (batch_size, n_channels, sequence_length, model_dim)
            
            self.chan_attention_dropout_layer.build(ongoing_shape)
            self.normalization_layer_chan1.build(ongoing_shape)

            # (batch_size, n_channels, sequence_length, chan_att_modeldim or model_dim)

            if self.do_chan_attention and self.chan_att_chandim_dense_layer is not None:  
                # transpose
                ongoing_shape_chandim =  list(np.array(ongoing_shape)[(0, 2, 3, 1),])
                # (batch_size, sequence_length, model_dim or chan_att_modeldim, n_channels)
                ongoing_shape_chandim = self.chan_att_chandim_dense_layer.compute_output_shape(ongoing_shape_chandim) 
                # (batch_size, sequence_length, model_dim or chan_att_modeldim, chan_att_chandim)
                # transpose
                ongoing_shape_chandim =  list(np.array(ongoing_shape_chandim)[(0, 3, 1, 2),])
                # (batch_size, chan_att_chandim, sequence_length, model_dim or chan_att_modeldim)
            else:
                ongoing_shape_chandim = ongoing_shape
                # (batch_size, n_channels, sequence_length, modeldim or chan_att_modeldim)

            self.gasp_chan_attention_layer.build([ongoing_shape, ongoing_shape_chandim, ongoing_shape_chandim]) 
            ongoing_shape = self.gasp_chan_attention_layer.compute_output_shape([ongoing_shape, ongoing_shape, ongoing_shape])
            # (batch_size, n_channels, sequence_length, model_dim or chan_att_modeldim)
            
            # tf.transpose
            ongoing_shape = list(np.array(ongoing_shape)[(0, 2, 1, 3),])
            # (batch_size, sequence_length, n_channels, model_dim)

            self.normalization_layer_chan2.build(ongoing_shape)
            self.feed_forward_layer_chan.build(ongoing_shape)

        # Build time attention
        self.time_attention_dropout_layer.build(ongoing_shape)
        self.normalization_layer_time1.build(ongoing_shape)
        self.gasp_time_attention_layer.build([ongoing_shape, ongoing_shape, ongoing_shape]) 
        # ongoing_shape[1] = self.gasp_time_attention_output_seqlen
        ongoing_shape = self.gasp_time_attention_layer.compute_output_shape([ongoing_shape, ongoing_shape, ongoing_shape]) 

        if self.do_chan_attention and self.do_cross_attention:

            self.gasp_cross_attention_layer.build([ongoing_shape, ongoing_shape, ongoing_shape]) 
            ongoing_shape = self.gasp_cross_attention_layer.compute_output_shape([ongoing_shape, ongoing_shape, ongoing_shape]) 

        # ongoing_shape = (batch_size, output_seqlens, n_channels, model_dim)

        self.normalization_layer_time2.build(ongoing_shape)
        self.feed_forward_layer_time.build(ongoing_shape)
        
        # Set the built flag to True
        super().build(input_shape)

    def compute_output_shape(self, input_shape):

        output_shape = input_shape
        output_shape[1] = self.seq_len_out

        return output_shape       

    def call(self, inputs, training=None, **kwargs):

        x, chan_attention_weight = inputs 
        # x is (batch_size, sequence_length, n_channels, model_dim)

        if self.do_chan_attention:

            # Channel attention
            xt = tf.transpose(x, perm=(0, 2, 1, 3)) # (batch_size, n_channels, sequence_length, model_dim)

            # Residual connection
            xt_residual = xt

            xt = self.normalization_layer_chan1(xt, training=training, **kwargs)

            if self.chan_att_chandim_dense_layer is not None:                
                xt_dim = tf.transpose(xt, perm=(0, 2, 3, 1))  # (batch_size, sequence_length, model_dim or chan_att_modeldim, n_channels)
                xt_dim = self.chan_att_chandim_dense_layer(xt_dim, training=training, **kwargs) # (batch_size, sequence_length, model_dim or chan_att_modeldim)
                xt_dim = tf.transpose(xt_dim, perm=(0, 3, 1, 2))  # (batch_size, chan_att_chandim, sequence_length, model_dim)
            else:
                xt_dim = xt # (batch_size, n_channels, sequence_length, modeldim)

            # input: query, key, value
            xt = self.gasp_chan_attention_layer([xt, xt_dim, xt_dim], training=training, **kwargs) # (batch_size, n_channels, sequence_length, model_dim)

            xt = self.chan_attention_dropout_layer(xt,  training=training, **kwargs)

            # Add residual
            xt = chan_attention_weight * xt + xt_residual

            x_chan = tf.transpose(xt, perm=(0, 2, 1, 3))

            # Residual connection
            x_residual = x_chan

            x_chan = self.normalization_layer_chan2(x_chan, training=training, **kwargs)

            x_chan = self.feed_forward_layer_chan(x_chan, training=training, **kwargs)

            # Add residual
            x_chan = chan_attention_weight * x_chan + x_residual

            if not self.do_cross_attention:
                x = x_chan

        #### Time attention

        # Residual connection
        x_residual = x

        x_time = self.normalization_layer_time1(x, training=training, **kwargs)

        # GASP Time attention layer
        x_time = self.gasp_time_attention_layer([x_time, x_time, x_time], training=training, **kwargs)
        x_time = self.time_attention_dropout_layer(x_time, training=training, **kwargs)

        # Add to residual, adjust seq_len of x_residual to match x 
        x_time = x_time + x_residual[:, -tf.shape(x_time)[1] :]

        #### Cross attention: https://arxiv.org/pdf/2503.17658v1

        if self.do_cross_attention and self.do_chan_attention:

            # Residual connection
            x_residual = x_time

            # GASP cross attention layer, inputs are  [query, key, value]
            x_time = self.gasp_cross_attention_layer([x_time, x_chan, x_chan], training=training, **kwargs)

            # Add to residual, adjust seq_len of x_residual to match x 
            x_time = x_time + x_residual[:, -tf.shape(x_time)[1] :]

        x = x_time

        ####

        # Residual connection
        x_residual = x

        x = self.normalization_layer_time2(x, training=training, **kwargs)

        x = self.feed_forward_layer_time(x, training=training, **kwargs)

        # Add to residual
        x = x + x_residual

        return x

#class TransformerDecoder(keras.Model):
class TransformerDecoder(keras.layers.Layer):
    """
    Transfomer decoder used in the EphysGPT model.

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
        Number of channels in the input data
    n_patches_out : List[int]
        Number of patches out for each layer
    patch_length_out : List[int]
        Length of the patches out   
    n_patches_in : List[int]
        Number of patches in for each layer
    patch_length_in : List[int]
        Length of the patches in  for each layer
    unpatched_length_in : List[int]
        Length of the unpatched sequence for each layer
        Length of the unpatched sequence for each layer
    l_patched_b : List[int]
        Length of the patched sequence for each layer
    do_chan_attention : List[bool]
        Whether to use channel attention for each layer
    do_cross_attention : List[int]
        Whether to use cross attention for each layer
    chan_attention_mask : List[np.array]
        Channel attention mask for each layer
    chan_att_chandim : int
        Channel attention dimension
    full_channel_attention_dropout : float
        Dropout rate for all channel attention for all layers.
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
        n_heads: int,
        model_dim: int,
        embedding_dim: int,
        n_channels: int,
        n_patches_out: list,
        patch_length_out: list,
        n_patches_in: list,
        patch_length_in: list,        
        unpatched_length_in: list,
        l_unpatched_b: list,
        l_patched_b: list,
        do_chan_attention: list,
        do_cross_attention: list,
        chan_attention_mask: list,
        chan_att_chandim: int,
        full_channel_attention_dropout: float,  
        feed_forward_dim: int,
        feed_forward_activation: str,
        dropout: float,
        norm_type: str,
        n_groups: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_layers = len(n_patches_out)
        self.model_dim = model_dim
        self.do_chan_attention = do_chan_attention
        self.do_cross_attention = do_cross_attention
        self.n_layers = n_layers

        # ---------- Input ----------

        # Input dropout layer
        self.input_dropout_layer = keras.layers.Dropout(dropout, name="input_dropout")
        self.input_projection_layer = (
            IdentityLayer()
            if model_dim == embedding_dim
            else keras.layers.Dense(model_dim, name="input_projection")
        )

        # ---------- Decoder layers ----------

        self.decoder_layers = []
        # Dropout rate for channel attention. Shared across all layers and heads.
        self.full_channel_attention_dropout = keras.Variable(
            full_channel_attention_dropout, trainable=False, dtype=tf.float32
        )
        self.chan_attention_weight = tf.Variable(1.0, trainable=False)
        self.chan_attention_weight_tracker = keras.metrics.Mean(
            name="chan_attention_weight"
        )
        self._metrics.append(self.chan_attention_weight_tracker)

        for ii in range(n_layers):

            print(f"Layer {ii}, time att seq_len: {n_patches_in[ii]*patch_length_in[ii]} -> {n_patches_out[ii]*patch_length_out[ii]}")
            if ii > 0:
                if (
                    n_patches_out[ii - 1] * patch_length_out[ii - 1]
                    != n_patches_in[ii] * patch_length_in[ii]
                ):
                    raise ValueError(
                        "Output sequence length of previous layer does not match "
                        "input sequence length of current layer."
                    )      
            
            self.decoder_layers.append(
                DecoderLayer(
                    n_heads = n_heads,
                    model_dim = model_dim,
                    n_channels = n_channels,
                    n_patches_out = n_patches_out[ii],
                    patch_length_out = patch_length_out[ii],
                    n_patches_in = n_patches_in[ii],
                    patch_length_in = patch_length_in[ii],        
                    unpatched_length_in = unpatched_length_in[ii],
                    l_unpatched_b = l_unpatched_b[ii],
                    l_patched_b = l_patched_b[ii],
                    do_chan_attention = do_chan_attention[ii],
                    do_cross_attention = do_cross_attention[ii],
                    chan_attention_mask = chan_attention_mask[ii],
                    feed_forward_dim = feed_forward_dim,
                    feed_forward_activation = feed_forward_activation,
                    chan_att_chandim = chan_att_chandim,
                    dropout = dropout,
                    norm_type = norm_type,
                    n_groups = n_groups,
                    name = f"decoder_layer_{ii}"
                )
            )

    def build(self, input_shape):

        # input_shape = (batch_size, sequence_length, n_channels, embedding_dim)

        # Build the projection and dropout layers
        self.input_projection_layer.build(input_shape)
        self.input_dropout_layer.build(input_shape)

        model_dim = self.model_dim

        proj_shape = list(input_shape)
        proj_shape[-1] = self.model_dim
        # proj_shape = (batch_size, sequence_length, n_channels, model_dim)

        for ii in range(self.n_layers):
            self.decoder_layers[ii].build(proj_shape)
            proj_shape = self.decoder_layers[ii].compute_output_shape(proj_shape)

        # Set the built flag to True
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        
        # inputs.shape = (batch_size, sequence_length, n_channels, embedding_dim)
        x = inputs

        x = self.input_projection_layer(x)
        # x.shape = (batch_size, sequence_length, n_channels, model_dim)

        x = self.input_dropout_layer(x, training=training, **kwargs)

        self.chan_attention_weight.assign(1)
        if training:
            tf.cond(
                pred=tf.random.uniform(()) < self.full_channel_attention_dropout,
                true_fn=lambda: self.chan_attention_weight.assign(0),
                false_fn=lambda: self.chan_attention_weight.assign(1),
            )

        self.chan_attention_weight_tracker.update_state(self.chan_attention_weight)

        for ii in range(self.n_layers):
            x = self.decoder_layers[ii]([x, self.chan_attention_weight], training=training, **kwargs)

        # x.shape = (batch_size, len_out, n_channels, model_dim)

        return x
