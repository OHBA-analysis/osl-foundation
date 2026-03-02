import tensorflow as tf
import keras
import numpy as np
from typing import List

from osl_foundation.config.generator_config import Label
from osl_foundation.inference.layers import (
    IdentityLayer,
    SinusoidalPositionalEncodingLayer,
    NormalizationLayer,
    PositionEmbedding,
)
from osl_foundation.inference.gasp_layers import MultiHeadGASPLayer


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
            pat_length_out = patch_length_in if do_cross_attention else patch_length_in

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

        output_shape = list(input_shape)
        output_shape[1] = self.seq_len_out

        return tuple(output_shape)      

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

        # Build the model with defined shapes
        self.model(self._build_inputs(), training=False)

        #self.model._metrics.append(self.transformer_decoder._metrics)

    def _extract_data_shape(self, obj):
        """Best-effort extraction of the shape for the 'data' input."""
        if isinstance(obj, tf.data.Dataset):
            return self._extract_data_shape(obj.element_spec)
        if isinstance(obj, dict):
            if "data" in obj and hasattr(obj["data"], "shape"):
                return tuple(obj["data"].shape)
            for value in obj.values():
                shape = self._extract_data_shape(value)
                if shape is not None:
                    return shape
            return None
        if isinstance(obj, (tuple, list)):
            for value in obj:
                shape = self._extract_data_shape(value)
                if shape is not None:
                    return shape
            return None
        if hasattr(obj, "shape"):
            return tuple(obj.shape)
        return None

    def _get_expected_data_shape(self):
        input_shape = self.model.input_shape
        if isinstance(input_shape, dict):
            return tuple(input_shape.get("data"))
        if isinstance(input_shape, list):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def _shapes_compatible(self, expected, found):
        if expected is None or found is None:
            return True
        if len(expected) != len(found):
            return False
        for e, f in zip(expected[1:], found[1:]):  # ignore batch dim
            if e is not None and f is not None and int(e) != int(f):
                return False
        return True

    def _validate_fit_data_shape(self, args, kwargs):
        x_fit = get_argument(self.model.fit, "x", args, kwargs)
        found_shape = self._extract_data_shape(x_fit)
        expected_shape = self._get_expected_data_shape()
        if self._shapes_compatible(expected_shape, found_shape):
            return

        raise ValueError(
            "Training data shape is incompatible with this model.\n"
            f"Expected 'data' shape: {expected_shape}\n"
            f"Found 'data' shape:    {found_shape}\n"
            "This usually means on-disk tokenized/TFRecord data was created with a different "
            "sequence length or channel count than the current model config.\n"
            "Fix: regenerate tokenized training data (or update config.yml) so shapes match."
        )

    def load_weights(self, filepath: str) -> keras.Model:
        """Load weights with backward-compatibility for decoder non-trainable vars."""
        try:
            return super().load_weights(filepath)
        except ValueError as e:
            msg = str(e)
            if "Layer 'decoder' expected" not in msg:
                raise
            _logger.warning(
                "Decoder variable mismatch while loading weights; retrying with skip_mismatch=True."
            )
            with self.model.distribute_strategy.scope():
                opt = self.model.optimizer
                opt.build(self.model.trainable_variables)
                return self.model.load_weights(filepath, skip_mismatch=True)

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
            x_val = get_argument(self.model.fit, "validation_data", args, kwargs)

            # Tokenise the data and build Data object
            tokenized_x = self.tokenizer.tokenize_data(x)
            tokenized_x = Data(
                tokenized_x,
                store_dir=f"{getattr(x, 'store_dir', 'tmp')}/tokenized",
                use_tfrecord=use_tfrecord,
                n_jobs=n_jobs,
            )
            tokenized_x.session_labels = x.session_labels

            if x_val is not None:
                tokenized_x_val = self.tokenizer.tokenize_data(x_val)
                tokenized_x_val = Data(
                    tokenized_x_val,
                    store_dir=f"{getattr(x_val, 'store_dir', 'tmp')}/tokenized_val",
                    use_tfrecord=use_tfrecord,
                    n_jobs=n_jobs,
                )
                tokenized_x_val.session_labels = x_val.session_labels
            else:
                tokenized_x_val = None
        else:
            tokenized_x = get_argument(self.model.fit, "x", args, kwargs)
            tokenized_x_val = get_argument(
                self.model.fit, "validation_data", args, kwargs
            )

        validation_split = get_argument(
            self.model.fit, "validation_split", args, kwargs
        )

        if validation_split is None:
            args, kwargs = replace_argument(
                self.model.fit,
                "x",
                tokenized_x,
                args,
                kwargs,
            )
            if tokenized_x_val:
                args, kwargs = replace_argument(
                    self.model.fit,
                    "validation_data",
                    tokenized_x_val,
                    args,
                    kwargs,
                )
        else:
            dataset = self.make_dataset(
                tokenized_x,
                shuffle=True,
                concatenate=True,
                step_size=step_size,
                drop_last_batch=True,
                validation_split=validation_split,
            )

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

        self._validate_fit_data_shape(args, kwargs)

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

    def _load_pretrained_model(self) -> keras.Model:
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

    def _build_inputs(self):
        config = self.config.model_config
        inputs = {
            "data": tf.zeros(
                [1, config.sequence_length + 1, config.n_channels], dtype=tf.int32
            )
        }
        for label in config.extra_labels:
            inputs[label.name] = tf.zeros(
                [1, config.sequence_length + 1], dtype=tf.int32
            )
        return inputs

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
            config.token_embedding_dim,
            config.pos_embedding_dim,
            config.pos_embedding_type,
            config.channel_embedding_dim,
            config.extra_labels,
            pretrained_layer=pretrained_layer,
            name="input_embedding",
        )

    def _get_transformer_decoder(self) -> TransformerDecoder:
        config = self.config.model_config
        if "decoder" in self.pretrained_layers:
            return self.pretrained_model.model.get_layer("decoder")
        else:
            return TransformerDecoder(
                config.n_heads,
                config.model_dim,
                config.embedding_dim,
                config.n_channels,
                config.n_patches_out,
                config.patch_length_out,
                config.n_patches_in,
                config.patch_length_in,
                config.unpatched_length_in,
                config.l_unpatched_b,
                config.l_patched_b,
                config.do_chan_attention,
                config.do_cross_attention,
                config.chan_attention_mask,
                config.chan_attention_chandim,
                config.full_channel_attention_dropout,
                config.feed_forward_dim,
                config.feed_forward_activation,
                config.dropout,
                config.norm_type,
                config.n_groups,
                name="decoder",
            )
    
    def _get_prediction_head_layer(self) -> keras.layers.Layer:
        config = self.config.model_config
        if "prediction_head" in self.pretrained_layers:
            return self.pretrained_model.model.get_layer("prediction_head")
        else:
            return keras.layers.Dense(config.n_tokens, name="prediction_head")
    
    def _get_refactored2original(self):

        if not hasattr(self, "refactored2original") or self.refactored2original is None:
            n_refactored_tokens = self.config.model_config.n_tokens
            if self.tokenizer is None:
                # No tokenizer remapping available. Keep token indices unchanged.
                self.refactored2original = tf.eye(n_refactored_tokens, dtype=tf.float32)
                return self.refactored2original

            n_original_tokens = self.tokenizer.config.model_config.n_tokens

            refactored2original = np.zeros((n_refactored_tokens, n_original_tokens))
            for i in range(n_refactored_tokens):
                orig_token = self._get_orig_token_index(i)
                refactored2original[i, orig_token] = 1.0

            # convert to keras tensor
            self.refactored2original = tf.convert_to_tensor(refactored2original, dtype=tf.float32)  

        return self.refactored2original
    
    def _get_orig_token_index(self, factored_token_index):
        if self.tokenizer is None:
            return factored_token_index

        # check if self.refactored2orig_token_index exists
        if not hasattr(self, "refactored2orig_token_index") or self.refactored2orig_token_index is None:
            n_tokens_refactored = self.config.model_config.n_tokens
            orig2refactored_token_index = self.tokenizer.vocab["label_map"] - 1
            # -1 indicates that the original token has been removed in the refactored tokenizer

            self.refactored2orig_token_index = -np.ones((n_tokens_refactored,), dtype=np.int32)
            for original_token_idx, refactored_token_idx in enumerate(orig2refactored_token_index):
                if refactored_token_idx > -1:
                    self.refactored2orig_token_index[refactored_token_idx] = original_token_idx

        return self.refactored2orig_token_index[factored_token_index]
    
    def reconstruct_data_from_tokens(self, yy):
        
        '''
        Reconstruct data from refactored tokens using the token basis layer.

        Parameters
        ----------
        yy : tf.Tensor
            Tensor of shape (B, l_out, n_channels, n_refactored_tokens)
            
        Returns
        -------
        y_recon : tf.Tensor
        Reconstructed data of shape (B, l_out, n_channels)
        '''

        if self.tokenizer is None:
            raise ValueError(
                "Cannot reconstruct data without a tokenizer. Set model_config.tokenizer_path."
            )

        # Apply TimeDistributed(Conv1D)
        # Conv1D will convolve over the T dimension (axis 2 of x_permuted)
        # The same Conv1D kernel will be applied to all C 'time steps' (axis 1)

        refactored2original_projection = self._get_refactored2original()  # shape (n_refactored_tokens, n_original_tokens)

        y_recon = ReconstructDataLayer(refactored2original_projection, self.tokenizer)(yy)

        return y_recon

    def _build_model() -> keras.Model:
        config = self.config.model_config

        # ---------- Inputs ---------- #
        l_in = config.n_patches_in[0] * config.patch_length_in[0]
        data = keras.layers.Input(
            shape=(l_in + 1, config.n_channels),
            dtype=tf.int32,
            name="data",
        )

        extra_labels = []
        for label in config.extra_labels:
            extra_labels.append(
                keras.layers.Input(
                    shape=(l_in + 1,),
                    dtype=tf.int32,
                    name=label.name,
                )
            )

        # ---------- Initialize layers ---------- #
        shift_token_layer = ShiftTokenLayer(name="shift_token")
        input_embedding_layer = self._get_input_embedding_layer()
        self.transformer_decoder = self._get_transformer_decoder()
        prediction_head_layer = self._get_prediction_head_layer()

        cross_entropy_loss_layer = CrossEntropyLossLayer(
            config.loss_sequence_length,
            config.top_k,
            name="cross_entropy_loss",
        )

        #sample_token_layer = SampleTokenLayer(name="sample_tokens")
        
        self.reconstruct_data_layer = ReconstructDataLayer(
            self._get_refactored2original(),  # shape (n_refactored_tokens, n_original_tokens)
            self.tokenizer,
            name="reconstruct_data"
        )   

        self.lyapunov_loss_layer = LyapunovLossLayer(
            config.loss_sequence_length,
            config.lyapunov_beta,
            config.lyapunov_mu,
            config.lyapunov_collapse_weight,
            config.lyapunov_collapse_target_var,
            config.lyapunov_collapse_target_mean,
            config.lyapunov_dim,
            name="lyapunov_loss",
        )

        # ---------- Forward Pass ---------- #

        # Shift the tokens
        x_input, y_true = shift_token_layer(data)
        # x_input.shape = (batch_size, l_in, n_channels)
        # y_true.shape = (batch_size, l_in, n_channels)

        # Get the input embeddings to the decoder
        x_input_embedded = input_embedding_layer([x_input, extra_labels])
        # x_input_embedded.shape = (batch_size, l_in, n_channels, model_dim)

        # Run the decoder
        x = self.transformer_decoder(x_input_embedded, training="decoder" in self.pretrained_layers)
        # x.shape = (batch_size, l_out, n_channels, model_dim)

        # Get the prediction of the next token
        y_pred_logits = prediction_head_layer(x, training="prediction_head" in self.pretrained_layers)
        # y_pred_logits.shape = (batch_size, l_out, n_channels, n_refactoredtokens)
        # n_refactoredtokens = vocab_size

        # Gumbel-softmax sampling to get the predicted tokens
        # hard=True to use the Straight-Through Estimator (STE) trick
        y_pred_soft = GumbelSoftmaxLayer(temperature=1.0, hard=True)(y_pred_logits)
    
        # y_pred_soft.shape = (batch_size, l_out, n_channels, n_refactoredtokens)   

        # Soft Embedding Lookup
        # We bypass the embedding layer's standard call method.
        # and explicitly multiply the soft vectors by the embedding weights.
        # [batch_size, l_out, n_channels, n_refactoredtokens] @ [n_refactoredtokens, embedding_dim] -> [batch_size, l_out, n_channels, embedding_dim]
        embedding_matrix = input_embedding_layer.token_embedding_layer.embeddings  # shape: (n_refactoredtokens, embedding_dim)
        y_pred_soft_embeddings = keras.ops.matmul(y_pred_soft, embedding_matrix)
        y_pred_soft_embeddings = input_embedding_layer.token_embedding_output_layer(y_pred_soft_embeddings)
        # y_pred_soft_embeddings.shape = (batch_size, l_out, n_channels, model_dim) 

        # Embed x_input directly via gather to avoid allocating a large one-hot tensor.
        x_input_token_embeddings = keras.ops.take(embedding_matrix, x_input, axis=0)
        x_input_token_embeddings = input_embedding_layer.token_embedding_output_layer(x_input_token_embeddings)
        # x_input_token_embeddings.shape = (batch_size, l_in, n_channels, model_dim)

        # ---------- Losses ---------- #

        inputs = {"data": data}
        for i, label in enumerate(config.extra_labels):
            inputs[label.name] = extra_labels[i]
        name = config.name

        # A lightweight model used only for autoregressive sampling.
        # Reuses the exact same shared layers/weights but avoids loss-branch work.
        self.sampling_model = keras.Model(
            inputs=inputs,
            outputs=y_pred_logits,
            name=f"{name}_sampling",
        )

        ### Compute Lyapunov loss in embedding space
        self.lyapunov_loss, _ = self.lyapunov_loss_layer(
            [y_pred_soft_embeddings, x_input_token_embeddings]
        )

        ### Cross entropy loss
        self.cross_entropy_loss, _ = cross_entropy_loss_layer([y_pred_logits, y_true])

        # ---------- Create Model ---------- #
        
        # Keep training outputs compact; large tensors (e.g. logits trajectories) are not
        # needed for fit because losses/metrics are added inside the loss layers.
        outputs = {
            "cross_entropy_loss": self.cross_entropy_loss,
            "lyapunov_loss": self.lyapunov_loss,
        }

        return keras.Model(inputs=inputs, outputs=outputs, name=name)

    def _embed_tokens(self, token_labels_or_logits):
        
        # token_labels_or_logits.shape = 
        # (batch_size, sequence_length, n_channels) 
        # or (batch_size, sequence_length, n_channels, n_tokens)

        # detect if token_labels_or_logits are labels or logits
        if token_labels_or_logits.shape[-1] == self.config.model_config.n_tokens:
            logits = token_labels_or_logits
        else:
            # tokens.shape = (batch_size, sequence_length, n_channels)
            # One-hot encode the tokens
            logits = tf.one_hot(token_labels_or_logits, depth=self.config.model_config.n_tokens, dtype=tf.float32)  

        if False:
            embeddings = self.token_embedding_output_layer(
                self.token_embedding_layer(x, training=training_1), training=training_1
            )
        else:
            input_embedding_layer = self.model.get_layer("input_embedding")
            embedding_matrix = input_embedding_layer.token_embedding_layer.embeddings  # shape: (n_tokens, embedding_dim)
            embeddings = tf.linalg.matmul(logits, embedding_matrix)
            embeddings = input_embedding_layer.token_embedding_output_layer(embeddings)
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        return embeddings
    
    @tf.function
    def one_step_sample(
        self,
        inputs: list,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        lyapunov_margin: float = 0.0,
        lyapunov_rollout_horizon: int = 1,
        lyapunov_adaptive_rollout: bool = False,
        lyapunov_adaptive_band: float = 0.0,
        return_diagnostics: bool = False,
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
        lyapunov_margin : float, optional
            Margin for the lyapunov function.
        lyapunov_rollout_horizon : int, optional
            Reserved for rollout-based Lyapunov scoring.
        lyapunov_adaptive_rollout : bool, optional
            Reserved for adaptive rollout scoring.
        lyapunov_adaptive_band : float, optional
            Reserved for adaptive rollout scoring band.
        return_diagnostics : bool, optional
            If True, return attempts/validity diagnostics per batch element.

        Returns
        -------
        sampled_tokens : tf.Tensor
            The sampled tokens. Shape: (*batch_dims).
        lyapunov_fn : tf.Tensor
            The lyapunov function values. Shape: (*batch_dims).
        """

        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        y_pred_logits = self.sampling_model(inputs, training=False)
        # y_pred_logits.shape = (batch_size, sequence_length_4loss, n_channels, n_refactored_tokens)

        margin = tf.cast(lyapunov_margin, tf.float32)
        if margin.shape.rank == 0:
            margin = tf.reshape(margin, [1, 1])
        elif margin.shape.rank == 1:
            margin = tf.reshape(margin, [-1, 1])

        # Get the predicted logits at last time step:
        logits = y_pred_logits[:, -1:] / temperature # shape: (batch_size, 1, n_channels, n_refactored_tokens)   

        # inputs["data"] is the tokenized data with
        # shape = (batch_size, sequence_length + 1, n_channels)
        # The last time step in the 2nd dim is the one to be predicted/generated
        # inp_dat = inputs["data"][:, -self.config.model_config.loss_sequence_length_recon_buffer-1:-1, :] # shape: (batch_size, buffer_sequence_length, n_channels)
        # recon_dat = self.reconstruct_data_layer(inp_dat) # shape: (batch_size, buffer_sequence_length, n_channels)
        # recon_dat = recon_dat[:, -1:, :] # shape: (batch_size, 1, n_channels)
        # V_dat = self._compute_V(recon_dat) # shape: (batch_size, 1)

        # Get the tokens at the tpt before the one being generated
        tokenised_inp_dat = inputs["data"][:, -2:-1, :] # shape: (batch_size, 1, n_channels)
        inp_dat_embeddings = self._embed_tokens(tokenised_inp_dat) # shape: (batch_size, 1, n_channels, embedding_dim)

        # Precompute V_dat from tokenised_inp_dat to speed up sampling
        V_dat = self.lyapunov_loss_layer._compute_V(inp_dat_embeddings) # shape: (batch_size, 1)
        adaptive_band = tf.cast(lyapunov_adaptive_band, tf.float32)

        def _score_candidate(tokenised_sample):
            # Score first generated step.
            tokenised_sample_embeddings = self._embed_tokens(tokenised_sample)
            V_curr = self.lyapunov_loss_layer._compute_V(tokenised_sample_embeddings) # shape: (batch_size, 1)
            score = tf.nn.relu(V_curr - V_dat) # shape: (batch_size, 1)
            first_step_score = score

            # Optional rollout scoring: evaluate additional imagined future steps.
            if lyapunov_rollout_horizon > 1:
                rollout_inputs = {k: v for k, v in inputs.items()}
                # If extra channels are present, keep their current window fixed during
                # imagined rollout steps (constant conditioning approximation).
                rollout_data = tf.concat([inputs["data"][:, 1:, :], tokenised_sample], axis=1)
                threshold = margin - adaptive_band
                active_mask = score > threshold

                for _ in range(1, lyapunov_rollout_horizon):
                    rollout_inputs["data"] = rollout_data
                    rollout_logits = self.sampling_model(rollout_inputs, training=False)
                    rollout_logits = rollout_logits[:, -1:] / temperature
                    token_next = sample_from_logits(rollout_logits, top_p, top_k)

                    token_next_embeddings = self._embed_tokens(token_next)
                    V_next = self.lyapunov_loss_layer._compute_V(token_next_embeddings)
                    step_score = tf.nn.relu(V_next - V_curr)

                    if lyapunov_adaptive_rollout:
                        score = tf.where(active_mask, tf.maximum(score, step_score), score)
                        active_mask = tf.logical_and(active_mask, score > threshold)
                    else:
                        score = tf.maximum(score, step_score)

                    V_curr = V_next
                    rollout_data = tf.concat([rollout_data[:, 1:, :], token_next], axis=1)

            return score, first_step_score

        tokenised_sample = sample_from_logits(logits, top_p, top_k) # shape: (batch_size, 1, n_channels)
        lyapunov_fn, first_step_lyapunov = _score_candidate(tokenised_sample)

        best_sample = tokenised_sample
        best_lyapunov_fn = lyapunov_fn
        unresolved_mask = best_lyapunov_fn > margin  # shape: (batch_size, 1)
        attempts_used = tf.ones_like(tf.cast(unresolved_mask, tf.int32))  # shape: (batch_size, 1)

        for attempt in range(2, 11):
            tokenised_candidate = sample_from_logits(logits, top_p, top_k) # shape: (batch_size, 1, n_channels)
            candidate_lyapunov_fn, _ = _score_candidate(tokenised_candidate)

            newly_accepted = tf.logical_and(
                unresolved_mask, candidate_lyapunov_fn <= margin
            )
            attempts_used = tf.where(
                newly_accepted,
                tf.cast(attempt, tf.int32) * tf.ones_like(attempts_used),
                attempts_used,
            )

            better_mask = tf.logical_and(
                unresolved_mask, candidate_lyapunov_fn < best_lyapunov_fn
            )  # shape: (batch_size, 1)
            best_lyapunov_fn = tf.where(
                better_mask, candidate_lyapunov_fn, best_lyapunov_fn
            )
            best_sample = tf.where(
                tf.expand_dims(better_mask, axis=-1), tokenised_candidate, best_sample
            )
            unresolved_mask = best_lyapunov_fn > margin

        tokenised_sample = best_sample
        lyapunov_fn = best_lyapunov_fn

        # remove time dimension
        tokenised_sample = tf.squeeze(tokenised_sample, axis=1) # shape: (batch_size, n_channels)

        lyapunov_fn = tf.squeeze(lyapunov_fn, axis=1) # shape: (batch_size,)
        accepted_mask = lyapunov_fn <= tf.squeeze(margin, axis=1)  # shape: (batch_size,)
        attempts_used = tf.where(
            accepted_mask,
            tf.squeeze(attempts_used, axis=1),
            10 * tf.ones_like(tf.squeeze(attempts_used, axis=1)),
        )
        initial_invalid_mask = tf.squeeze(first_step_lyapunov > margin, axis=1)
        rejection_count = attempts_used - 1

        if return_diagnostics:
            return (
                tokenised_sample,
                lyapunov_fn,
                attempts_used,
                accepted_mask,
                initial_invalid_mask,
                rejection_count,
            )
        return tokenised_sample, lyapunov_fn

    def generate_tokens(
        self,
        n_samples: int,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        lyapunov_margin: float = 0.00001,
        lyapunov_adaptive_margin: bool = False,
        lyapunov_margin_quantile: float = 0.9,
        lyapunov_margin_window: int = 256,
        lyapunov_margin_min: float = 0.0,
        lyapunov_margin_max: float = np.inf,
        lyapunov_rollout_horizon: int = 1,
        lyapunov_adaptive_rollout: bool = False,
        lyapunov_adaptive_band: float = 0.0,
        lyapunov_diagnostics: bool = False,
        batch_size: int = 1,
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
        lyapunov_adaptive_margin : bool, optional
            If True, adapt margin per session from recent Lyapunov values.
        lyapunov_margin_quantile : float, optional
            Quantile for adaptive margin updates.
        lyapunov_margin_window : int, optional
            Rolling window length (in generated steps) for adaptive margin updates.
        lyapunov_margin_min : float, optional
            Lower bound for adaptive margins.
        lyapunov_margin_max : float, optional
            Upper bound for adaptive margins.
        lyapunov_rollout_horizon : int, optional
            Number of future imagined steps to include in Lyapunov candidate scoring.
        lyapunov_adaptive_rollout : bool, optional
            If True, only keep rollout accumulation active for near-threshold samples.
        lyapunov_adaptive_band : float, optional
            Band around margin used by adaptive rollout.
        lyapunov_diagnostics : bool, optional
            If True, prints summary diagnostics for Lyapunov validity.
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
        extra_channels : Dict[str, np.ndarray], optional
            Dictionary of extra channels. Keys are the names of the extra channels.
            Each value must have shape (batch_size, > sequence_length + n_samples).

        Returns
        -------
        generated_tokens : np.ndarray
            Generated tokens. Shape is (batch_size, n_samples, n_channels).

        FAQ
        ---
        What is relationship between n_samples and sequence_length?
        1. The model generates tokens in chunks of size sequence_length.
        2. The total number of tokens to generate is n_samples.
        3. Therefore, the model will generate ceil(n_samples / sequence_length) chunks.
        4. The final output will be trimmed to n_samples.
        5. For example, if n_samples=1500 and sequence_length=512,
           the model will generate 3 chunks of 512 tokens each (total 1536 tokens),
           and then trim the output to the first 1500 tokens.
        
        Does this mean that there will be discontinuities in the generated sequence?
            No. The model uses the previously generated tokens as context
            for generating the next tokens, so it can maintain continuity across chunks.

        What is relationship between batch_size and n_samples?
        1. batch_size determines how many sequences are generated in parallel (e.g.
           each batch might correspond to different subjects/sessions).
        2. n_samples determines the length of each generated sequence.
        3. The output shape is (batch_size, n_samples, n_channels).
        4. For example, if batch_size=4 and n_samples=1500,
           the model will generate 4 sequences, each of length 1500 tokens.

        """

        batch_size = batch_size or self.config.training_config.batch_size

        n_channels = self.config.model_config.n_channels

        sequence_length = self.config.model_config.sequence_length

        if lyapunov_rollout_horizon < 1:
            raise ValueError("lyapunov_rollout_horizon must be >= 1.")
        if lyapunov_adaptive_band < 0.0:
            raise ValueError("lyapunov_adaptive_band must be >= 0.")
        if lyapunov_adaptive_margin:
            if not (0.0 < lyapunov_margin_quantile < 1.0):
                raise ValueError("lyapunov_margin_quantile must be in (0, 1).")
            if lyapunov_margin_window < 1:
                raise ValueError("lyapunov_margin_window must be >= 1.")
            if lyapunov_margin_min > lyapunov_margin_max:
                raise ValueError("lyapunov_margin_min must be <= lyapunov_margin_max.")


        # ---------- Helper functions ---------- #
        def _random_tokens() -> np.ndarray:
            _rng = np.random.default_rng()
            try:
                token_weights = self.tokenizer.vocab["total_token_counts"].astype(
                    np.float32
                )
                #token_weights = np.ones(self.tokenizer.vocab["total_token_counts"].shape[0]).astype(np.float32)
            except AttributeError:
                token_weights = np.ones(
                    max(1, self.config.model_config.n_tokens - 1), dtype=np.float32
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
        if len(extra_channels) > 0 and lyapunov_rollout_horizon > 1:
            _logger.warning(
                "Rollout scoring with extra_channels reuses the current extra-channel window "
                "for imagined future rollout steps."
            )

        # ---------- Generate tokens ---------- #

        generated_tokens = np.zeros(
            (batch_size, sequence_length + n_samples, n_channels), dtype=np.int32
        )

        lyapunov = np.zeros(
            [batch_size, sequence_length + n_samples + 1], dtype=np.float32
        )
        current_margin = np.full((batch_size,), lyapunov_margin, dtype=np.float32)
        if lyapunov_diagnostics:
            invalid_mask = np.zeros([batch_size, n_samples], dtype=bool)
            initial_invalid_mask = np.zeros([batch_size, n_samples], dtype=bool)
            attempts_used = np.ones([batch_size, n_samples], dtype=np.int32)
            rejection_count = np.zeros([batch_size, n_samples], dtype=np.int32)
            margin_used = np.zeros([batch_size, n_samples], dtype=np.float32)
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
            if lyapunov_diagnostics:
                (
                    next_token,
                    next_lyapunov_fn,
                    step_attempts_used,
                    step_accepted_mask,
                    step_initial_invalid_mask,
                    step_rejection_count,
                ) = self.one_step_sample(
                    model_inputs,
                    top_p,
                    top_k,
                    temperature,
                    current_margin,
                    lyapunov_rollout_horizon,
                    lyapunov_adaptive_rollout,
                    lyapunov_adaptive_band,
                    True,
                )
            else:
                next_token, next_lyapunov_fn = self.one_step_sample(
                    model_inputs,
                    top_p,
                    top_k,
                    temperature,
                    current_margin,
                    lyapunov_rollout_horizon,
                    lyapunov_adaptive_rollout,
                    lyapunov_adaptive_band,
                )

            # Add the next token to the prompt
            generated_tokens[:, i] = next_token.numpy().astype(np.int32)
            lyapunov[:, i] = next_lyapunov_fn.numpy()
            if lyapunov_diagnostics:
                idx = i - sequence_length
                invalid_mask[:, idx] = ~step_accepted_mask.numpy()
                initial_invalid_mask[:, idx] = step_initial_invalid_mask.numpy()
                attempts_used[:, idx] = step_attempts_used.numpy()
                rejection_count[:, idx] = step_rejection_count.numpy()
                margin_used[:, idx] = current_margin

            if lyapunov_adaptive_margin:
                recent_start = max(sequence_length, i - lyapunov_margin_window + 1)
                recent_vals = lyapunov[:, recent_start : i + 1]
                new_margin = np.quantile(
                    recent_vals, q=lyapunov_margin_quantile, axis=1
                ).astype(np.float32)
                current_margin = np.clip(
                    new_margin,
                    np.float32(lyapunov_margin_min),
                    np.float32(lyapunov_margin_max),
                )
            
            # shape of generated_tokens: (1, time, n_channels)                

        if lyapunov_diagnostics:
            invalid_count = int(np.sum(invalid_mask))
            total_count = int(np.prod(invalid_mask.shape))
            invalid_rate = invalid_count / max(1, total_count)
            mean_attempts = float(np.mean(attempts_used))
            rejection_rate = float(np.sum(rejection_count) / max(1, np.sum(attempts_used)))
            _logger.info(
                (
                    "Lyapunov diagnostics: initial_invalid_rate=%.4f, "
                    "final_invalid_rate=%.4f (%d/%d), mean_attempts=%.3f, rejection_rate=%.4f, "
                    "mean_margin=%.6g"
                ),
                float(np.mean(initial_invalid_mask)),
                invalid_rate,
                invalid_count,
                total_count,
                mean_attempts,
                rejection_rate,
                float(np.mean(margin_used)) if lyapunov_diagnostics else float(np.mean(current_margin)),
            )
            if invalid_count > 0:
                _logger.warning(
                    "Some generated samples remained above Lyapunov margin after max attempts."
                )
            diagnostics = {
                "lyapunov_margin": float(lyapunov_margin),
                "lyapunov_adaptive_margin": bool(lyapunov_adaptive_margin),
                "lyapunov_margin_quantile": float(lyapunov_margin_quantile),
                "lyapunov_margin_window": int(lyapunov_margin_window),
                "lyapunov_margin_min": float(lyapunov_margin_min),
                "lyapunov_margin_max": float(lyapunov_margin_max),
                "lyapunov_rollout_horizon": int(lyapunov_rollout_horizon),
                "lyapunov_adaptive_rollout": bool(lyapunov_adaptive_rollout),
                "lyapunov_adaptive_band": float(lyapunov_adaptive_band),
                "max_attempts": 10,
                "invalid_mask": invalid_mask,
                "initial_invalid_mask": initial_invalid_mask,
                "attempts_used": attempts_used,
                "rejection_count": rejection_count,
                "margin_used": margin_used,
                "final_margin_per_session": current_margin.copy(),
                "final_invalid_rate_per_session": np.mean(invalid_mask, axis=1),
                "initial_invalid_rate_per_session": np.mean(initial_invalid_mask, axis=1),
                "mean_attempts_per_session": np.mean(attempts_used, axis=1),
            }
            return generated_tokens[:, sequence_length:], lyapunov[:, sequence_length:], diagnostics

        return generated_tokens[:, sequence_length:], lyapunov[:, sequence_length:]

    def generate_data(self, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate data using the model.
        
        Parameters
        ----------
        See generate_tokens() for parameters.

        Returns
        -------
        generated_data

        """

        if self.tokenizer is None:
            raise ValueError(
                "Cannot generate reconstructed data without a tokenizer. "
                "Use generate_tokens() or set model_config.tokenizer_path."
            )

        generated = self.generate_tokens(**kwargs)
        if len(generated) == 3:
            tokens, lyapunov, diagnostics = generated
            return self.tokenizer.reconstruct_data(list(tokens)), lyapunov, diagnostics

        tokens, lyapunov = generated  # (batch_size, n_samples, n_channels)
        return self.tokenizer.reconstruct_data(list(tokens)), lyapunov # (batch_size, n_samples, n_channels)

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
