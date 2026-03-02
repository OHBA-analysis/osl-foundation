import tensorflow as tf
import numpy as np
import keras

class AttentionLayer(keras.layers.Layer):
    """
    Layer for performing attention.

    Parameters
    ----------
    key_dim : int
        Key dimension.
    """

    def __init__(
        self,
        key_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key_dim = key_dim


    def call(self, inputs, mask_in=None, **kwargs):
        # q (Query): (batch_size, n_heads, l_out, l_other, key_dim)
        # k (Key): (batch_size, n_heads, l_in, l_other, key_dim)
        # v (Value): (batch_size, n_heads, l_in, l_other, key_dim)
        
        q, k, v = inputs
        
        # Transpose inputs for time attention

        q = tf.transpose(q, perm=(0, 1, 3, 2, 4))
        # q: (batch_size, n_heads, l_other, l_out, key_dim)

        k = tf.transpose(k, perm=(0, 1, 3, 2, 4))
        # k: (batch_size, n_heads, l_other, l_in, key_dim)

        # Compute attention
        attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )
        # attention: (batch_size, n_heads, l_other, l_out, l_in)

        # Apply mask
        if mask_in is not None:
            attention += -1e9 * mask_in

        # Normalise attention with softmax
        attention = tf.nn.softmax(attention, axis=-1)

        if mask_in is not None:
            attention = attention * (1 - mask_in)

        # Apply attention to value
        
        v = tf.transpose(v, perm=(0, 1, 3, 2, 4))
        # v (Value): (batch_size, n_heads, l_other, l_in, key_dim)        

        output_t = tf.matmul(attention, v)
        # output: (batch_size, n_heads, l_other, l_out, key_dim)
        
        output = tf.transpose(output_t, perm=(0, 1, 3, 2, 4))
        # output: (batch_size, n_heads, l_out, l_other, key_dim)

        return output

class GASPLayer(keras.layers.Layer):
    """
    This layer performs attention on the input tensor.

    Parameters
    ----------
    n_heads : int
        Number of heads.
    l_other: int
    n_patches_out : int
    n_patches_in : int
        Number of patches to attend to.
    patch_length : int
        Patch length.
    unpatched_length_in : int
        Number of unpatched elements to attend to.
    key_dim : int
        Key dimension.
    causal : bool
        Whether to use causal attention.
    l_unpatched_b : int
        Number of unpatched elements to attend to before and after the current element. 
        If None, no limit is applied.
    l_patched_b : int
        Number of patched elements to attend to before and after the current element. 
        If None, no limit is applied.
    attention_mask : np.array   
        Attention mask of shape (l_out, n_patches_in + unpatched_length_in).

    """
    
    def __init__(
        self,
        n_heads: int,
        l_other: int,
        n_patches_out: int,
        patch_length_out: int,
        n_patches_in: int,
        patch_length_in: int,
        unpatched_length_in: int,
        key_dim: int,
        causal: bool,
        l_unpatched_b: int,
        l_patched_b: int,
        attention_mask: np.array,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.l_other = l_other 
        self.n_patches_out = n_patches_out
        self.patch_length_out = patch_length_out
        self.n_patches_in = n_patches_in
        self.patch_length_in = patch_length_in
        self.unpatched_length_in = unpatched_length_in
        self.key_dim = key_dim
        self.causal = causal
        self.l_unpatched_b = l_unpatched_b
        self.l_patched_b = l_patched_b

        self.l_in = self.n_patches_in * self.patch_length_in
        self.l_out = self.n_patches_out * self.patch_length_out

        self.attention_mask = self._compute_attention_mask(attention_mask)

        self.attention_layer = AttentionLayer(key_dim)    

    def _compute_attention_mask(self, mask=None) -> tf.Tensor:
        """
        Compute the mask for  attention.

        Returns
        -------
        mask : tf.Tensor
            Mask for attention.
            Shape: (l_out, n_patches_in + unpatched_length_in).
        """

        if mask is None:
            mask = np.zeros((self.l_out, self.n_patches_in + self.unpatched_length_in)) 

        if self.l_unpatched_b is not None:
            # i indexes columns (unpatched input) in mask from self.n_patches_in to (self.n_patches_in+self.unpatched_length_in)
            # j indexes rows in mask from 0 to self.l_out-1
            
            # Set mask to 1 if i < j-self.l_unpatched_b
            for i in range(self.n_patches_in, self.n_patches_in + self.unpatched_length_in):
                for j in range(self.l_out):
                    if i < j - self.l_unpatched_b or i > j + self.l_unpatched_b:
                        mask[j, i] = 1

        if self.l_patched_b is not None:
            # i indexes columns (patched input) in mask from 0 to self.n_patches_in-1
            # j indexes rows in mask from 0 to self.l_out-1
            for i in range(self.n_patches_in):
                for j in range(self.l_out):
                    if i <= j // self.patch_length_in - self.l_patched_b or i >= j // self.patch_length_in + self.l_patched_b:
                        mask[j, i] = 1

        if self.causal:
            # Patch masking
            for i in range(self.n_patches_in):
                m_indx = max(
                    0,
                    (i + 1 - self.n_patches_in) * self.patch_length_in
                    + self.l_out
                    - 1,
                )
                mask[:m_indx, i] = 1

            # Unpatched masking
            for i in range(self.n_patches_in, self.n_patches_in + self.unpatched_length_in):
                mask[
                    : self.l_out
                    - (self.n_patches_in + self.unpatched_length_in)
                    + i,
                    i,
                ] = 1
        
        mask = tf.constant(mask, dtype=tf.float32)

        return mask

    def build(self, input_shape):
        q_shape, k_shape, v_shape = input_shape
        
        # Build time-attention sub-layer

        self.attention_layer.build([q_shape, k_shape, v_shape])

        # Set the built flag to True
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # ---------- Unpack Inputs ---------- #

        # q (Query): (batch_size, n_heads, l_out, l_other, key_dim)
        # k (Key): (batch_size, n_heads, n_patches_in + unpatched_length_in, l_other, key_dim)
        # v (Value): (batch_size, n_heads, n_patches_in + unpatched_length_in, l_other, key_dim)
        q, k, v = inputs

        # ---------- Attention ---------- #
        output = self.attention_layer(
            [q, k, v], mask_in=self.attention_mask, training=training, **kwargs
        )

        return output


class MultiHeadGASPLayer(keras.layers.Layer):
    """
    The Multi-head GASP layer.

    Parameters
    ----------
    n_heads : int
        Number of heads.
    model_dim : int
        Model dimension.
    l_other : int
    n_patches_out : int
        Number of patches to attend to.
    patch_length_out : int
        Patch length.
    n_patches_in : int
        Number of patches to attend to.
    patch_length_in : int
        Patch length.
    unpatched_length_in : int
        Number of unpatched elements to attend to.
    causal : bool
        Whether to use causal attention.
    l_unpatched_b : int
        Number of unpatched elements to attend to before and after the current element. 
        If None, no limit is applied.
    l_patched_b : int
        Number of patched elements to attend to before and after the current element. 
        If None, no limit is applied.
    attention_mask : np.array
        Attention mask of shape (l_out, n_patches_in + unpatched_length_in).
    """

    # For chan attention:
    # l_other = n_patches_in*patch_length_in,
    # n_patches_out = n_channels,
    # patch_length_out = 1,
    # n_patches_in = n_channels
    # patch_length_in = 1
    # unpatched_length_in = 0
    # causal = False
    # l_unpatched_b = None
    # l_patched_b = None

    def __init__(
        self,
        n_heads: int,
        model_dim: int,
        l_other: int,
        n_patches_out: int,
        patch_length_out: int,
        n_patches_in: int,
        patch_length_in: int,
        unpatched_length_in: int,
        causal: bool,
        l_unpatched_b: int,
        l_patched_b: int,
        attention_mask: np.array,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.key_dim = model_dim // n_heads
        self.l_other = l_other
        self.n_patches_out = n_patches_out
        self.patch_length_out = patch_length_out
        self.n_patches_in = n_patches_in
        self.patch_length_in = patch_length_in
        self.unpatched_length_in = unpatched_length_in
        self.causal = causal
        self.l_unpatched_b = l_unpatched_b
        self.l_patched_b = l_patched_b

        # Patch projection
        if patch_length_in > 1:
            self.patch_projection_k = keras.layers.Dense(n_heads, name="patch_projection_k")
            self.patch_projection_v = keras.layers.Dense(n_heads, name="patch_projection_v")
        else:
            self.patch_projection_k = None
            self.patch_projection_v = None

        # Input projections
        self.patched_projection_k = keras.layers.Dense(self.model_dim, name="patched_projection_k")
        self.patched_projection_v = keras.layers.Dense(self.model_dim, name="patched_projection_v")
        
        if unpatched_length_in > 0:
            self.unpatched_projection_k = keras.layers.Dense(self.model_dim, name="unpatched_projection_k")
            self.unpatched_projection_v = keras.layers.Dense(self.model_dim, name="unpatched_projection_v")  
        else:
            self.unpatched_projection_k = None
            self.unpatched_projection_v = None

        self.query_projection = keras.layers.Dense(self.model_dim, name="query_projection")

        self.l_in = self.n_patches_in*self.patch_length_in
        self.l_out = self.n_patches_out*self.patch_length_out

        # GASP layer for time and channel attention
        self.gasp_layer = GASPLayer(
            n_heads,
            l_other,
            n_patches_out,
            patch_length_out,
            n_patches_in,
            patch_length_in,
            unpatched_length_in,
            self.key_dim,
            causal,
            l_unpatched_b,
            l_patched_b,
            attention_mask,
            name="gasp_layer",
        )

        # Output projection
        self.output_projection = keras.layers.Dense(self.model_dim, name="output_projection")

    def _patch_x(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, l_in, l_other, model_dim)

        x = tf.transpose(x, perm=(0, 2, 3, 1))
        # x.shape: (batch_size, l_other, model_dim, l_in)
        
        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                self.l_other,
                self.model_dim,
                self.n_patches_in,
                self.patch_length_in,
            ),
        )
        # x.shape: (batch_size, l_other, model_dim, n_patches, patch_length_in)

        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # x.shape: (batch_size, n_patches, l_other, model_dim, patch_length_in)

        return x

    def _perceiver_x(self, x: tf.Tensor) -> tf.Tensor:
        """Get the last l_out elements of the sequence."""
        # x.shape: (batch_size, sequence_length, l_other, model_dim)

        if self.l_in > self.l_out:
            x = tf.slice(
                x,
                [0, self.l_in - self.l_out, 0, 0],
                [-1, -1, -1, -1],
            )
            # x.shape: (batch_size, l_out, l_other, model_dim)

        return x

    def _unpatch_x(self, x: tf.Tensor) -> tf.Tensor:
        """Get the last unpatched_length_in elements of the sequence."""
        # x.shape: (batch_size, sequence_length, l_other, model_dim)

        x = tf.slice(
            x,
            [0, self.l_in - self.unpatched_length_in, 0, 0],
            [-1, -1, -1, -1],
        )
        # x.shape: (batch_size, unpatched_length_in, l_other, model_dim)

        return x

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, time_length, l_other, model_dim)
        # Here time_length is either n_patches_in + unpatched_length_in or l_out

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                tf.shape(x)[1],
                self.l_other,
                self.n_heads,
                self.key_dim,
            ),
        )
        # x.shape: (batch_size, time_length, l_other, n_heads, key_dim)

        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # x.shape: (batch_size, n_heads, time_length, l_other, key_dim)

        return x

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x.shape: (batch_size, n_heads, l_out, l_other, key_dim)

        x = tf.transpose(x, perm=(0, 2, 3, 1, 4))
        # x.shape: (batch_size, l_out, l_other, n_heads, key_dim)

        x = tf.reshape(
            x,
            (
                tf.shape(x)[0],
                self.l_out,
                self.l_other,
                self.model_dim,
            ),
        )
        # x.shape: (batch_size, l_out, l_other, model_dim)

        return x
    
    def build(self, input_shape):

        if self.patch_length_in > 1:
            # Build patch projection layer
            self.patch_projection_v.build(
                (None, self.n_patches_in, self.l_other, self.key_dim, self.n_heads * self.patch_length_in)
            )
            self.patch_projection_k.build(
                (None, self.n_patches_in, self.l_other, self.key_dim, self.n_heads * self.patch_length_in)
            )

        # Build input projection layers
        self.patched_projection_v.build((None, self.n_patches_in, self.l_other, self.model_dim))

        if self.unpatched_length_in > 0:
            self.unpatched_projection_v.build((None, self.unpatched_length_in, self.l_other, self.model_dim))
            self.unpatched_projection_k.build((None, self.unpatched_length_in, self.l_other, self.model_dim))

        self.patched_projection_k.build((None, self.n_patches_in, self.l_other, self.model_dim))
        self.query_projection.build((None, self.l_out, self.l_other, self.model_dim))

        # Build the GASP Layer
        q_c_shape = (
            None,
            self.n_heads,
            self.l_out,
            self.l_other,
            self.key_dim,
        )  # shape for q, c_q, and c_k
        
        k_v_shape = (
            None,
            self.n_heads,
            self.n_patches_in + self.unpatched_length_in,
            self.l_other,
            self.key_dim,
        )  # shape for k and v

        self.gasp_layer.build([q_c_shape, k_v_shape, k_v_shape])

        # Build output projection layer
        self.output_projection.build((None, self.l_out, self.l_other, self.model_dim))

        # Set the built flag to True
        super().build(input_shape)

    def compute_output_shape(self, input_shape):

        input_shape = [input_shape[0][0], self.l_out, self.l_other, self.model_dim]
        return self.output_projection.compute_output_shape(input_shape)

    def _patch_inputs(self, x, patch_projection_x):

        # ---------- Patch inputs ---------- #
        # Input is processed into 2 parts:
        # 1. Patched input: patched_x
        # 2. Unpatched input: unpatched_x

        # x.shape: (batch_size, l_in, l_other, model_dim)

        patched_x = self._patch_x(x)
        # patched_x.shape: (batch_size, n_patches_in, l_other, model_dim, patch_length_in)

        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches_in,
                self.l_other,
                self.key_dim,
                self.n_heads,
                self.patch_length_in,
            ),
        )
        # patched_x.shape: (batch_size, n_patches_in, l_other, key_dim, n_heads, patch_length_in)
        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches_in,
                self.l_other,
                self.key_dim,
                self.n_heads * self.patch_length_in,
            ),
        )
        # patched_x.shape: (batch_size, n_patches_in, l_other, key_dim, n_heads * patch_length_in)

        if self.patch_length_in > 1:
            patched_x = patch_projection_x(patched_x)
            # patched_x.shape: (batch_size, n_patches_in, l_other, key_dim, n_heads)
        
        patched_x = tf.reshape(
            patched_x,
            (
                tf.shape(patched_x)[0],
                self.n_patches_in,
                self.l_other,
                self.model_dim,
            ),
        )
        # patched_x.shape: (batch_size, n_patches_in, l_other, model_dim)

        unpatched_x = self._unpatch_x(x)
        # unpatched_x.shape: (batch_size, unpatched_length_in, l_other, model_dim)

        return patched_x, unpatched_x
    
    def call(self, inputs, training=None, **kwargs):
        
        query, key, value = inputs
        
        # query.shape: (batch_size, l_out, l_other, model_dim)
        # key.shape: (batch_size, l_in, l_other, model_dim)
        # value.shape: (batch_size, l_in, l_other, model_dim)

        # ---------- Project inputs to Q, K, V ---------- #

        patched_k, unpatched_k = self._patch_inputs(key, self.patch_projection_k)
        k_patched = self.patched_projection_k(patched_k)
        # k_patched.shape: (batch_size, n_patches, l_other, model_dim)
        
        if self.unpatched_length_in > 0:
            k_unpatched = self.unpatched_projection_k(unpatched_k)
            # k_unpatched.shape: (batch_size, unpatched_length_in, l_other, model_dim)
            k = tf.concat([k_patched, k_unpatched], axis=1)
            # k.shape: (batch_size, n_patches_in + unpatched_length_in, l_other, model_dim)
        else:
            k = k_patched

        patched_v, unpatched_v = self._patch_inputs(value, self.patch_projection_v)
        v_patched = self.patched_projection_v(patched_v)
        # v_patched.shape: (batch_size, n_patches, l_other, model_dim)

        if self.unpatched_length_in > 0:
            v_unpatched = self.unpatched_projection_v(unpatched_v)
            # v_unpatched.shape: (batch_size, unpatched_length_in, l_other, model_dim)
            v = tf.concat([v_patched, v_unpatched], axis=1)
            # v.shape: (batch_size, n_patches_in + unpatched_length_in, l_other, model_dim)
        else:
            v = v_patched

        # query.shape: (batch_size, l_out, l_other, model_dim)
        perceiver_q = self._perceiver_x(query)
        # perceiver_q.shape: (batch_size, l_out, l_other, model_dim)
        q = self.query_projection(perceiver_q)
        # q.shape: (batch_size, l_out, l_other, model_dim)

        # ---------- Split heads ---------- #

        q = self._split_heads(q)
        # (batch_size, n_heads, l_out, l_other, key_dim)

        k = self._split_heads(k)
        # (batch_size, n_heads, n_patches_in + unpatched_length_in, l_other, key_dim)

        v = self._split_heads(v)
        # (batch_size, n_heads, n_patches_in + unpatched_length_in, l_other, key_dim)

        # ---------- gasp Layer ---------- #
        output = self.gasp_layer([q, k, v], training=training, **kwargs)
        # output.shape: (batch_size, n_heads, l_out, l_other, key_dim)

        # ---------- Combine heads ---------- #
        output = self._combine_heads(output)
        # output.shape: (batch_size, l_out, l_other, model_dim)

        # ---------- Output projection ---------- #
        output = self.output_projection(output)
        # output.shape: (batch_size, l_out, l_other, model_dim)

        return output

def plot_attention_mask(gasp_layer):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(1-gasp_layer.attention_mask.numpy().transpose(), cmap="gray")
    # set color range to be 0 to 1
    plt.clim(0, 1)
    plt.grid(True)
    plt.xlabel('to')
    plt.ylabel('from')
    plt.xticks(np.arange(gasp_layer.attention_mask.shape[0]))
    plt.yticks(np.arange(gasp_layer.attention_mask.shape[1]))
    plt.show()

#######

n_heads=2
n_chans=10
key_dim=1
n_batches=1
model_dim=n_heads*key_dim

if False:

    causal=True

    # layer 1: time attention
    patch_length_in=1
    n_patches_in=12
    patch_length_out=1
    n_patches_out=12
    unpatched_length_in=0

    mh_gasp_layer1 = MultiHeadGASPLayer(n_heads,
                model_dim,
                n_chans,
                n_patches_out,
                patch_length_out,
                n_patches_in,
                patch_length_in,
                unpatched_length_in,
                causal,
                l_unpatched_b,
                l_patched_b
    )

    # layer 2: time attention, lin=12
    patch_length_in2=3
    n_patches_in2=4
    patch_length_out2=3
    n_patches_out2=3
    unpatched_length_in2=3

    mh_gasp_layer2 = MultiHeadGASPLayer(n_heads,
                model_dim,
                n_chans,
                n_patches_out2,
                patch_length_out2,
                n_patches_in2,
                patch_length_in2,
                unpatched_length_in2,
                causal,
                l_unpatched_b,
                l_patched_b
    )

    # layer 3: chan attention, tin=9, nchans=10
    patch_length_in3=1
    patch_length_out3=1

    n_patches_in3=n_chans
    n_patches_out3=n_chans
    unpatched_length_in3=0

    mh_gasp_layer3 = MultiHeadGASPLayer(n_heads,
                model_dim,
                n_patches_out2*patch_length2,
                n_patches_out3,
                patch_length_out3,
                n_patches_in3,
                patch_length_in3,
                unpatched_length_in3,
                False,
                l_unpatched_b,
                l_patched_b
    )
    ###

    # gen data
    l_in = n_patches_in*patch_length_in
    l_out = n_patches_out*patch_length_out
    x = np.random.normal(size=(n_batches, l_in, n_chans, model_dim))

    x.shape

    x1 = mh_gasp_layer1(x)

    x1.shape

    x2 = mh_gasp_layer2(x1)

    x2.shape

    x2t = tf.transpose(x2, perm=(0, 2, 1, 3))

    x3 = mh_gasp_layer3(x2t)

    x3t = tf.transpose(x3, perm=(0, 2, 1, 3))

    plot_attention_mask(mh_gasp_layer2.gasp_layer)

    ###

    n_patches_out = [12, 3]
    patch_length_out = [1, 2]
    n_patches_in = [12, 3]
    patch_length_in = [1, 4]    
    unpatched_length_in = [0, 3]
    l_unpatched_b = [-1, -1]
    l_patched_b = [3, -1]
    do_chan_attention = True
    chan_attention_mask = np.zeros((n_chans, n_chans))
    chan_attention_mask[:3, 4:8] = 1
    chan_attention_mask[4:8, :3] = 1

    decoder_layer = DecoderLayer(
            n_heads = n_heads,
            model_dim = model_dim,
            embedding_dim = model_dim,
            n_channels = n_chans,
            n_patches_out = n_patches_out,
            patch_length_out = patch_length_out,
            n_patches_in = n_patches_in,
            patch_length_in = patch_length_in,        
            unpatched_length_in = unpatched_length_in,
            l_unpatched_b = l_unpatched_b,
            l_patched_b = l_patched_b,
            do_chan_attention = do_chan_attention,
            chan_attention_mask = chan_attention_mask,
            feed_forward_dim = model_dim,
            feed_forward_activation="relu",
            dropout=0.2,
            norm_type="layer",
            )

    # gen data
    l_in = n_patches_in[0]*patch_length_in[0]
    embedding_dim = model_dim
    input = np.random.normal(size=(n_batches, l_in, n_chans, embedding_dim))

    output = decoder_layer(input)

    plot_attention_mask(decoder_layer.gasp_time_attention_layers[0].gasp_layer)
    plot_attention_mask(decoder_layer.gasp_chan_attention_layers[0].gasp_layer)
