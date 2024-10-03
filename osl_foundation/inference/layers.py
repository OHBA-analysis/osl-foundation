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

    def __init__(self, output_dim, activation="linear", **kwargs):
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
