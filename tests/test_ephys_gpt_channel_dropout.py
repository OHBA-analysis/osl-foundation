import unittest
from unittest.mock import patch

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

if tf is not None:
    from osl_foundation.models.ephys_gpt import InputEmbeddingLayer
else:  # pragma: no cover
    InputEmbeddingLayer = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TestInputChannelDropout(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(0)
        self.layer = InputEmbeddingLayer(
            embedding_dim=8,
            n_tokens=16,
            sequence_length=4,
            n_channels=3,
            channel_dropout_rate=0.5,
            extra_labels=[],
        )
        self.data = tf.constant(
            [
                [[1, 2, 3], [4, 5, 6], [1, 1, 1], [2, 2, 2]],
                [[3, 2, 1], [6, 5, 4], [1, 2, 3], [4, 3, 2]],
            ],
            dtype=tf.int32,
        )

    def test_no_dropout_when_not_training(self):
        embeddings = self.layer([self.data, []], training=False)
        self.assertFalse(bool(tf.reduce_all(tf.equal(embeddings, 0.0)).numpy()))

    def test_drops_full_channels_when_training(self):
        with patch(
            "tensorflow.random.uniform",
            return_value=tf.ones([2, 1, 3, 1], dtype=tf.float32),
        ):
            embeddings = self.layer([self.data, []], training=True)
        self.assertTrue(bool(tf.reduce_all(tf.equal(embeddings, 0.0)).numpy()))


if __name__ == "__main__":
    unittest.main()
