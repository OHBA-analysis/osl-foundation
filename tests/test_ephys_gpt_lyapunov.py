import unittest
from unittest.mock import patch

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

if tf is not None:
    from osl_foundation.models.ephys_gpt import EphysGPT, LyapunovLossLayer
else:  # pragma: no cover
    EphysGPT = None
    LyapunovLossLayer = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TestLyapunovLossLayer(unittest.TestCase):
    def test_beta_zero_disables_lyapunov_loss(self):
        layer = LyapunovLossLayer(loss_sequence_length=4, beta=0.0, dim=8)
        y_t_plus_one = tf.random.normal((2, 4, 3, 5))
        y_t = tf.random.normal((2, 4, 3, 5))

        loss, lyapunov_fn = layer([y_t_plus_one, y_t], training=False)

        self.assertEqual(tuple(lyapunov_fn.shape), (2, 4))
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=7)

    def test_lyapunov_fn_matches_loss_sequence_length(self):
        layer = LyapunovLossLayer(loss_sequence_length=5, beta=1.0, dim=4)
        y_t_plus_one = tf.zeros((1, 5, 2, 3), dtype=tf.float32)
        y_t = tf.zeros((1, 5, 2, 3), dtype=tf.float32)

        _, lyapunov_fn = layer([y_t_plus_one, y_t], training=False)

        self.assertEqual(tuple(lyapunov_fn.shape), (1, 5))


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TestOneStepSampleLyapunovSelection(unittest.TestCase):
    class _DummyModel:
        def __call__(self, inputs, training=False):
            batch_size = tf.shape(inputs["data"])[0]
            # Shape: (batch_size, time=1, channels=1, n_tokens=3)
            return {"y_pred": tf.zeros((batch_size, 1, 1, 3), dtype=tf.float32)}

    class _DummyLyapunovLoss:
        @staticmethod
        def _compute_V(embeddings):
            # embeddings shape: (batch, time=1, channels=1, embedding_dim=1)
            return tf.reduce_mean(embeddings, axis=[2, 3])

    class _DummySampler:
        def __init__(self):
            self.model = TestOneStepSampleLyapunovSelection._DummyModel()
            self.lyapunov_loss_layer = (
                TestOneStepSampleLyapunovSelection._DummyLyapunovLoss()
            )

        def _embed_tokens(self, token_labels_or_logits):
            return tf.expand_dims(tf.cast(token_labels_or_logits, tf.float32), axis=-1)

        def one_step_sample(self, *args, **kwargs):
            return EphysGPT.one_step_sample.python_function(self, *args, **kwargs)

    def test_generation_keeps_best_per_sample_with_margin(self):
        sampler = self._DummySampler()
        inputs = {"data": tf.constant([[[0], [0], [0]], [[0], [0], [0]]], tf.int32)}
        margin = 0.5

        # 10 draws total: 1 initial + 9 retries.
        draws = [
            tf.constant([[[5]], [[1]]], tf.int32),  # initial
            tf.constant([[[4]], [[2]]], tf.int32),
            tf.constant([[[3]], [[0]]], tf.int32),  # second sample resolves here
            tf.constant([[[2]], [[9]]], tf.int32),
            tf.constant([[[1]], [[8]]], tf.int32),  # first sample best improves here
            tf.constant([[[7]], [[7]]], tf.int32),
            tf.constant([[[6]], [[6]]], tf.int32),
            tf.constant([[[5]], [[5]]], tf.int32),
            tf.constant([[[4]], [[4]]], tf.int32),
            tf.constant([[[3]], [[3]]], tf.int32),
        ]

        with patch(
            "osl_foundation.models.ephys_gpt.sample_from_logits",
            side_effect=draws,
        ) as mocked_sampler:
            tokens, lyapunov_fn = sampler.one_step_sample(
                inputs,
                top_p=None,
                top_k=None,
                temperature=1.0,
                lyapunov_margin=margin,
            )

        self.assertEqual(tokens.numpy().reshape(-1).tolist(), [1, 0])
        self.assertEqual(lyapunov_fn.numpy().tolist(), [1.0, 0.0])
        self.assertEqual(mocked_sampler.call_count, 10)


if __name__ == "__main__":
    unittest.main()
