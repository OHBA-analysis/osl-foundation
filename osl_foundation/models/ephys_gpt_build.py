from typing import List
import logging

import keras
import numpy as np
import tensorflow as tf

from osl_dynamics.data import Data
from osl_dynamics.utils.misc import get_argument, replace_argument

from osl_foundation.config import get_config
from osl_foundation.models.ephys_gpt_layers import (
    CrossEntropyLossLayer,
    GumbelSoftmaxLayer,
    InputEmbeddingLayer,
    LyapunovLossLayer,
    ReconstructDataLayer,
    ShiftTokenLayer,
    TransformerDecoder,
)
from osl_foundation.models.tokenizers import MuTransformTokenizer, OSLTokenizer
from osl_foundation.utils.testing import create_random_tokens

_logger = logging.getLogger("osl-foundation")


class EphysGPTBuildMixin:
    def build_model(self) -> None:
        self.pretrained_model = self._load_pretrained_model()
        self.tokenizer = self._load_tokenizer()
        self.model = self._build_model()

        # Build the model with defined shapes
        self.model(self._build_inputs(), training=False)

        # self.model._metrics.append(self.transformer_decoder._metrics)

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

    def load_weights(
        self, filepath: str, allow_skip_mismatch: bool = False
    ) -> keras.Model:
        """Load weights with backward-compatibility for decoder non-trainable vars."""
        try:
            return super().load_weights(filepath)
        except ValueError as e:
            msg = str(e)
            if "Layer 'decoder' expected" not in msg:
                raise
            if not allow_skip_mismatch:
                raise ValueError(
                    "Decoder variable mismatch while loading weights. "
                    "This usually indicates a config/tokenizer mismatch between "
                    "the saved model and the current config. "
                    "To force a partial load, call load_weights(..., allow_skip_mismatch=True)."
                ) from e
            _logger.warning(
                "Decoder variable mismatch while loading weights; retrying with "
                "skip_mismatch=True (weights may be partially loaded)."
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
        pretrained_model = self.__class__.load_model(
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
        if (
            not hasattr(self, "refactored2original")
            or self.refactored2original is None
        ):
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
            self.refactored2original = tf.convert_to_tensor(
                refactored2original, dtype=tf.float32
            )

        return self.refactored2original

    def _get_orig_token_index(self, factored_token_index):
        if self.tokenizer is None:
            return factored_token_index

        # check if self.refactored2orig_token_index exists
        if (
            not hasattr(self, "refactored2orig_token_index")
            or self.refactored2orig_token_index is None
        ):
            n_tokens_refactored = self.config.model_config.n_tokens
            orig2refactored_token_index = self.tokenizer.vocab["label_map"] - 1
            # -1 indicates that the original token has been removed in the refactored tokenizer

            self.refactored2orig_token_index = -np.ones(
                (n_tokens_refactored,), dtype=np.int32
            )
            for original_token_idx, refactored_token_idx in enumerate(
                orig2refactored_token_index
            ):
                if refactored_token_idx > -1:
                    self.refactored2orig_token_index[
                        refactored_token_idx
                    ] = original_token_idx

        return self.refactored2orig_token_index[factored_token_index]

    def reconstruct_data_from_tokens(self, yy):
        """
        Reconstruct data from refactored tokens using the token basis layer.

        Parameters
        ----------
        yy : tf.Tensor
            Tensor of shape (B, l_out, n_channels, n_refactored_tokens)

        Returns
        -------
        y_recon : tf.Tensor
        Reconstructed data of shape (B, l_out, n_channels)
        """

        if self.tokenizer is None:
            raise ValueError(
                "Cannot reconstruct data without a tokenizer. Set model_config.tokenizer_path."
            )

        # Apply TimeDistributed(Conv1D)
        # Conv1D will convolve over the T dimension (axis 2 of x_permuted)
        # The same Conv1D kernel will be applied to all C 'time steps' (axis 1)

        refactored2original_projection = (
            self._get_refactored2original()
        )  # shape (n_refactored_tokens, n_original_tokens)

        y_recon = ReconstructDataLayer(
            refactored2original_projection, self.tokenizer
        )(yy)

        return y_recon

    def _build_model(self) -> keras.Model:
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

        # sample_token_layer = SampleTokenLayer(name="sample_tokens")

        self.reconstruct_data_layer = ReconstructDataLayer(
            self._get_refactored2original(),  # shape (n_refactored_tokens, n_original_tokens)
            self.tokenizer,
            name="reconstruct_data",
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
        x = self.transformer_decoder(
            x_input_embedded, training="decoder" in self.pretrained_layers
        )
        # x.shape = (batch_size, l_out, n_channels, model_dim)

        # Get the prediction of the next token
        y_pred_logits = prediction_head_layer(
            x, training="prediction_head" in self.pretrained_layers
        )
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
        embedding_matrix = (
            input_embedding_layer.token_embedding_layer.embeddings
        )  # shape: (n_refactoredtokens, embedding_dim)
        y_pred_soft_embeddings = keras.ops.matmul(y_pred_soft, embedding_matrix)
        y_pred_soft_embeddings = input_embedding_layer.token_embedding_output_layer(
            y_pred_soft_embeddings
        )
        # y_pred_soft_embeddings.shape = (batch_size, l_out, n_channels, model_dim)

        # Embed x_input directly via gather to avoid allocating a large one-hot tensor.
        x_input_token_embeddings = keras.ops.take(embedding_matrix, x_input, axis=0)
        x_input_token_embeddings = input_embedding_layer.token_embedding_output_layer(
            x_input_token_embeddings
        )
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

        # Compute Lyapunov loss in embedding space
        self.lyapunov_loss, _ = self.lyapunov_loss_layer(
            [y_pred_soft_embeddings, x_input_token_embeddings]
        )

        # Cross entropy loss
        self.cross_entropy_loss, _ = cross_entropy_loss_layer([y_pred_logits, y_true])

        # ---------- Create Model ---------- #

        # Keep training outputs compact; large tensors (e.g. logits trajectories) are not
        # needed for fit because losses/metrics are added inside the loss layers.
        outputs = {
            "cross_entropy_loss": self.cross_entropy_loss,
            "lyapunov_loss": self.lyapunov_loss,
        }

        return keras.Model(inputs=inputs, outputs=outputs, name=name)

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
