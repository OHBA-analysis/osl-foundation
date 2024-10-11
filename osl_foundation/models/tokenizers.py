import logging
from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from osl_dynamics.array_ops import get_one_hot
from osl_dynamics.utils.misc import get_argument

from osl_foundation.models.base import BaseModel
from osl_foundation.data import Data
from osl_foundation.config import Config
from osl_foundation.inference.layers import TokenWeightsLayer, MSELossLayer

_logger = logging.getLogger("osl-foundation")


class OSLTokenizer(BaseModel):
    """
    OSL Tokenizer model.

    Parameters
    ----------
    config : Config
        Config object.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.vocab = {}

    def build_model(self) -> None:
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Model definition."""
        config = self.config.model_config

        # ---------- Layers ---------- #

        encoder_layer = tf.keras.layers.GRU(
            config.rnn_n_units,
            return_sequences=True,
            stateful=False,
            name="encoder",
        )

        token_weights_layer = TokenWeightsLayer(
            config.n_tokens,
            name="token_weights",
        )

        class Decoder(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.token_basis_layer = tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=config.token_dim,
                    padding="same",
                )

            def call(self, inputs):
                return self.token_basis_layer(inputs)

            # def __init__(self, **kwargs):
            #     super().__init__(**kwargs)
            #     self.token_basis_layer = tf.keras.layers.Conv1D(
            #         filters=config.n_tokens,
            #         kernel_size=config.token_dim,
            #         padding="same",
            #     )

            # def call(self, inputs):
            #     output = self.token_basis_layer(inputs)
            #     return tf.reduce_sum(output, axis=-1, keepdims=True)

        decoder_layer = Decoder(name="decoder")

        mse_loss_layer = MSELossLayer(name="mse_loss")

        # ---------- Forward Pass ---------- #

        inputs = tf.keras.layers.Input(shape=(config.sequence_length, 1), name="data")
        # Shape: (batch_size, sequence_length, 1)

        encoder_output = encoder_layer(inputs)
        # Shape: (batch_size, sequence_length, rnn_n_units)

        token_weights = token_weights_layer(encoder_output)
        # Shape: (batch_size, sequence_length, n_tokens)

        reconstructed_data = decoder_layer(token_weights)
        # Shape: (batch_size, sequence_length, 1)

        mse_loss = mse_loss_layer(inputs, reconstructed_data)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[mse_loss, reconstructed_data, token_weights],
            name=config.name,
        )

    # def fit(self, *args, **kwargs):
    #     """Fit the model and refactor the vocabulary."""
    #     super().fit(*args, **kwargs)
    #     x = get_argument(self.model.fit, "x", args, kwargs)
    #     self.refactor_vocab(x)

    def tokenize_data(
        self,
        data: Data,
        concatenate: bool = False,
        split_channels: bool = True,
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
    ]:
        """
        Tokenize data using the model.

        Parameters
        ----------
        data : osl_foundation.data.Data
            The data to tokenize.
        concatenate : bool, optional
            Whether to concatenate the tokens over all sessions, by default False.
        split_channels : bool, optional
            Whether to split the sequences back to different channels, by default True.

        Returns
        -------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens for each session.
        token_weights : Union[np.ndarray, List[np.ndarray]]
            The token weights for each token in each session.
        """
        # Concatenate channels if it's not done already
        if getattr(data, "original_n_channels", None) is None:
            data.concatenate_channels()

        dataset = self.make_dataset(data, shuffle=False, concatenate=False)

        def _tokenize_data(d):
            # Tokenize for a single session
            tw = []
            for x in d:
                tw.append(self.model(x)[2])
            # Concatenate over batches and sequences
            tw = np.concatenate(np.concatenate(tw))
            return tw

        _logger.info("Tokenizing data...")
        token_weights = []
        for d in tqdm(dataset, desc="Tokenizing data", total=len(dataset)):
            token_weights.append(_tokenize_data(d))

        tokens = [np.argmax(tw, axis=-1, keepdims=True) for tw in token_weights]

        if split_channels:
            tokens = [
                np.reshape(t, (-1, data.original_n_channels), order="F") for t in tokens
            ]
            token_weights = [
                np.reshape(
                    tw,
                    (-1, data.original_n_channels, self.config.model_config.n_tokens),
                    order="F",
                )
                for tw in token_weights
            ]

        if concatenate:
            tokens = np.concatenate(tokens)
            token_weights = np.concatenate(token_weights)

        return tokens, token_weights

    def refactor_vocab(
        self,
        data: Data,
        sort: bool = True,
        trim: bool = True,
    ) -> None:
        """
        Refactor the vocabulary based on the data.

        Parameters
        ----------
        data : osl_foundation.data.Data
            Time series data.
        sort : bool, optional
            Should we sort the tokens by frequency?, by default True.
        trim : bool, optional
            Should we remove tokens with zero frequency?, by default True.
        """

        config = self.config.model_config

        _logger.info("Refactoring vocabulary...")
        tokens = self.tokenize_data(data)[0]

        token_counts = np.array(
            [np.bincount(t, minlength=config.n_tokens) for t in tokens], dtype=np.int32
        )
        token_order = (
            np.argsort(np.sum(token_counts, axis=0))[::-1]
            if sort
            else np.arange(config.n_tokens)
        )
        if trim:
            token_order = token_order[np.sum(token_counts, axis=0) > 0]

        label_map = np.zeros(config.n_tokens, dtype=np.int32)
        label_map[token_order] = np.arange(len(token_order)) + 1

        self.vocab = {
            "token_order": token_order,
            "token_counts": token_counts,
            "total_token_counts": np.sum(token_counts, axis=0),
            "label_map": label_map,
        }

    def get_pve(self, data: Data) -> np.ndarray:
        """
        Get the percentage of variance explained by the tokens.

        Parameters
        ----------
        data : osl_foundation.data.Data
            Time series data.

        Returns
        -------
        pve : np.ndarray
            The percentage of variance explained by the tokens for each session.
        """
        dataset = self.make_dataset(data, shuffle=False, concatenate=False)

        def _get_pve(d):
            # PVE for a single session
            pve_ = []
            for x in d:
                original_x = x["data"]
                reconstructed_x = self.model(x)[1]
                pve_.append(
                    100
                    * (
                        1
                        - np.sum((original_x - reconstructed_x) ** 2)
                        / np.sum(original_x**2)
                    )
                )
            return np.mean(pve_)

        _logger.info("Calculating Percentage of Variance Explained...")
        pve = []
        for d in tqdm(
            dataset,
            desc="Calculating Percentage of Variance Explained",
            total=len(dataset),
        ):
            pve.append(_get_pve(d))

        return np.array(pve)

    def reconstruct_data(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        concatenate: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstruct data from tokens.

        Parameters
        ----------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens to reconstruct.
            Shape of tokens of each session: (n_samples, n_channels) or (n_samples,)
        concatenate : bool, optional
            Whether to concatenate the reconstructed data over all sessions, by default False.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
            Shape of reconstructed data of each session: (n_samples, n_channels) or (n_samples,)
        """

        if not isinstance(tokens, list):
            tokens = [tokens]

        decoder = self.model.get_layer("decoder")

        def _reconstruct_data(t):
            # Reconstruct for a single array of tokens
            # t.shape = (n_samples, n_channels) or (n_samples,)

            original_shape = t.shape
            t = np.reshape(t, -1, order="F")
            # Shape: (n_samples, )

            t_one_hot = get_one_hot(t, self.config.model_config.n_tokens).astype(
                np.float32
            )
            # Shape: (n_samples, n_tokens)

            # Add batch dimension
            t_one_hot = np.expand_dims(t_one_hot, axis=0)
            # Shape: (1, n_samples, n_tokens)

            reconstructed_data = np.squeeze(decoder(t_one_hot))
            # Shape: (n_samples, )

            return np.reshape(reconstructed_data, original_shape, order="F")

        reconstructed_data = []
        for t in tqdm(tokens, desc="Reconstructing data", total=len(tokens)):
            reconstructed_data.append(_reconstruct_data(t))

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)

        return reconstructed_data
