import os
import logging
from typing import Tuple, Union, List

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pqdm.threads import pqdm
from tqdm.auto import trange, tqdm

from osl_dynamics.data import Data
from osl_dynamics.utils.misc import get_argument
from osl_dynamics.utils.plotting import rough_square_axes

from osl_foundation.models.base import BaseModel
from osl_foundation.config import Config, get_config
from osl_foundation.inference.layers import rnn_layer, TokenWeightsLayer, MSELossLayer
from osl_foundation.utils import plotting

_logger = logging.getLogger("osl-foundation")


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer.
    This layer treats the channel dimension as the batch dimension
    and uses a GRU to encode the data.

    Parameters
    ----------
    rnn_type : str
        Type of the RNN layer. Either 'gru' or 'lstm'.
    rnn_n_layers : int
        Number of layers in the RNN.
    rnn_n_units : int
        Number of units in the RNN.
    """

    def __init__(self, rnn_type: str, rnn_n_layers: int, rnn_n_units: int, **kwargs):
        super().__init__(**kwargs)
        self.rnn = tf.keras.Sequential(
            [rnn_layer(rnn_type, rnn_n_units) for _ in range(rnn_n_layers)]
        )

    def call(self, inputs, **kwargs):
        x = inputs
        # x.shape = (batch_size, sequence_length, n_channels)

        # Prepare for RNN
        x = tf.transpose(x, perm=(0, 2, 1))
        # x.shape = (batch_size, n_channels, sequence_length)

        # Reshape for RNN
        x = tf.reshape(x, (-1, x.shape[-1], 1))
        # x.shape = (batch_size * n_channels, sequence_length, 1)

        # Encode
        x = self.rnn(x)
        # x.shape = (batch_size * n_channels, sequence_length, rnn_n_units)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder layer.
    This layer decodes the token weights to reconstruct the data.

    Parameters
    ----------
    n_channels : int
        Number of channels in the data.
    sequence_length : int
        Length of the sequence.
    n_tokens : int
        Number of tokens.
    token_dim : int
        Dimension of the token.
    token_kernel_padding : str
        Padding for the token kernel. Can take "valid", "same", or "causal".
    """

    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_tokens: int,
        token_dim: int,
        token_kernel_padding: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_tokens = n_tokens
        self.token_basis_layer = tf.keras.layers.Conv1D(
            filters=n_tokens,
            kernel_size=token_dim,
            padding=token_kernel_padding,
        )

    def call(self, inputs):
        x = inputs
        # x.shape = (batch_size * n_channels, sequence_length, n_tokens)

        x = tf.reduce_sum(self.token_basis_layer(x), axis=-1, keepdims=True)
        # x.shape = (batch_size * n_channels, sequence_length, 1)

        x = tf.reshape(x, (-1, self.n_channels, self.sequence_length))
        # x.shape = (batch_size, n_channels, sequence_length)

        x = tf.transpose(x, perm=(0, 2, 1))
        # x.shape = (batch_size, sequence_length, n_channels)

        token_weights = tf.reshape(
            inputs, (-1, self.n_channels, self.sequence_length, self.n_tokens)
        )
        # token_weights.shape = (batch_size, n_channels, sequence_length, n_tokens)

        token_weights = tf.transpose(token_weights, perm=(0, 2, 1, 3))
        # token_weights.shape = (batch_size, sequence_length, n_channels, n_tokens)

        return x, token_weights


class OSLTokenizer(BaseModel):
    """
    OSL Tokenizer model.

    Parameters
    ----------
    config : Config
        Config object.
    """

    def __init__(self, config: Config, strategy: tf.distribute.Strategy = None):
        super().__init__(config, strategy)
        self.vocab = {}

    def build_model(self) -> None:
        self.model = self._build_model()

    def fit(self, *args, n_jobs=1, **kwargs) -> None:
        """
        Wrapper for the standard keras fit method.

        Parameters
        ----------
        args : arguments
            Arguments for :code:`tf.keras.Model.fit()`.
        n_jobs : int, optional
            Number of jobs to refactor vocab, by default 1.
        kwargs : keyword arguments, optional
            Keyword arguments for :code:`tf.keras.Model.fit()`.
        """
        super().fit(*args, **kwargs)
        x = get_argument(self.model.fit, "x", args, kwargs)
        # Refactor vocabularies
        self.refactor_vocab(x, n_jobs=n_jobs)

    def save(self, dirname: str) -> None:
        super().save(dirname)
        with open(f"{dirname}/vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def _build_model(self) -> tf.keras.Model:
        """Model definition."""
        config = self.config.model_config

        # ---------- Layers ---------- #
        encoder_layer = EncoderLayer(
            config.rnn_type,
            config.rnn_n_layers,
            config.rnn_n_units,
            name="encoder",
        )

        token_weights_layer = TokenWeightsLayer(
            config.n_tokens,
            name="token_weights",
        )

        decoder_layer = Decoder(
            config.n_channels,
            config.sequence_length,
            config.n_tokens,
            config.token_dim,
            config.token_kernel_padding,
            name="decoder",
        )
        mse_loss_layer = MSELossLayer(name="mse_loss")

        # ---------- Forward Pass ---------- #

        inputs = tf.keras.layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )
        # Shape: (batch_size, sequence_length, n_channels)

        encoder_output = encoder_layer(inputs)
        # Shape: (batch_size * n_channels, sequence_length, rnn_n_units)

        token_weights = token_weights_layer(encoder_output)
        # Shape: (batch_size * n_channels, sequence_length, n_tokens)

        reconstructed_data, token_weights = decoder_layer(token_weights)
        # reconstructed_data.shape = (batch_size, sequence_length, n_channels)
        # token_weights.shape = (batch_size, sequence_length, n_channels, n_tokens)

        mse_loss = mse_loss_layer(inputs, reconstructed_data)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[mse_loss, reconstructed_data, token_weights],
            name=config.name,
        )

    def _tokenize_data(
        self,
        data: Data,
        concatenate: bool = False,
        return_weights: bool = False,
        n_jobs: int = 1,
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
    ]:
        """
        Tokenize data using the model.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            The data to tokenize.
        concatenate : bool, optional
            Whether to concatenate the tokens over all sessions, by default False.
        return_weights : bool, optional
            Whether the token weights are returned.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.

        Returns
        -------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens for each session. Shape is (n_samples, n_channels).
        token_weights : Union[np.ndarray, List[np.ndarray]]
            The token weights for each token in each session.
            Only returned if :code:`return_weights=True`.
            Shape is (n_samples, n_channels, n_tokens).
        """

        dataset = self.make_dataset(data, shuffle=False, concatenate=False)

        def _tokenize_data_per_session(d):
            # Tokenize for a single session
            tw = np.concatenate([self.model(x, training=False)[2].numpy() for x in d])

            # Concatenate over sequences
            tw = np.concatenate(tw)
            # tw.shape = (n_samples, n_channels, n_tokens)

            # Get the token with the maximum weight
            t = np.argmax(tw, axis=-1)
            # t.shape = (n_samples, n_channels)

            if return_weights:
                return t, tw
            return t

        # Keywords for parallel processing
        kwargs = [{"d": d} for d in dataset]

        if len(dataset) == 1:
            _logger.info("Tokenizing data...")
            results = [_tokenize_data_per_session(**kwargs[0])]

        elif n_jobs == 1:
            results = []
            for i in trange(len(dataset), desc="Tokenizing data"):
                results.append(_tokenize_data_per_session(**kwargs[i]))

        else:
            results = pqdm(
                kwargs,
                _tokenize_data_per_session,
                n_jobs=n_jobs,
                desc="Tokenizing data",
                argument_type="kwargs",
                exception_behaviour="immediate",
            )

        # Unpack results
        if return_weights:
            tokens, token_weights = zip(*results)
            tokens = list(tokens)
            token_weights = list(token_weights)
        else:
            tokens = results

        if concatenate:
            tokens = np.concatenate(tokens)
            if return_weights:
                token_weights = np.concatenate(token_weights)

        if return_weights:
            return tokens, token_weights

        return tokens

    def refactor_vocab(
        self,
        data: Data,
        sort: bool = True,
        trim: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """
        Refactor the vocabulary based on the data.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Time series data.
        sort : bool, optional
            Should we sort the tokens by frequency?, by default True.
        trim : bool, optional
            Should we remove tokens with zero frequency?, by default True.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.
        """

        config = self.config.model_config

        _logger.info("Refactoring vocabulary...")
        tokens = self._tokenize_data(data, n_jobs=n_jobs)

        # Count the tokens across samples and channels for each session
        token_counts = np.array(
            [np.bincount(t.flatten(), minlength=config.n_tokens) for t in tokens],
            dtype=np.int32,
        )

        # Get token order based on token counts
        token_order = (
            np.argsort(np.sum(token_counts, axis=0))[::-1]
            if sort
            else np.arange(config.n_tokens)
        )

        # Remove all token indices with zero counts
        if trim:
            token_order = token_order[np.sum(token_counts, axis=0)[token_order] > 0]

        # Apply trimming and ordering to token counts
        token_counts = token_counts[:, token_order]

        # Get labels from token orders
        label_map = np.zeros(config.n_tokens, dtype=np.int32)
        label_map[token_order] = np.arange(len(token_order)) + 1

        self.vocab = {
            "token_order": token_order,
            "token_counts": token_counts,
            "total_token_counts": np.sum(token_counts, axis=0),
            "label_map": label_map,
        }

    def tokenize_data(
        self, data: Data, concatenate: bool = False, n_jobs: int = 1
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Tokenize data using the model and the refactored vocabularies.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            The data to tokenize.
        concatenate : bool, optional
            Whether to concatenate the tokens over all sessions, by default False.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.

        Returns
        -------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens for each session. Shape is (n_samples, n_channels).
        """
        if not self.vocab:
            raise ValueError("Vocabularies are not refactored.")
        tokens = self._tokenize_data(data, n_jobs=n_jobs)

        # Remap tokens to refactored tokens
        tokens = [self.vocab["label_map"][t] for t in tokens]

        if concatenate:
            tokens = np.concatenate(tokens)

        return tokens

    def get_pve(self, data: Data) -> np.ndarray:
        """
        Get the percentage of variance explained by the tokens.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Time series data.

        Returns
        -------
        pve : np.ndarray
            The percentage of variance explained by the tokens for each session.
        """
        dataset = self.make_dataset(data, shuffle=False, concatenate=False)

        def _get_pve(d):
            # PVE for a single session
            original_x = np.concatenate([x["data"].numpy() for x in d])
            reconstructed_x = np.concatenate(
                [self.model(x, training=False)[1].numpy() for x in d]
            )
            pve = 100 * (
                1 - np.sum((original_x - reconstructed_x) ** 2) / np.sum(original_x**2)
            )
            return pve

        _logger.info("Calculating Percentage of Variance Explained...")
        pve = []
        for d in tqdm(
            dataset,
            desc="Calculating Percentage of Variance Explained",
            total=len(dataset),
        ):
            pve.append(_get_pve(d))

        return np.array(pve)

    def _reconstruct_data(
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
            Shape of tokens of each session: (n_samples, n_channels).
        concatenate : bool, optional
            Whether to concatenate the reconstructed data over all sessions, by default False.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
            Shape of reconstructed data of each session: (n_samples, n_channels).
        """

        if not isinstance(tokens, list):
            tokens = [tokens]

        n_tokens = self.config.model_config.n_tokens
        n_channels = self.config.model_config.n_channels
        sequence_length = self.config.model_config.sequence_length

        tokens = Data(tokens)
        tokens = self.make_dataset(
            tokens, shuffle=False, concatenate=False, drop_last_batch=False
        )

        token_basis_layer = self.model.get_layer("decoder").token_basis_layer

        def _reconstruct_data_per_session(t):
            # Reconstruct for a single session
            _reconstructed_data = []
            for x in t:
                x = x["data"].numpy()
                # x.shape = (batch_size, sequence_length, n_channels)

                x = np.transpose(x, axes=(0, 2, 1))
                # x.shape = (batch_size, n_channels, sequence_length)

                x = np.reshape(x, (-1, x.shape[-1]))
                # x.shape = (batch_size * n_channels, sequence_length)

                x = tf.one_hot(x, n_tokens).numpy().astype(np.float32)
                # x.shape = (batch_size * n_channels, sequence_length, n_tokens)

                x = np.sum(token_basis_layer(x), axis=-1)
                # x.shape = (batch_size * n_channels, sequence_length)

                x = np.reshape(x, (-1, n_channels, sequence_length))
                # x.shape = (batch_size, n_channels, sequence_length)

                x = np.transpose(x, axes=(0, 2, 1))
                # x.shape = (batch_size, sequence_length, n_channels)

                _reconstructed_data.append(np.concatenate(x))

            return np.concatenate(_reconstructed_data)

        reconstructed_data = []
        for t in tqdm(tokens, desc="Reconstructing data", total=len(tokens)):
            reconstructed_data.append(_reconstruct_data_per_session(t))

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)

        return reconstructed_data

    def reconstruct_data(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        concatenate: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstruct data from tokens using the refactored vocabularies.

        Parameters
        ----------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens to reconstruct.
            Shape of tokens of each session: (n_samples, n_channels).
        concatenate : bool, optional
            Whether to concatenate the reconstructed data over all sessions, by default False.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
            Shape of reconstructed data of each session: (n_samples, n_channels).
        """
        if not self.vocab:
            raise ValueError("Vocabularies are not refactored.")

        # Remap refactored tokens to original tokens
        remapped_tokens = []
        for t in tokens:
            remapped_t = (
                np.ones(t.shape, dtype=np.int32)
                * np.where(self.vocab["label_map"] == 0)[0][0]
            )
            t -= 1
            remapped_t[t >= 0] = self.vocab["token_order"][t[t >= 0]]
            remapped_tokens.append(remapped_t)

        return self._reconstruct_data(remapped_tokens, concatenate)

    def get_token_kernel_response(
        self, data: Data = None, input: Union[str, np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns stimulus response of tokens to passed input.

        Parameters
        ----------
        data : osl_dynamics.data.Data, optional
            Time series data for refactoring tokens.
        input : Union[str, np.ndarray], optional
            Stimulus input to get kernel response for. Should be "impulse",
            "tophat", or a 1D array. Defaults to "impulse".

        Returns
        -------
        token_response : np.ndarray
            An containing a stimulus response for each token.
        input : np.ndarray
            Input used to get stimulus response for tokens.
        """

        token_dim = self.config.model_config.token_dim
        n_tokens = self.config.model_config.n_tokens

        if not self.vocab:
            # Refactor vocabularies
            if data is None:
                raise ValueError("Data is required to refactor vocabularies.")
            self.refactor_vocab(data)

        # Make a stimulus
        if input in [None, "impulse"]:
            input = np.zeros(token_dim * 2)
            input[token_dim] = 1
        elif input == "tophat":
            input = np.zeros(token_dim * 6)
            input[token_dim : token_dim * 5] = 1
        elif isinstance(input, np.ndarray) and input.ndim != 1:
            raise ValueError("Input should be a 1D array.")
        else:
            raise ValueError("Invalid input.")

        n_samples = input.shape[0]

        # Get stimulus response for each token
        token_basis_layer = self.model.get_layer("decoder").token_basis_layer

        kernel_response = []
        for n in range(n_tokens):
            token_weights = np.zeros((1, n_samples, n_tokens))
            token_weights[0, :, n] = input
            response = np.squeeze(
                np.sum(token_basis_layer(token_weights).numpy(), axis=-1)
            )  # resposne.shape = (n_samples)
            kernel_response.append(response)

        # Remap to refactored tokens
        token_response = np.array(
            [kernel_response[order] for order in self.vocab["token_order"]]
        )  # token_response.shape = (n_refactored_tokens, n_samples)

        return token_response, input

    def plot_pve(self, data: Data, plot_dir: str = None) -> None:
        """
        Plots a histogram of the percentage of variance explained by the tokens.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Time series data.
        plot_dir : str, optional
            Directory to save the plot.
        """
        # Calculate PVEs across all sessions
        pves = self.get_pve(data)
        plotting.plot_pve(pve=pves, plot_dir=plot_dir)

    def plot_token_counts(self, data: Data = None, plot_dir: str = None) -> None:
        """
        Plots a histogram of token counts over all sessions.

        Parameters
        ----------
        data : osl_dynamics.data.Data, optional
            Time series data for refactoring tokens.
        plot_dir : str, optional
            Directory to save the plot.
        """
        if not self.vocab:
            # Refactor vocabularies
            if data is None:
                raise ValueError("Data is required to refactor vocabularies.")
            self.refactor_vocab(data)

        total_token_counts = self.vocab["total_token_counts"]

        # Plot a histogram of token counts
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes.bar(
            range(1, total_token_counts.shape[0] + 1),
            total_token_counts,
            color="skyblue",
            edgecolor="black",
        )
        axes.set_xlabel("Token Index")
        axes.set_ylabel("Number of Occurrences")
        axes.set_title(f"Token Histogram (N={len(total_token_counts)})")
        plt.tight_layout()

        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/token_counts.png")
            plt.close(fig)

    def plot_token_response(
        self, data: Data = None, input: str = "impulse", plot_dir: str = None
    ) -> None:
        """
        Plots a stimulus response of each token kernel.

        Parameters
        ----------
        data : osl_dynamics.data.Data, optional
            Time series data for refactoring tokens.
        input : str, optional
            Stimulus input to get kernel response for. Should be "impulse"
            or "tophat". Defaults to "impulse".
        plot_dir : str, optional
            Directory to save the plot.
        """

        # Get token kernel responses
        token_response, input = self.get_token_kernel_response(data, input)

        # Number of tokens
        n_tokens = len(token_response)

        # Limit number of tokens to plot
        if n_tokens > 30:
            n_tokens = 30
            token_response = token_response[:n_tokens]  # select top 30 tokens

        # Plot stimulus responses for each token
        short, long, _ = rough_square_axes(n_tokens)
        fig, axes = plt.subplots(nrows=short, ncols=long, figsize=(2 * short, 3 * long))
        axes = axes.flatten()
        for n, resp in enumerate(token_response):
            axes[n].plot(resp, label="Token Response" if n == 0 else "")
            axes[n].plot(input, "r", label="Input" if n == 0 else "")
            axes[n].set_ylim([-1.1, 1.1])
        for ax in axes[n_tokens:]:
            ax.axis("off")
        fig.legend()
        plt.tight_layout()

        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/token_response.png")
            plt.close(fig)

    def plot_fitted_signal(
        self, data_path: str, ground_truth_path: str = None, plot_dir: str = None
    ) -> None:
        """
        Plots a signal reconstructed from tokenized data and its token weights.

        Parameters
        ----------
        data_path : str
            A path to a file containing the original data.
        ground_truth_path : str, optional
            A path to a file containing the ground truth data.
        plot_dir : str, optional
            Directory to save the plot.
        """

        def _load_and_normalize_data(file_path, tmp_dir):
            """Load and normalize data from a given path."""

            # Define anonymous function
            normalize = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)

            # Get data
            dataset = Data(file_path, store_dir=tmp_dir)
            data = normalize(dataset.arrays[0])
            dataset.delete_dir()  # delete temporary directory
            return data

        # Get simulated data and its ground truth
        original_data = _load_and_normalize_data(data_path, "tmp_org_data")
        if ground_truth_path:
            true_data = _load_and_normalize_data(ground_truth_path, "tmp_true_data")

        # Get data reconstructed from tokens
        tokenized_data, token_weights = self._tokenize_data(
            original_data, return_weights=True
        )
        fitted_data = self._reconstruct_data(tokenized_data)

        # Plot data signals and token weights
        n_channels = min(original_data.shape[1], 3)  # number of channels to plot
        start_idx, end_idx = 200, 500  # start and end indices to plot
        for n in range(n_channels):
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
            axes[0].plot(original_data[start_idx:end_idx, n], label="Original")
            if ground_truth_path:
                axes[0].plot(true_data[start_idx:end_idx, n], label="True")
            axes[0].plot(fitted_data[start_idx:end_idx, n], label="Fitted")
            axes[0].set_title(f"Channel {n}: Data Signals")
            axes[0].legend()
            axes[1].plot(token_weights[0][start_idx:end_idx, n, :])
            axes[1].set_title(f"Token Weights")
            plt.tight_layout()

            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
                fig.savefig(f"{plot_dir}/fitted_signal_ch{n}.png")
                plt.close(fig)


class MuTransformTokenizer:
    def __init__(self, config: Config, strategy: tf.distribute.Strategy = None):
        """
        MuTransformTokenizer class.

        Parameters
        ----------
        config : Config
            Configuration object.
        strategy : tf.distribute.Strategy
            For API compatibility, not used.
        """
        self.config = config
        self.vocab = {}

    def fit(self, x: Union[Data, np.ndarray, List[np.ndarray]]) -> None:
        if isinstance(x, Data):
            x = x.time_series(concatenate=False)

        if not isinstance(x, list):
            x = [x]

        self.vocab["data_range"] = self.get_data_range(x)
        self.vocab["bins"] = self.get_bins()
        self.vocab["bins_average"] = self.get_bins_average()
        self.vocab["total_token_counts"] = self.get_token_counts(x)

    def get_token_counts(
        self, data: Union[Data, np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Get the token counts for the given data.

        Parameters
        ----------
        data : Union[Data, np.ndarray, List[np.ndarray]]
            The data to get token counts for.
        """
        tokens = self.tokenize_data(data, concatenate=True)
        token_counts = np.bincount(
            tokens.flatten(), minlength=self.config.model_config.n_tokens
        )
        return token_counts

    def save_config(self, dirname: str) -> None:
        """
        Save the config to a directory.
        The config is saved as a YAML file as 'config.yml'.

        Parameters
        ----------
        dirname : str
            Directory to save the configuration.
        """
        self.config.save_config(dirname)

    def save(self, dirname: str) -> None:
        """Save the model config and attributes.

        Parameters
        ----------
        dirname : str
            Directory to save the model.
        """
        os.makedirs(dirname, exist_ok=True)
        self.save_config(dirname)
        with open(f"{dirname}/vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def get_data_range(
        self, data: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[float, float]:

        if not isinstance(data, list):
            data = [data]

        min_ = np.inf
        max_ = -np.inf
        for x in tqdm(data, desc="Calculating data range", total=len(data)):
            min_ = min(min_, np.min(x))
            max_ = max(max_, np.max(x))

        return min_, max_

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalise data to the range of (-1, 1).
        """
        if self.vocab["data_range"] is None:
            raise ValueError("Data range is not set. Call fit() first.")

        min_, max_ = self.vocab["data_range"]
        x = (x - min_) / (max_ - min_)
        x = 2 * x - 1
        return x

    def reverse_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalise data to the original range.
        """
        if self.vocab["data_range"] is None:
            raise ValueError("Data range is not set. Call fit() first.")

        min_, max_ = self.vocab["data_range"]
        x = (x + 1) / 2
        x = x * (max_ - min_) + min_
        return x

    def mu_transform(self, x: np.ndarray) -> np.ndarray:
        mu = self.config.model_config.mu
        return np.sign(x) * np.log1p(np.abs(x) * mu) / np.log1p(mu)

    def reverse_mu_transform(self, x: np.ndarray) -> np.ndarray:
        mu = self.config.model_config.mu
        return np.sign(x) * (np.expm1(np.abs(x) * np.log1p(mu)) / mu)

    def _get_bins(self) -> np.ndarray:
        n_tokens = self.config.model_config.n_tokens

        # tokens 0 and n_tokens are for values < -1 and >= 1
        # the rest n_tokens - 2 are for values in between
        # the bins are equally spaced in the range of (-1, 1)

        bins = np.linspace(-1, 1, n_tokens - 1)
        return bins

    def get_bins(self) -> np.ndarray:
        bins = self._get_bins()
        bins = self.reverse_mu_transform(bins)
        bins = self.reverse_normalize(bins)
        return bins

    def _get_bins_average(self) -> np.ndarray:
        bins = self._get_bins()
        bins_average = np.zeros(len(bins) + 1)
        bins_average[0] = -1
        bins_average[-1] = 1

        # Calculate the average of each bin
        for i in range(len(bins) - 1):
            bins_average[i + 1] = (bins[i] + bins[i + 1]) / 2
        return bins_average

    def get_bins_average(self) -> np.ndarray:
        bins_average = self._get_bins_average()
        bins_average = self.reverse_mu_transform(bins_average)
        bins_average = self.reverse_normalize(bins_average)
        return bins_average

    def tokenize_data(
        self,
        data: Union[Data, np.ndarray, List[np.ndarray]],
        concatenate: bool = False,
        n_jobs: int = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(data, Data):
            data = data.time_series(concatenate=False)

        if not isinstance(data, list):
            data = [data]

        def _tokenize_data_per_session(d):
            return np.digitize(d, self.vocab["bins"])

        # Keywords for parallel processing
        kwargs = [{"d": d} for d in data]
        if len(data) == 1:
            _logger.info("Tokenizing data...")
            tokens = [_tokenize_data_per_session(**kwargs[0])]

        elif n_jobs == 1:
            tokens = []
            for i in trange(len(data), desc="Tokenizing data"):
                tokens.append(_tokenize_data_per_session(**kwargs[i]))

        else:
            tokens = pqdm(
                kwargs,
                _tokenize_data_per_session,
                n_jobs=n_jobs,
                desc="Tokenizing data",
                argument_type="kwargs",
                exception_behaviour="immediate",
            )

        if concatenate or len(tokens) == 1:
            tokens = np.concatenate(tokens)

        return tokens

    def reconstruct_data(
        self,
        tokens: Union[Data, np.ndarray, List[np.ndarray]],
        concatenate: bool = False,
        n_jobs: int = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(tokens, Data):
            tokens = tokens.time_series(concatenate=False)

        if not isinstance(tokens, list):
            tokens = [tokens]

        def _reconstruct_data_per_session(t):
            x = self.vocab["bins_average"][t]
            return x

        # Keywords for parallel processing
        kwargs = [{"t": t} for t in tokens]
        if len(tokens) == 1:
            _logger.info("Reconstructing data...")
            reconstructed_data = [_reconstruct_data_per_session(**kwargs[0])]

        elif n_jobs == 1:
            reconstructed_data = []
            for i in trange(len(tokens), desc="Reconstructing data"):
                reconstructed_data.append(_reconstruct_data_per_session(**kwargs[i]))

        else:
            reconstructed_data = pqdm(
                kwargs,
                _reconstruct_data_per_session,
                n_jobs=n_jobs,
                desc="Reconstructing data",
                argument_type="kwargs",
                exception_behaviour="immediate",
            )

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)
        return reconstructed_data

    def get_pve(
        self, data: Union[Data, np.ndarray, List[np.ndarray]], n_jobs: int = 1
    ) -> np.ndarray:
        """
        Get the percentage of variance explained by the tokens.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Time series data.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.

        Returns
        -------
        pve : np.ndarray
            The percentage of variance explained by the tokens for each session.
        """
        if isinstance(data, Data):
            data = data.time_series(concatenate=False)
        if not isinstance(data, list):
            data = [data]

        tokens = self.tokenize_data(data, n_jobs=n_jobs)
        reconstructed_data = self.reconstruct_data(tokens, n_jobs=n_jobs)

        if not isinstance(reconstructed_data, list):
            reconstructed_data = [reconstructed_data]

        pve = []
        for i in range(len(data)):
            original_x = data[i]
            reconstructed_x = reconstructed_data[i]
            pve.append(
                100
                * (
                    1
                    - np.sum((original_x - reconstructed_x) ** 2)
                    / np.sum(original_x**2)
                )
            )

        if len(pve) == 1:
            pve = pve[0]
        return np.array(pve)

    def plot_fitted_signal(
        self,
        data_path: str,
        ground_truth_path: str = None,
        plot_bins: bool = False,
        plot_dir: str = None,
    ) -> None:
        """
        Plots a signal reconstructed from tokenized data

        Parameters
        ----------
        data_path : str
            A path to a file containing the original data.
        ground_truth_path : str, optional
            A path to a file containing the ground truth data.
        plot_bins : bool, optional
            Whether to plot the bins, by default False.
        plot_dir : str, optional
            Directory to save the plot.
        """

        def _load_and_normalize_data(file_path, tmp_dir):
            """Load and normalize data from a given path."""

            # Define anonymous function
            normalize = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)

            # Get data
            dataset = Data(file_path, store_dir=tmp_dir)
            data = normalize(dataset.arrays[0])
            dataset.delete_dir()  # delete temporary directory
            return data

        # Get simulated data and its ground truth
        original_data = _load_and_normalize_data(data_path, "tmp_org_data")
        if ground_truth_path:
            true_data = _load_and_normalize_data(ground_truth_path, "tmp_true_data")

        # Get data reconstructed from tokens
        tokenized_data = self.tokenize_data(original_data)
        fitted_data = self.reconstruct_data(tokenized_data)

        # Plot data signals and token weights
        n_channels = min(original_data.shape[1], 3)  # number of channels to plot
        start_idx, end_idx = 200, 500  # start and end indices to plot

        fig, axes = plt.subplots(nrows=n_channels, ncols=1, figsize=(20, 12))
        for i in range(n_channels):
            axes[i].plot(original_data[start_idx:end_idx, i], label="Original")
            if ground_truth_path:
                axes[i].plot(true_data[start_idx:end_idx, i], label="True")
            axes[i].plot(fitted_data[start_idx:end_idx, i], label="Fitted")
            axes[i].set_title(f"Channel {i}: Data Signals")
            if plot_bins:
                for j in range(len(self.vocab["bins"])):
                    axes[i].axhline(
                        self.vocab["bins"][j], color="red", linestyle="--", linewidth=1
                    )
            axes[i].legend()
        plt.tight_layout()
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/fitted_signal.png")
            plt.close(fig)

    @staticmethod
    def load_config(dirname: str) -> Config:
        """
        Load the config from a directory.

        Parameters
        ----------
        dirname : str
            Directory to load the configuration from.

        Returns
        -------
        config : Config
            Configuration object.
        """
        return get_config(f"{dirname}/config.yml")

    @classmethod
    def load_model(
        cls,
        dirname: str,
        checkpoint: str = None,
        strategy: tf.distribute.Strategy = None,
    ):
        """
        Load a saved model.

        Parameters
        ----------
        dirname : str
            Directory containing the saved model.
        checkpoint : str
            For API compatibility, not used.
        strategy : tf.distribute.Strategy
            For API compatibility, not used.

        Returns
        -------
        model : MuTransformTokenizer
            The loaded model.
        """
        config = cls.load_config(dirname)
        model = cls(config)
        with open(f"{dirname}/vocab.pkl", "rb") as f:
            model.vocab = pickle.load(f)
        return model
