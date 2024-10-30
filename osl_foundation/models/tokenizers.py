import os
import logging
from typing import Tuple, Union, List

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.auto import tqdm

from osl_dynamics.data import Data
from osl_dynamics.array_ops import get_one_hot
from osl_dynamics.utils.plotting import rough_square_axes

from osl_foundation.models.base import BaseModel
from osl_foundation.config import Config, get_config
from osl_foundation.inference.layers import TokenWeightsLayer, MSELossLayer

_logger = logging.getLogger("osl-foundation")


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer.
    This layer treats the channel dimension as the batch dimension
    and uses a GRU to encode the data.

    Parameters
    ----------
    rnn_n_units : int
        Number of units in the RNN.
    """

    def __init__(self, rnn_n_units: int, **kwargs):
        super().__init__(**kwargs)
        self.rnn = tf.keras.layers.GRU(
            rnn_n_units,
            return_sequences=True,
            stateful=False,
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
    """

    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_tokens: int,
        token_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_tokens = n_tokens
        self.token_basis_layer = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=token_dim,
            padding="same",
        )

    def call(self, inputs):
        x = inputs
        # x.shape = (batch_size * n_channels, sequence_length, n_tokens)

        x = self.token_basis_layer(x)
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

    def __init__(self, config: Config):
        super().__init__(config)
        self.vocab = {}

    def build_model(self) -> None:
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Model definition."""
        config = self.config.model_config

        # ---------- Layers ---------- #
        encoder_layer = EncoderLayer(
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

    def tokenize_data(
        self,
        data: Data,
        concatenate: bool = False,
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

        Returns
        -------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokens for each session.
        token_weights : Union[np.ndarray, List[np.ndarray]]
            The token weights for each token in each session.
        """

        dataset = self.make_dataset(data, shuffle=False, concatenate=False)

        def _tokenize_data(d):
            # Tokenize for a single session
            tw = []
            for x in d:
                tw.append(self.model(x)[2])
            # Concatenate over batches and sequences
            tw = np.concatenate(np.concatenate(tw))
            # tw.shape = (n_samples, n_channels, n_tokens)
            return tw

        _logger.info("Tokenizing data...")
        token_weights = []
        for d in tqdm(dataset, desc="Tokenizing data", total=len(dataset)):
            token_weights.append(_tokenize_data(d))

        tokens = [np.argmax(tw, axis=-1) for tw in token_weights]

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
        data : osl_dynamics.data.Data
            Time series data.
        sort : bool, optional
            Should we sort the tokens by frequency?, by default True.
        trim : bool, optional
            Should we remove tokens with zero frequency?, by default True.
        """

        config = self.config.model_config

        _logger.info("Refactoring vocabulary...")
        tokens = self.tokenize_data(data)[0]

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

        token_basis_layer = self.model.get_layer("decoder").token_basis_layer

        def _reconstruct_data(t):
            n_tokens = self.config.model_config.n_tokens

            # Reconstruct for a single array of tokens
            # t.shape = (n_samples, n_channels) or (n_samples,)

            if t.ndim == 1:
                t = t[:, np.newaxis]

            n_channels = t.shape[1]
            t_one_hot = [
                get_one_hot(t[:, n], n_tokens).astype(
                    np.float32
                )  # Shape: (n_samples, n_tokens)
                for n in range(n_channels)
            ]

            # Add batch dimension
            t_one_hot = np.array(t_one_hot)  # having channel in batch dimension
            # Shape: (n_channels, n_samples, n_tokens)

            reconstructed_data = np.squeeze(token_basis_layer(t_one_hot))
            # x.shape = (n_channels, n_samples)

            # Reorder dimensions for consistency
            reconstructed_data = np.transpose(reconstructed_data, axes=(1, 0))

            return reconstructed_data

        reconstructed_data = []
        for t in tqdm(tokens, desc="Reconstructing data", total=len(tokens)):
            reconstructed_data.append(_reconstruct_data(t))

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)

        return reconstructed_data

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
                token_basis_layer(token_weights).numpy()
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

        # Plot a histogram of PVEs
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes.hist(pves, bins=20, color="skyblue", edgecolor="black")
        axes.set_xlabel("PVE (%)")
        axes.set_ylabel("Number of Sessions")
        axes.set_title(
            "Percentage of Variance Explained (Avg: {:.2f}%)".format(pves.mean())
        )
        plt.tight_layout()
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(f"{plot_dir}/pve_histogram.png")
            plt.close(fig)

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
        self, data_dir, sess_id: int = 0, plot_dir: str = None
    ) -> None:
        """
        Plots a signal reconstructed from tokenized data and its token weights.
        This method assumes that data are stored as {data_dir}/x_{sess_id}.npy,
        and the ground truth is stored as {data_dir}/ground_truth/true_signal_{sess_id}.npy.

        Parameters
        ----------
        data_dir : str
            Directory containing the data files.
        sess_id : int, optional
            Session ID to read a data file from. Defaults to 0.
        plot_dir : str, optional
            Directory to save the plot.
        """

        # Get simulated data and its ground truth
        data_path = f"{data_dir}/x_{sess_id}.npy"
        original_data = np.load(data_path)
        true_data = np.load(data_path.replace("x", "ground_truth/true_signal"))

        # Normalize data
        normalize = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        original_data = normalize(original_data)
        true_data = normalize(true_data)

        # Get data reconstructed from tokens
        data = Data(data_path)
        tokenized_data, token_weights = self.tokenize_data(data)
        fitted_data = self.reconstruct_data(tokenized_data)

        # Plot data signals and token weights
        n_channels = min(original_data.shape[1], 3)  # number of channels to plot
        start_idx, end_idx = 200, 500  # start and end indices to plot
        for n in range(n_channels):
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
            axes[0].plot(original_data[start_idx:end_idx, n], label="Original")
            axes[0].plot(true_data[start_idx:end_idx, n], label="True")
            axes[0].plot(fitted_data[start_idx:end_idx, n], label="Fitted")
            axes[0].set_title(f"Channel {n}: Data Signals")
            axes[0].legend()
            axes[1].plot(token_weights[0][start_idx:end_idx, n, :])
            axes[1].set_title(f"Token Weights")
            plt.tight_layout()

            if plot_dir is not None:
                os.makedirs(plot_dir, exist_ok=True)
                fig.savefig(f"{plot_dir}/fitted_signal_sess{sess_id}_ch{n}.png")
                plt.close(fig)


def load_tokenizer(model_dir: str) -> OSLTokenizer:
    """
    Load a tokenizer from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the tokenizer model.

    Returns
    -------
    tokenizer : OSLTokenizer
        The loaded tokenizer.
    """
    config = get_config(f"{model_dir}/config.yml")
    if config.model_config.name == "osl_tokenizer":
        tokenizer = OSLTokenizer(config)
    else:
        raise NotImplementedError(f"Model {config.model_config.name} not implemented.")
    tokenizer.load_weights(f"{model_dir}/weights").expect_partial()
    with open(f"{model_dir}/history.pkl", "rb") as f:
        tokenizer.history = pickle.load(f)
    return tokenizer
