from dataclasses import dataclass

from osl_foundation.config.base import (
    BaseModelConfig,
)


@dataclass
class OSLTokenizerModelConfig(BaseModelConfig):
    name: str = "osl_tokenizer"
    n_tokens: int = None
    token_dim: int = None
    token_kernel_padding: str = None
    token_use_bias: bool = None
    token_groups: int = None
    rnn_n_units: int = None
    rnn_type: str = None
    rnn_n_layers: int = None

    def validate(self) -> None:
        super().validate()
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.token_dim is not None, "token_dim must be set"
        assert self.token_kernel_padding is not None, "token_kernel_padding must be set"
        assert self.token_use_bias is not None, "token_use_bias must be set"
        assert self.token_groups is not None, "token_groups must be set"
        assert self.rnn_n_units is not None, "rnn_n_units must be set"
        assert self.rnn_type is not None, "rnn_type must be set"
        assert self.rnn_n_layers is not None, "rnn_n_layers must be set"

        assert self.rnn_type in ["gru", "lstm"], "rnn_type must be 'gru' or 'lstm'"
        assert self.rnn_n_layers > 0, "rnn_n_layers must be greater than 0"

    def set_config(self, config: dict) -> None:
        self.n_tokens = config.get("n_tokens", 16)
        self.token_dim = config.get("token_dim", 32)
        self.token_kernel_padding = config.get("token_kernel_padding", "same")
        self.token_use_bias = config.get("token_use_bias", True)
        self.token_groups = config.get("token_groups", 1)
        self.rnn_n_units = config.get("rnn_n_units", 64)
        self.rnn_type = config.get("rnn_type", "gru")
        self.rnn_n_layers = config.get("rnn_n_layers", 1)


@dataclass
class MuTransformTokenizerModelConfig:
    name: str = "mu_transform_tokenizer"
    n_tokens: int = None
    mu: int = None
    normalization: str = "max_abs"

    def validate(self) -> None:
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.mu is not None, "mu must be set"
        assert self.mu > 0, "mu must be greater than 0"
        assert self.n_tokens > 0, "n_tokens must be greater than 0"
        assert self.normalization in ["max_abs", "min_max"], \
        "normalization must be 'max_abs' or 'min_max'"

    def set_config(self, config: dict) -> None:
        self.n_tokens = config.get("n_tokens", 64)
        self.mu = config.get("mu", 255)
