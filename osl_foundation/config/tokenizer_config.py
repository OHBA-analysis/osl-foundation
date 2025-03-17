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
    rnn_n_units: int = None
    rnn_type: str = None
    rnn_n_layers: int = None

    def validate(self) -> None:
        super().validate()
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.token_dim is not None, "token_dim must be set"
        assert self.token_kernel_padding is not None, "token_kernel_padding must be set"
        assert self.rnn_n_units is not None, "rnn_n_units must be set"
        assert self.rnn_type is not None, "rnn_type must be set"
        assert self.rnn_n_layers is not None, "rnn_n_layers must be set"

        assert self.rnn_type in ["gru", "lstm"], "rnn_type must be 'gru' or 'lstm'"
        assert self.rnn_n_layers > 0, "rnn_n_layers must be greater than 0"

    def set_config(self, config: dict) -> None:
        self.n_tokens = config.get("n_tokens", 16)
        self.token_dim = config.get("token_dim", 32)
        self.token_kernel_padding = config.get("token_kernel_padding", "same")
        self.rnn_n_units = config.get("rnn_n_units", 64)
        self.rnn_type = config.get("rnn_type", "gru")
        self.rnn_n_layers = config.get("rnn_n_layers", 1)
