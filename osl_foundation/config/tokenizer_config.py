from dataclasses import dataclass

from osl_foundation.config.base import (
    BaseModelConfig,
)


@dataclass
class OSLTokenizerModelConfig(BaseModelConfig):
    name: str = "osl_tokenizer"
    n_tokens: int = None
    token_dim: int = None
    rnn_n_units: int = None

    def validate(self):
        super().validate()
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.token_dim is not None, "token_dim must be set"
        assert self.rnn_n_units is not None, "rnn_n_units must be set"

    def set_n_tokens(self, n_tokens: int):
        self.n_tokens = n_tokens

    def set_token_dim(self, token_dim: int):
        self.token_dim = token_dim

    def set_rnn_n_units(self, rnn_n_units: int):
        self.rnn_n_units = rnn_n_units

    def get_config(self) -> dict:
        return (
            super()
            .get_config()
            .update(
                {
                    "n_tokens": self.n_tokens,
                    "token_dim": self.token_dim,
                    "rnn_n_units": self.rnn_n_units,
                }
            )
        )
