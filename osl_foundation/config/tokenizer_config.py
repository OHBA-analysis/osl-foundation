from dataclasses import dataclass

from osl_foundation.config.base import BaseModelConfig, BaseTrainingConfig, BaseConfig


@dataclass
class OSLTokenizerModelConfig(BaseModelConfig):
    name: str = "osl_tokenizer"
    n_tokens: int = None
    token_dim: int = None
    rnn_u_units: int = None

    def validate(self):
        super().validate()
        assert self.n_tokens is not None, "n_tokens must be set"
        assert self.token_dim is not None, "token_dim must be set"
        assert self.rnn_u_units is not None, "rnn_u_units must be set"

    def set_n_tokens(self, n_tokens: int):
        self.n_tokens = n_tokens

    def set_token_dim(self, token_dim: int):
        self.token_dim = token_dim

    def set_rnn_u_units(self, rnn_u_units: int):
        self.rnn_u_units = rnn_u_units

    def get_config(self) -> dict:
        return (
            super()
            .get_config()
            .update(
                {
                    "n_tokens": self.n_tokens,
                    "token_dim": self.token_dim,
                    "rnn_u_units": self.rnn_u_units,
                }
            )
        )


@dataclass
class OSLTokenizerConfig(BaseConfig):

    def set_model_config(self, **kwargs):
        self.model_config = OSLTokenizerModelConfig(**kwargs)

    def set_training_config(self, **kwargs):
        self.training_config = BaseTrainingConfig(**kwargs)
