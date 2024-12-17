from dataclasses import dataclass
from typing import List

from osl_foundation.config.base import BaseModelConfig


@dataclass
class Label:
    """Class for extra labels to be used in the model."""

    name: str
    n_classes: int
    label_dim: int = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        assert self.name is not None, "name must be set"
        assert self.n_classes is not None, "n_classes must be set"

        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.label_dim is not None:
            assert self.label_dim > 0, "label_dim must be greater than 0"


@dataclass
class EphysGPTModelConfig(BaseModelConfig):
    name: str = "ephys_gpt"

    # ---------- Tokenizer ---------- #
    tokenizer_path: str = None

    # ---------- Input parameters ---------- #
    embedding_dim: int = None
    n_tokens: int = None

    token_embedding_dim: int = None
    pos_embedding_dim: int = None
    pos_embedding_type: str = "absolute"
    channel_embedding_dim: int = None
    extra_labels: List[Label] = None

    # ---------- Decoder parameters ---------- #
    n_layers: int = 1
    n_heads: int = 1
    model_dim: int = None
    latent_sequence_length = None
    n_patches: int = None
    patch_length: int = None
    unpatched_length: int = None
    channel_attention_dropout: float = 0.0
    within_channel_attention_dropout: float = 0.0
    feed_forward_dim: int = None
    feed_forward_activation: str = "relu"
    dropout: float = 0.0
    norm_type: str = "layer"
    n_groups: int = None

    # ---------- Loss parameters ---------- #
    loss_sequence_length: int = None
    top_k: list = None

    def validate(self) -> None:
        super().validate()
        self._validate_tokenizer_path()
        self._validate_input_parameters()
        self._validate_decoder_parameters()
        self._validate_loss_parameters()

    def _validate_tokenizer_path(self) -> None:
        if self.tokenizer_path is not None:
            assert isinstance(
                self.tokenizer_path, str
            ), "tokenizer_path must be a string"

    def _validate_input_parameters(self) -> None:
        assert self.embedding_dim is not None, "embedding_dim must be set"
        if self.n_tokens is None:
            assert (
                self.tokenizer_path is not None
            ), "n_tokens must be set if tokenizer_path is not set"
        else:
            assert self.n_tokens > 0, "n_tokens must be greater than 0"
        assert self.n_channels is not None, "n_channels must be set"
        VALID_POS_EMBEDDING_TYPES = ["absolute", "sinusoidal", "rope", "alibi"]
        assert self.pos_embedding_type in VALID_POS_EMBEDDING_TYPES, (
            f"pos_embedding_type must be one of {VALID_POS_EMBEDDING_TYPES}"
        )
        self.extra_labels = self.extra_labels or []

    def _validate_decoder_parameters(self) -> None:
        assert self.model_dim is not None, "model_dim must be set"
        assert (
            self.latent_sequence_length is not None
        ), "latent_sequence_length must be set"
        assert self.n_patches is not None, "n_patches must be set"
        assert self.patch_length is not None, "patch_length must be set"
        assert self.unpatched_length is not None, "unpatched_length must be set"
        assert self.feed_forward_dim is not None, "feed_forward_dim must be set"

        assert self.n_layers > 0, "n_layers must be greater than 0"
        assert self.n_heads > 0, "n_heads must be greater than 0"
        assert (
            self.latent_sequence_length > 0
        ), "latent_sequence_length must be greater than 0"
        assert self.n_patches > 0, "n_patches must be greater than 0"
        assert self.patch_length > 0, "patch_length must be greater than 0"
        assert self.unpatched_length >= 0, "unpatched_length must be non-negative"
        assert self.feed_forward_dim > 0, "feed_forward_dim must be greater than 0"
        assert self.n_groups is None or self.n_groups > 0

        assert (
            self.model_dim % self.n_heads == 0
        ), "model_dim must be divisible by n_heads"
        assert (
            self.sequence_length == self.n_patches * self.patch_length
        ), "sequence_length must be equal to n_patches * patch_length"
        if self.n_groups is not None:
            assert (
                self.model_dim % self.n_groups == 0
            ), "model_dim must be divisible by n_groups"

    def _validate_loss_parameters(self) -> None:
        assert self.loss_sequence_length is not None, "loss_sequence_length must be set"
        assert (
            self.loss_sequence_length > 0
        ), "loss_sequence_length must be greater than 0"
        assert (
            self.loss_sequence_length <= self.latent_sequence_length
        ), "loss_sequence_length must be less or equal to latent_sequence_length"

    def set_config(self, config: dict) -> None:
        self._set_tokenizer_path(config)
        self._set_input_parameters(config.get("input_parameters", {}))
        self._set_decoder_parameters(config.get("decoder_parameters", {}))
        self._set_loss_parameters(config.get("loss_parameters", {}))

    def _set_tokenizer_path(self, config: dict) -> None:
        self.tokenizer_path = config.get("tokenizer_path", None)

    def _set_input_parameters(self, config: dict) -> None:
        self.embedding_dim = config.get("embedding_dim", 64)
        self.n_tokens = config.get("n_tokens", 128)
        self.token_embedding_dim = config.get("token_embedding_dim", None)
        self.pos_embedding_dim = config.get("pos_embedding_dim", None)
        self.pos_embedding_type = config.get(
            "pos_embedding_type", "absolute"
        )
        self.channel_embedding_dim = config.get("channel_embedding_dim", None)
        self.extra_labels = [Label(**label) for label in config.get("extra_labels", [])]

    def _set_decoder_parameters(self, config: dict) -> None:
        self.n_layers = config.get("n_layers", 1)
        self.n_heads = config.get("n_heads", 1)
        self.model_dim = config.get("model_dim", 64)
        self.latent_sequence_length = config.get(
            "latent_sequence_length", self.sequence_length // 2
        )
        self.patch_length = config.get("patch_length", 4)
        self.n_patches = config.get(
            "n_patches", self.sequence_length // self.patch_length
        )
        self.unpatched_length = config.get("unpatched_length", self.patch_length)
        self.channel_attention_dropout = config.get("channel_attention_dropout", 0.0)
        self.within_channel_attention_dropout = config.get(
            "within_channel_attention_dropout", 0.0
        )
        self.feed_forward_dim = config.get("feed_forward_dim", 64)
        self.feed_forward_activation = config.get("feed_forward_activation", "relu")
        self.dropout = config.get("dropout", 0.0)
        self.norm_type = config.get("norm_type", "layer")
        self.n_groups = config.get("n_groups", None)

    def _set_loss_parameters(self, config: dict) -> None:
        self.loss_sequence_length = config.get(
            "loss_sequence_length", self.sequence_length // 4
        )
        self.top_k = config.get("top_k", None)
