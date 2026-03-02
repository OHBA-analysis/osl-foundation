from dataclasses import dataclass
from typing import List
import numpy as np

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

    # ---------- Pretrained model ---------- #
    pretrained_model_path: str = None
    pretrained_model_checkpoint: str = None
    pretrained_layers: List[str] = None

    # ---------- Tokenizer ---------- #
    tokenizer_path: str = None

    # ---------- Input parameters ---------- #
    embedding_dim: int = None
    n_tokens: int = None

    token_embedding_dim: int = None
    pos_embedding_dim: int = None
    pos_embedding_type: str = "absolute"
    channel_embedding_dim: int = None
    channel_dropout_rate: float = 0.0
    extra_labels: List[Label] = None

    # ---------- Decoder parameters ---------- #
    n_heads: int = 1
    model_dim: int = None
    n_patches_out: List[int] = None
    patch_length_out: List[int] = None
    n_patches_in: List[int] = None
    patch_length_in: List[int] = None
    unpatched_length_in: List[int] = None
    l_unpatched_b: List[int] = None
    l_patched_b: List[int] = None
    do_chan_attention: List[int] = None
    do_cross_attention: List[int] = None
    chan_attention_mask: List[str] = None

    full_channel_attention_dropout: float = None
    chan_attention_channel_dropout_rate: float = 0.0
    chan_attention_chandim: int = None
    feed_forward_dim: int = None
    feed_forward_activation: str = "relu"
    dropout: float = 0.0
    norm_type: str = "layer"
    n_groups: int = None

    # ---------- Loss parameters ---------- #
    loss_sequence_length: int = None
    lyapunov_beta: float = 0
    lyapunov_mu: float = 10
    lyapunov_collapse_weight: float = 0
    lyapunov_collapse_target_var: float = 1e-4
    lyapunov_collapse_target_mean: float = 1e-4
    lyapunov_dim: int = 16

    top_k: list = None

    def validate(self) -> None:
        super().validate()
        self._validate_tokenizer_path()
        self._validate_input_parameters()
        self._validate_decoder_parameters()
        self._validate_loss_parameters()

    def _validate_pretrained_model_parameters(self) -> None:
        if self.pretrained_model_path is not None:
            assert isinstance(
                self.pretrained_model_path, str
            ), "pretrained_model_path must be a string"
        if self.pretrained_model_checkpoint is not None:
            assert isinstance(
                self.pretrained_model_checkpoint, str
            ), "pretrained_model_checkpoint must be a string"
        if self.pretrained_layers is not None:
            assert isinstance(
                self.pretrained_layers, list
            ), "pretrained_layers must be a list"

        for layer in self.pretrained_layers:
            assert isinstance(layer, str), "pretrained_layers must be a list of strings"
            assert layer in ["input_embedding", "decoder", "prediction_head"], (
                "pretrained_layers must be one of "
                + "['input_embedding', 'decoder', 'prediction_head']"
            )

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
        assert (
            self.pos_embedding_type in VALID_POS_EMBEDDING_TYPES
        ), f"pos_embedding_type must be one of {VALID_POS_EMBEDDING_TYPES}"
        assert (
            0.0 <= self.channel_dropout_rate < 1.0
        ), "channel_dropout_rate must be in [0, 1)."
        self.extra_labels = self.extra_labels or []

    def _validate_decoder_parameters(self) -> None:

        assert self.model_dim is not None, "model_dim must be set"
        assert len(self.n_patches_in) > 0, "n_patches_in must be a non-empty list (where length is the number of layers)"
        assert len(self.patch_length_in) > 0, "patch_length_in must be a non-empty list (where length is the number of layers)"
        assert len(self.n_patches_out) > 0, "n_patches_out must be a non-empty list (where length is the number of layers)"
        assert len(self.patch_length_out) > 0, "patch_length_out must be a non-empty list (where length is the number of layers)"        
        assert len(self.unpatched_length_in) > 0, "unpatched_length_in must be a non-empty list (where length is the number of layers)"
        assert len(self.l_unpatched_b) > 0, "l_unpatched_b must be a non-empty list (where length is the number of layers)"
        assert len(self.l_patched_b) > 0, "l_patched_b must be a non-empty list (where length is the number of layers)"
        
        assert self.feed_forward_dim is not None, "feed_forward_dim must be set"

        assert self.n_heads > 0, "n_heads must be greater than 0"
        assert self.feed_forward_dim > 0, "feed_forward_dim must be greater than 0"
        assert (
            0.0 <= self.chan_attention_channel_dropout_rate < 1.0
        ), "chan_attention_channel_dropout_rate must be in [0, 1)."
        assert self.n_groups is None or self.n_groups > 0

        assert (
            self.model_dim % self.n_heads == 0
        ), "model_dim must be divisible by n_heads"
        
        if self.n_groups is not None:
            assert (
                self.model_dim % self.n_groups == 0
            ), "model_dim must be divisible by n_groups"

        print(f"n_patches_in: {len(self.n_patches_in)}")
        print(f"patch_length_in: {len(self.patch_length_in)}")
        print(f"n_patches_out: {len(self.n_patches_out)}")
        print(f"patch_length_out: {len(self.patch_length_out)}")
        print(f"unpatched_length_in: {len(self.unpatched_length_in)}")
        print(f"l_unpatched_b: {len(self.l_unpatched_b)}")
        print(f"l_patched_b: {len(self.l_patched_b)}")
        print(f"do_chan_attention: {len(self.do_chan_attention)}")
        print(f"do_cross_attention: {len(self.do_cross_attention)}")
        print(f"chan_attention_mask: {len(self.chan_attention_mask)}")
        assert (
            len(self.n_patches_in) == len(self.patch_length_in) == len(self.n_patches_out) == len(self.patch_length_out) == len(self.unpatched_length_in) == len(self.l_unpatched_b) == len(self.l_patched_b) == len(self.do_chan_attention) == len(self.do_cross_attention) == len(self.chan_attention_mask)
        ), "n_patches_in, patch_length_in, n_patches_out, patch_length_out, unpatched_length_in, l_unpatched_b, l_patched_b, do_chan_attention, do_cross_attention, chan_attention_mask must have the same length (i.e. the number of layers)"

        for i in range(len(self.n_patches_in)):
            assert self.n_patches_in[i] > 0, "n_patches_in must be greater than 0"
            assert self.patch_length_in[i] > 0, "patch_length_in must be greater than 0"

            assert (
                    self.n_patches_out[i] * self.patch_length_out[i]
                    <= self.n_patches_in[i] * self.patch_length_in[i]
                ), f"Output sequence length of layer {i} must be less or equal to input sequence length of layer {i}"

            assert (
                self.unpatched_length_in[i] <= self.n_patches_in[i] * self.patch_length_in[i]
            ), f"unpatched_length_in of layer {i} must be less or equal to input sequence length of layer {i}"

            assert (
                self.unpatched_length_in[i] <= self.n_patches_in[i] * self.patch_length_in[i]
            ), f"unpatched_length_in of layer {i} must be less or equal to input sequence length of layer {i}"

            if i > 0:
                assert (
                    self.n_patches_out[i - 1] * self.patch_length_out[i - 1]== self.n_patches_in[i] * self.patch_length_in[i]
                ), f"""Output sequence length of layer {i - 1} does not match input sequence length of layer {i}
                Layer {i-1}, time att seq_len: {self.n_patches_in[i-1]*self.patch_length_in[i-1]} -> {self.n_patches_out[i-1]*self.patch_length_out[i-1]}         
                Layer {i}, time att seq_len: {self.n_patches_in[i]*self.patch_length_in[i]} -> {self.n_patches_out[i]*self.patch_length_out[i]}"""
                            
    def _validate_loss_parameters(self) -> None:

        assert self.loss_sequence_length is not None, "loss_sequence_length must be set"
        assert (
            self.loss_sequence_length > 0
        ), "loss_sequence_length must be greater than 0"
        assert (
            self.lyapunov_beta >= 0
        ), "lyapunov_beta must be greater than or equal to 0"
        assert (
            self.lyapunov_mu >= 0
        ), "lyapunov_mu must be greater than or equal to 0"
        assert (
            self.lyapunov_collapse_weight >= 0
        ), "lyapunov_collapse_weight must be greater than or equal to 0"
        assert (
            self.lyapunov_collapse_target_var >= 0
        ), "lyapunov_collapse_target_var must be greater than or equal to 0"
        assert (
            self.lyapunov_collapse_target_mean >= 0
        ), "lyapunov_collapse_target_mean must be greater than or equal to 0"
        assert (
            self.lyapunov_dim > 0
        ), "lyapunov_dim must be greater than 0"

    def set_config(self, config: dict) -> None:
        self._set_pretrained_model_parameters(config.get("pretrained_model", {}))
        self._set_tokenizer_path(config)
        self._set_input_parameters(config.get("input_parameters", {}))
        self._set_loss_parameters(config.get("loss_parameters", {}))
        self._set_decoder_parameters(config.get("decoder_parameters", {}))

    def _set_pretrained_model_parameters(self, config: dict) -> None:
        self.pretrained_model_path = config.get("model_path", None)
        self.pretrained_model_checkpoint = config.get("checkpoint", None)
        if self.pretrained_model_path is not None:
            self.pretrained_layers = config.get(
                "pretrained_layers", ["input_embedding", "decoder", "prediction_head"]
            )
        else:
            self.pretrained_layers = config.get("pretrained_layers", None)

    def _set_tokenizer_path(self, config: dict) -> None:
        self.tokenizer_path = config.get("tokenizer_path", None)

    def _set_input_parameters(self, config: dict) -> None:
        self.embedding_dim = config.get("embedding_dim", 64)
        self.n_tokens = config.get("n_tokens", 128)
        self.token_embedding_dim = config.get("token_embedding_dim", None)
        self.pos_embedding_dim = config.get("pos_embedding_dim", None)
        self.pos_embedding_type = config.get("pos_embedding_type", "absolute")
        self.channel_embedding_dim = config.get("channel_embedding_dim", None)
        self.channel_dropout_rate = config.get("channel_dropout_rate", 0.0)
        self.extra_labels = [Label(**label) for label in config.get("extra_labels", [])]

    def _set_decoder_parameters(self, config: dict) -> None:
        self.n_heads = config.get("n_heads", 1)
        self.model_dim = config.get("model_dim", 64)
        self.n_patches_in = config.get("n_patches_in", None)
        self.patch_length_in = config.get("patch_length_in", None)
        self.unpatched_length_in = config.get("unpatched_length_in", None)
        self.l_unpatched_b = config.get("l_unpatched_b", None)
        self.l_patched_b = config.get("l_patched_b", None)
        self.do_chan_attention = config.get("do_chan_attention", None)
        self.full_channel_attention_dropout = config.get("full_channel_attention_dropout", 0)        
        self.chan_attention_channel_dropout_rate = config.get(
            "chan_attention_channel_dropout_rate", 0.0
        )
        self.do_cross_attention = config.get("do_cross_attention", None)
        self.chan_attention_mask = config.get("chan_attention_mask", None) 
        self.chan_attention_chandim = config.get("chan_attention_chandim", None) 
        self.feed_forward_dim = config.get("feed_forward_dim", 64)
        self.feed_forward_activation = config.get("feed_forward_activation", "relu")
        self.dropout = config.get("dropout", 0.0)
        self.norm_type = config.get("norm_type", "layer")
        self.n_groups = config.get("n_groups", None)

        self.n_patches_out = self.n_patches_in[1:]
        self.n_patches_out.append(self.loss_sequence_length)
        self.patch_length_out = self.patch_length_in[1:]
        self.patch_length_out.append(1)

        for i in range(len(self.l_unpatched_b)):
            if self.l_unpatched_b[i] == -1 or self.l_unpatched_b[i] == 'None' or self.l_unpatched_b[i] is None:
                self.l_unpatched_b[i] = self.n_patches_in[i] * self.patch_length_in[i]

            if self.l_patched_b[i] == -1 or self.l_patched_b[i] == 'None' or self.l_patched_b[i] is None:
                self.l_patched_b[i] = self.n_patches_in[i]

            if self.chan_attention_mask[i] == 'None' or self.chan_attention_mask[i] is None:
                self.chan_attention_mask[i] = None
            else:
                self.chan_attention_mask[i] = np.load(self.chan_attention_mask[i])

        if self.chan_attention_chandim == 'None' or self.chan_attention_chandim is None:
            self.chan_attention_chandim = None

        self.sequence_length = self.n_patches_in[0] * self.patch_length_in[0]

    def _set_loss_parameters(self, config: dict) -> None:
        self.loss_sequence_length = config.get(
            "loss_sequence_length", None    
        )
        self.lyapunov_beta = config.get("lyapunov_beta", 0)
        self.lyapunov_mu = config.get("lyapunov_mu", 10)
        self.lyapunov_collapse_weight = config.get("lyapunov_collapse_weight", 0)
        self.lyapunov_collapse_target_var = config.get(
            "lyapunov_collapse_target_var", 1e-4
        )
        self.lyapunov_collapse_target_mean = config.get(
            "lyapunov_collapse_target_mean", 1e-4
        )
        self.lyapunov_dim = config.get("lyapunov_dim", 16)

        self.top_k = config.get("top_k", None)
