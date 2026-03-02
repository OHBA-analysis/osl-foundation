from typing import Dict, List, Union
import logging

import keras
import numpy as np
import tensorflow as tf
from tqdm.auto import trange

from osl_foundation.utils.sampling import sample_from_logits

_logger = logging.getLogger("osl-foundation")


class EphysGPTSamplingMixin:
    def _spectral_peakiness(
        self,
        tokens: tf.Tensor,
        sampling_rate: float,
        min_freq: float,
        max_freq: float,
        eps: float = 1e-6,
    ) -> tf.Tensor:
        data_window = self.reconstruct_data_layer(tokens)
        data_window = data_window - tf.reduce_mean(data_window, axis=1, keepdims=True)
        data_window = tf.transpose(data_window, (0, 2, 1))
        spectrum = tf.signal.rfft(tf.cast(data_window, tf.float32))
        power = tf.abs(spectrum) ** 2
        power = tf.reduce_mean(power, axis=1)

        n_freq = tf.shape(power)[1]
        freqs = tf.linspace(
            tf.cast(0.0, tf.float32),
            tf.cast(sampling_rate / 2.0, tf.float32),
            n_freq,
        )
        min_freq = tf.cast(min_freq, tf.float32)
        max_freq = tf.cast(max_freq, tf.float32)
        mask = tf.logical_and(freqs >= min_freq, freqs <= max_freq)
        mask_f = tf.cast(mask, power.dtype)
        masked_power = power * mask_f
        denom = tf.reduce_sum(mask_f)
        denom = tf.maximum(denom, tf.cast(1.0, power.dtype))
        mean_power = tf.reduce_sum(masked_power, axis=1) / (denom + eps)
        peak_power = tf.reduce_max(masked_power, axis=1)
        return peak_power / (mean_power + eps)

    def _embed_tokens(self, token_labels_or_logits):
        # token_labels_or_logits.shape =
        # (batch_size, sequence_length, n_channels)
        # or (batch_size, sequence_length, n_channels, n_tokens)

        # detect if token_labels_or_logits are labels or logits
        if token_labels_or_logits.shape[-1] == self.config.model_config.n_tokens:
            logits = token_labels_or_logits
        else:
            # tokens.shape = (batch_size, sequence_length, n_channels)
            # One-hot encode the tokens
            logits = tf.one_hot(
                token_labels_or_logits,
                depth=self.config.model_config.n_tokens,
                dtype=tf.float32,
            )

        if False:
            embeddings = self.token_embedding_output_layer(
                self.token_embedding_layer(x, training=training_1), training=training_1
            )
        else:
            input_embedding_layer = self.model.get_layer("input_embedding")
            embedding_matrix = (
                input_embedding_layer.token_embedding_layer.embeddings
            )  # shape: (n_tokens, embedding_dim)
            embeddings = tf.linalg.matmul(logits, embedding_matrix)
            embeddings = input_embedding_layer.token_embedding_output_layer(embeddings)
        # embeddings.shape = (batch_size, sequence_length, n_channels, embedding_dim)

        return embeddings

    @tf.function
    def one_step_sample(
        self,
        inputs: list,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        lyapunov_margin: float = 0.0,
        lyapunov_rollout_horizon: int = 1,
        lyapunov_adaptive_rollout: bool = False,
        lyapunov_adaptive_band: float = 0.0,
        spectral_penalty_weight: float = 0.0,
        spectral_penalty_active: tf.Tensor = None,
        spectral_window_samples: int = None,
        spectral_sampling_rate: float = None,
        spectral_min_freq: float = 0.5,
        spectral_max_freq: float = None,
        spectral_peakiness_threshold: float = 5.0,
        spectral_penalty_normalize: bool = True,
        spectral_penalty_cap: float = 2.0,
        return_diagnostics: bool = False,
    ) -> tf.Tensor:
        """
        Generate a single token using the model.

        Parameters
        ----------
        inputs : list
            List of inputs to the model.
        top_p : float, optional
            Top p proportion of values to keep.
        top_k : int, optional
            Top k number of values to keep.
        temperature : float, optional
            Temperature for sampling from the logits.
        lyapunov_margin : float, optional
            Margin for the lyapunov function.
        lyapunov_rollout_horizon : int, optional
            Reserved for rollout-based Lyapunov scoring.
        lyapunov_adaptive_rollout : bool, optional
            Reserved for adaptive rollout scoring.
        lyapunov_adaptive_band : float, optional
            Reserved for adaptive rollout scoring band.
        spectral_penalty_weight : float, optional
            Weight for the dominant-band spectral penalty.
        spectral_penalty_active : tf.Tensor, optional
            Boolean or 0/1 mask per batch indicating whether to apply the penalty.
        spectral_window_samples : int, optional
            Window length (samples) for spectral scoring.
        spectral_sampling_rate : float, optional
            Sampling rate in Hz used to compute frequency bins.
        spectral_min_freq : float, optional
            Minimum frequency for spectral scoring.
        spectral_max_freq : float, optional
            Maximum frequency for spectral scoring.
        spectral_peakiness_threshold : float, optional
            Threshold for peakiness ratio before penalty is applied.
        spectral_penalty_normalize : bool, optional
            If True, normalize peakiness by the threshold before applying the penalty.
        spectral_penalty_cap : float, optional
            Optional upper bound on the penalty term (in normalized units).
        return_diagnostics : bool, optional
            If True, return attempts/validity diagnostics per batch element.

        Returns
        -------
        sampled_tokens : tf.Tensor
            The sampled tokens. Shape: (*batch_dims).
        lyapunov_fn : tf.Tensor
            The lyapunov function values. Shape: (*batch_dims).
        """

        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        y_pred_logits = self.sampling_model(inputs, training=False)
        # y_pred_logits.shape = (batch_size, sequence_length_4loss, n_channels, n_refactored_tokens)

        margin = tf.cast(lyapunov_margin, tf.float32)
        if margin.shape.rank == 0:
            margin = tf.reshape(margin, [1, 1])
        elif margin.shape.rank == 1:
            margin = tf.reshape(margin, [-1, 1])

        # Get the predicted logits at last time step:
        logits = y_pred_logits[:, -1:] / temperature
        # shape: (batch_size, 1, n_channels, n_refactored_tokens)

        # inputs["data"] is the tokenized data with
        # shape = (batch_size, sequence_length + 1, n_channels)
        # The last time step in the 2nd dim is the one to be predicted/generated
        # inp_dat = inputs["data"][:, -self.config.model_config.loss_sequence_length_recon_buffer-1:-1, :]
        # shape: (batch_size, buffer_sequence_length, n_channels)
        # recon_dat = self.reconstruct_data_layer(inp_dat) # shape: (batch_size, buffer_sequence_length, n_channels)
        # recon_dat = recon_dat[:, -1:, :] # shape: (batch_size, 1, n_channels)
        # V_dat = self._compute_V(recon_dat) # shape: (batch_size, 1)

        # Get the tokens at the tpt before the one being generated
        tokenised_inp_dat = inputs["data"][:, -2:-1, :]
        # shape: (batch_size, 1, n_channels)
        inp_dat_embeddings = self._embed_tokens(tokenised_inp_dat)
        # shape: (batch_size, 1, n_channels, embedding_dim)

        # Precompute V_dat from tokenised_inp_dat to speed up sampling
        V_dat = self.lyapunov_loss_layer._compute_V(inp_dat_embeddings)
        # shape: (batch_size, 1)
        adaptive_band = tf.cast(lyapunov_adaptive_band, tf.float32)
        spectral_enabled = (
            spectral_penalty_weight is not None
            and spectral_penalty_weight > 0.0
            and spectral_penalty_active is not None
            and spectral_window_samples is not None
            and spectral_sampling_rate is not None
        )
        if spectral_max_freq is None and spectral_sampling_rate is not None:
            spectral_max_freq = spectral_sampling_rate / 2.0

        if not spectral_enabled:
            def _score_candidate_simple(tokenised_sample):
                tokenised_sample_embeddings = self._embed_tokens(tokenised_sample)
                V_curr = self.lyapunov_loss_layer._compute_V(
                    tokenised_sample_embeddings
                )
                score = tf.nn.relu(V_curr - V_dat)
                first_step_score = score

                if lyapunov_rollout_horizon > 1:
                    rollout_inputs = {k: v for k, v in inputs.items()}
                    rollout_data = tf.concat(
                        [inputs["data"][:, 1:, :], tokenised_sample], axis=1
                    )
                    threshold = margin - adaptive_band
                    active_mask = score > threshold

                    for _ in range(1, lyapunov_rollout_horizon):
                        rollout_inputs["data"] = rollout_data
                        rollout_logits = self.sampling_model(
                            rollout_inputs, training=False
                        )
                        rollout_logits = rollout_logits[:, -1:] / temperature
                        token_next = sample_from_logits(rollout_logits, top_p, top_k)

                        token_next_embeddings = self._embed_tokens(token_next)
                        V_next = self.lyapunov_loss_layer._compute_V(
                            token_next_embeddings
                        )
                        step_score = tf.nn.relu(V_next - V_curr)

                        if lyapunov_adaptive_rollout:
                            score = tf.where(
                                active_mask, tf.maximum(score, step_score), score
                            )
                            active_mask = tf.logical_and(active_mask, score > threshold)
                        else:
                            score = tf.maximum(score, step_score)

                        V_curr = V_next
                        rollout_data = tf.concat(
                            [rollout_data[:, 1:, :], token_next], axis=1
                        )

                return score, first_step_score

            tokenised_sample = sample_from_logits(logits, top_p, top_k)
            lyapunov_fn, first_step_lyapunov = _score_candidate_simple(
                tokenised_sample
            )

            best_sample = tokenised_sample
            best_lyapunov_fn = lyapunov_fn
            unresolved_mask = best_lyapunov_fn > margin
            attempts_used = tf.ones_like(tf.cast(unresolved_mask, tf.int32))

            for attempt in range(2, 11):
                tokenised_candidate = sample_from_logits(logits, top_p, top_k)
                candidate_lyapunov_fn, _ = _score_candidate_simple(
                    tokenised_candidate
                )

                newly_accepted = tf.logical_and(
                    unresolved_mask, candidate_lyapunov_fn <= margin
                )
                attempts_used = tf.where(
                    newly_accepted,
                    tf.cast(attempt, tf.int32) * tf.ones_like(attempts_used),
                    attempts_used,
                )

                better_mask = tf.logical_and(
                    unresolved_mask, candidate_lyapunov_fn < best_lyapunov_fn
                )
                best_lyapunov_fn = tf.where(
                    better_mask, candidate_lyapunov_fn, best_lyapunov_fn
                )
                best_sample = tf.where(
                    tf.expand_dims(better_mask, axis=-1),
                    tokenised_candidate,
                    best_sample,
                )
                unresolved_mask = best_lyapunov_fn > margin

            tokenised_sample = best_sample
            lyapunov_fn = best_lyapunov_fn
        else:
            def _score_candidate(tokenised_sample):
                tokenised_sample_embeddings = self._embed_tokens(tokenised_sample)
                V_curr = self.lyapunov_loss_layer._compute_V(
                    tokenised_sample_embeddings
                )
                score = tf.nn.relu(V_curr - V_dat)
                first_step_score = score

                if lyapunov_rollout_horizon > 1:
                    rollout_inputs = {k: v for k, v in inputs.items()}
                    rollout_data = tf.concat(
                        [inputs["data"][:, 1:, :], tokenised_sample], axis=1
                    )
                    threshold = margin - adaptive_band
                    active_mask = score > threshold

                    for _ in range(1, lyapunov_rollout_horizon):
                        rollout_inputs["data"] = rollout_data
                        rollout_logits = self.sampling_model(
                            rollout_inputs, training=False
                        )
                        rollout_logits = rollout_logits[:, -1:] / temperature
                        token_next = sample_from_logits(rollout_logits, top_p, top_k)

                        token_next_embeddings = self._embed_tokens(token_next)
                        V_next = self.lyapunov_loss_layer._compute_V(
                            token_next_embeddings
                        )
                        step_score = tf.nn.relu(V_next - V_curr)

                        if lyapunov_adaptive_rollout:
                            score = tf.where(
                                active_mask, tf.maximum(score, step_score), score
                            )
                            active_mask = tf.logical_and(active_mask, score > threshold)
                        else:
                            score = tf.maximum(score, step_score)

                        V_curr = V_next
                        rollout_data = tf.concat(
                            [rollout_data[:, 1:, :], token_next], axis=1
                        )

                context_tokens = inputs["data"][:, :-1, :]
                context_len = tf.shape(context_tokens)[1]
                prefix_len = tf.minimum(
                    context_len, tf.cast(spectral_window_samples - 1, context_len.dtype)
                )
                prefix_tokens = context_tokens[:, context_len - prefix_len :, :]
                window_tokens = tf.concat([prefix_tokens, tokenised_sample], axis=1)
                peakiness = self._spectral_peakiness(
                    window_tokens,
                    sampling_rate=spectral_sampling_rate,
                    min_freq=spectral_min_freq,
                    max_freq=spectral_max_freq,
                )
                if spectral_penalty_normalize:
                    penalty = tf.nn.relu(
                        peakiness
                        / tf.cast(spectral_peakiness_threshold, peakiness.dtype)
                        - 1.0
                    )
                else:
                    penalty = tf.nn.relu(
                        peakiness
                        - tf.cast(spectral_peakiness_threshold, peakiness.dtype)
                    )
                if spectral_penalty_cap is not None:
                    penalty = tf.minimum(
                        penalty, tf.cast(spectral_penalty_cap, penalty.dtype)
                    )
                penalty = penalty * tf.cast(
                    spectral_penalty_active, penalty.dtype
                ) * tf.cast(spectral_penalty_weight, penalty.dtype)
                combined_score = score + tf.expand_dims(penalty, axis=1)

                return score, combined_score, first_step_score

            tokenised_sample = sample_from_logits(logits, top_p, top_k)
            lyapunov_fn, combined_score, first_step_lyapunov = _score_candidate(
                tokenised_sample
            )

            best_sample = tokenised_sample
            best_lyapunov_fn = lyapunov_fn
            best_combined_score = combined_score
            best_valid = best_lyapunov_fn <= margin
            unresolved_mask = tf.logical_not(best_valid)
            attempts_used = tf.ones_like(tf.cast(unresolved_mask, tf.int32))

            for attempt in range(2, 11):
                tokenised_candidate = sample_from_logits(logits, top_p, top_k)
                candidate_lyapunov_fn, candidate_combined, _ = _score_candidate(
                    tokenised_candidate
                )
                candidate_valid = candidate_lyapunov_fn <= margin

                newly_accepted = tf.logical_and(
                    unresolved_mask, candidate_valid
                )
                attempts_used = tf.where(
                    newly_accepted,
                    tf.cast(attempt, tf.int32) * tf.ones_like(attempts_used),
                    attempts_used,
                )

                valid_beats_invalid = tf.logical_and(
                    tf.logical_not(best_valid), candidate_valid
                )
                better_valid = tf.logical_and(best_valid, candidate_valid)
                better_valid = tf.logical_and(
                    better_valid, candidate_combined < best_combined_score
                )
                better_invalid = tf.logical_and(
                    tf.logical_not(best_valid), tf.logical_not(candidate_valid)
                )
                better_invalid = tf.logical_and(
                    better_invalid, candidate_combined < best_combined_score
                )
                override_mask = tf.logical_or(
                    valid_beats_invalid, tf.logical_or(better_valid, better_invalid)
                )

                best_lyapunov_fn = tf.where(
                    override_mask, candidate_lyapunov_fn, best_lyapunov_fn
                )
                best_combined_score = tf.where(
                    override_mask, candidate_combined, best_combined_score
                )
                best_sample = tf.where(
                    tf.expand_dims(override_mask, axis=-1),
                    tokenised_candidate,
                    best_sample,
                )
                best_valid = tf.where(override_mask, candidate_valid, best_valid)
                unresolved_mask = tf.logical_not(best_valid)

            tokenised_sample = best_sample
            lyapunov_fn = best_lyapunov_fn

        # remove time dimension
        tokenised_sample = tf.squeeze(tokenised_sample, axis=1)
        # shape: (batch_size, n_channels)

        lyapunov_fn = tf.squeeze(lyapunov_fn, axis=1)  # shape: (batch_size,)
        accepted_mask = lyapunov_fn <= tf.squeeze(margin, axis=1)
        # shape: (batch_size,)
        attempts_used = tf.where(
            accepted_mask,
            tf.squeeze(attempts_used, axis=1),
            10 * tf.ones_like(tf.squeeze(attempts_used, axis=1)),
        )
        initial_invalid_mask = tf.squeeze(first_step_lyapunov > margin, axis=1)
        rejection_count = attempts_used - 1

        if return_diagnostics:
            return (
                tokenised_sample,
                lyapunov_fn,
                attempts_used,
                accepted_mask,
                initial_invalid_mask,
                rejection_count,
            )
        return tokenised_sample, lyapunov_fn

    def generate_tokens(
        self,
        n_samples: int,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        lyapunov_margin: float = 0.00001,
        lyapunov_adaptive_margin: bool = False,
        lyapunov_margin_quantile: float = 0.9,
        lyapunov_margin_window: int = 256,
        lyapunov_margin_min: float = 0.0,
        lyapunov_margin_max: float = np.inf,
        lyapunov_rollout_horizon: int = 1,
        lyapunov_adaptive_rollout: bool = False,
        lyapunov_adaptive_band: float = 0.0,
        lyapunov_diagnostics: bool = False,
        batch_size: int = 1,
        prompt: np.ndarray = None,
        extra_labels: Dict[str, np.ndarray] = None,
        extra_channels: Dict[str, np.ndarray] = None,
        spectral_penalty_weight: float = 0.0,
        sampling_rate: float = None,
        burst_window_s: float = None,
        max_burst_s: float = None,
        spectral_peakiness_threshold: float = 5.0,
        spectral_min_freq: float = 0.5,
        spectral_max_freq: float = None,
        spectral_penalty_normalize: bool = True,
        spectral_penalty_cap: float = 2.0,
        spectral_penalty_auto_weight: bool = False,
        spectral_penalty_target_frac: float = 0.2,
        spectral_penalty_calibrate_steps: int = 0,
    ) -> np.ndarray:
        """
        Generate tokens using the model.

        Parameters
        ----------
        n_samples : int
            Number of samples per sequence to generate.
        top_p : float, optional
            Top p proportion of values to keep.
        top_k : int, optional
            Top k number of values to keep.
        temperature : float, optional
            Temperature for sampling from the logits.
        lyapunov_adaptive_margin : bool, optional
            If True, adapt margin per session from recent Lyapunov values.
        lyapunov_margin_quantile : float, optional
            Quantile for adaptive margin updates.
        lyapunov_margin_window : int, optional
            Rolling window length (in generated steps) for adaptive margin updates.
        lyapunov_margin_min : float, optional
            Lower bound for adaptive margins.
        lyapunov_margin_max : float, optional
            Upper bound for adaptive margins.
        lyapunov_rollout_horizon : int, optional
            Number of future imagined steps to include in Lyapunov candidate scoring.
        lyapunov_adaptive_rollout : bool, optional
            If True, only keep rollout accumulation active for near-threshold samples.
        lyapunov_adaptive_band : float, optional
            Band around margin used by adaptive rollout.
        lyapunov_diagnostics : bool, optional
            If True, prints summary diagnostics for Lyapunov validity.
        batch_size : int, optional
            Batch size for generating the samples.
            If None, batch size in the config is used.
        prompt : str or np.ndarray, optional
            Prompt to start the generation.
            If None, a random sequence is used.
            If np.ndarray, the shape must be (sequence_length, n_channels)
            or (batch_size, sequence_length, n_channels).
        extra_labels : Dict[np.ndarray], optional
            Dictionary of extra labels.
            Keys are the names of the extra labels. Each value must have shape (batch_size,).
        extra_channels : Dict[str, np.ndarray], optional
            Dictionary of extra channels. Keys are the names of the extra channels.
            Each value must have shape (batch_size, > sequence_length + n_samples).
        spectral_penalty_weight : float, optional
            Weight for the dominant-band spectral penalty.
        sampling_rate : float, optional
            Sampling rate in Hz used for spectral scoring.
        burst_window_s : float, optional
            Window length in seconds used to score spectral peakiness.
        max_burst_s : float, optional
            Maximum allowed burst duration in seconds before penalty activates.
        spectral_peakiness_threshold : float, optional
            Threshold for peakiness ratio before penalty is applied.
        spectral_min_freq : float, optional
            Minimum frequency for spectral scoring.
        spectral_max_freq : float, optional
            Maximum frequency for spectral scoring.
        spectral_penalty_normalize : bool, optional
            If True, normalize peakiness by the threshold before applying the penalty.
        spectral_penalty_cap : float, optional
            Optional upper bound on the penalty term (in normalized units).
        spectral_penalty_auto_weight : bool, optional
            If True, derive spectral_penalty_weight from Lyapunov scale and cap.
        spectral_penalty_target_frac : float, optional
            Target fraction of Lyapunov scale used when auto-calibrating the weight.
        spectral_penalty_calibrate_steps : int, optional
            Number of initial steps to estimate Lyapunov scale before setting the weight.

        Returns
        -------
        generated_tokens : np.ndarray
            Generated tokens. Shape is (batch_size, n_samples, n_channels).

        FAQ
        ---
        What is relationship between n_samples and sequence_length?
        1. The model generates tokens in chunks of size sequence_length.
        2. The total number of tokens to generate is n_samples.
        3. Therefore, the model will generate ceil(n_samples / sequence_length) chunks.
        4. The final output will be trimmed to n_samples.
        5. For example, if n_samples=1500 and sequence_length=512,
           the model will generate 3 chunks of 512 tokens each (total 1536 tokens),
           and then trim the output to the first 1500 tokens.

        Does this mean that there will be discontinuities in the generated sequence?
            No. The model uses the previously generated tokens as context
            for generating the next tokens, so it can maintain continuity across chunks.

        What is relationship between batch_size and n_samples?
        1. batch_size determines how many sequences are generated in parallel (e.g.
           each batch might correspond to different subjects/sessions).
        2. n_samples determines the length of each generated sequence.
        3. The output shape is (batch_size, n_samples, n_channels).
        4. For example, if batch_size=4 and n_samples=1500,
           the model will generate 4 sequences, each of length 1500 tokens.

        """

        batch_size = batch_size or self.config.training_config.batch_size

        n_channels = self.config.model_config.n_channels

        sequence_length = self.config.model_config.sequence_length

        if lyapunov_rollout_horizon < 1:
            raise ValueError("lyapunov_rollout_horizon must be >= 1.")
        if lyapunov_adaptive_band < 0.0:
            raise ValueError("lyapunov_adaptive_band must be >= 0.")
        if lyapunov_adaptive_margin:
            if not (0.0 < lyapunov_margin_quantile < 1.0):
                raise ValueError("lyapunov_margin_quantile must be in (0, 1).")
            if lyapunov_margin_window < 1:
                raise ValueError("lyapunov_margin_window must be >= 1.")
            if lyapunov_margin_min > lyapunov_margin_max:
                raise ValueError("lyapunov_margin_min must be <= lyapunov_margin_max.")

        auto_weight_pending = (
            spectral_penalty_auto_weight and spectral_penalty_calibrate_steps > 0
        )
        if spectral_penalty_auto_weight and not auto_weight_pending:
            if spectral_penalty_target_frac <= 0:
                raise ValueError("spectral_penalty_target_frac must be > 0.")
            if spectral_penalty_normalize:
                if spectral_penalty_cap is None:
                    raise ValueError(
                        "spectral_penalty_cap must be set when auto-calibrating with normalization."
                    )
                denom = float(spectral_penalty_cap)
            else:
                denom = float(spectral_peakiness_threshold)
            denom = max(denom, 1e-6)
            spectral_penalty_weight = (
                float(lyapunov_margin) * float(spectral_penalty_target_frac) / denom
            )
            _logger.info(
                "Auto spectral_penalty_weight=%.6g (target_frac=%.3f, lyapunov_margin=%.6g, denom=%.6g).",
                spectral_penalty_weight,
                float(spectral_penalty_target_frac),
                float(lyapunov_margin),
                denom,
            )

        spectral_penalty_intended = (
            (spectral_penalty_weight is not None and spectral_penalty_weight > 0.0)
            or spectral_penalty_auto_weight
        )
        if spectral_penalty_intended:
            if sampling_rate is None or burst_window_s is None or max_burst_s is None:
                raise ValueError(
                    "sampling_rate, burst_window_s, and max_burst_s must be provided "
                    "when spectral_penalty_weight > 0."
                )
            if self.tokenizer is None:
                raise ValueError(
                    "Spectral penalty requires a tokenizer for reconstruction."
                )
            if burst_window_s <= 0:
                raise ValueError("burst_window_s must be > 0.")
            if max_burst_s <= 0:
                raise ValueError("max_burst_s must be > 0.")
            if sampling_rate <= 0:
                raise ValueError("sampling_rate must be > 0.")
            if spectral_min_freq < 0:
                raise ValueError("spectral_min_freq must be >= 0.")
            if spectral_max_freq is not None and spectral_max_freq <= spectral_min_freq:
                raise ValueError("spectral_max_freq must be > spectral_min_freq.")

        # ---------- Helper functions ---------- #
        def _random_tokens() -> np.ndarray:
            _rng = np.random.default_rng()
            try:
                token_weights = self.tokenizer.vocab["total_token_counts"].astype(
                    np.float32
                )
                # token_weights = np.ones(self.tokenizer.vocab["total_token_counts"].shape[0]).astype(np.float32)
            except AttributeError:
                token_weights = np.ones(
                    max(1, self.config.model_config.n_tokens - 1), dtype=np.float32
                )
            token_weights /= np.sum(token_weights)

            tokens = (
                _rng.choice(
                    len(token_weights),
                    size=(batch_size, sequence_length, n_channels),
                    p=token_weights,
                )
                + 1
            )
            return tokens.astype(np.int32)

        # ---------- Validation ---------- #
        if prompt is None:
            prompt = _random_tokens()
        elif isinstance(prompt, np.ndarray):
            if prompt.shape != (batch_size, sequence_length, n_channels):
                if prompt.shape == (sequence_length, n_channels):
                    prompt = np.array([prompt] * batch_size)
                else:
                    raise ValueError(
                        "Prompt must have shape (batch_size, sequence_length, n_channels) or (sequence_length, n_channels)."
                    )
        else:
            raise ValueError("Prompt must be a numpy array.")

        if extra_labels is None:
            extra_labels = {}
        for k in extra_labels.keys():
            if extra_labels[k].shape != (batch_size,):
                raise ValueError(f"Extra label {k} must have shape (batch_size,).")
            extra_labels[k] = np.broadcast_to(
                extra_labels[k][:, None], (batch_size, sequence_length + 1)
            )

        if extra_channels is None:
            extra_channels = {}
        for k in extra_channels.keys():
            if extra_channels[k].ndim != 2:
                raise ValueError(
                    f"Extra channel {k} must have shape (batch_size, n_samples)."
                )
            if extra_channels[k].shape[0] != batch_size:
                raise ValueError(
                    f"Extra channel {k} must have shape (batch_size, n_samples)."
                )
            if not extra_channels[k].shape[1] > sequence_length + n_samples:
                raise ValueError(
                    f"Extra channel {k} must have at least n_samples + sequence_length + 1 samples."
                )
            extra_channels[k] = extra_channels[k][:, : sequence_length + n_samples + 1]
        if len(extra_channels) > 0 and lyapunov_rollout_horizon > 1:
            _logger.warning(
                "Rollout scoring with extra_channels reuses the current extra-channel window "
                "for imagined future rollout steps."
            )

        # ---------- Generate tokens ---------- #

        generated_tokens = np.zeros(
            (batch_size, sequence_length + n_samples, n_channels), dtype=np.int32
        )

        lyapunov = np.zeros(
            [batch_size, sequence_length + n_samples + 1], dtype=np.float32
        )
        current_margin = np.full((batch_size,), lyapunov_margin, dtype=np.float32)
        if spectral_penalty_intended:
            spectral_window_samples = max(
                4, int(round(burst_window_s * sampling_rate))
            )
            max_burst_windows = max(
                1, int(np.ceil(max_burst_s / burst_window_s))
            )
            peakiness_run = np.zeros((batch_size,), dtype=np.int32)
            calibration_lyapunov = []
        if lyapunov_diagnostics:
            invalid_mask = np.zeros([batch_size, n_samples], dtype=bool)
            initial_invalid_mask = np.zeros([batch_size, n_samples], dtype=bool)
            attempts_used = np.ones([batch_size, n_samples], dtype=np.int32)
            rejection_count = np.zeros([batch_size, n_samples], dtype=np.int32)
            margin_used = np.zeros([batch_size, n_samples], dtype=np.float32)
            if spectral_penalty_intended:
                spectral_peakiness = np.full(
                    [batch_size, n_samples], np.nan, dtype=np.float32
                )
                spectral_penalty_active_mask = np.zeros(
                    [batch_size, n_samples], dtype=bool
                )
                spectral_penalty_weight_used = np.zeros(
                    [batch_size, n_samples], dtype=np.float32
                )
                spectral_penalty_value = np.zeros(
                    [batch_size, n_samples], dtype=np.float32
                )
        generated_tokens[:, :sequence_length] = prompt
        for i in trange(
            sequence_length, sequence_length + n_samples, desc="Generating tokens"
        ):
            model_inputs = {"data": generated_tokens[:, i - sequence_length : i + 1]}
            # model_input["data"].shape = (batch_size, sequence_length + 1, n_channels)

            # Add extra labels to the model inputs
            model_inputs.update(extra_labels)

            # Add extra channels to the model inputs
            for k in extra_channels.keys():
                model_inputs[k] = extra_channels[k][:, i - sequence_length : i + 1]

            effective_weight = spectral_penalty_weight
            if auto_weight_pending:
                effective_weight = 0.0

            if spectral_penalty_intended and effective_weight > 0.0:
                penalty_active = (peakiness_run >= max_burst_windows).astype(
                    np.float32
                )
                penalty_active = tf.convert_to_tensor(penalty_active, dtype=tf.float32)
            else:
                penalty_active = None

            # Prediction logits for the next token
            if lyapunov_diagnostics:
                (
                    next_token,
                    next_lyapunov_fn,
                    step_attempts_used,
                    step_accepted_mask,
                    step_initial_invalid_mask,
                    step_rejection_count,
                ) = self.one_step_sample(
                    model_inputs,
                    top_p,
                    top_k,
                    temperature,
                    current_margin,
                    lyapunov_rollout_horizon,
                    lyapunov_adaptive_rollout,
                    lyapunov_adaptive_band,
                    effective_weight,
                    penalty_active,
                    spectral_window_samples if spectral_penalty_intended else None,
                    sampling_rate if spectral_penalty_intended else None,
                    spectral_min_freq,
                    spectral_max_freq,
                    spectral_peakiness_threshold,
                    spectral_penalty_normalize,
                    spectral_penalty_cap,
                    True,
                )
            else:
                next_token, next_lyapunov_fn = self.one_step_sample(
                    model_inputs,
                    top_p,
                    top_k,
                    temperature,
                    current_margin,
                    lyapunov_rollout_horizon,
                    lyapunov_adaptive_rollout,
                    lyapunov_adaptive_band,
                    effective_weight,
                    penalty_active,
                    spectral_window_samples if spectral_penalty_intended else None,
                    sampling_rate if spectral_penalty_intended else None,
                    spectral_min_freq,
                    spectral_max_freq,
                    spectral_peakiness_threshold,
                    spectral_penalty_normalize,
                    spectral_penalty_cap,
                )

            # Add the next token to the prompt
            generated_tokens[:, i] = next_token.numpy().astype(np.int32)
            lyapunov[:, i] = next_lyapunov_fn.numpy()
            if lyapunov_diagnostics:
                idx = i - sequence_length
                invalid_mask[:, idx] = ~step_accepted_mask.numpy()
                initial_invalid_mask[:, idx] = step_initial_invalid_mask.numpy()
                attempts_used[:, idx] = step_attempts_used.numpy()
                rejection_count[:, idx] = step_rejection_count.numpy()
                margin_used[:, idx] = current_margin

            if lyapunov_adaptive_margin:
                recent_start = max(sequence_length, i - lyapunov_margin_window + 1)
                recent_vals = lyapunov[:, recent_start : i + 1]
                new_margin = np.quantile(
                    recent_vals, q=lyapunov_margin_quantile, axis=1
                ).astype(np.float32)
                current_margin = np.clip(
                    new_margin,
                    np.float32(lyapunov_margin_min),
                    np.float32(lyapunov_margin_max),
                )

            spectral_peakiness_value = None
            if spectral_penalty_intended and (effective_weight > 0.0 or lyapunov_diagnostics):
                window_end = i + 1
                window_start = max(0, window_end - spectral_window_samples)
                window_tokens = generated_tokens[:, window_start:window_end, :]
                spectral_peakiness_value = self._spectral_peakiness(
                    tf.convert_to_tensor(window_tokens, dtype=tf.int32),
                    sampling_rate=sampling_rate,
                    min_freq=spectral_min_freq,
                    max_freq=spectral_max_freq or sampling_rate / 2.0,
                ).numpy()
                if effective_weight > 0.0:
                    is_peaky = spectral_peakiness_value > spectral_peakiness_threshold
                    peakiness_run = np.where(is_peaky, peakiness_run + 1, 0)
                if lyapunov_diagnostics and spectral_penalty_intended:
                    idx = i - sequence_length
                    spectral_peakiness[:, idx] = spectral_peakiness_value
                    penalty_active_values = (
                        penalty_active.numpy()
                        if penalty_active is not None
                        else np.zeros((batch_size,), dtype=np.float32)
                    )
                    spectral_penalty_active_mask[:, idx] = penalty_active_values > 0
                    spectral_penalty_weight_used[:, idx] = effective_weight
                    if spectral_penalty_normalize:
                        penalty_raw = np.maximum(
                            spectral_peakiness_value / spectral_peakiness_threshold
                            - 1.0,
                            0.0,
                        )
                    else:
                        penalty_raw = np.maximum(
                            spectral_peakiness_value - spectral_peakiness_threshold, 0.0
                        )
                    if spectral_penalty_cap is not None:
                        penalty_raw = np.minimum(penalty_raw, spectral_penalty_cap)
                    spectral_penalty_value[:, idx] = (
                        penalty_raw * penalty_active_values * effective_weight
                    )
            if auto_weight_pending:
                calibration_lyapunov.append(next_lyapunov_fn.numpy())
                if len(calibration_lyapunov) >= spectral_penalty_calibrate_steps:
                    cal_values = np.concatenate(calibration_lyapunov, axis=0)
                    median_lyap = float(np.median(cal_values))
                    if spectral_penalty_normalize:
                        denom = (
                            float(spectral_penalty_cap)
                            if spectral_penalty_cap is not None
                            else 1.0
                        )
                    else:
                        denom = float(spectral_peakiness_threshold)
                    denom = max(denom, 1e-6)
                    spectral_penalty_weight = (
                        median_lyap * float(spectral_penalty_target_frac) / denom
                    )
                    _logger.info(
                        "Calibrated spectral_penalty_weight=%.6g (median_lyapunov=%.6g, target_frac=%.3f, denom=%.6g).",
                        spectral_penalty_weight,
                        median_lyap,
                        float(spectral_penalty_target_frac),
                        denom,
                    )
                    auto_weight_pending = False

            # shape of generated_tokens: (1, time, n_channels)

        if lyapunov_diagnostics:
            invalid_count = int(np.sum(invalid_mask))
            total_count = int(np.prod(invalid_mask.shape))
            invalid_rate = invalid_count / max(1, total_count)
            mean_attempts = float(np.mean(attempts_used))
            rejection_rate = float(
                np.sum(rejection_count) / max(1, np.sum(attempts_used))
            )
            _logger.info(
                (
                    "Lyapunov diagnostics: initial_invalid_rate=%.4f, "
                    "final_invalid_rate=%.4f (%d/%d), mean_attempts=%.3f, rejection_rate=%.4f, "
                    "mean_margin=%.6g"
                ),
                float(np.mean(initial_invalid_mask)),
                invalid_rate,
                invalid_count,
                total_count,
                mean_attempts,
                rejection_rate,
                float(np.mean(margin_used))
                if lyapunov_diagnostics
                else float(np.mean(current_margin)),
            )
            if invalid_count > 0:
                _logger.warning(
                    "Some generated samples remained above Lyapunov margin after max attempts."
                )
            diagnostics = {
                "lyapunov_margin": float(lyapunov_margin),
                "lyapunov_adaptive_margin": bool(lyapunov_adaptive_margin),
                "lyapunov_margin_quantile": float(lyapunov_margin_quantile),
                "lyapunov_margin_window": int(lyapunov_margin_window),
                "lyapunov_margin_min": float(lyapunov_margin_min),
                "lyapunov_margin_max": float(lyapunov_margin_max),
                "lyapunov_rollout_horizon": int(lyapunov_rollout_horizon),
                "lyapunov_adaptive_rollout": bool(lyapunov_adaptive_rollout),
                "lyapunov_adaptive_band": float(lyapunov_adaptive_band),
                "max_attempts": 10,
                "invalid_mask": invalid_mask,
                "initial_invalid_mask": initial_invalid_mask,
                "attempts_used": attempts_used,
                "rejection_count": rejection_count,
                "margin_used": margin_used,
                "final_margin_per_session": current_margin.copy(),
                "final_invalid_rate_per_session": np.mean(invalid_mask, axis=1),
                "initial_invalid_rate_per_session": np.mean(
                    initial_invalid_mask, axis=1
                ),
                "mean_attempts_per_session": np.mean(attempts_used, axis=1),
            }
            if spectral_penalty_intended:
                diagnostics.update(
                    {
                        "spectral_peakiness": spectral_peakiness,
                        "spectral_penalty_active": spectral_penalty_active_mask,
                        "spectral_penalty_weight": spectral_penalty_weight_used,
                        "spectral_penalty_value": spectral_penalty_value,
                        "spectral_penalty_threshold": float(
                            spectral_peakiness_threshold
                        ),
                        "spectral_penalty_cap": float(spectral_penalty_cap)
                        if spectral_penalty_cap is not None
                        else None,
                        "spectral_penalty_normalize": bool(
                            spectral_penalty_normalize
                        ),
                        "spectral_penalty_calibrate_steps": int(
                            spectral_penalty_calibrate_steps
                        ),
                    }
                )
            return (
                generated_tokens[:, sequence_length:],
                lyapunov[:, sequence_length:],
                diagnostics,
            )

        return generated_tokens[:, sequence_length:], lyapunov[:, sequence_length:]

    def generate_data(self, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate data using the model.

        Parameters
        ----------
        See generate_tokens() for parameters.

        Returns
        -------
        generated_data

        """

        if self.tokenizer is None:
            raise ValueError(
                "Cannot generate reconstructed data without a tokenizer. "
                "Use generate_tokens() or set model_config.tokenizer_path."
            )

        generated = self.generate_tokens(**kwargs)
        if len(generated) == 3:
            tokens, lyapunov, diagnostics = generated
            return self.tokenizer.reconstruct_data(list(tokens)), lyapunov, diagnostics

        tokens, lyapunov = generated  # (batch_size, n_samples, n_channels)
        return (
            self.tokenizer.reconstruct_data(list(tokens)),
            lyapunov,
        )  # (batch_size, n_samples, n_channels)
