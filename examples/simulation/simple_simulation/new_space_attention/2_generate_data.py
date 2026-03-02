import numpy as np
import pickle
import gc
from tqdm.auto import tqdm

from osl_dynamics.utils.misc import set_random_seed
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model

# python /Users/woolrich/dev/projects/osl-foundation/examples/simulation/simple_simulation/new_space_attention/2_generate_data.py 

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"

# Set GPU memory growth
tf_ops.gpu_growth()

set_random_seed(42)

sample_frequency = 100
spectral_penalty_weight = 0.003
spectral_penalty_normalize = True
spectral_penalty_cap = 2.0
spectral_peakiness_threshold = 10.0
spectral_min_freq = 1.0
spectral_max_freq = 40.0
spectral_penalty_auto_weight = True
spectral_penalty_target_frac = 0.05
spectral_penalty_calibrate_steps = 200
burst_window_s = 1
max_burst_s = 4
# When auto weight is enabled, spectral_penalty_weight below is treated as a fallback.
# With the normalized + capped penalty, 0.005 is a good starting point.
# Stronger: 0.008–0.01. Weaker: 0.002–0.003.

# ---------- Directories ---------- #
generator_dir = f"{dev_dir}/results/osl-foundation/simple_simulation/generator"

# ---------- Load generator ---------- #
try:
    generator = load_model(generator_dir, checkpoint="latest")
except Exception as e:
    print(
        "Falling back to model.weights.h5 load because latest checkpoint restore failed: "
        f"{e}"
    )
    generator = load_model(generator_dir)

# ---------- Generate data using the generator ---------- #

n_sessions = 6
session_labels = np.arange(n_sessions)
generation_batch_size = 3

# Generate and save in chunks to keep peak memory usage bounded.
for start in range(0, n_sessions, generation_batch_size):
    labels_chunk = session_labels[start : start + generation_batch_size]
    generated_data, lyapunov_data, lyapunov_diagnostics = generator.generate_data(
        n_samples=2 * 60 * sample_frequency,
        top_p=0.99,
        batch_size=len(labels_chunk),
        extra_labels={"session_id": labels_chunk.astype(np.int32)},
        temperature=1,
        # Low-intrusion Lyapunov-on generation settings.
        lyapunov_margin=0.05,
        lyapunov_adaptive_margin=False,
        lyapunov_margin_quantile=0.9,
        lyapunov_margin_window=500,
        lyapunov_margin_min=1e-4,
        lyapunov_margin_max=5e-2,
        lyapunov_rollout_horizon=1,
        lyapunov_adaptive_rollout=False,
        lyapunov_adaptive_band=2e-4,
        lyapunov_diagnostics=True,
        # Adaptive dominant-band penalty with persistence gate (band-agnostic).
        spectral_penalty_weight=spectral_penalty_weight,
        sampling_rate=sample_frequency,
        burst_window_s=burst_window_s,
        max_burst_s=max_burst_s,
        spectral_penalty_normalize=spectral_penalty_normalize,
        spectral_penalty_cap=spectral_penalty_cap,
        spectral_peakiness_threshold=spectral_peakiness_threshold,
        spectral_min_freq=spectral_min_freq,
        spectral_max_freq=spectral_max_freq,
        spectral_penalty_auto_weight=spectral_penalty_auto_weight,
        spectral_penalty_target_frac=spectral_penalty_target_frac,
        spectral_penalty_calibrate_steps=spectral_penalty_calibrate_steps,
    )

    def _safe_session_value(diag, key, idx):
        if key not in diag:
            return None
        value = diag[key]
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        if arr.shape[0] <= idx:
            return None
        return arr[idx]

    for i, label in enumerate(
        tqdm(labels_chunk, desc=f"Saving generated data [{start}:{start + len(labels_chunk)}]", total=len(labels_chunk))
    ):
        with open(f"{generator_dir}/generated_data_{label}.pkl", "wb") as f:
            pickle.dump(generated_data[i], f)
        with open(f"{generator_dir}/lyapunov_data_{label}.pkl", "wb") as f:
            pickle.dump(lyapunov_data[i], f)
        with open(f"{generator_dir}/lyapunov_diagnostics_{label}.pkl", "wb") as f:
            pickle.dump(
                {
                    "lyapunov_margin": lyapunov_diagnostics["lyapunov_margin"],
                    "lyapunov_adaptive_margin": lyapunov_diagnostics["lyapunov_adaptive_margin"],
                    "lyapunov_margin_quantile": lyapunov_diagnostics["lyapunov_margin_quantile"],
                    "lyapunov_margin_window": lyapunov_diagnostics["lyapunov_margin_window"],
                    "lyapunov_margin_min": lyapunov_diagnostics["lyapunov_margin_min"],
                    "lyapunov_margin_max": lyapunov_diagnostics["lyapunov_margin_max"],
                    "lyapunov_rollout_horizon": lyapunov_diagnostics["lyapunov_rollout_horizon"],
                    "lyapunov_adaptive_rollout": lyapunov_diagnostics["lyapunov_adaptive_rollout"],
                    "lyapunov_adaptive_band": lyapunov_diagnostics["lyapunov_adaptive_band"],
                    "spectral_peakiness": _safe_session_value(lyapunov_diagnostics, "spectral_peakiness", i),
                    "spectral_penalty_active": _safe_session_value(lyapunov_diagnostics, "spectral_penalty_active", i),
                    "spectral_penalty_weight": _safe_session_value(lyapunov_diagnostics, "spectral_penalty_weight", i),
                    "spectral_penalty_value": _safe_session_value(lyapunov_diagnostics, "spectral_penalty_value", i),
                    "spectral_penalty_threshold": lyapunov_diagnostics.get("spectral_penalty_threshold", None),
                    "spectral_penalty_cap": lyapunov_diagnostics.get("spectral_penalty_cap", None),
                    "spectral_penalty_normalize": lyapunov_diagnostics.get("spectral_penalty_normalize", None),
                    "spectral_penalty_calibrate_steps": lyapunov_diagnostics.get("spectral_penalty_calibrate_steps", None),
                    "spectral_penalty_settings": {
                        "spectral_penalty_weight": spectral_penalty_weight,
                        "spectral_penalty_normalize": spectral_penalty_normalize,
                        "spectral_penalty_cap": spectral_penalty_cap,
                        "spectral_peakiness_threshold": spectral_peakiness_threshold,
                        "spectral_min_freq": spectral_min_freq,
                        "spectral_max_freq": spectral_max_freq,
                        "spectral_penalty_auto_weight": spectral_penalty_auto_weight,
                        "spectral_penalty_target_frac": spectral_penalty_target_frac,
                        "spectral_penalty_calibrate_steps": spectral_penalty_calibrate_steps,
                        "sampling_rate": sample_frequency,
                        "burst_window_s": burst_window_s,
                        "max_burst_s": max_burst_s,
                    },n
                    "sampling_rate": sample_frequency,
                    "burst_window_s": burst_window_s,
                    "max_burst_s": max_burst_s,
                    "max_attempts": lyapunov_diagnostics["max_attempts"],
                    "invalid_mask": lyapunov_diagnostics["invalid_mask"][i],
                    "initial_invalid_mask": lyapunov_diagnostics["initial_invalid_mask"][i],
                    "attempts_used": lyapunov_diagnostics["attempts_used"][i],
                    "rejection_count": lyapunov_diagnostics["rejection_count"][i],
                    "margin_used": lyapunov_diagnostics["margin_used"][i],
                    "final_margin": lyapunov_diagnostics["final_margin_per_session"][i],
                    "final_invalid_rate": lyapunov_diagnostics["final_invalid_rate_per_session"][i],
                    "initial_invalid_rate": lyapunov_diagnostics["initial_invalid_rate_per_session"][i],
                    "mean_attempts": lyapunov_diagnostics["mean_attempts_per_session"][i],
                },
                f,
            )

    # Drop chunk outputs promptly to keep peak memory bounded.
    del generated_data, lyapunov_data, lyapunov_diagnostics
    gc.collect()
