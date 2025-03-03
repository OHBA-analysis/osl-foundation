import tensorflow as tf
import tensorflow_probability as tfp


def sample_from_logits(
    logits: tf.Tensor, top_p: float = None, top_k: int = None
) -> tf.Tensor:
    """
    Sample from the logits using the specified method.

    Parameters
    ----------
    logits : tf.Tensor
        The logits to sample from. Shape: (*batch_dims, n_tokens)
    top_p : float, optional
        The probability threshold for top-p sampling.
    top_k : int, optional
        The number of top logits to sample from.

    Returns
    -------
    sampled_tokens : tf.Tensor
        The sampled tokens. Shape: (*batch_dims).
    """
    if top_p is not None and top_k is not None:
        raise ValueError("Only one of 'top_p' or 'top_k' can be provided.")
    elif top_p is not None:
        sampled_tokens = top_p_sampling(logits, top_p)
    elif top_k is not None:
        sampled_tokens = top_k_sampling(logits, top_k)
    else:
        sampled_tokens = argmax_sampling(logits)

    return tf.cast(sampled_tokens, tf.int32)


def top_k_sampling(logits: tf.Tensor, k: int) -> tf.Tensor:
    """
    Top-k sampling. Only sample from the largest k logits.

    Parameters
    ----------
    logits : tf.Tensor
        The logits to sample from. Shape: (*batch_dims, n_tokens).
    k : int
        The number of top logits to sample from.

    Returns
    -------
    sampled_tokens : tf.Tensor
        The sampled tokens. Shape: (*batch_dims).
    """
    # logits.shape = (*batch_dims, n_tokens)
    n_dim = len(tf.shape(logits))

    # Get the top k logits and their indices
    top_k_logits, top_k_tokens = tf.math.top_k(logits, k)
    # top_k_logits.shape = (*batch_dims, k)
    # top_k_tokens.shape = (*batch_dims, k)

    # Sample from the top k logits
    cat = tfp.distributions.Categorical(logits=top_k_logits)
    sampled_indices = cat.sample()
    # sampled_indices.shape = (*batch_dims)

    # Get the sampled tokens
    sampled_tokens = tf.gather(top_k_tokens, sampled_indices, batch_dims=n_dim - 1)
    # sampled_logits.shape = (*batch_dims)

    return sampled_tokens


def argmax_sampling(logits: tf.Tensor) -> tf.Tensor:
    """
    Argmax sampling. Choose the token with the highest logit.

    Parameters
    ----------
    logits : tf.Tensor
        The logits to sample from. Shape: (*batch_dims, n_tokens).

    Returns
    -------
    sampled_tokens : tf.Tensor
        The sampled tokens. Shape: (*batch_dims).
    """
    sampled_tokens = tf.argmax(logits, axis=-1)
    return sampled_tokens


def top_p_sampling(logits: tf.Tensor, p: float) -> tf.Tensor:
    """
    Top-p sampling. Only sample from the smallest set of tokens whose cumulative probability exceeds p.

    Parameters
    ----------
    logits : tf.Tensor
        The logits to sample from. Shape: (*batch_dims, n_tokens).
    p : float
        The probability threshold.

    Returns
    -------
    sampled_tokens : tf.Tensor
        The sampled tokens. Shape: (*batch_dims).
    """
    # logits.shape = (*batch_dims, n_tokens)
    n_dim = len(tf.shape(logits))

    # sort logits in descending order
    sorted_logits = tf.sort(logits, direction="DESCENDING")
    sorted_indices = tf.argsort(logits, direction="DESCENDING")

    # compute cumulative probabilities
    cum_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Find the index where cumulative probability exceeds p
    cutoff_indices = tf.reduce_sum(tf.cast(cum_probs < p, tf.int32), axis=-1) + 1

    # Mask logits beyond the cutoff index
    mask = tf.logical_not(
        tf.sequence_mask(cutoff_indices, maxlen=tf.shape(logits)[-1], dtype=tf.bool)
    )
    masked_logits = tf.where(mask, float("-inf"), sorted_logits)

    # Sample from the masked logits
    cat = tfp.distributions.Categorical(logits=masked_logits)
    sampled_indices = cat.sample()

    # Get the sampled tokens
    sampled_tokens = tf.gather(sorted_indices, sampled_indices, batch_dims=n_dim - 1)

    return sampled_tokens
