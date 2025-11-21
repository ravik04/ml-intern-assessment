import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention (single-head, no batching).

    Args:
        Q: np.ndarray of shape (seq_len_q, d_k)
        K: np.ndarray of shape (seq_len_k, d_k)
        V: np.ndarray of shape (seq_len_k, d_v)
        mask: optional np.ndarray of shape (seq_len_q, seq_len_k)
              where masked positions are True/1 and will be ignored
              (driven towards zero after softmax).

    Returns:
        output: np.ndarray of shape (seq_len_q, d_v)
        attention_weights: np.ndarray of shape (seq_len_q, seq_len_k)
    """

    # ---- 1. Basic shape checks ----
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be 2D arrays of shapes "
                         "(seq_len_q, d_k), (seq_len_k, d_k), (seq_len_k, d_v)")

    seq_len_q, d_k_q = Q.shape
    seq_len_k, d_k_k = K.shape
    seq_len_v, d_v = V.shape

    if d_k_q != d_k_k:
        raise ValueError(f"Q and K must have the same depth d_k, got {d_k_q} and {d_k_k}")
    if seq_len_k != seq_len_v:
        raise ValueError(f"K and V must have the same seq_len, got {seq_len_k} and {seq_len_v}")

    d_k = d_k_q

    # ---- 2. Raw attention scores: Q K^T / sqrt(d_k) ----
    # scores[i, j] = dot(Q_i, K_j) / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)

    # ---- 3. Apply mask (optional) ----
    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match scores shape {scores.shape}"
            )
        # Assume mask is boolean or {0,1}. True/1 means "mask this position".
        scores = np.where(mask, -1e9, scores)

    # ---- 4. Softmax over keys dimension (axis=1) ----
    # numerical stability: subtract per-row max before exp
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)

    # avoid division by zero if a whole row was masked
    sum_exp_safe = np.where(sum_exp == 0.0, 1.0, sum_exp)
    attention_weights = exp_scores / sum_exp_safe  # (seq_len_q, seq_len_k)

    # ---- 5. Weighted sum of values ----
    # output[i] = Σ_j attention_weights[i, j] * V[j]
    output = attention_weights @ V  # (seq_len_q, d_v)

    return output, attention_weights
