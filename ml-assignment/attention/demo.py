import numpy as np
from .scaled_attention import scaled_dot_product_attention


def main():
    # Tiny toy example
    seq_len_q = 2
    seq_len_k = 3
    d_k = 4
    d_v = 5

    rng = np.random.default_rng(42)

    # Random queries, keys, values
    Q = rng.normal(size=(seq_len_q, d_k))
    K = rng.normal(size=(seq_len_k, d_k))
    V = rng.normal(size=(seq_len_k, d_v))

    # Optional mask: mask out the last key for the second query
    mask = np.zeros((seq_len_q, seq_len_k), dtype=bool)
    mask[1, -1] = True  # mask (query 1, key 2)

    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("Q:\n", Q)
    print("\nK:\n", K)
    print("\nV:\n", V)
    print("\nMask:\n", mask.astype(int))
    print("\nAttention weights:\n", attention_weights)
    print("\nOutput:\n", output)


if __name__ == "__main__":
    main()
