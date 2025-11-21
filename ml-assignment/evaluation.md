# Evaluation

## Trigram Language Model – Design Summary

This project implements a trigram-based language model capable of learning token-level probabilities and generating new sequences. The design emphasizes clarity, robustness, and correctness.

---

## N-gram Storage

Trigram counts are stored using a nested dictionary backed by `defaultdict(Counter)`:

```
trigram_counts[(w1, w2)][w3] = count
```

This structure enables:

- Fast count updates  
- Simple retrieval of candidate next words  
- Direct mapping from counts to probability weights  

Using `(w1, w2)` as a tuple key keeps the dictionary compact and efficient.

---

## Text Cleaning & Tokenization

The following preprocessing steps are applied:

- Lowercasing  
- Extracting words via regex pattern `\b\w+\b`  
- Splitting sentences using `[.!?]+`  

This avoids external NLP dependencies while ensuring clean, predictable input for trigram training.

---

## Padding Strategy

Each sentence is padded like:

```
<s> <s> word1 word2 ... wordN </s>
```

Reasons for this design:

- `<s>` provides context for the first two predicted words  
- `</s>` signals sequence termination during generation  
- Padding prevents mixing of sentence contexts  

This ensures consistency between training and generation.

---

## Unknown Word Handling

Low-frequency words (`min_freq`) are replaced with `<unk>`.  
During generation and training, all tokens pass through `_normalize_token()`.

This avoids errors from unseen words and stabilizes the model for small datasets.

---

## Generation & Probabilistic Sampling

Generation initializes with:

```
(<s>, <s>)
```

At each step:

1. Retrieve all next-word candidates  
2. Convert counts to probability weights  
3. Sample using:

```python
random.choices(words, weights=counts)
```

4. Shift context (w1, w2 → w2, new_word)  
5. Stop when generating `</s>` or when `max_length` is reached  

This sampling introduces variability and avoids deterministic text.

---

## Additional Design Choices

- No smoothing (e.g., Laplace/Kneser–Ney) to keep the implementation aligned with assignment requirements  
- Lightweight design without unnecessary dependencies  
- Separate `generate.py` provided for quick manual testing  

---

# Task 2 – Scaled Dot-Product Attention (Optional)

The implementation is located in:

```
attention/scaled_attention.py
```

Demo script:

```
attention/demo.py
```

---

## Computation Steps

1. **Raw attention scores:**

\[
\text{scores} = \frac{QK^T}{\sqrt{d_k}}
\]

2. **Masking:**  
   Masked values are replaced with `-1e9`.

3. **Stable softmax:**  

\[
e^{(scores - \max(scores))}
\]

4. **Normalize rows** to obtain attention weights.

5. **Final output computation:**

\[
output = attention\_weights \cdot V
\]

---

## Demo Script

The demo prints:

- Q, K, V matrices  
- Mask (optional)  
- Attention weights  
- Final attended output  

Run using:

```bash
python -m attention.demo
```

This verifies the correctness of the NumPy-based attention mechanism.

---
