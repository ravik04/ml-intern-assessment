# Scaled Dot-Product Attention (Optional Task 2)

This directory contains the NumPy implementation of Scaled Dot-Product Attention, the core mechanism behind Transformer architectures such as BERT and GPT.

---

## File Structure

```
attention/
├── scaled_attention.py   # Core implementation
└── demo.py               # Demonstration script
```

---

## Formula

The implementation follows the standard attention equation from *Attention Is All You Need*:

\[
\text{Attention}(Q, K, V)
= \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

---

## Usage

Run the demo script from inside the **ml-assignment/** directory:

```bash
python -m attention.demo
```

The script will display:

- Input matrices: **Q**, **K**, **V**
- Optional **mask**
- **Attention weights** (softmax over scaled dot products)
- **Final attended output**

---

## Implementation Notes

- Written entirely using **NumPy** (`import numpy as np`)
- Includes **stable softmax**, subtracting max value to prevent overflow
- Supports **optional masking** (set masked logits to `-1e9`)
- Produces attention weights of shape `(seq_len_q, seq_len_k)` and output of `(seq_len_q, d_v)`

---

## Summary

This folder completes the optional bonus task by implementing a mathematically correct, NumPy-based version of the Scaled Dot-Product Attention mechanism, along with a reproducible demo for testing.

