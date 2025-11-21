# Trigram Language Model – ML Intern Assignment

This repository contains my implementation of the N-Gram Language Model (Trigram Model) and the optional Scaled Dot-Product Attention task.

---

## Project Structure

```
ml-assignment/
│
├── data/
│   └── example_corpus.txt
│
├── src/
│   ├── ngram_model.py
│   ├── generate.py
│   └── utils.py
│
├── tests/
│   └── test_ngram.py
│
├── attention/
│   ├── scaled_attention.py
│   └── demo.py
│
├── README.md
└── evaluation.md
```

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate       # Linux / Mac
.venv\Scripts\activate          # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Trigram Model

Generate text using the example corpus:

```bash
python src/generate.py
```

This loads `data/example_corpus.txt`, trains a trigram model, and prints generated text.

---

## Run Tests

```bash
pytest tests/test_ngram.py
```

Expected:

```
3 passed in X.XXs
```

---

## Optional Task 2 – Scaled Dot-Product Attention

Run the demo:

```bash
python -m attention.demo
```

The script prints:

- Q, K, V matrices  
- Mask  
- Attention weights  
- Final output  

This verifies correctness of the NumPy-based implementation.

---

## Evaluation Report

See `evaluation.md` for full design explanations and implementation decisions.

---

## Status

- ✔ Completed Trigram Model  
- ✔ All tests pass  
- ✔ Optional attention task implemented  
- ✔ Clean, documented structure  