import re
import random
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self, min_freq: int = 1):
        """
        Simple trigram language model (N=3).

        Args:
            min_freq (int): Minimum frequency a word must have to
                            NOT be treated as <unk>. Defaults to 1
                            (i.e., no UNKing based on frequency).
        """
        self.min_freq = min_freq

        # trigram_counts[(w1, w2)][w3] = count
        self.trigram_counts = defaultdict(lambda: Counter())

        # context_totals[(w1, w2)] = total number of trigrams starting with (w1, w2)
        self.context_totals = Counter()

        # vocabulary
        self.vocab = set()

        # special tokens
        self.START = "<s>"
        self.END = "</s>"
        self.UNK = "<unk>"

        self._is_fitted = False

    # ---------- internal helpers ----------

    def _clean_text(self, text: str) -> str:
        """
        Basic cleaning:
        - lowercasing
        """
        text = text.lower()
        return text

    def _tokenize(self, sentence: str):
        """
        Tokenize a single sentence into word tokens.
        """
        return re.findall(r"\b\w+\b", sentence)

    def _split_sentences(self, text: str):
        """
        Split text into sentences using punctuation as delimiters.
        """
        raw_sentences = re.split(r"[.!?]+", text)
        sentences = []
        for s in raw_sentences:
            tokens = self._tokenize(s)
            if tokens:
                sentences.append(tokens)
        return sentences

    def _build_vocab(self, tokens):
        """
        Build vocabulary and handle <unk> based on min_freq.
        """
        freq = Counter(tokens)

        self.vocab = {self.START, self.END, self.UNK}
        for w, c in freq.items():
            if c >= self.min_freq:
                self.vocab.add(w)

    def _normalize_token(self, token: str) -> str:
        """
        Map token to itself if in vocab, else to <unk>.
        """
        return token if token in self.vocab else self.UNK

    # ---------- public API ----------

    def fit(self, text: str):
        """
        Train the trigram model on the given text.
        """
        # reset state
        self.trigram_counts.clear()
        self.context_totals.clear()
        self.vocab.clear()
        self._is_fitted = False

        if not text or not text.strip():
            # handle empty text gracefully
            return

        # 1. clean
        cleaned = self._clean_text(text)

        # 2. sentences
        sentences = self._split_sentences(cleaned)
        if not sentences:
            return

        # 3. build vocab
        all_tokens = [tok for sent in sentences for tok in sent]
        self._build_vocab(all_tokens)

        # 4. pad and count trigrams
        for sent in sentences:
            sent_norm = [self._normalize_token(t) for t in sent]
            tokens = [self.START, self.START] + sent_norm + [self.END]

            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                context = (w1, w2)
                self.trigram_counts[context][w3] += 1
                self.context_totals[context] += 1

        self._is_fitted = True

    def _sample_next(self, context):
        """
        Sample next word from P(w3 | w1, w2).
        """
        candidates = self.trigram_counts.get(context)
        if not candidates:
            return None

        words = list(candidates.keys())
        counts = [candidates[w] for w in words]
        total = sum(counts)
        probs = [c / total for c in counts]

        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, max_length: int = 50) -> str:
        """
        Generate a sequence of text up to max_length tokens.
        """
        if not self._is_fitted or not self.trigram_counts:
            return ""

        context = (self.START, self.START)
        generated = []

        for _ in range(max_length):
            next_word = self._sample_next(context)
            if next_word is None:
                break
            if next_word == self.END:
                break

            generated.append(next_word)
            context = (context[1], next_word)

        return " ".join(generated)
