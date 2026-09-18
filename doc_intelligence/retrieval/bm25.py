from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def tokenize(text: str | None) -> list[str]:
    """Lowercase word tokens; keeps unit-like tokens such as 'ml/day' and '3.2' intact."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Small in-memory BM25 (Okapi) over a chunk corpus. Exact-term matching complements
    embedding similarity for legal/regulatory wording."""

    def __init__(self, documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self._tf = {doc_id: Counter(tokens) for doc_id, tokens in self._tokens.items()}
        self._len = {doc_id: len(tokens) for doc_id, tokens in self._tokens.items()}
        self._avg_len = (sum(self._len.values()) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tokens in self._tokens.values():
            df.update(set(tokens))
        n = len(self._tokens)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def __len__(self) -> int:
        return len(self._tokens)

    def score(self, query: str, doc_id: str) -> float:
        tf = self._tf[doc_id]
        length = self._len[doc_id]
        score = 0.0
        for term in tokenize(query):
            freq = tf.get(term)
            if not freq:
                continue
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len) if self._avg_len else self.k1
            score += self._idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scored = ((doc_id, self.score(query, doc_id)) for doc_id in self._tokens)
        ranked = sorted((item for item in scored if item[1] > 0), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
