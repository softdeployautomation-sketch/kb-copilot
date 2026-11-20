"""
Lightweight BM25 retrieval over local text chunks (no ML deps).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def iter_chunks_from_file(path: Path, max_chars: int = 900, overlap: int = 120) -> Iterable[tuple[str, str]]:
    """Yield (source_label, chunk_text) from a text file."""
    label = str(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return
    parts = re.split(r"\n\s*\n", raw)
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return None
        text = "\n\n".join(buf).strip()
        buf = []
        buf_len = 0
        return text if text else None

    for para in parts:
        para = para.strip()
        if not para:
            continue
        if buf_len + len(para) > max_chars and buf:
            done = flush()
            if done:
                yield label, done
            # overlap: keep tail of previous for context
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para) + 2

    last = flush()
    if last:
        yield label, last

    # Very long single paragraphs: hard-split
    if not parts:
        for i in range(0, len(raw), max_chars - overlap):
            chunk = raw[i : i + max_chars].strip()
            if chunk:
                yield label, chunk


def load_corpus(kb_root: Path) -> list[tuple[str, str]]:
    corpus: list[tuple[str, str]] = []
    patterns = ("*.md", "*.txt", "*.markdown")
    seen: set[Path] = set()
    for pat in patterns:
        for path in sorted(kb_root.rglob(pat)):
            if not path.is_file() or path.name.startswith(".") or path in seen:
                continue
            seen.add(path)
            try:
                for label, chunk in iter_chunks_from_file(path):
                    corpus.append((label, chunk))
            except OSError:
                continue
    return corpus


@dataclass
class BM25Index:
    chunks: list[tuple[str, str]]
    tokens_per_doc: list[list[str]]
    doc_freq: dict[str, int]
    idf: dict[str, float]
    avgdl: float
    N: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunks: list[tuple[str, str]]) -> BM25Index:
        tokens_per_doc = [tokenize(c[1]) for c in chunks]
        N = len(tokens_per_doc)
        doc_freq: dict[str, int] = {}
        for tokens in tokens_per_doc:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        idf: dict[str, float] = {}
        for t, df in doc_freq.items():
            idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        total_len = sum(len(t) for t in tokens_per_doc)
        avgdl = total_len / N if N else 0.0
        return cls(
            chunks=chunks,
            tokens_per_doc=tokens_per_doc,
            doc_freq=doc_freq,
            idf=idf,
            avgdl=avgdl,
            N=N,
        )

    def scores(self, query: str) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = [0.0] * self.N
        for qi, tokens in enumerate(self.tokens_per_doc):
            dl = len(tokens)
            freqs: dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            s = 0.0
            for qt in q_tokens:
                if qt not in freqs:
                    continue
                f = freqs[qt]
                idf = self.idf.get(qt, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += idf * (f * (self.k1 + 1)) / (denom or 1.0)
            scores[qi] = s
        indexed = [(i, scores[i]) for i in range(self.N)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        ranked = self.scores(query)
        out: list[dict] = []
        for idx, sc in ranked[:top_k]:
            if sc <= 0 and ranked and ranked[0][1] <= 0:
                # fall back: return first chunks if nothing matched
                break
            src, text = self.chunks[idx]
            out.append({"score": round(sc, 4), "source": src, "text": text})
        if not out and ranked:
            for idx, sc in ranked[:top_k]:
                src, text = self.chunks[idx]
                out.append({"score": round(sc, 4), "source": src, "text": text})
        return out
