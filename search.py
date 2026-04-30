"""
BM25 search over wiki markdown files.
Pure Python, zero external dependencies, O(1) per query after index build.
"""

import logging
import math
import re
from pathlib import Path

log = logging.getLogger("search")

# BM25 parameters
K1 = 1.5
B  = 0.75


class BM25Search:
    def __init__(self, wiki_dir: Path):
        self.wiki_dir = wiki_dir
        self._docs:   list[dict]         = []   # {"path": str, "tokens": list[str]}
        self._index:  dict[str, list]    = {}   # term → [(doc_idx, tf), ...]
        self._avgdl:  float              = 0.0
        self._idf:    dict[str, float]   = {}

    def rebuild(self):
        """Re-index all wiki markdown files."""
        if not self.wiki_dir.exists():
            return
        docs = []
        for p in sorted(self.wiki_dir.rglob("*.md")):
            if p.name.startswith("."):
                continue
            try:
                text   = p.read_text(encoding="utf-8")
                tokens = _tokenize(text)
                docs.append({
                    "path":   str(p.relative_to(self.wiki_dir)),
                    "tokens": tokens,
                    "text":   text[:300],   # snippet
                })
            except Exception:
                pass

        self._docs  = docs
        N           = len(docs)
        if N == 0:
            return

        # Build inverted index
        inv: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            freq: dict[str, int] = {}
            for t in doc["tokens"]:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                inv.setdefault(t, {})[i] = f

        self._index = {t: list(d.items()) for t, d in inv.items()}
        self._avgdl = sum(len(d["tokens"]) for d in docs) / N
        self._idf   = {
            t: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for t, postings in self._index.items()
        }
        log.info(f"Search index: {N} pages, {len(self._index)} terms")

    def search(self, query: str, k: int = 6) -> list[dict]:
        """Return top-k results sorted by BM25 score."""
        if not self._docs:
            return []

        q_terms  = _tokenize(query)
        scores:  dict[int, float] = {}
        avgdl    = self._avgdl or 1.0

        for term in q_terms:
            if term not in self._index:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in self._index[term]:
                dl = len(self._docs[doc_idx]["tokens"])
                numerator   = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:k]:
            d = self._docs[doc_idx]
            results.append({
                "path":    d["path"],
                "score":   round(score, 3),
                "snippet": d["text"],
            })
        return results


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip markdown syntax, split on non-word chars, remove stopwords."""
    # Strip frontmatter
    text = re.sub(r"^---[\s\S]*?---\n", "", text)
    # Strip markdown syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[#*`_>~|]", " ", text)
    # Tokenize
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "so", "if", "then",
    "than", "more", "also", "into", "about", "which", "what", "when",
    "how", "all", "each", "other", "one", "two", "new", "see", "use",
    "used", "using", "page", "file", "note", "notes",
}
