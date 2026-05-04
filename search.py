"""
BM25 search over Obsidian vault markdown files.
Pure Python, zero external dependencies, O(1) per query after index build.

Improvements over v1:
- CJK tokenization support (Chinese, Japanese, Korean)
- Better snippet generation with context around matching terms
- Title-boosted scoring
- Improved stopword list
- Frontmatter-aware parsing
"""

import logging
import math
import re
from pathlib import Path

from config_loader import CFG

log = logging.getLogger("search")

# BM25 parameters from config
K1 = CFG["search"]["k1"]
B  = CFG["search"]["b"]

# Supported image file extensions (not indexed, just tracked)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


class BM25Search:
    def __init__(self, vault_dir: Path):
        self.vault_dir = vault_dir
        self._docs: list[dict]         = []  # {"path", "tokens", "title_tokens", "text", "title"}
        self._index: dict[str, list]    = {}  # term → [(doc_idx, tf), ...]
        self._avgdl: float             = 0.0
        self._idf: dict[str, float]    = {}
        self._title_index: dict[str, list] = {}  # title term → [(doc_idx, tf)]

    def rebuild(self):
        """Re-index all markdown files in the vault."""
        if not self.vault_dir.exists():
            return
        docs = []
        for p in sorted(self.vault_dir.rglob("*.md")):
            # Skip hidden files, .obsidian config, and schema files
            if p.name.startswith(".") or ".obsidian" in str(p):
                continue
            try:
                text = p.read_text(encoding="utf-8")
                # Skip very short files (likely templates)
                if len(text.strip()) < 20:
                    continue

                body_text = _strip_frontmatter(text)
                title = _extract_title_from_text(text)

                tokens = _tokenize(body_text)
                title_tokens = _tokenize(title)

                docs.append({
                    "path":         str(p.relative_to(self.vault_dir)),
                    "tokens":       tokens,
                    "title_tokens": title_tokens,
                    "text":         body_text[:500],  # longer snippet
                    "title":        title,
                })
            except Exception:
                pass

        self._docs = docs
        N = len(docs)
        if N == 0:
            return

        # Build inverted index for body content
        inv: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            freq: dict[str, int] = {}
            for t in doc["tokens"]:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                inv.setdefault(t, {})[i] = f

        self._index = {t: list(d.items()) for t, d in inv.items()}

        # Build inverted index for titles (for title boosting)
        title_inv: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            freq: dict[str, int] = {}
            for t in doc["title_tokens"]:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                title_inv.setdefault(t, {})[i] = f

        self._title_index = {t: list(d.items()) for t, d in title_inv.items()}

        self._avgdl = sum(len(d["tokens"]) for d in docs) / N
        self._idf = {
            t: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for t, postings in self._index.items()
        }

        log.info(f"Search index: {N} pages, {len(self._index)} terms")

    def search(self, query: str, k: int = 6) -> list[dict]:
        """Return top-k results sorted by BM25 score with title boosting."""
        if not self._docs:
            return []

        q_terms = _tokenize(query)
        scores: dict[int, float] = {}
        avgdl = self._avgdl or 1.0

        # Body scoring
        for term in q_terms:
            if term not in self._index:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in self._index[term]:
                dl = len(self._docs[doc_idx]["tokens"])
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * numerator / denominator

        # Title boosting — matches in title get 2x weight
        for term in q_terms:
            if term not in self._title_index:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in self._title_index[term]:
                dl = len(self._docs[doc_idx]["tokens"])
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 2.0 * idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:k]:
            d = self._docs[doc_idx]
            snippet = _generate_snippet(d["text"], q_terms)
            results.append({
                "path":    d["path"],
                "title":   d["title"],
                "score":   round(score, 3),
                "snippet": snippet,
            })
        return results


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from text for indexing."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end >= 0:
            return text[end + 3:].strip()
    return text


def _extract_title_from_text(text: str) -> str:
    """Extract title from frontmatter or first heading."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    Handles English and CJK (Chinese, Japanese, Korean) text.
    """
    # Strip frontmatter
    text = _strip_frontmatter(text)
    # Strip markdown syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[#*`_>~|]", " ", text)

    # CJK tokenization: split on character boundaries for Chinese/Japanese/Korean
    # Extract CJK character sequences and split into bigrams
    cjk_tokens = []
    cjk_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+")
    for match in cjk_pattern.finditer(text):
        seq = match.group()
        # Generate unigrams and bigrams for CJK text
        for ch in seq:
            cjk_tokens.append(ch)
        for i in range(len(seq) - 1):
            cjk_tokens.append(seq[i:i + 2])

    # English tokenization
    eng_tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    eng_tokens = [t for t in eng_tokens if t not in _STOPWORDS]

    return eng_tokens + cjk_tokens


def _generate_snippet(text: str, query_terms: list[str], max_len: int = 200) -> str:
    """Generate a snippet with context around the first matching query term."""
    text_lower = text.lower()
    best_pos = -1

    for term in query_terms:
        pos = text_lower.find(term)
        if pos >= 0:
            best_pos = pos
            break

    if best_pos < 0:
        return text[:max_len] + ("…" if len(text) > max_len else "")

    # Show context around the match
    start = max(0, best_pos - 60)
    end = min(len(text), best_pos + max_len - 60)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "so", "if", "then",
    "than", "more", "also", "into", "about", "which", "what", "when",
    "how", "all", "each", "other", "one", "two", "new", "see", "use",
    "used", "using", "page", "file", "note", "notes", "like", "just",
    "very", "even", "still", "well", "back", "over", "after", "before",
    "between", "through", "during", "without", "within", "along",
    "following", "across", "behind", "beyond", "plus", "except",
    "up", "out", "around", "down", "off", "above", "near",
}
