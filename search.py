"""
BM25 search over Obsidian vault markdown files.
Pure Python, zero external dependencies, English-only.

Features:
- Title-boosted scoring (2x weight for title matches)
- Context-aware snippet generation around matching terms
- Frontmatter-aware parsing (YAML stripped before tokenizing)
- Incremental add/remove without full rebuild
- Wikilink adjacency list for neighbor-expanded search
- Async rebuild via asyncio.to_thread()
"""

import asyncio
import json
import logging
import math
import re
from pathlib import Path

from config_loader import CFG

log = logging.getLogger("search")

# ── BM25 parameters from config ───────────────────────────────────────────────
K1 = CFG["search"]["k1"]  # 1.5
B  = CFG["search"]["b"]   # 0.75

# ── English stopword list ─────────────────────────────────────────────────────
_STOPWORDS = frozenset({
    # Articles
    "a", "an", "the",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so", "for", "because", "although",
    "though", "while", "whereas", "unless", "until", "since",
    # Prepositions
    "in", "on", "at", "to", "of", "with", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "against", "without", "within", "along",
    "across", "behind", "beyond", "except", "toward", "towards", "upon",
    "around", "among", "throughout", "beside", "beneath", "near",
    # Pronouns
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "this", "that",
    "these", "those", "who", "whom", "whose", "which", "what", "whatever",
    "whoever", "whomever", "whichever",
    # Be / have / do verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    # Modals
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "need", "dare",
    # Other function words
    "not", "no", "nor", "as", "if", "then", "than", "too", "very",
    "just", "also", "more", "most", "much", "many", "some", "any",
    "each", "every", "all", "both", "few", "several", "such",
    "only", "own", "same", "other", "another",
    # Common but uninformative
    "like", "even", "still", "well", "back", "out", "off", "down",
    "here", "there", "when", "where", "how", "why", "now",
    "get", "got", "go", "going", "gone", "went", "come", "came",
    "make", "made", "take", "took", "give", "gave", "say", "said",
    "know", "knew", "think", "thought", "see", "look", "want", "let",
    "try", "use", "used", "using", "put", "set", "keep", "kept",
    "begin", "began", "seem", "help", "show", "shown",
    # Domain-noise words (obsidian/wiki)
    "page", "file", "note", "notes",
})

# ── Regex patterns (compiled once) ────────────────────────────────────────────
_RE_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_RE_IMAGE_REF   = re.compile(r"!\[.*?\]\(.*?\)")
_RE_LINK_REF    = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_WIKILINK    = re.compile(r"\[\[([^\]]+)\]\]")
_RE_MD_SYNTAX   = re.compile(r"[#*`_>~|\-]")
_RE_WORD_SPLIT  = re.compile(r"[^a-z0-9]+")
_RE_HEADING     = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_RE_WIKILINK_PARSE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")


# ── Text processing helpers ───────────────────────────────────────────────────

def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block from text."""
    m = _RE_FRONTMATTER.match(text)
    if m:
        return text[m.end():]
    return text


def _extract_title(text: str) -> str:
    """Extract title from frontmatter 'title:' field or first # heading."""
    # Check frontmatter first
    m = _RE_FRONTMATTER.match(text)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    # Fall back to first heading
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _slug_from_path(path: str) -> str:
    """Derive a page slug from a file path (stem, lowercased, spaces normalized)."""
    stem = Path(path).stem
    return stem.lower().replace("-", " ").replace("_", " ")


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing. English only.
    - Strip frontmatter
    - Strip markdown syntax
    - Lowercase
    - Split on non-word chars
    - Filter stopwords
    - Only keep words with 2+ chars
    """
    text = _strip_frontmatter(text)
    # Strip markdown syntax
    text = _RE_IMAGE_REF.sub(" ", text)
    text = _RE_LINK_REF.sub(r"\1", text)
    text = _RE_WIKILINK.sub(r"\1", text)
    text = _RE_MD_SYNTAX.sub(" ", text)
    # Lowercase and split
    words = _RE_WORD_SPLIT.split(text.lower())
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def _generate_snippet(text: str, query_terms: list[str], max_len: int = 200) -> str:
    """Generate a snippet with context around the best cluster of matching terms."""
    if not text:
        return ""

    text_lower = text.lower()

    # Find positions of all query terms in the text
    match_positions = []
    for term in query_terms:
        start = 0
        while True:
            pos = text_lower.find(term, start)
            if pos < 0:
                break
            match_positions.append((pos, pos + len(term)))
            start = pos + 1

    if not match_positions:
        # No matches: return the beginning of the text
        snippet = text[:max_len]
        return snippet + ("\u2026" if len(text) > max_len else "")

    # Find the best window — the region with the densest cluster of matches
    # Score each possible window start by counting matches that fall within it
    best_start = 0
    best_count = 0
    window_size = max_len

    # Sort match positions
    match_positions.sort()

    # Sliding window: for each match as potential window anchor, count matches
    for i, (m_start, _) in enumerate(match_positions):
        count = 0
        for j in range(i, len(match_positions)):
            if match_positions[j][0] - m_start < window_size:
                count += 1
            else:
                break
        if count > best_count:
            best_count = count
            best_start = max(0, m_start - 40)  # slight left padding

    snippet = text[best_start:best_start + window_size]

    # Clean cut at word boundaries
    if best_start > 0:
        # Don't cut mid-word at the start
        space_pos = snippet.find(" ")
        if space_pos > 0 and space_pos < 30:
            snippet = snippet[space_pos + 1:]
        snippet = "\u2026" + snippet

    if best_start + window_size < len(text):
        # Don't cut mid-word at the end
        last_space = snippet.rfind(" ")
        if last_space > window_size // 2:
            snippet = snippet[:last_space]
        snippet = snippet + "\u2026"

    return snippet


# ── BM25 Search ───────────────────────────────────────────────────────────────

class BM25Search:
    """BM25 search engine over an Obsidian vault of markdown files."""

    def __init__(self, vault_dir: Path):
        self.vault_dir = vault_dir
        # doc_id (int) → document record
        self._docs: dict[int, dict] = {}
        # path (str) → doc_id — for quick lookup by path
        self._path_to_id: dict[str, int] = {}
        # Next available doc id
        self._next_id: int = 0
        # Inverted index: term → {doc_id: tf}
        self._index: dict[str, dict[int, int]] = {}
        # Title inverted index: term → {doc_id: tf}
        self._title_index: dict[str, dict[int, int]] = {}
        # Average document length (body tokens)
        self._avgdl: float = 0.0
        # Total body token count across all docs
        self._total_dl: int = 0
        # IDF cache: term → idf value
        self._idf: dict[str, float] = {}
        # Adjacency list: page_slug → [linked_slug, ...]
        self._adjacency: dict[str, list[str]] = {}

    # ── Rebuild ────────────────────────────────────────────────────────────

    def rebuild(self):
        """Full rebuild of the index from all .md files in the vault."""
        if not self.vault_dir.exists():
            log.warning("Vault directory does not exist: %s", self.vault_dir)
            return

        # Reset all state
        self._docs.clear()
        self._path_to_id.clear()
        self._next_id = 0
        self._index.clear()
        self._title_index.clear()
        self._total_dl = 0
        self._avgdl = 0.0
        self._idf.clear()

        for p in sorted(self.vault_dir.rglob("*.md")):
            # Skip hidden files and .obsidian directory
            if p.name.startswith(".") or ".obsidian" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
                if len(text.strip()) < 20:
                    continue
                rel_path = str(p.relative_to(self.vault_dir))
                self.add_document(rel_path, text)
            except Exception as exc:
                log.debug("Skipping %s: %s", p, exc)

        N = len(self._docs)
        log.info("Search index built: %d pages, %d terms", N, len(self._index))

    async def async_rebuild(self):
        """Non-blocking rebuild using asyncio.to_thread()."""
        await asyncio.to_thread(self.rebuild)

    # ── Incremental add ───────────────────────────────────────────────────

    def add_document(self, path: str, content: str):
        """
        Add a single document to the index incrementally.
        Updates inverted index, avgdl, and IDF efficiently without full rebuild.
        If the path already exists, it is replaced.
        """
        # Remove old version if present
        if path in self._path_to_id:
            self.remove_document(path)

        doc_id = self._next_id
        self._next_id += 1
        self._path_to_id[path] = doc_id

        body_text = _strip_frontmatter(content)
        title = _extract_title(content)
        tokens = _tokenize(body_text)
        title_tokens = _tokenize(title)

        # Store the document record
        self._docs[doc_id] = {
            "path":         path,
            "tokens":       tokens,
            "title_tokens": title_tokens,
            "text":         body_text[:2000],  # keep more text for snippets
            "title":        title or Path(path).stem.replace("-", " ").replace("_", " "),
        }

        dl = len(tokens)
        self._total_dl += dl
        self._avgdl = self._total_dl / len(self._docs) if self._docs else 0.0

        # Update body inverted index
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        for t, f in freq.items():
            self._index.setdefault(t, {})[doc_id] = f

        # Update title inverted index
        title_freq: dict[str, int] = {}
        for t in title_tokens:
            title_freq[t] = title_freq.get(t, 0) + 1
        for t, f in title_freq.items():
            self._title_index.setdefault(t, {})[doc_id] = f

        # Recompute IDF for affected terms
        self._recompute_idf(set(freq.keys()) | set(title_freq.keys()))

    # ── Incremental remove ────────────────────────────────────────────────

    def remove_document(self, path: str):
        """Remove a document from the index by its vault-relative path."""
        doc_id = self._path_to_id.pop(path, None)
        if doc_id is None:
            return

        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return

        dl = len(doc["tokens"])
        self._total_dl -= dl
        self._avgdl = self._total_dl / len(self._docs) if self._docs else 0.0

        # Collect terms that need IDF recomputation
        affected_terms: set[str] = set()

        # Remove from body inverted index
        for t in doc["tokens"]:
            postings = self._index.get(t)
            if postings is not None:
                postings.pop(doc_id, None)
                affected_terms.add(t)
                if not postings:
                    del self._index[t]

        # Remove from title inverted index
        for t in doc["title_tokens"]:
            postings = self._title_index.get(t)
            if postings is not None:
                postings.pop(doc_id, None)
                affected_terms.add(t)
                if not postings:
                    del self._title_index[t]

        # Recompute IDF for affected terms
        self._recompute_idf(affected_terms)

    # ── IDF computation ───────────────────────────────────────────────────

    def _recompute_idf(self, terms: set[str]):
        """Recompute IDF values for a set of terms."""
        N = len(self._docs)
        if N == 0:
            for t in terms:
                self._idf.pop(t, None)
            return
        for t in terms:
            df = len(self._index.get(t, {}))
            if df == 0:
                self._idf.pop(t, None)
            else:
                self._idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _ensure_idf(self):
        """Ensure IDF is computed for all terms (called after bulk rebuild)."""
        N = len(self._docs)
        if N == 0:
            self._idf.clear()
            return
        self._idf = {
            t: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1.0)
            for t, postings in self._index.items()
        }

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 6) -> list[dict]:
        """Return top-k results sorted by BM25 score with title boosting."""
        if not self._docs:
            return []

        q_terms = _tokenize(query)
        if not q_terms:
            return []

        scores: dict[int, float] = {}
        avgdl = self._avgdl or 1.0
        N = len(self._docs)

        # Body scoring — standard BM25
        for term in q_terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            postings = self._index.get(term, {})
            for doc_id, tf in postings.items():
                dl = len(self._docs[doc_id]["tokens"])
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * numerator / denominator

        # Title boosting — matches in title get 2x weight
        for term in q_terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            postings = self._title_index.get(term, {})
            for doc_id, tf in postings.items():
                dl = len(self._docs[doc_id]["tokens"])
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + 2.0 * idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:k]:
            d = self._docs[doc_id]
            snippet = _generate_snippet(d["text"], q_terms)
            results.append({
                "path":    d["path"],
                "title":   d["title"],
                "score":   round(score, 4),
                "snippet": snippet,
            })
        return results

    # ── Adjacency list (wikilink graph) ───────────────────────────────────

    def build_adjacency_list(self) -> dict[str, list[str]]:
        """
        Parse all .md files for [[wikilinks]] and return
        {page_slug: [linked_slug1, linked_slug2, ...]}.
        """
        adjacency: dict[str, list[str]] = {}

        if not self.vault_dir.exists():
            return adjacency

        # Build a map from slug → normalized slug for resolution
        slug_set: set[str] = set()
        for doc_id, doc in self._docs.items():
            slug = _slug_from_path(doc["path"])
            slug_set.add(slug)
            # Also index by title slug
            title_slug = doc["title"].lower().strip()
            if title_slug:
                slug_set.add(title_slug)

        for doc_id, doc in self._docs.items():
            source_slug = _slug_from_path(doc["path"])
            links: list[str] = []

            # Extract wikilinks from the raw text (not the stored snippet)
            p = self.vault_dir / doc["path"]
            if not p.exists():
                continue
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception:
                continue

            for m in _RE_WIKILINK_PARSE.finditer(raw):
                target = m.group(1).strip()
                target_slug = target.lower()
                # Normalize: try to match against known slugs
                if target_slug in slug_set:
                    links.append(target_slug)
                else:
                    # Also try the path-style slug
                    alt_slug = target.lower().replace("-", " ").replace("_", " ")
                    if alt_slug in slug_set:
                        links.append(alt_slug)
                    else:
                        # Keep the link even if target doesn't exist (orphan link)
                        links.append(target_slug)

            adjacency[source_slug] = links

        self._adjacency = adjacency
        log.info("Adjacency list built: %d nodes", len(adjacency))
        return adjacency

    def save_adjacency_list(self, path: Path):
        """Save adjacency list to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._adjacency, f, indent=2, ensure_ascii=False)
        log.info("Adjacency list saved to %s", path)

    def load_adjacency_list(self, path: Path):
        """Load adjacency list from a JSON file."""
        if not path.exists():
            log.warning("Adjacency file not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            self._adjacency = json.load(f)
        log.info("Adjacency list loaded from %s (%d nodes)", path, len(self._adjacency))

    # ── Neighbor-expanded search ──────────────────────────────────────────

    def search_with_neighbors(self, query: str, k: int = 6) -> list[dict]:
        """
        BM25 search + 1-degree neighbor expansion from adjacency list.
        Returns the top-k BM25 results plus any direct neighbors of those
        results, with neighbors getting a discounted score.
        """
        base_results = self.search(query, k)
        if not base_results or not self._adjacency:
            return base_results

        # Map from slug → doc for quick neighbor lookup
        slug_to_doc: dict[str, dict] = {}
        for doc_id, doc in self._docs.items():
            slug_to_doc[_slug_from_path(doc["path"])] = doc

        # Collect already-included paths
        seen_paths: set[str] = {r["path"] for r in base_results}

        # Expand with 1-degree neighbors
        neighbor_results: list[dict] = []
        for result in base_results:
            slug = _slug_from_path(result["path"])
            neighbors = self._adjacency.get(slug, [])
            for neighbor_slug in neighbors:
                # Find the neighbor document
                neighbor_doc = slug_to_doc.get(neighbor_slug)
                if neighbor_doc is None:
                    # Try matching by iterating (slower fallback)
                    for doc_id, doc in self._docs.items():
                        if _slug_from_path(doc["path"]) == neighbor_slug:
                            neighbor_doc = doc
                            break
                if neighbor_doc is None or neighbor_doc["path"] in seen_paths:
                    continue
                seen_paths.add(neighbor_doc["path"])
                # Neighbor gets a discounted score (half of the source result)
                q_terms = _tokenize(query)
                snippet = _generate_snippet(neighbor_doc["text"], q_terms)
                neighbor_results.append({
                    "path":    neighbor_doc["path"],
                    "title":   neighbor_doc["title"],
                    "score":   round(result["score"] * 0.5, 4),
                    "snippet": snippet,
                })

        # Merge and sort: base results first, then neighbors interleaved by score
        all_results = base_results + neighbor_results
        all_results.sort(key=lambda r: r["score"], reverse=True)
        return all_results[:k * 2]  # return up to 2k to include neighbors

    # ── Page snippet ──────────────────────────────────────────────────────

    def get_page_snippet(self, path: str, max_len: int = 200) -> str:
        """Get a short text snippet for a page by its vault-relative path."""
        doc_id = self._path_to_id.get(path)
        if doc_id is not None:
            doc = self._docs[doc_id]
            text = doc["text"]
            if len(text) <= max_len:
                return text
            # Try to cut at a sentence or word boundary
            snippet = text[:max_len]
            last_period = snippet.rfind(".")
            last_space = snippet.rfind(" ")
            if last_period > max_len // 2:
                snippet = snippet[:last_period + 1]
            elif last_space > max_len // 2:
                snippet = snippet[:last_space]
            return snippet + ("\u2026" if len(text) > len(snippet) else "")

        # Fallback: read from disk
        p = self.vault_dir / path
        if not p.exists():
            return ""
        try:
            raw = p.read_text(encoding="utf-8")
            body = _strip_frontmatter(raw)
            if len(body) <= max_len:
                return body
            snippet = body[:max_len]
            last_period = snippet.rfind(".")
            last_space = snippet.rfind(" ")
            if last_period > max_len // 2:
                snippet = snippet[:last_period + 1]
            elif last_space > max_len // 2:
                snippet = snippet[:last_space]
            return snippet + "\u2026"
        except Exception:
            return ""
