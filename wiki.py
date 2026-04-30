"""
Wiki engine implementing the Karpathy LLM-Wiki pattern.

Three operations:
  ingest(raw_text, filename) → write structured wiki pages
  query(question, context)   → stream LLM answer
  lint()                     → health-check the wiki
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx

from config_loader import CFG

log = logging.getLogger("wiki")

OLLAMA_URL = CFG["ollama_url"] + "/api/generate"
LLM_MODEL  = CFG["llm_model"]
MAX_CHARS  = CFG["wiki"]["max_source_chars"]


# ─── Prompts ──────────────────────────────────────────────────────────────────

INGEST_SYSTEM = """\
You are a wiki librarian. Your job is to ingest a source document and generate wiki pages.

WIKI STRUCTURE:
  wiki/sources/{slug}.md  — summary of this source document
  wiki/entities/{name}.md — real-world things: people, places, organizations, products
  wiki/concepts/{name}.md — abstract ideas: theories, methods, techniques, principles

FRONTMATTER (required on every page):
  ---
  title: Human Readable Title
  type: source | entity | concept
  tags: [tag1, tag2]
  sources: ["original_filename.pdf"]
  created: "YYYY-MM-DD"
  ---

WIKILINKS: Use [[Page Name]] to link between pages. The slug is the filename without .md.

RULES:
- Always create exactly 1 source page
- Create 2 to 5 entity pages for important real-world things mentioned
- Create 2 to 5 concept pages for important ideas or methods mentioned
- If an entity/concept is minor, skip it — quality over quantity
- Keep pages concise (200-400 words each)
- Use [[wikilinks]] liberally — they build the graph
- Never fabricate information not in the source

OUTPUT FORMAT: Return ONLY valid JSON, nothing else before or after.
"""

INGEST_USER = """\
EXISTING WIKI INDEX:
{index}

SOURCE FILENAME: {filename}
DATE: {date}

SOURCE CONTENT (may be truncated):
{content}

Return JSON in this exact schema:
{{
  "pages": [
    {{
      "path": "sources/slug-name.md",
      "content": "---\\ntitle: ...\\ntype: source\\ntags: [...]\\nsources: [\"{filename}\"]\\ncreated: \"{date}\"\\n---\\n\\n# Title\\n\\nContent with [[wikilinks]]..."
    }}
  ],
  "log_entry": "## [{date}] ingest | {filename}\\n\\nBrief summary of what was ingested.\\n"
}}
"""

QUERY_SYSTEM = """\
You are a wiki assistant. Answer questions using the provided wiki pages.
- Be precise and cite sources using [[Page Name]] notation
- If the wiki doesn't contain relevant info, say so clearly
- Keep answers focused and well-structured
"""

LINT_SYSTEM = """\
You are a wiki health inspector. Analyze the wiki and report:
1. Pages that mention concepts/entities without linking them ([[wikilinks]])
2. Important topics that appear frequently but have no dedicated page
3. Contradictions or outdated information between pages
4. Orphan pages (never linked to from other pages)
5. Suggestions for new pages that would improve connectivity

Be specific. List actual page names and missing links.
"""


# ─── WikiEngine ───────────────────────────────────────────────────────────────

class WikiEngine:
    def __init__(self, vault: Path, cfg: dict):
        self.vault    = vault
        self.wiki_dir = vault / "wiki"
        self.cfg      = cfg
        self._ensure_scaffold()

    def _ensure_scaffold(self):
        """Create index.md, log.md, overview.md if missing."""
        idx = self.wiki_dir / "index.md"
        if not idx.exists():
            idx.write_text(
                "# Wiki Index\n\nThis file is maintained by the wiki engine.\n\n"
                "## Sources\n\n## Entities\n\n## Concepts\n\n",
                encoding="utf-8",
            )
        log_p = self.wiki_dir / "log.md"
        if not log_p.exists():
            log_p.write_text(
                "# Operation Log\n\n_Append-only. Format: `## [YYYY-MM-DD] op | title`_\n\n",
                encoding="utf-8",
            )

    def _read_index(self) -> str:
        p = self.wiki_dir / "index.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _write_page(self, rel_path: str, content: str) -> Path:
        """Write a wiki page, creating parent dirs as needed."""
        p = (self.wiki_dir / rel_path).resolve()
        if not str(p).startswith(str(self.wiki_dir.resolve())):
            raise ValueError(f"Path escape attempt: {rel_path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def _update_index(self, pages: list[dict]):
        """Append new page entries to index.md."""
        idx_path = self.wiki_dir / "index.md"
        current = idx_path.read_text(encoding="utf-8")

        # Group by type
        groups = {"sources": [], "entities": [], "concepts": []}
        for pg in pages:
            path = pg["path"]
            title = _extract_title(pg["content"])
            link  = f"[[{path.replace('.md', '')}]]"
            entry = f"- {link} — {title}"
            if path.startswith("sources/"):
                groups["sources"].append(entry)
            elif path.startswith("entities/"):
                groups["entities"].append(entry)
            elif path.startswith("concepts/"):
                groups["concepts"].append(entry)

        for section, entries in groups.items():
            if not entries:
                continue
            marker = f"## {section.capitalize()}"
            if marker in current:
                current = current.replace(
                    marker, marker + "\n" + "\n".join(entries)
                )
            else:
                current += f"\n{marker}\n" + "\n".join(entries) + "\n"

        idx_path.write_text(current, encoding="utf-8")

    def _append_log(self, entry: str):
        log_path = self.wiki_dir / "log.md"
        current  = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        log_path.write_text(current + "\n" + entry.strip() + "\n", encoding="utf-8")

    # ── Ingest ────────────────────────────────────────────────────────────────
    async def ingest(
        self,
        raw_text:  str,
        filename:  str,
        job_id:    str,
        emit_fn:   Callable,
    ) -> int:
        """
        Call LLM to enrich raw text → structured wiki pages.
        Returns number of pages written.
        """
        today   = datetime.now().strftime("%Y-%m-%d")
        content = raw_text[:MAX_CHARS]
        if len(raw_text) > MAX_CHARS:
            content += f"\n\n[... truncated to {MAX_CHARS} chars ...]"

        index = self._read_index()

        user_msg = INGEST_USER.format(
            index=index[:3000],
            filename=filename,
            date=today,
            content=content,
        )

        emit_fn(job_id, "status", {"status": "running", "step": "llm_enriching",
                                    "detail": f"Calling {LLM_MODEL}…"})

        raw_response = await _ollama_call(
            prompt=user_msg,
            system=INGEST_SYSTEM,
            model=LLM_MODEL,
            json_mode=True,
        )

        # ── Parse response ───────────────────────────────────────────────────
        try:
            data = _parse_json(raw_response)
            pages     = data.get("pages", [])
            log_entry = data.get("log_entry", "")
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}\nRaw:\n{raw_response[:500]}")
            emit_fn(job_id, "warning", {"detail": f"LLM parse error, fallback mode: {e}"})
            # Fallback: create a minimal source page
            slug = _to_slug(Path(filename).stem)
            pages = [{
                "path": f"sources/{slug}.md",
                "content": (
                    f"---\ntitle: {Path(filename).stem}\ntype: source\n"
                    f"tags: []\nsources: [\"{filename}\"]\ncreated: \"{today}\"\n---\n\n"
                    f"# {Path(filename).stem}\n\n*OCR extracted content:*\n\n"
                    + content[:2000]
                ),
            }]
            log_entry = f"## [{today}] ingest | {filename}\n\nFallback page created.\n"

        # ── Write pages ───────────────────────────────────────────────────────
        written = 0
        for pg in pages:
            path    = pg.get("path", "")
            content = pg.get("content", "")
            if not path or not content:
                continue
            try:
                self._write_page(path, content)
                written += 1
                emit_fn(job_id, "page_written", {"path": path})
                log.info(f"  Wrote wiki/{path}")
            except Exception as e:
                log.warning(f"  Could not write {path}: {e}")

        if log_entry:
            self._append_log(log_entry)
        self._update_index(pages)

        return written

    # ── Query ─────────────────────────────────────────────────────────────────
    async def query(
        self, question: str, context_pages: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Stream an answer from the LLM using wiki pages as context."""
        ctx_parts = []
        for i, pg in enumerate(context_pages, 1):
            ctx_parts.append(f"**[{i}] {pg['path']}**\n```\n{pg['content'][:1500]}\n```")
        context = "\n\n".join(ctx_parts) if ctx_parts else "*(no relevant pages found)*"

        prompt = f"WIKI CONTEXT:\n{context}\n\nQUESTION: {question}"

        async for chunk in _ollama_stream(prompt, system=QUERY_SYSTEM, model=LLM_MODEL):
            yield chunk

    # ── Lint ──────────────────────────────────────────────────────────────────
    async def lint(self) -> str:
        """Health-check the wiki. Returns a markdown report."""
        pages = list(self.wiki_dir.rglob("*.md"))
        if not pages:
            return "Wiki is empty."

        # Build a compact summary of the wiki for the LLM
        summary_parts = []
        for p in pages[:40]:  # cap at 40 pages
            rel = str(p.relative_to(self.wiki_dir))
            text = p.read_text(encoding="utf-8")
            links = re.findall(r"\[\[([^\]]+)\]\]", text)
            summary_parts.append(f"- **{rel}** — links: {links[:8]}")
        summary = "\n".join(summary_parts)

        prompt = f"WIKI PAGES SUMMARY:\n{summary}\n\nRun a health check and produce a markdown report."
        chunks = []
        async for chunk in _ollama_stream(prompt, system=LINT_SYSTEM, model=LLM_MODEL):
            chunks.append(chunk)
        return "".join(chunks)


# ─── Ollama helpers ───────────────────────────────────────────────────────────

async def _ollama_call(
    prompt: str,
    system: str,
    model:  str,
    json_mode: bool = False,
) -> str:
    """Blocking Ollama call. Returns full response string."""
    payload: dict = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient() as client:
        r = await client.post(OLLAMA_URL, json=payload, timeout=300.0)
        r.raise_for_status()
        return r.json()["response"]


async def _ollama_stream(
    prompt: str,
    system: str,
    model:  str,
) -> AsyncGenerator[str, None]:
    """Streaming Ollama call. Yields text chunks."""
    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  True,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", OLLAMA_URL, json=payload, timeout=300.0) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("response"):
                        yield d["response"]
                    if d.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ─── Utilities ────────────────────────────────────────────────────────────────

def _to_slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:60] or "untitled"


def _extract_title(content: str) -> str:
    """Extract the first # heading or frontmatter title."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling common noise."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)
