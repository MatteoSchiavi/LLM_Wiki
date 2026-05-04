"""
Wiki engine: Qwen reads raw OCR output → creates polished .md pages
with wikilinks, categorization, typo fixes, and proper folder placement.

Operations:
  ingest(raw_text, filename) → structured wiki pages in .md
  query(question, context)   → stream LLM answer
  lint()                     → health-check the wiki
  reprocess(page_path)       → re-run Qwen on an existing page
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

OLLAMA_URL    = CFG["ollama_url"] + "/api/generate"
LLM_MODEL     = CFG["llm_model"]
MAX_CHARS     = CFG["wiki"]["max_source_chars"]
KEEP_ALIVE    = CFG["wiki"]["keep_alive"]
AUTO_ENTITIES = CFG["wiki"]["auto_create_entities"]
AUTO_CONCEPTS = CFG["wiki"]["auto_create_concepts"]
MAX_ENTITIES  = CFG["wiki"]["max_entity_pages"]
MAX_CONCEPTS  = CFG["wiki"]["max_concept_pages"]

# ─── Category map for quick lookup ────────────────────────────────────────────

_CATEGORIES = {cat["slug"]: cat["name"] for cat in CFG["categories"]}
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    (cat["slug"], cat["keywords"]) for cat in CFG["categories"]
]


# ─── Prompts ──────────────────────────────────────────────────────────────────

INGEST_SYSTEM = """\
You are a wiki librarian and knowledge architect for an Obsidian vault. Your job is to process a raw document transcription and create polished, well-organized wiki pages.

THE VAULT STRUCTURE:
  Sources/{category-slug}/{slug}.md  — processed source document
  Entities/{name}.md                 — real-world things: people, organizations, tools, products
  Concepts/{name}.md                 — abstract ideas: theories, methods, techniques, principles
  MOC/{topic}.md                     — Maps of Content (hub pages for major topics)
  index.md                           — master catalog of all pages

YAML FRONTMATTER (required on EVERY page):
  ---
  title: Human Readable Title
  type: source | entity | concept | moc
  category: Category Slug
  tags: [tag1, tag2, tag3]
  sources: ["original_filename.ext"]
  created: "YYYY-MM-DD"
  updated: "YYYY-MM-DD"
  ---

WIKILINKS: Use [[Page Name]] to link between pages.
- Link the FIRST mention of any concept/entity that already has a wiki page
- Do NOT over-link — only link important, substantive references
- Wikilinks create the knowledge graph — they are the most important part
- Use the exact page title (without .md) inside double brackets
- Examples: [[Machine Learning]], [[Attention Mechanism]], [[OpenAI]]

CATEGORIZATION: Choose the best category for source pages:
  ai-ml       — AI, machine learning, deep learning, neural networks, NLP, computer vision
  science     — physics, chemistry, biology, mathematics, academic research
  technology  — software, hardware, programming, web, systems, APIs
  health      — medicine, nutrition, fitness, mental health, clinical
  business    — economics, management, investing, startups, finance
  philosophy  — ethics, consciousness, cognition, philosophy, psychology
  arts        — history, literature, art, music, culture, language
  personal    — personal notes, journals, reflections, goals

TYPO AND ERROR CORRECTION:
- Fix OCR misreads: "rn" → "m", "0" → "O" (context-dependent), "l" → "I" (context-dependent)
- Fix broken or merged words
- Correct heading levels if OCR got them wrong
- Fix missing or wrong punctuation
- Normalize inconsistent formatting
- Do NOT change the meaning — only fix clear errors

PAGE WRITING RULES:
- Source pages: 300-800 words, well-structured with headings
- Entity pages: 150-300 words, factual, concise
- Concept pages: 200-400 words, explanatory, with examples if possible
- Every page must have frontmatter
- Every page must link to related pages with [[wikilinks]]
- Never fabricate information not in the source
- Preserve all important data, numbers, and specific details

OUTPUT FORMAT: Return ONLY valid JSON. No markdown fences, no preamble.
"""

INGEST_USER = """\
EXISTING WIKI PAGES I CAN LINK TO:
{existing_pages}

AVAILABLE CATEGORIES:
{categories}

SOURCE FILENAME: {filename}
PROCESSING DATE: {date}

RAW OCR TRANSCRIPTION (may contain OCR errors — fix them):
{content}

---

TASK: Process this raw transcription into wiki pages.

1. Create ONE source page with the full processed content
2. Fix any typos or OCR errors you detect through context
3. Add [[wikilinks]] to existing pages where relevant
4. Choose the correct category folder for the source page
5. {entity_instruction}
6. {concept_instruction}

Return JSON in this exact schema:
{{
  "source_page": {{
    "title": "Descriptive Title",
    "category_slug": "category-slug",
    "folder": "Sources/category-slug",
    "filename": "slug-name.md",
    "content": "---\\ntitle: ...\\ntype: source\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{date}\\"\\nupdated: \\"{date}\\"\\n---\\n\\n# Title\\n\\nFull content with [[wikilinks]]..."
  }},
  "entity_pages": [
    {{
      "title": "Entity Name",
      "filename": "entity-name.md",
      "content": "---\\ntitle: Entity Name\\ntype: entity\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{date}\\"\\nupdated: \\"{date}\\"\\n---\\n\\n# Entity Name\\n\\nDescription with [[wikilinks]]..."
    }}
  ],
  "concept_pages": [
    {{
      "title": "Concept Name",
      "filename": "concept-name.md",
      "content": "---\\ntitle: Concept Name\\ntype: concept\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{date}\\"\\nupdated: \\"{date}\\"\\n---\\n\\n# Concept Name\\n\\nExplanation with [[wikilinks]]..."
    }}
  ],
  "log_entry": "## [{date}] ingest | Title\\n\\nBrief summary of what was ingested.\\n"
}}
"""

QUERY_SYSTEM = """\
You are a wiki assistant for a personal knowledge base. Answer questions using the provided wiki pages.

RULES:
- Be precise and cite sources using [[Page Name]] notation
- If the wiki doesn't contain relevant info, say so clearly — never fabricate
- Keep answers focused, well-structured, and informative
- Use Markdown formatting for readability
- When referencing specific data, include the exact numbers from the wiki
- If multiple pages provide relevant information, synthesize them coherently
"""

LINT_SYSTEM = """\
You are a wiki health inspector for an Obsidian vault. Analyze the wiki and report:

1. BROKEN LINKS: [[wikilinks]] that point to pages that don't exist
2. ORPHAN PAGES: Pages that are never linked to from other pages
3. MISSING LINKS: Concepts/entities mentioned in text that should be [[wikilinked]] but aren't
4. CONTENT GAPS: Important topics that appear frequently across pages but have no dedicated page
5. DUPLICATES: Pages that cover essentially the same topic
6. CONTRADICTIONS: Information that conflicts between pages
7. STALE PAGES: Pages that haven't been updated despite new relevant information being added
8. CATEGORIZATION: Pages in wrong category folders

Be specific. List actual page names, file paths, and missing links. Format as a clear Markdown report with sections.
"""

REPROCESS_SYSTEM = """\
You are a wiki page processor. You will receive an existing wiki page and your task is to improve it:

1. Fix any typos or errors
2. Add [[wikilinks]] to other existing wiki pages where relevant
3. Improve structure and readability
4. Ensure the frontmatter is correct and complete
5. Update the 'updated' date in frontmatter to today

Do NOT remove any substantive content. Only improve and enhance.

Return the COMPLETE updated page content as a single string in JSON format:
{"content": "full updated markdown content with frontmatter"}
"""


# ─── WikiEngine ───────────────────────────────────────────────────────────────

class WikiEngine:
    def __init__(self, vault: Path, cfg: dict):
        self.vault       = vault
        self.sources_dir = vault / "Sources"
        self.entities_dir = vault / "Entities"
        self.concepts_dir = vault / "Concepts"
        self.moc_dir     = vault / "MOC"
        self.attachments = vault / "attachments"
        self.cfg         = cfg
        self._ensure_scaffold()

    def _ensure_scaffold(self):
        """Create vault directory structure and foundational files if missing."""
        # Category subfolders under Sources/
        for cat in self.cfg.get("categories", []):
            (self.sources_dir / cat["slug"]).mkdir(parents=True, exist_ok=True)

        # Other top-level folders
        for d in [self.entities_dir, self.concepts_dir, self.moc_dir, self.attachments]:
            d.mkdir(parents=True, exist_ok=True)

        # index.md — master catalog
        idx = self.vault / "index.md"
        if not idx.exists():
            idx.write_text(
                "# LLM Wiki Index\n\n"
                "This file is auto-maintained by Secondo Cervello.\n\n"
                "## Sources\n\n"
                "## Entities\n\n"
                "## Concepts\n\n"
                "## Maps of Content\n\n",
                encoding="utf-8",
            )

        # log.md — operation log
        log_p = self.vault / "log.md"
        if not log_p.exists():
            log_p.write_text(
                "# Operation Log\n\n"
                "_Append-only. Format: `## [YYYY-MM-DD] operation | title`_\n\n",
                encoding="utf-8",
            )

        # AGENTS.md — AI schema governance
        agents = self.vault / "AGENTS.md"
        if not agents.exists():
            agents.write_text(
                "# AGENTS.md — Wiki Schema\n\n"
                "This file tells AI agents how the wiki is structured.\n\n"
                "## Structure\n\n"
                "- `Sources/{category}/` — Processed source documents\n"
                "- `Entities/` — Real-world things (people, orgs, tools)\n"
                "- `Concepts/` — Abstract ideas (theories, methods)\n"
                "- `MOC/` — Maps of Content (hub pages)\n"
                "- `index.md` — Master catalog\n"
                "- `log.md` — Operation log\n\n"
                "## Conventions\n\n"
                "- All pages have YAML frontmatter\n"
                "- Wikilinks use `[[Title]]` format\n"
                "- Source pages go in `Sources/{category}/` folders\n"
                "- Never modify files in `attachments/`\n"
                "- The `log.md` is append-only\n"
                "- Preserve and extend — never discard existing knowledge\n",
                encoding="utf-8",
            )

    # ── Page listing for wikilink context ────────────────────────────────────

    def _get_existing_pages(self) -> list[dict]:
        """Get a compact list of all existing wiki pages for linking context."""
        pages = []
        for p in sorted(self.vault.rglob("*.md")):
            if p.name.startswith(".") or p.name in ("index.md", "log.md", "AGENTS.md"):
                continue
            try:
                rel = str(p.relative_to(self.vault))
                title = _extract_title(p.read_text(encoding="utf-8"))
                pages.append({"path": rel, "title": title})
            except Exception:
                pass
        return pages

    def _build_existing_pages_text(self) -> str:
        """Build a text summary of existing pages for the LLM prompt."""
        pages = self._get_existing_pages()
        if not pages:
            return "(No pages yet — this is the first document)"
        lines = []
        for pg in pages:
            lines.append(f"- [[{pg['title']}]] → {pg['path']}")
        return "\n".join(lines)

    # ── File writing ─────────────────────────────────────────────────────────

    def _write_page(self, rel_path: str, content: str) -> Path:
        """Write a wiki page, creating parent dirs as needed."""
        p = (self.vault / rel_path).resolve()
        # Security: ensure path stays within vault
        if not str(p).startswith(str(self.vault.resolve())):
            raise ValueError(f"Path escape attempt: {rel_path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def _update_index(self, source_page: dict, entity_pages: list, concept_pages: list):
        """Update index.md with new page entries."""
        idx_path = self.vault / "index.md"
        current = idx_path.read_text(encoding="utf-8") if idx_path.exists() else ""

        # Source entry
        if source_page:
            title = source_page.get("title", "Untitled")
            folder = source_page.get("folder", "Sources")
            filename = source_page.get("filename", "")
            entry = f"- [[{title}]] — {folder}/{filename}"
            marker = "## Sources"
            if marker in current:
                current = current.replace(marker, marker + "\n" + entry)

        # Entity entries
        for pg in entity_pages:
            title = pg.get("title", "Untitled")
            filename = pg.get("filename", "")
            entry = f"- [[{title}]] — Entities/{filename}"
            marker = "## Entities"
            if marker in current:
                current = current.replace(marker, marker + "\n" + entry)

        # Concept entries
        for pg in concept_pages:
            title = pg.get("title", "Untitled")
            filename = pg.get("filename", "")
            entry = f"- [[{title}]] — Concepts/{filename}"
            marker = "## Concepts"
            if marker in current:
                current = current.replace(marker, marker + "\n" + entry)

        idx_path.write_text(current, encoding="utf-8")

    def _append_log(self, entry: str):
        log_path = self.vault / "log.md"
        current = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        log_path.write_text(current + "\n" + entry.strip() + "\n", encoding="utf-8")

    # ── Categorization ───────────────────────────────────────────────────────

    def _classify_category(self, text: str) -> str:
        """Simple keyword-based category classification as a fallback."""
        text_lower = text.lower()
        best_slug = "uncategorized"
        best_score = 0

        for slug, keywords in _CATEGORY_KEYWORDS:
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_slug = slug

        return best_slug

    # ── Ingest ───────────────────────────────────────────────────────────────

    async def ingest(
        self,
        raw_text: str,
        filename: str,
        job_id: str,
        emit_fn: Callable,
    ) -> int:
        """
        Call Qwen to process raw OCR text → structured wiki pages.
        Returns number of pages written.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        content = raw_text[:MAX_CHARS]
        if len(raw_text) > MAX_CHARS:
            content += f"\n\n[... truncated to {MAX_CHARS} chars ...]"

        existing_pages = self._build_existing_pages_text()
        categories_text = "\n".join(
            f"  {cat['slug']:15s} — {cat['name']}"
            for cat in self.cfg.get("categories", [])
        )

        entity_instruction = (
            f"Create up to {MAX_ENTITIES} entity pages for important real-world things mentioned"
            if AUTO_ENTITIES else "Do NOT create entity pages"
        )
        concept_instruction = (
            f"Create up to {MAX_CONCEPTS} concept pages for important ideas or methods mentioned"
            if AUTO_CONCEPTS else "Do NOT create concept pages"
        )

        user_msg = INGEST_USER.format(
            existing_pages=existing_pages[:4000],
            categories=categories_text,
            filename=filename,
            date=today,
            content=content,
            entity_instruction=entity_instruction,
            concept_instruction=concept_instruction,
        )

        emit_fn(job_id, "status", {
            "status": "running",
            "step": "llm_enriching",
            "detail": f"Calling {LLM_MODEL} for wiki processing…",
        })

        raw_response = await _ollama_call(
            prompt=user_msg,
            system=INGEST_SYSTEM,
            model=LLM_MODEL,
            json_mode=True,
        )

        # ── Parse response ───────────────────────────────────────────────────
        try:
            data = _parse_json(raw_response)
            source_page  = data.get("source_page", {})
            entity_pages = data.get("entity_pages", [])
            concept_pages = data.get("concept_pages", [])
            log_entry    = data.get("log_entry", "")
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}\nRaw:\n{raw_response[:500]}")
            emit_fn(job_id, "warning", {"detail": f"LLM parse error, fallback mode: {e}"})

            # Fallback: create a minimal source page
            slug = _to_slug(Path(filename).stem)
            category = self._classify_category(content)
            source_page = {
                "title": Path(filename).stem.replace("_", " "),
                "category_slug": category,
                "folder": f"Sources/{category}",
                "filename": f"{slug}.md",
                "content": (
                    f"---\ntitle: {Path(filename).stem.replace('_', ' ')}\n"
                    f"type: source\ncategory: {category}\ntags: []\n"
                    f'sources: ["{filename}"]\n'
                    f'created: "{today}"\nupdated: "{today}"\n---\n\n'
                    f"# {Path(filename).stem.replace('_', ' ')}\n\n"
                    f"*OCR extracted content:*\n\n" + content[:3000]
                ),
            }
            entity_pages = []
            concept_pages = []
            log_entry = f"## [{today}] ingest | {filename}\n\nFallback page created.\n"

        # ── Write source page ────────────────────────────────────────────────
        written = 0
        if source_page:
            folder = source_page.get("folder", "Sources/uncategorized")
            fname = source_page.get("filename", "untitled.md")
            cont = source_page.get("content", "")
            if cont:
                rel_path = f"{folder}/{fname}"
                try:
                    self._write_page(rel_path, cont)
                    written += 1
                    emit_fn(job_id, "page_written", {"path": rel_path})
                    log.info(f"  Wrote vault/{rel_path}")
                except Exception as e:
                    log.warning(f"  Could not write source page: {e}")

        # ── Write entity pages ───────────────────────────────────────────────
        for pg in entity_pages[:MAX_ENTITIES]:
            fname = pg.get("filename", "")
            cont = pg.get("content", "")
            if fname and cont:
                rel_path = f"Entities/{fname}"
                try:
                    self._write_page(rel_path, cont)
                    written += 1
                    emit_fn(job_id, "page_written", {"path": rel_path})
                    log.info(f"  Wrote vault/{rel_path}")
                except Exception as e:
                    log.warning(f"  Could not write entity page {fname}: {e}")

        # ── Write concept pages ──────────────────────────────────────────────
        for pg in concept_pages[:MAX_CONCEPTS]:
            fname = pg.get("filename", "")
            cont = pg.get("content", "")
            if fname and cont:
                rel_path = f"Concepts/{fname}"
                try:
                    self._write_page(rel_path, cont)
                    written += 1
                    emit_fn(job_id, "page_written", {"path": rel_path})
                    log.info(f"  Wrote vault/{rel_path}")
                except Exception as e:
                    log.warning(f"  Could not write concept page {fname}: {e}")

        # ── Update index and log ─────────────────────────────────────────────
        if log_entry:
            self._append_log(log_entry)
        self._update_index(source_page, entity_pages, concept_pages)

        return written

    # ── Query ────────────────────────────────────────────────────────────────

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

    # ── Lint ─────────────────────────────────────────────────────────────────

    async def lint(self) -> str:
        """Health-check the wiki. Returns a markdown report."""
        pages = list(self.vault.rglob("*.md"))
        # Filter out non-content files
        pages = [p for p in pages if p.name not in (".obsidian",)]
        if not pages:
            return "Wiki is empty. Upload some documents first."

        # Build compact summary for the LLM
        summary_parts = []
        for p in pages[:50]:
            rel = str(p.relative_to(self.vault))
            try:
                text = p.read_text(encoding="utf-8")
                links = re.findall(r"\[\[([^\]]+)\]\]", text)
                summary_parts.append(f"- **{rel}** — links: {links[:10]}")
            except Exception:
                pass
        summary = "\n".join(summary_parts)

        prompt = f"WIKI PAGES SUMMARY:\n{summary}\n\nRun a health check and produce a markdown report."
        chunks = []
        async for chunk in _ollama_stream(prompt, system=LINT_SYSTEM, model=LLM_MODEL):
            chunks.append(chunk)
        return "".join(chunks)

    # ── Reprocess ────────────────────────────────────────────────────────────

    async def reprocess(self, page_path: str) -> str:
        """Re-run Qwen on an existing page to improve it."""
        full_path = (self.vault / page_path).resolve()
        if not str(full_path).startswith(str(self.vault.resolve())):
            raise ValueError(f"Path escape attempt: {page_path}")
        if not full_path.exists():
            raise FileNotFoundError(f"Page not found: {page_path}")

        existing_content = full_path.read_text(encoding="utf-8")
        existing_pages = self._build_existing_pages_text()
        today = datetime.now().strftime("%Y-%m-%d")

        prompt = (
            f"EXISTING WIKI PAGES I CAN LINK TO:\n{existing_pages[:3000]}\n\n"
            f"TODAY'S DATE: {today}\n\n"
            f"CURRENT PAGE CONTENT:\n{existing_content}"
        )

        raw_response = await _ollama_call(
            prompt=prompt,
            system=REPROCESS_SYSTEM,
            model=LLM_MODEL,
            json_mode=True,
        )

        try:
            data = _parse_json(raw_response)
            new_content = data.get("content", existing_content)
            full_path.write_text(new_content, encoding="utf-8")
            return new_content
        except Exception as e:
            log.error(f"Reprocess failed: {e}")
            raise


# ─── Ollama helpers ───────────────────────────────────────────────────────────

async def _ollama_call(
    prompt: str,
    system: str,
    model: str,
    json_mode: bool = False,
) -> str:
    """Blocking Ollama call. Returns full response string."""
    payload: dict = {
        "model":      model,
        "prompt":     prompt,
        "system":     system,
        "stream":     False,
        "options":    {"temperature": 0.1, "num_ctx": 8192},
        "keep_alive": KEEP_ALIVE,
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
    model: str,
) -> AsyncGenerator[str, None]:
    """Streaming Ollama call. Yields text chunks."""
    payload = {
        "model":      model,
        "prompt":     prompt,
        "system":     system,
        "stream":     True,
        "options":    {"temperature": 0.2, "num_ctx": 8192},
        "keep_alive": KEEP_ALIVE,
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
    """Convert a string to a URL/filename-safe slug."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:80] or "untitled"


def _extract_title(content: str) -> str:
    """Extract the first # heading or frontmatter title."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
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
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)
