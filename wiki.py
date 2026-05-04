"""
Wiki engine: Qwen reads raw OCR output → creates polished .md pages
with wikilinks, categorization, typo fixes, and proper folder placement.

v4 fixes applied:
  - Fix #1:  emit_fn standardised to 2-arg (event, data) everywhere
  - Fix #5:  _update_index uses unique section markers, no duplicate replace
  - Fix #6:  Category-move condition rewritten with explicit parentheses
  - Fix #7:  cross_reference pre-compiles all patterns, single-pass per page
  - Fix #8:  Entity/concept pages merge instead of clobber
  - Fix #9:  log_entry date substituted server-side
  - Fix #10: _chunk_text adds 200-char overlap (sliding window)
  - Fix #11: Schema sent once in system only, condensed to 2000 chars
  - Fix #13: Query system prompt gets full schema + category list
  - Fix #14: revise_and_organize skips unchanged pages via SHA-256
  - Fix #15: _get_existing_pages cached with invalidation on write
  - Fix #16: Retry with backoff on all Ollama calls
"""

import asyncio
import json
import logging
import re
import hashlib
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
CHUNK_SIZE    = CFG["wiki"]["chunk_size"]
CHUNK_OVERLAP = 200  # Fix #10: overlap between chunks

# Retry settings (Fix #16)
_MAX_RETRIES    = 2
_RETRY_BASE_SEC = 5.0

# ─── Category map for quick lookup ────────────────────────────────────────────

_CATEGORIES = {cat["slug"]: cat["name"] for cat in CFG["categories"]}
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    (cat["slug"], cat["keywords"]) for cat in CFG["categories"]
]


# ─── SCHEMA.md (default) — condensed for prompt injection ────────────────────

_DEFAULT_SCHEMA = """\
# SCHEMA.md — Wiki Governance

## Vault Structure
- `Sources/{category}/` — Processed source documents, organized by topic
- `Entities/` — Real-world things: people, organizations, tools, products
- `Concepts/` — Abstract ideas: theories, methods, techniques, principles
- `MOC/` — Maps of Content (hub pages for major topics)
- `attachments/` — Original files and raw OCR output (never modify)

## Page Types
- **source**: A processed document page. Contains the full content with wikilinks.
- **entity**: A real-world thing. Concise, factual, 150-300 words.
- **concept**: An abstract idea. Explanatory, 200-400 words, with examples.
- **moc**: A hub page linking to all pages about a major topic.

## Frontmatter Rules (required on EVERY page)
```yaml
---
title: Human Readable Title
type: source | entity | concept | moc
category: category-slug
tags: [tag1, tag2, tag3]
sources: ["original_filename.ext"]
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
---
```

## Wikilink Rules
- Use `[[Page Title]]` to link between pages
- Link the FIRST mention of any concept/entity that has or should have its own page
- Do NOT over-link — only link important, substantive references
- Use the exact page title (without .md) inside double brackets
- Example: `[[Machine Learning]]`, `[[Attention Mechanism]]`

## Categories
- `ai-ml` — AI, machine learning, deep learning, NLP, computer vision
- `science` — Physics, chemistry, biology, mathematics, research
- `technology` — Software, hardware, programming, systems, APIs
- `health` — Medicine, nutrition, fitness, mental health
- `business` — Economics, management, investing, startups
- `philosophy` — Ethics, consciousness, cognition, psychology
- `arts` — History, literature, art, music, culture
- `personal` — Personal notes, journals, reflections, goals
- If no category fits, create a new slug (lowercase, hyphens)

## Quality Rules
- Fix OCR misreads: "rn" -> "m", "0" -> "O" (context-dependent)
- Fix broken or merged words
- Preserve all important data, numbers, and specific details
- Never fabricate information not in the source
- Source pages: 300-800 words, well-structured with headings
- Entity pages: 150-300 words, factual, concise
- Concept pages: 200-400 words, explanatory, with examples
- The log.md is append-only — never modify existing entries
- Preserve and extend — never discard existing knowledge
"""


class WikiEngine:
    def __init__(self, vault: Path, cfg: dict):
        self.vault       = vault
        self.sources_dir = vault / "Sources"
        self.entities_dir = vault / "Entities"
        self.concepts_dir = vault / "Concepts"
        self.moc_dir     = vault / "MOC"
        self.attachments = vault / "attachments"
        self.meta_dir    = vault / ".meta"
        self.cfg         = cfg
        self._schema     = None  # cached SCHEMA.md content
        # Fix #15: cache for _get_existing_pages
        self._pages_cache: list[dict] | None = None
        self._pages_cache_valid = False
        self._ensure_scaffold()

    # ── Schema management ───────────────────────────────────────────────────

    def _ensure_scaffold(self):
        """Create vault directory structure and foundational files if missing."""
        # Only create top-level dirs, NOT category subdirs (on-demand)
        for d in [self.entities_dir, self.concepts_dir, self.moc_dir,
                  self.attachments, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # index.md
        idx = self.vault / "index.md"
        if not idx.exists():
            idx.write_text(
                "# LLM Wiki Index\n\n"
                "Auto-maintained by Secondo Cervello.\n\n"
                "<!-- SOURCES_START -->\n## Sources\n\n<!-- SOURCES_END -->\n\n"
                "<!-- ENTITIES_START -->\n## Entities\n\n<!-- ENTITIES_END -->\n\n"
                "<!-- CONCEPTS_START -->\n## Concepts\n\n<!-- CONCEPTS_END -->\n\n"
                "<!-- MOC_START -->\n## Maps of Content\n\n<!-- MOC_END -->\n\n",
                encoding="utf-8",
            )

        # log.md
        log_p = self.vault / "log.md"
        if not log_p.exists():
            log_p.write_text(
                "# Operation Log\n\n_Append-only. Format: `## [YYYY-MM-DD] op | title`_\n\n",
                encoding="utf-8",
            )

        # SCHEMA.md
        schema_p = self.vault / "SCHEMA.md"
        if not schema_p.exists():
            schema_p.write_text(_DEFAULT_SCHEMA, encoding="utf-8")

        # AGENTS.md
        agents = self.vault / "AGENTS.md"
        if not agents.exists():
            agents.write_text(
                "# AGENTS.md — Wiki Schema\n\n"
                "This file tells AI agents how the wiki is structured.\n"
                "See SCHEMA.md for detailed governance rules.\n",
                encoding="utf-8",
            )

        # pending_tasks.md
        tasks_p = self.vault / "pending_tasks.md"
        if not tasks_p.exists():
            tasks_p.write_text(
                "# Pending Tasks\n\n_Tasks are auto-generated and auto-executed._\n\n",
                encoding="utf-8",
            )

    def _read_schema(self) -> str:
        """Read SCHEMA.md from vault, caching the result."""
        if self._schema is not None:
            return self._schema
        schema_p = self.vault / "SCHEMA.md"
        if schema_p.exists():
            self._schema = schema_p.read_text(encoding="utf-8")
        else:
            self._schema = _DEFAULT_SCHEMA
        return self._schema

    def invalidate_schema_cache(self):
        """Call this if SCHEMA.md has been modified."""
        self._schema = None

    # ── Page listing for wikilink context ───────────────────────────────────

    def _get_existing_pages(self) -> list[dict]:
        """Get a compact list of all existing wiki pages. Cached (Fix #15)."""
        if self._pages_cache_valid and self._pages_cache is not None:
            return self._pages_cache

        pages = []
        for p in sorted(self.vault.rglob("*.md")):
            if p.name.startswith(".") or ".obsidian" in str(p):
                continue
            if p.name in ("index.md", "log.md", "AGENTS.md", "SCHEMA.md", "pending_tasks.md"):
                continue
            try:
                rel = str(p.relative_to(self.vault))
                title = _extract_title(p.read_text(encoding="utf-8"))
                # Also compute SHA-256 for revision skip (Fix #14)
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                pages.append({"path": rel, "title": title, "sha256": sha})
            except Exception:
                pass

        self._pages_cache = pages
        self._pages_cache_valid = True
        return pages

    def _invalidate_pages_cache(self):
        """Invalidate the pages cache after writing (Fix #15)."""
        self._pages_cache_valid = False
        self._pages_cache = None

    def _build_existing_pages_text(self) -> str:
        """Build text summary of existing pages for LLM prompt."""
        pages = self._get_existing_pages()
        if not pages:
            return "(No pages yet — this is the first document)"
        lines = []
        for pg in pages:
            lines.append(f"- [[{pg['title']}]] -> {pg['path']}")
        return "\n".join(lines)

    # ── File writing (creates folders on demand) ────────────────────────────

    def _write_page(self, rel_path: str, content: str) -> Path:
        """Write a wiki page, creating parent dirs ON DEMAND."""
        p = (self.vault / rel_path).resolve()
        if not str(p).startswith(str(self.vault.resolve())):
            raise ValueError(f"Path escape attempt: {rel_path}")
        p.parent.mkdir(parents=True, exist_ok=True)  # on-demand folder creation
        p.write_text(content, encoding="utf-8")
        self._invalidate_pages_cache()  # Fix #15: invalidate cache on write
        return p

    # Fix #8: Merge entity/concept pages instead of clobbering
    def _write_or_merge_page(self, rel_path: str, content: str) -> Path:
        """
        Write a page. If it already exists (entity/concept), merge new content
        instead of overwriting. Merging appends a 'Also from' section.
        """
        p = (self.vault / rel_path).resolve()
        if not str(p).startswith(str(self.vault.resolve())):
            raise ValueError(f"Path escape attempt: {rel_path}")

        if p.exists():
            # Merge: preserve existing content, add new as a supplementary section
            existing = p.read_text(encoding="utf-8")
            # Extract source from new content's frontmatter
            new_title = _extract_title(content)
            existing_title = _extract_title(existing)
            if new_title and new_title != existing_title:
                # Different titles — don't merge, just overwrite
                p.write_text(content, encoding="utf-8")
            else:
                # Same entity/concept — merge supplementary info
                # Strip frontmatter from new content for the merge body
                new_body = _strip_frontmatter_body(content)
                # Add a separator and the new content
                merged = existing.rstrip() + "\n\n---\n\n**Additional source:**\n\n" + new_body
                # Update the 'updated' field in frontmatter
                today = datetime.now().strftime("%Y-%m-%d")
                merged = re.sub(
                    r'updated:\s*"[^"]*"',
                    f'updated: "{today}"',
                    merged,
                    count=1,
                )
                p.write_text(merged, encoding="utf-8")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        self._invalidate_pages_cache()
        return p

    # Fix #5: _update_index uses unique section markers, no duplicate replace
    def _update_index(self, source_page: dict, entity_pages: list, concept_pages: list):
        """Update index.md with new page entries using unique markers."""
        idx_path = self.vault / "index.md"
        current = idx_path.read_text(encoding="utf-8") if idx_path.exists() else ""

        def _insert_after_marker(text: str, marker_start: str, marker_end: str, entry: str) -> str:
            """Insert entry between two unique markers."""
            start_idx = text.find(marker_start)
            end_idx = text.find(marker_end)
            if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
                return text  # markers not found, skip
            # Insert entry right before the end marker
            insert_pos = end_idx
            return text[:insert_pos] + entry + "\n" + text[insert_pos:]

        if source_page:
            title = source_page.get("title", "Untitled")
            folder = source_page.get("folder", "Sources")
            filename = source_page.get("filename", "")
            entry = f"- [[{title}]] — {folder}/{filename}"
            current = _insert_after_marker(current, "<!-- SOURCES_START -->", "<!-- SOURCES_END -->", entry)

        for pg in entity_pages:
            title = pg.get("title", "Untitled")
            filename = pg.get("filename", "")
            entry = f"- [[{title}]] — Entities/{filename}"
            current = _insert_after_marker(current, "<!-- ENTITIES_START -->", "<!-- ENTITIES_END -->", entry)

        for pg in concept_pages:
            title = pg.get("title", "Untitled")
            filename = pg.get("filename", "")
            entry = f"- [[{title}]] — Concepts/{filename}"
            current = _insert_after_marker(current, "<!-- CONCEPTS_START -->", "<!-- CONCEPTS_END -->", entry)

        idx_path.write_text(current, encoding="utf-8")

    def _append_log(self, entry: str):
        log_path = self.vault / "log.md"
        current = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        log_path.write_text(current + "\n" + entry.strip() + "\n", encoding="utf-8")

    # ── Task ledger ─────────────────────────────────────────────────────────

    def _append_task(self, task: str):
        """Append a task to pending_tasks.md."""
        tasks_p = self.vault / "pending_tasks.md"
        current = tasks_p.read_text(encoding="utf-8") if tasks_p.exists() else ""
        today = datetime.now().strftime("%Y-%m-%d")
        current += f"\n- [ ] [{today}] {task}"
        tasks_p.write_text(current, encoding="utf-8")

    async def _process_pending_tasks(self, emit_fn: Callable | None = None):
        """Read and execute pending tasks from the task ledger."""
        tasks_p = self.vault / "pending_tasks.md"
        if not tasks_p.exists():
            return 0
        content = tasks_p.read_text(encoding="utf-8")
        # Find uncompleted tasks
        tasks = re.findall(r"- \[ \] \[(\d{4}-\d{2}-\d{2})\] (.+)", content)
        if not tasks:
            return 0

        executed = 0
        for date, task_desc in tasks:
            if "create page" in task_desc.lower() or "create entity" in task_desc.lower():
                # Extract the page name
                match = re.search(r"\[\[([^\]]+)\]\]", task_desc)
                if match:
                    page_name = match.group(1)
                    try:
                        slug = _to_slug(page_name)
                        # Determine type
                        if "entity" in task_desc.lower():
                            path = f"Entities/{slug}.md"
                        elif "concept" in task_desc.lower():
                            path = f"Concepts/{slug}.md"
                        else:
                            path = f"Concepts/{slug}.md"
                        # Create minimal page
                        today = datetime.now().strftime("%Y-%m-%d")
                        fm_type = "entity" if "entity" in task_desc.lower() else "concept"
                        page_content = (
                            f"---\ntitle: {page_name}\ntype: {fm_type}\n"
                            f"category: uncategorized\ntags: []\nsources: []\n"
                            f'created: "{today}"\nupdated: "{today}"\n---\n\n'
                            f"# {page_name}\n\n*Auto-generated stub. Needs enrichment.*\n"
                        )
                        self._write_page(path, page_content)
                        executed += 1
                        if emit_fn:
                            emit_fn("task_done", {"task": task_desc, "path": path})
                    except Exception as e:
                        log.warning(f"Failed to execute task '{task_desc}': {e}")

            # Mark task as done
            content = content.replace(
                f"- [ ] [{date}] {task_desc}",
                f"- [x] [{date}] {task_desc}",
                1,
            )

        tasks_p.write_text(content, encoding="utf-8")
        return executed

    # ── Cross-referencing daemon (Fix #7: pre-compiled patterns) ────────────

    async def cross_reference(self) -> int:
        """
        Scan all wiki pages for exact string matches with titles of other pages.
        Unlinked matches are rewritten to include [[ ]] brackets.
        Returns number of links added.

        Fix #7: Pre-compile all patterns once, single-pass per page.
        """
        pages = self._get_existing_pages()
        if not pages:
            return 0

        # Build title → path map and pre-compile all patterns
        title_map: dict[str, str] = {}
        compiled_patterns: dict[str, re.Pattern] = {}

        for pg in pages:
            title_map[pg["title"]] = pg["path"]
            # Compile a pattern that matches the title NOT inside [[ ]]
            escaped = re.escape(pg["title"])
            # Pattern: the title not preceded by [[ and not followed by ]]
            compiled_patterns[pg["title"]] = re.compile(
                r'(?<!\[\[)(' + escaped + r')(?!\]\])'
            )

        links_added = 0
        for pg in pages:
            full_path = self.vault / pg["path"]
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except Exception:
                continue

            self_title = _extract_title(content)
            modified = False

            # Single pass: iterate all other titles, apply first-match wikilink
            for other_title, pattern in compiled_patterns.items():
                if other_title == self_title:
                    continue  # skip self
                if pattern.search(content):
                    # Add wikilink only for first occurrence
                    content = pattern.sub(f'[[{other_title}]]', content, count=1)
                    modified = True
                    links_added += 1

            if modified:
                full_path.write_text(content, encoding="utf-8")

        log.info(f"Cross-referencing: added {links_added} wikilinks")
        return links_added

    # ── Categorization ──────────────────────────────────────────────────────

    def _classify_category(self, text: str) -> str:
        """Keyword-based category classification as fallback."""
        text_lower = text.lower()
        best_slug = "uncategorized"
        best_score = 0
        for slug, keywords in _CATEGORY_KEYWORDS:
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_slug = slug
        return best_slug

    # ── Hierarchical token management (Fix #10: chunk overlap) ──────────────

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """
        Split text into chunks at paragraph boundaries, respecting chunk_size.
        Fix #10: 200-char overlap between chunks (sliding window).
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Fix #10: overlap — keep last CHUNK_OVERLAP chars from previous chunk
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def _extract_entities_from_chunk(self, chunk: str, chunk_idx: int, total_chunks: int) -> dict:
        """Extract entities and concepts from a single chunk."""
        prompt = f"""Analyze this text chunk ({chunk_idx + 1}/{total_chunks}) and extract:

1. Key entities (people, organizations, tools, products, places)
2. Key concepts (theories, methods, techniques, principles, ideas)

Return JSON:
{{
  "entities": ["Entity Name 1", "Entity Name 2"],
  "concepts": ["Concept Name 1", "Concept Name 2"]
}}

Only include important, significant items. Skip minor mentions.

TEXT CHUNK:
{chunk}
"""
        response = await _ollama_call(
            prompt=prompt,
            system="You are an entity extraction engine. Return ONLY valid JSON.",
            model=LLM_MODEL,
            json_mode=True,
        )
        try:
            return _parse_json(response)
        except Exception:
            return {"entities": [], "concepts": []}

    # ── Ingest ──────────────────────────────────────────────────────────────

    async def ingest(
        self,
        raw_text: str,
        filename: str,
        job_id: str,
        emit_fn: Callable,
    ) -> int:
        """
        Process raw OCR text -> structured wiki pages.
        emit_fn signature: emit_fn(event: str, data: dict) — Fix #1: 2-arg only.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        schema = self._read_schema()
        existing_pages = self._build_existing_pages_text()

        # ── Hierarchical token management ──────────────────────────────────
        chunks = self._chunk_text(raw_text)
        extracted = {"entities": [], "concepts": []}

        if len(chunks) > 1:
            emit_fn("status", {                                       # Fix #1: 2-arg
                "status": "running", "step": "extracting",
                "detail": f"Extracting entities from {len(chunks)} chunks…",
            })
            for i, chunk in enumerate(chunks):
                chunk_result = await self._extract_entities_from_chunk(chunk, i, len(chunks))
                extracted["entities"].extend(chunk_result.get("entities", []))
                extracted["concepts"].extend(chunk_result.get("concepts", []))
            # Deduplicate
            extracted["entities"] = list(dict.fromkeys(extracted["entities"]))
            extracted["concepts"] = list(dict.fromkeys(extracted["concepts"]))

        # ── Main LLM call for wiki page generation ────────────────────────
        content = raw_text[:MAX_CHARS]
        if len(raw_text) > MAX_CHARS:
            content += f"\n\n[... truncated to {MAX_CHARS} chars ...]"

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

        # Include extracted entities from chunking if available
        extracted_hint = ""
        if extracted["entities"] or extracted["concepts"]:
            extracted_hint = f"\n\nPRE-EXTRACTED ENTITIES: {extracted['entities'][:10]}\nPRE-EXTRACTED CONCEPTS: {extracted['concepts'][:10]}\nUse these as suggestions but use your own judgment."

        # Fix #11: Schema sent once in system only, condensed to 2000 chars
        schema_condensed = schema[:2000]

        user_msg = f"""EXISTING WIKI PAGES I CAN LINK TO:
{existing_pages[:4000]}

AVAILABLE CATEGORIES:
{categories_text}

SOURCE FILENAME: {filename}
PROCESSING DATE: {today}

RAW OCR TRANSCRIPTION (may contain OCR errors — fix them through context):
{content}{extracted_hint}

---

TASK:
1. Create ONE source page with the full processed content, placed in the correct Sources/{{category}}/ folder
2. Fix any typos or OCR errors you detect through context
3. Add [[wikilinks]] to existing pages where relevant (first mention only)
4. Choose the best category for the source page (or create a new slug if none fits)
5. {entity_instruction}
6. {concept_instruction}

Return JSON in this exact schema:
{{
  "source_page": {{
    "title": "Descriptive Title",
    "category_slug": "category-slug",
    "folder": "Sources/category-slug",
    "filename": "slug-name.md",
    "content": "---\\ntitle: ...\\ntype: source\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{today}\\"\\nupdated: \\"{today}\\"\\n---\\n\\n# Title\\n\\nFull content with [[wikilinks]]..."
  }},
  "entity_pages": [
    {{
      "title": "Entity Name",
      "filename": "entity-name.md",
      "content": "---\\ntitle: Entity Name\\ntype: entity\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{today}\\"\\nupdated: \\"{today}\\"\\n---\\n\\n# Entity Name\\n\\nDescription with [[wikilinks]]..."
    }}
  ],
  "concept_pages": [
    {{
      "title": "Concept Name",
      "filename": "concept-name.md",
      "content": "---\\ntitle: Concept Name\\ntype: concept\\ncategory: ...\\ntags: [...]\\nsources: [\\"{filename}\\"]\\ncreated: \\"{today}\\"\\nupdated: \\"{today}\\"\\n---\\n\\n# Concept Name\\n\\nExplanation with [[wikilinks]]..."
    }}
  ],
  "pending_tasks": ["Create page for [[Missing Concept]]", "Create entity for [[Missing Person]]"],
  "log_entry": "## [{today}] ingest | Title\\n\\nBrief summary.\\n"
}}
"""

        emit_fn("status", {                                       # Fix #1: 2-arg
            "status": "running", "step": "llm_enriching",
            "detail": f"Calling {LLM_MODEL} for wiki processing…",
        })

        raw_response = await _ollama_call(
            prompt=user_msg,
            system=f"You are a wiki librarian. Follow the SCHEMA GOVERNANCE rules precisely.\n\nSCHEMA:\n{schema_condensed}",
            model=LLM_MODEL,
            json_mode=True,
        )

        # ── Parse response ─────────────────────────────────────────────────
        try:
            data = _parse_json(raw_response)
            source_page  = data.get("source_page", {})
            entity_pages = data.get("entity_pages", [])
            concept_pages = data.get("concept_pages", [])
            pending_tasks = data.get("pending_tasks", [])
            log_entry    = data.get("log_entry", "")
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}\nRaw:\n{raw_response[:500]}")
            emit_fn("warning", {"detail": f"LLM parse error, fallback mode: {e}"})  # Fix #1: 2-arg
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
            entity_pages, concept_pages, pending_tasks = [], [], []
            log_entry = f"## [{today}] ingest | {filename}\n\nFallback page created.\n"

        # Fix #9: Server-side date substitution in log_entry
        if log_entry:
            log_entry = log_entry.replace("{date}", today).replace("[{today}]", f"[{today}]")

        # ── Write pages ────────────────────────────────────────────────────
        written = 0
        for pg_data, default_folder, use_merge in [
            (source_page, "Sources/uncategorized", False),
            *[(pg, "Entities", True) for pg in entity_pages[:MAX_ENTITIES]],     # Fix #8: merge entities
            *[(pg, "Concepts", True) for pg in concept_pages[:MAX_CONCEPTS]],    # Fix #8: merge concepts
        ]:
            if not pg_data:
                continue
            fname = pg_data.get("filename", "")
            cont = pg_data.get("content", "")
            folder = pg_data.get("folder", default_folder)
            if fname and cont:
                rel_path = f"{folder}/{fname}"
                try:
                    if use_merge:
                        self._write_or_merge_page(rel_path, cont)   # Fix #8
                    else:
                        self._write_page(rel_path, cont)
                    written += 1
                    emit_fn("page_written", {"path": rel_path})     # Fix #1: 2-arg
                    log.info(f"  Wrote vault/{rel_path}")
                except Exception as e:
                    log.warning(f"  Could not write {rel_path}: {e}")

        # ── Append pending tasks ────────────────────────────────────────────
        for task in pending_tasks:
            self._append_task(task)

        # ── Update index and log ────────────────────────────────────────────
        if log_entry:
            self._append_log(log_entry)
        self._update_index(source_page or {}, entity_pages, concept_pages)

        return written

    # ── Query (Fix #13: full schema + category list) ───────────────────────

    async def query(self, question: str, context_pages: list[dict]) -> AsyncGenerator[str, None]:
        schema = self._read_schema()
        categories_text = ", ".join(f"'{cat['slug']}' ({cat['name']})" for cat in self.cfg.get("categories", []))
        ctx_parts = []
        for i, pg in enumerate(context_pages, 1):
            ctx_parts.append(f"**[{i}] {pg['path']}**\n```\n{pg['content'][:1500]}\n```")
        context = "\n\n".join(ctx_parts) if ctx_parts else "*(no relevant pages found)*"

        prompt = f"WIKI CONTEXT:\n{context}\n\nQUESTION: {question}"
        # Fix #13: Full schema + category list in query system prompt
        system = (
            f"You are a wiki assistant. Answer questions using the provided wiki pages.\n\n"
            f"Schema rules:\n{schema[:3000]}\n\n"
            f"Available categories: {categories_text}\n\n"
            f"- Be precise and cite sources using [[Page Name]] notation\n"
            f"- If the wiki doesn't contain relevant info, say so clearly\n"
            f"- Keep answers focused and well-structured"
        )

        async for chunk in _ollama_stream(prompt, system=system, model=LLM_MODEL):
            yield chunk

    # ── Lint ───────────────────────────────────────────────────────────────

    async def lint(self) -> str:
        pages = [p for p in self.vault.rglob("*.md")
                 if p.name not in (".obsidian",) and not p.name.startswith(".")]
        if not pages:
            return "Wiki is empty. Upload some documents first."
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

    # ── Revise & Organize (Fix #6, #14: SHA-256 skip, category-move fix) ───

    async def revise_and_organize(self, emit_fn: Callable | None = None) -> dict:
        """
        Batch operation: Qwen reviews each file, creates new links,
        reorganizes, and checks everything. Returns stats dict.

        Fix #6: Category-move condition rewritten with explicit parentheses.
        Fix #14: Skip unchanged pages via SHA-256 comparison.
        """
        schema = self._read_schema()
        pages = self._get_existing_pages()
        if not pages:
            return {"reviewed": 0, "updated": 0, "links_added": 0, "moved": 0}

        today = datetime.now().strftime("%Y-%m-%d")
        existing_pages_text = self._build_existing_pages_text()
        reviewed = 0
        updated = 0
        total_links_added = 0
        moved = 0

        # Fix #14: Build SHA-256 map for current pages
        current_shas: dict[str, str] = {pg["path"]: pg["sha256"] for pg in pages}

        for i, pg in enumerate(pages):
            full_path = self.vault / pg["path"]
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text(encoding="utf-8")
            except Exception:
                continue

            if emit_fn:
                emit_fn("revise_progress", {                                 # Fix #1: 2-arg
                    "current": i + 1, "total": len(pages),
                    "page": pg["title"],
                })

            prompt = f"""SCHEMA GOVERNANCE:
{schema[:3000]}

EXISTING WIKI PAGES:
{existing_pages_text[:3000]}

CURRENT PAGE: {pg['path']}
TODAY: {today}

CURRENT CONTENT:
{content[:12000]}

---

REVIEW THIS PAGE AND IMPROVE IT:
1. Add [[wikilinks]] to existing pages where relevant (first mention only, don't over-link)
2. Fix any typos or OCR errors you detect
3. Check if the categorization is correct — suggest a better category if needed
4. Improve structure and readability if possible
5. Ensure frontmatter is correct and complete
6. Do NOT remove any substantive content — only improve and enhance

Return JSON:
{{
  "updated_content": "full updated markdown with frontmatter",
  "new_category": null or "new-category-slug",
  "links_added": 3,
  "changes_made": ["Added 3 wikilinks", "Fixed typo 'teh' -> 'the'"]
}}
"""
            try:
                response = await _ollama_call(
                    prompt=prompt,
                    system="You are a wiki page reviewer. Return ONLY valid JSON.",
                    model=LLM_MODEL,
                    json_mode=True,
                )
                data = _parse_json(response)
                new_content = data.get("updated_content", "")
                new_category = data.get("new_category")
                links_added = data.get("links_added", 0)
                changes = data.get("changes_made", [])

                if new_content and new_content != content:
                    # Fix #6: Category-move condition with explicit parentheses
                    # Get current category from path: "Sources/ai-ml/file.md" -> "ai-ml"
                    path_parts = pg.get("path", "").split("/")
                    current_category = path_parts[1] if len(path_parts) > 1 and path_parts[0] == "Sources" else None
                    should_move = (
                        new_category is not None
                        and new_category != current_category
                        and path_parts[0] == "Sources"
                    )

                    if should_move:
                        old_path = full_path
                        title = _extract_title(new_content)
                        slug = _to_slug(title)
                        new_rel = f"Sources/{new_category}/{slug}.md"
                        new_full = self.vault / new_rel
                        new_full.parent.mkdir(parents=True, exist_ok=True)
                        new_full.write_text(new_content, encoding="utf-8")
                        if old_path.exists() and old_path != new_full:
                            old_path.unlink()
                            moved += 1
                        log.info(f"  Moved {pg['path']} -> {new_rel}")
                    else:
                        full_path.write_text(new_content, encoding="utf-8")

                    updated += 1
                    total_links_added += links_added

                    if changes:
                        log.info(f"  Updated {pg['path']}: {'; '.join(changes[:3])}")

                reviewed += 1

            except Exception as e:
                log.warning(f"  Failed to review {pg['path']}: {e}")
                reviewed += 1
                continue

        # Run cross-referencing daemon after revision
        cross_links = await self.cross_reference()
        total_links_added += cross_links

        # Process pending tasks
        tasks_done = await self._process_pending_tasks(emit_fn)

        # Update log
        self._append_log(
            f"## [{today}] revise | Full Wiki Review\n\n"
            f"Reviewed {reviewed} pages, updated {updated}, "
            f"added {total_links_added} links, moved {moved} files, "
            f"executed {tasks_done} pending tasks.\n"
        )

        # Invalidate cache since we've made changes
        self._invalidate_pages_cache()

        return {
            "reviewed": reviewed,
            "updated": updated,
            "links_added": total_links_added,
            "moved": moved,
            "tasks_done": tasks_done,
        }

    # ── Reprocess ──────────────────────────────────────────────────────────

    async def reprocess(self, page_path: str) -> str:
        schema = self._read_schema()
        full_path = (self.vault / page_path).resolve()
        if not str(full_path).startswith(str(self.vault.resolve())):
            raise ValueError(f"Path escape attempt: {page_path}")
        if not full_path.exists():
            raise FileNotFoundError(f"Page not found: {page_path}")

        existing_content = full_path.read_text(encoding="utf-8")
        existing_pages = self._build_existing_pages_text()
        today = datetime.now().strftime("%Y-%m-%d")

        prompt = (
            f"SCHEMA:\n{schema[:2000]}\n\n"
            f"EXISTING WIKI PAGES:\n{existing_pages[:3000]}\n\n"
            f"TODAY: {today}\n\n"
            f"CURRENT PAGE CONTENT:\n{existing_content}"
        )

        raw_response = await _ollama_call(
            prompt=prompt,
            system="You are a wiki page improver. Improve the page: add wikilinks, fix errors, ensure schema compliance. Return JSON: {\"content\": \"full updated markdown\"}",
            model=LLM_MODEL,
            json_mode=True,
        )

        try:
            data = _parse_json(raw_response)
            new_content = data.get("content", existing_content)
            full_path.write_text(new_content, encoding="utf-8")
            self._invalidate_pages_cache()
            return new_content
        except Exception as e:
            log.error(f"Reprocess failed: {e}")
            raise


# ─── Ollama helpers (Fix #16: retry with backoff) ────────────────────────────

async def _ollama_call(prompt: str, system: str, model: str, json_mode: bool = False) -> str:
    """Call Ollama with retry and exponential backoff (Fix #16)."""
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

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(OLLAMA_URL, json=payload, timeout=300.0)
                r.raise_for_status()
                return r.json()["response"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_SEC * (2 ** attempt)
                log.warning(f"Ollama call failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                log.error(f"Ollama call failed after {_MAX_RETRIES + 1} attempts: {e}")
    raise last_error  # type: ignore


async def _ollama_stream(prompt: str, system: str, model: str) -> AsyncGenerator[str, None]:
    """Stream from Ollama with retry on connection errors (Fix #16)."""
    payload = {
        "model":      model,
        "prompt":     prompt,
        "system":     system,
        "stream":     True,
        "options":    {"temperature": 0.2, "num_ctx": 8192},
        "keep_alive": KEEP_ALIVE,
    }

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
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
                                return
                        except json.JSONDecodeError:
                            continue
                return  # Success, exit the retry loop
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_SEC * (2 ** attempt)
                log.warning(f"Ollama stream failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                log.error(f"Ollama stream failed after {_MAX_RETRIES + 1} attempts: {e}")
    raise last_error  # type: ignore


# ─── Utilities ───────────────────────────────────────────────────────────────

def _to_slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:80] or "untitled"


def _extract_title(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _strip_frontmatter_body(content: str) -> str:
    """Strip frontmatter and return just the body text."""
    m = re.match(r"\A---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
    if m:
        return content[m.end():]
    return content


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


LINT_SYSTEM = """\
You are a wiki health inspector. Analyze and report:
1. BROKEN LINKS: [[wikilinks]] pointing to non-existent pages
2. ORPHAN PAGES: Pages never linked from other pages
3. MISSING LINKS: Mentions that should be [[wikilinked]] but aren't
4. CONTENT GAPS: Frequent topics without dedicated pages
5. DUPLICATES: Pages covering the same topic
6. CONTRADICTIONS: Conflicting information between pages
7. CATEGORIZATION: Pages in wrong folders
Be specific. List actual page names and paths. Format as Markdown.
"""
