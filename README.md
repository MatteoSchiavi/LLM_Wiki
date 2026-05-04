# Secondo Cervello v3

**The Ultimate Local AI Knowledge Base.**

Upload PDFs, images, audio, and text → OCR extracts everything → Qwen organizes it into your Obsidian wiki with wikilinks, categories, and typo fixes. All local, all private, zero cloud.

Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

```
http://localhost:7337
```

---

## Quick start

```bash
pip install -r requirements.txt
ollama pull glm-ocr:latest
ollama pull qwen2.5:7b-instruct
# Edit config.yaml → set vault_dir to your Obsidian vault
python main.py
```

---

## What's new in v3

### Technical Optimizations
- **Incremental Compilation**: SHA-256 hashes for all source files. Only new/modified files are processed.
- **Adjacency List**: Wikilink graph built from all .md files. Search retrieves BM25 matches + 1-degree neighbors.
- **Hierarchical Token Management**: Large documents are chunked, entities extracted per chunk, then merged.
- **VRAM Management**: OCR model uses `keep_alive="5m"` during page loop, then explicitly unloaded before Qwen loads.
- **Database Concurrency**: `aiosqlite` with connection-per-operation isolation. No more locking.
- **Non-Blocking Indexing**: BM25 rebuild runs via `asyncio.to_thread()`.

### Architectural Improvements
- **Schema-Driven Prompts**: `SCHEMA.md` in vault root contains all governance rules, injected as LLM system prompt.
- **Strict State Separation**: `raw_sources_log` and `wiki_entities_log` tables track file provenance. Malformed pages can be regenerated.
- **Folders On Demand**: Category folders created only when a document is categorized into them.

### Feature Additions
- **Web Search Ingestion**: Search the web, fetch results, auto-ingest into wiki.
- **Durable Task Ledger**: `pending_tasks.md` tracks auto-generated tasks (create missing pages).
- **Automated Cross-Referencing**: Scans pages for unlinked title matches, adds [[ ]] brackets.
- **Programmatic API**: REST endpoints for external scripts to push data into the vault.
- **Audio Transcription**: MP3/WAV/M4A/FLAC/OGG/WEBM → Whisper → text → wiki.
- **Revise & Organize**: Dashboard button triggers full wiki review (fix links, reorganize, check everything).

---

## Pipeline

```
File upload (PDF/image/audio/TXT/MD)
  → OCR/Transcription (glm-ocr or Whisper, keep_alive=5m)
  → Explicit model unload (keep_alive=0) — frees VRAM
  → Raw .md saved to attachments/
  → SHA-256 hash computed and recorded in DB
  → Qwen loads (qwen2.5:7b-instruct)
      · Reads SCHEMA.md for governance rules
      · Reads existing wiki page list for linking
      · Reads raw OCR (chunked if large)
      · Extracts entities per chunk (hierarchical token mgmt)
      · Fixes typos through context
      · Adds [[wikilinks]] to existing pages
      · Categorizes document into topic folder (on-demand)
      · Creates source + entity + concept pages
      · Appends pending tasks for missing pages
      · Writes .md files to vault
      · Updates index.md and log.md
  → Cross-referencing daemon (auto-adds missing wikilinks)
  → Pending task execution (creates stub pages)
  → Search index incrementally updated
  → Adjacency list rebuilt
```

---

## Vault Structure

```
D:\obsidian\LLM_Wiki/
├── Sources/                    ← On-demand category folders
│   ├── ai-ml/
│   ├── science/
│   ├── technology/
│   └── ...
├── Entities/                   ← People, orgs, tools
├── Concepts/                   ← Ideas, methods, theories
├── MOC/                        ← Maps of Content
├── attachments/                ← Original files + raw OCR
├── .meta/                      ← adjacency.json (graph data)
├── index.md                    ← Master catalog
├── log.md                      ← Append-only operation log
├── SCHEMA.md                   ← Governance rules (LLM reads this)
├── AGENTS.md                   ← AI agent instructions
└── pending_tasks.md            ← Auto-generated task ledger
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload file for processing |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/events` | SSE stream for real-time updates |
| GET | `/api/wiki/tree` | File tree of vault |
| GET | `/api/wiki/page` | Get page content |
| POST | `/api/wiki/reprocess` | Re-run Qwen on a page |
| POST | `/api/query` | Query wiki with streaming LLM |
| POST | `/api/revise` | Trigger full wiki revision |
| POST | `/api/lint` | Health check the wiki |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/health` | Ollama connectivity check |
| GET | `/api/config` | Non-sensitive config |
| POST | `/api/web-search` | Search web & ingest results |
| POST | `/api/v1/ingest` | Push structured content into vault |
| POST | `/api/v1/search` | Programmatic wiki search |
| GET | `/api/v1/graph` | Get adjacency list (wikilink graph) |

---

## Supported Input Formats

| Format | Processing |
|--------|-----------|
| `.pdf` | Full OCR via glm-ocr (parallel pages, VRAM-managed) |
| `.png`, `.jpg`, etc. | Direct OCR via glm-ocr |
| `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm` | Whisper audio transcription |
| `.md` | Direct ingest (Qwen adds links + categorizes) |
| `.txt` | Direct ingest (Qwen adds links + categorizes) |

---

## Project Structure

```
secondo-cervello/
├── main.py           ← FastAPI app: routes, SSE, pipeline, APIs
├── ocr.py            ← PDF/Image OCR + Audio transcription + VRAM mgmt
├── wiki.py           ← Wiki engine: schema-driven, chunking, cross-ref, revise
├── search.py         ← BM25 + adjacency list + incremental indexing
├── db.py             ← aiosqlite database layer with state separation
├── config_loader.py  ← YAML config with deep-merge defaults
├── config.yaml       ← All tunable parameters
├── requirements.txt
├── templates/
│   └── index.html    ← Full SPA UI
└── D:\obsidian\LLM_Wiki/  ← Obsidian vault
```
