# 🧠 Secondo Cervello

**One process. One port. PDF → Wiki → Knowledge Graph.**  
Local, private, zero cloud. Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

```
http://localhost:7337
      ↑
  FastAPI app
      │
  ┌───┴────────────────────────────────────────┐
  │  Upload panel  →  OCR (glm-ocr)            │
  │                →  Wiki Engine (qwen2.5:7b) │
  │                →  vault/wiki/*.md           │
  │                                            │
  │  Wiki panel    →  browse + wikilinks        │
  │  Query panel   →  BM25 search + LLM stream │
  │  Dashboard     →  stats + lint             │
  └────────────────────────────────────────────┘
```

---

## Quick start

### 1. Install Python deps

```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
# OCR (vision) — handles scanned PDFs and handwriting
ollama pull glm-ocr:latest

# Wiki enrichment + query — RECOMMENDED for RTX 3070 Laptop (8GB VRAM)
ollama pull qwen2.5:7b-instruct
```

> **Why qwen2.5:7b?**
> At q8_0 it fits in ~7.7GB. Outperforms Mistral 7B and Llama 3.1 8B on structured
> tasks (JSON output, Italian, instruction-following). glm-ocr and qwen2.5 never
> run in parallel — OCR finishes before the wiki LLM starts — so no VRAM conflict.

### 3. Run

```bash
python main.py
# → http://localhost:7337
```

---

## How it works (Karpathy 3-layer pattern)

```
vault/
├── raw/sources/{doc}/          ← immutable originals (never modified)
│   ├── document.pdf
│   └── document_raw.md         ← raw OCR output
│
└── wiki/                       ← LLM-owned, always evolving
    ├── index.md                ← catalog of all pages + 1-line summaries
    ├── log.md                  ← append-only operation log
    ├── sources/{slug}.md       ← 1 page per ingested document
    ├── entities/{name}.md      ← real things: people, places, orgs
    └── concepts/{name}.md      ← ideas: theories, methods, techniques
```

**Ingest flow:**
1. Drop PDF in the Upload panel
2. glm-ocr converts pages to Markdown (parallel, semaphore-controlled)
3. qwen2.5:7b reads the raw text, generates structured wiki pages as JSON
4. Pages are written to `vault/wiki/`, index.md and log.md updated
5. BM25 search index rebuilt automatically

**Query flow:**
1. Your question is tokenized
2. BM25 finds top-6 relevant wiki pages (sub-millisecond)
3. Pages are sent to qwen2.5:7b as context
4. Answer streams back to the UI

---

## Tuning

All parameters in `config.yaml`. The most important ones:

| Parameter | Default | Change if… |
|-----------|---------|------------|
| `ocr.max_concurrency` | 2 | OOM errors → lower to 1 |
| `ocr.dpi` | 200 | Bad quality → raise to 300 |
| `wiki.max_source_chars` | 12000 | Want richer pages → raise to 20000 |
| `llm_model` | qwen2.5:7b-instruct | Try `phi4:latest` for faster responses |

---

## Project structure

```
secondo_cervello/
├── main.py           ← FastAPI app: routes, SSE, pipeline orchestration
├── ocr.py            ← Async PDF → Markdown via glm-ocr
├── wiki.py           ← Karpathy wiki engine: ingest, query, lint
├── search.py         ← Pure-Python BM25 over wiki/*.md
├── config_loader.py  ← YAML config with defaults
├── config.yaml       ← All tunable parameters
├── requirements.txt
├── templates/
│   └── index.html    ← Full single-page UI (upload, queue, wiki, query, dash)
└── vault/            ← Created on first run
    ├── raw/sources/
    └── wiki/
```

---

## Supported input formats

| Format | Processing |
|--------|-----------|
| `.pdf` | Full OCR via glm-ocr (vision model) |
| `.md`  | Direct ingest (no OCR needed) |
| `.txt` | Direct ingest (no OCR needed) |

---

## Troubleshooting

**Ollama not connecting:**  
Make sure Ollama is running: `ollama serve`

**Out of memory during OCR:**  
Set `ocr.max_concurrency: 1` in `config.yaml`

**LLM produces bad JSON:**  
This is rare with qwen2.5:7b. If it happens, a fallback minimal page is created
automatically. The raw OCR markdown is always saved to `vault/raw/sources/`.

**Wiki is empty after upload:**  
Check `cervello.log` for errors. The raw OCR output is always preserved even if
the LLM enrichment step fails.
