# Secondo Cervello v2

**The Ultimate Local AI Knowledge Base.**

Upload PDFs, images, and text files → OCR extracts everything → Qwen organizes it into your Obsidian wiki with wikilinks, categories, and typo fixes. All local, all private, zero cloud.

Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

```
http://localhost:7337
      ↑
  FastAPI app
      │
  ┌───┴──────────────────────────────────────────────────┐
  │  Upload → OCR (glm-ocr, auto-unload)                │
  │         → Qwen processes raw text:                   │
  │           · Fixes typos and OCR errors               │
  │           · Adds [[wikilinks]] to existing pages     │
  │           · Categorizes into topic folders           │
  │           · Creates entity & concept pages           │
  │           · Writes .md files to Obsidian vault       │
  │                                                      │
  │  Wiki panel   → browse + wikilink navigation         │
  │  Chat panel   → BM25 search + streaming LLM          │
  │  Dashboard    → stats + category breakdown + lint    │
  │  Settings     → vault path + model config            │
  └──────────────────────────────────────────────────────┘
```

---

## Quick start

### 1. Install Python deps

```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
# OCR (vision) — handles scanned PDFs, images, handwriting
ollama pull glm-ocr:latest

# Wiki enrichment + query — recommended for RTX 3070 Laptop (8GB VRAM)
ollama pull qwen2.5:7b-instruct
```

> **Why qwen2.5:7b?** At q8_0 it fits in ~7.7GB. Outperforms Mistral 7B and Llama 3.1 8B on structured tasks (JSON output, instruction-following). glm-ocr and qwen2.5 never run in parallel — OCR finishes first, then its model is unloaded (keep_alive=0), then Qwen loads. No VRAM conflict.

### 3. Configure vault path

Edit `config.yaml` and set `vault_dir` to your Obsidian vault:

```yaml
vault_dir: "D:\\obsidian\\LLM_Wiki"
```

### 4. Run

```bash
python main.py
# → http://localhost:7337
```

---

## How it works

### Pipeline Flow

```
File upload (PDF/image/TXT/MD)
  → OCR (glm-ocr, keep_alive=0)
      · Renders PDF pages or sends images to vision model
      · Comprehensive prompt handles text, tables, graphs, infographics, diagrams
      · Model auto-unloads after OCR completes
  → Raw markdown saved to vault/attachments/
  → Qwen loads (qwen2.5:7b-instruct)
      · Reads raw OCR output + existing wiki page list
      · Fixes typos and OCR errors through context
      · Adds [[wikilinks]] to existing pages
      · Categorizes document into topic folder
      · Creates source page + entity/concept pages
      · Writes .md files to correct vault folders
      · Updates index.md and log.md
  → Search index rebuilt automatically
```

### Vault Structure (Obsidian-compatible)

```
D:\obsidian\LLM_Wiki/
├── Sources/                    ← Source documents organized by category
│   ├── ai-ml/                  ← AI & Machine Learning
│   ├── science/                ← Science & Research
│   ├── technology/             ← Technology
│   ├── health/                 ← Health & Medicine
│   ├── business/               ← Business & Finance
│   ├── philosophy/             ← Philosophy & Psychology
│   ├── arts/                   ← Arts & Humanities
│   └── personal/               ← Personal
├── Entities/                   ← People, organizations, tools, products
├── Concepts/                   ← Ideas, theories, methods, techniques
├── MOC/                        ← Maps of Content (hub pages)
├── attachments/                ← Original files + raw OCR output
├── index.md                    ← Master catalog (auto-updated)
├── log.md                      ← Append-only operation log
└── AGENTS.md                   ← AI schema governance
```

### Query Flow

1. Your question is tokenized (supports English + CJK)
2. BM25 finds top-k relevant wiki pages (sub-millisecond, title-boosted)
3. Pages are sent to qwen2.5 as context
4. Answer streams back with [[citations]]
5. Optionally re-process pages to improve them

---

## Key improvements over v1

| Feature | v1 | v2 |
|---------|----|----|
| Vault format | JSON intermediary | Pure .md files in Obsidian vault |
| Vault location | Local `vault/` subfolder | Configurable Obsidian vault path |
| OCR model management | Stays loaded | Auto-unloads after OCR (keep_alive=0) |
| OCR prompt | Basic text extraction | Comprehensive: tables, graphs, infographics, diagrams, equations |
| Image support | PDF only | PDF + PNG, JPG, GIF, BMP, TIFF, WEBP |
| Wikilinks | Random | Qwen reads existing pages to create accurate links |
| Categorization | All in `sources/` | Automatic categorization into topic folders |
| Typo correction | None | Context-based OCR error fixing |
| Search | English only | English + CJK tokenization, title boosting |
| Health check | 30s polling (spam) | 60s polling |
| UI | Top tabs | Sidebar navigation, modern design |
| Reprocess | None | Re-run Qwen on existing pages |

---

## Config reference

All parameters in `config.yaml`. Key ones:

| Parameter | Default | Change if… |
|-----------|---------|------------|
| `vault_dir` | `D:\obsidian\LLM_Wiki` | Point to your Obsidian vault |
| `ocr.max_concurrency` | 2 | OOM errors → lower to 1 |
| `ocr.dpi` | 200 | Bad quality → raise to 300 |
| `ocr.keep_alive` | 0 | Keep OCR model loaded → increase |
| `wiki.max_source_chars` | 16000 | Richer pages → raise to 20000 |
| `wiki.auto_create_entities` | true | Don't want entity pages → false |
| `wiki.auto_create_concepts` | true | Don't want concept pages → false |
| `llm_model` | qwen2.5:7b-instruct | Try `phi4:latest` for speed |

---

## Supported input formats

| Format | Processing |
|--------|-----------|
| `.pdf` | Full OCR via glm-ocr (vision model), parallel pages |
| `.png`, `.jpg`, etc. | Direct OCR via glm-ocr |
| `.md` | Direct ingest (Qwen still adds links + categorizes) |
| `.txt` | Direct ingest (Qwen still adds links + categorizes) |

---

## Project structure

```
secondo-cervello/
├── main.py           ← FastAPI app: routes, SSE, pipeline orchestration
├── ocr.py            ← Async PDF/Image → Markdown via glm-ocr
├── wiki.py           ← Wiki engine: ingest, query, lint, reprocess
├── search.py         ← BM25 with CJK support + title boosting
├── config_loader.py  ← YAML config with deep-merge defaults
├── config.yaml       ← All tunable parameters
├── requirements.txt
├── templates/
│   └── index.html    ← Full single-page UI
└── D:\obsidian\LLM_Wiki/  ← Obsidian vault (created on first run)
```

---

## Troubleshooting

**Ollama not connecting:**
Make sure Ollama is running: `ollama serve`

**Out of memory during OCR:**
Set `ocr.max_concurrency: 1` in `config.yaml`

**LLM produces bad JSON:**
A fallback minimal page is created automatically. The raw OCR markdown is always saved to `attachments/`.

**Wiki is empty after upload:**
Check `cervello.log` for errors. Raw OCR output is preserved even if enrichment fails.

**Want to re-process a page:**
Open the page in the Wiki panel and click "Re-process" to run Qwen on it again with updated context.

**Add a new category:**
Add it to the `categories` list in `config.yaml` and restart the server.
