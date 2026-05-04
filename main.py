"""
Secondo Cervello v3 — The Ultimate Local AI Knowledge Base

Pipeline:
  Upload → OCR (glm-ocr, keep_alive=5m, explicit unload) → raw .md
         → Qwen loads → reads existing wiki + SCHEMA.md → fixes typos
         → adds [[wikilinks]] → categorizes → writes to correct folder
         → cross-references → processes pending tasks → updates index

New in v3:
  - Incremental compilation (SHA-256 hashing)
  - aiosqlite database with state separation
  - Non-blocking BM25 indexing
  - Adjacency list for neighbor-expanded search
  - Schema-driven prompts (SCHEMA.md)
  - Web search ingestion
  - Audio transcription (Whisper)
  - Programmatic API endpoints
  - Revise & Organize batch operation
  - Durable task ledger
  - Automated cross-referencing
"""

import asyncio
import hashlib
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from config_loader import CFG
from db import Database
from ocr import ocr_pdf, ocr_image, transcribe_audio, unload_ocr_model
from ocr import PDF_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS
from wiki import WikiEngine, compute_sha256
from search import BM25Search

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cervello.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

# ─── Paths ────────────────────────────────────────────────────────────────────
VAULT       = Path(CFG["vault_dir"])
RAW_DIR     = VAULT / "attachments"
UPLOADS_TMP = Path(CFG["uploads_tmp"])
DB_PATH     = Path(CFG["db_path"])
TMPL_DIR    = Path("templates")
META_DIR    = VAULT / ".meta"
ADJ_PATH    = META_DIR / "adjacency.json"

for d in [RAW_DIR, UPLOADS_TMP, DB_PATH.parent, TMPL_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Global state ─────────────────────────────────────────────────────────────
wiki   = WikiEngine(VAULT, CFG)
search = BM25Search(VAULT)
db     = Database(str(DB_PATH))
_sse_clients: list[asyncio.Queue] = []

SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | {".txt", ".md"}


def emit(job_id: str, event: str, data: dict):
    payload = {"job_id": job_id, "event": event, "data": data,
               "ts": datetime.now().isoformat()}
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _sse_clients.remove(q)
        except ValueError:
            pass


# ─── Background pipeline ──────────────────────────────────────────────────────
async def run_pipeline(job_id: str, filename: str, tmp_path: Path):
    """Full pipeline: OCR → unload → Qwen enrichment → write wiki pages."""
    try:
        await db.upsert_job(job_id, filename=filename, status="running", step="starting")
        emit(job_id, "status", {"status": "running", "step": "starting"})

        suffix = tmp_path.suffix.lower()

        # ── Step 1: OCR / text extraction / audio transcription ──────────
        if suffix in PDF_EXTENSIONS:
            def on_page(done, total):
                emit(job_id, "progress", {"step": "ocr", "done": done, "total": total})
            emit(job_id, "status", {"status": "running", "step": "ocr"})
            raw_md = await ocr_pdf(tmp_path, progress_cb=lambda d, t: (
                db.upsert_job(job_id, step="ocr", progress=d, total=t),
                on_page(d, t)
            ))

        elif suffix in IMAGE_EXTENSIONS:
            emit(job_id, "status", {"status": "running", "step": "ocr"})
            raw_md = await ocr_image(tmp_path, progress_cb=lambda d, t: None)

        elif suffix in AUDIO_EXTENSIONS:
            emit(job_id, "status", {"status": "running", "step": "transcribe"})
            raw_md = await transcribe_audio(tmp_path, progress_cb=lambda d, t: None)

        elif suffix in {".md", ".txt"}:
            raw_md = tmp_path.read_text(encoding="utf-8", errors="replace")
            emit(job_id, "status", {"status": "running", "step": "reading"})
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Save raw output + original to attachments
        raw_md_path = RAW_DIR / (Path(filename).stem + "_raw.md")
        raw_md_path.write_text(raw_md, encoding="utf-8")
        shutil.copy2(tmp_path, RAW_DIR / filename)

        # Compute SHA-256 and record in DB
        sha256 = compute_sha256(tmp_path)
        await db.upsert_source(
            file_id=job_id,
            file_path=str(RAW_DIR / filename),
            sha256=sha256,
            ocr_status="done",
            ocr_completed_at=datetime.now().isoformat(),
        )

        # ── Step 2: Qwen enrichment ─────────────────────────────────────
        await db.upsert_job(job_id, step="enriching", progress=0, total=1)
        emit(job_id, "status", {"status": "running", "step": "enriching"})

        pages_written = await wiki.ingest(
            raw_text=raw_md,
            filename=filename,
            job_id=job_id,
            emit_fn=lambda event, data: emit(job_id, event, data),
        )

        # Update DB: wiki status
        await db.upsert_source(
            file_id=job_id,
            file_path=str(RAW_DIR / filename),
            sha256=sha256,
            wiki_status="done",
            wiki_completed_at=datetime.now().isoformat(),
        )

        # Record entity pages in DB
        # (handled within wiki.ingest, but we could also track here)

        # ── Step 3: Rebuild search index (non-blocking) + adjacency list ─
        await search.async_rebuild()
        search.build_adjacency_list()
        search.save_adjacency_list(ADJ_PATH)

        await db.upsert_job(job_id, status="done", step="done", progress=1, total=1)
        emit(job_id, "done", {"pages_written": pages_written})
        log.info(f"Job {job_id[:8]} done — {pages_written} pages written")

    except Exception as e:
        log.error(f"Job {job_id[:8]} failed: {e}", exc_info=True)
        await db.upsert_job(job_id, status="failed", step="error", error=str(e))
        emit(job_id, "error", {"message": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── FastAPI setup ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()
    await search.async_rebuild()
    search.build_adjacency_list()
    search.load_adjacency_list(ADJ_PATH)
    log.info(f"Vault: {VAULT.resolve()}")
    log.info(f"Ready → http://localhost:{CFG['server']['port']}")
    yield


app = FastAPI(title="Secondo Cervello v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory=str(TMPL_DIR))


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
            status_code=400,
        )
    job_id = str(uuid.uuid4())
    tmp_path = UPLOADS_TMP / f"{job_id}_{file.filename}"
    content = await file.read()
    tmp_path.write_bytes(content)

    # Check SHA-256 for incremental compilation
    sha256 = compute_sha256(tmp_path)
    existing = await db.get_source_by_hash(sha256)
    if existing and existing.get("wiki_status") == "done":
        tmp_path.unlink(missing_ok=True)
        return {"job_id": None, "filename": file.filename, "status": "duplicate", "message": "Already processed"}

    await db.upsert_job(job_id, filename=file.filename, status="queued", step="queued")
    background_tasks.add_task(run_pipeline, job_id, file.filename, tmp_path)
    log.info(f"Queued job {job_id[:8]} for {file.filename}")
    return {"job_id": job_id, "filename": file.filename}


@app.get("/api/jobs")
async def get_jobs():
    return await db.get_all_jobs()


@app.get("/api/events")
async def sse_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_clients.append(q)

    async def generator():
        try:
            yield f"data: {json.dumps({'event': 'connected'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/wiki/tree")
def wiki_tree():
    def _tree(path: Path, base: Path) -> dict:
        if path.is_file():
            return {"name": path.name, "path": str(path.relative_to(base)), "type": "file"}
        children = sorted(
            [_tree(c, base) for c in path.iterdir()
             if not c.name.startswith(".") and c.name != ".obsidian" and c.name != ".meta"],
            key=lambda x: (x["type"] == "file", x["name"].lower()),
        )
        return {"name": path.name, "path": str(path.relative_to(base)),
                "type": "dir", "children": children}

    if not VAULT.exists():
        return {"name": "LLM_Wiki", "type": "dir", "children": []}
    return _tree(VAULT, VAULT)


@app.get("/api/wiki/page")
def wiki_page(path: str):
    p = (VAULT / path).resolve()
    if not str(p).startswith(str(VAULT.resolve())):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"path": path, "content": p.read_text(encoding="utf-8")}


@app.post("/api/wiki/reprocess")
async def reprocess_page(request: Request):
    body = await request.json()
    page_path = body.get("path", "").strip()
    if not page_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        await wiki.reprocess(page_path)
        await search.async_rebuild()
        search.build_adjacency_list()
        search.save_adjacency_list(ADJ_PATH)
        return {"status": "ok", "path": page_path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/revise")
async def revise_and_organize(background_tasks: BackgroundTasks):
    """Trigger a full wiki revision and organization."""
    job_id = str(uuid.uuid4())
    await db.upsert_job(job_id, filename="wiki-revision", status="running", step="revise")
    emit(job_id, "status", {"status": "running", "step": "revise",
                             "detail": "Starting full wiki revision…"})

    async def _run_revise():
        try:
            def emit_fn(event, data):
                emit(job_id, event, data)

            result = await wiki.revise_and_organize(emit_fn=emit_fn)

            # Rebuild search index and adjacency list
            await search.async_rebuild()
            search.build_adjacency_list()
            search.save_adjacency_list(ADJ_PATH)

            await db.upsert_job(job_id, status="done", step="done")
            emit(job_id, "done", {
                "pages_written": 0,
                "revision_stats": result,
            })
            log.info(f"Revise job {job_id[:8]} done: {result}")
        except Exception as e:
            log.error(f"Revise job {job_id[:8]} failed: {e}", exc_info=True)
            await db.upsert_job(job_id, status="failed", step="error", error=str(e))
            emit(job_id, "error", {"message": str(e)})

    background_tasks.add_task(_run_revise)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)

    # Use neighbor-expanded search for richer context
    results = search.search_with_neighbors(question, k=CFG["wiki"]["context_pages"])
    context_pages = []
    for r in results:
        p = VAULT / r["path"]
        if p.exists():
            context_pages.append({"path": r["path"], "content": p.read_text(encoding="utf-8")})

    async def stream():
        async for chunk in wiki.query(question, context_pages):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/lint")
async def lint():
    result = await wiki.lint()
    await search.async_rebuild()
    return {"report": result}


@app.get("/api/stats")
async def stats():
    def count_md(d: Path) -> int:
        return len(list(d.rglob("*.md"))) if d.exists() else 0

    entities  = count_md(VAULT / "Entities")
    concepts  = count_md(VAULT / "Concepts")
    sources   = count_md(VAULT / "Sources")
    mocs      = count_md(VAULT / "MOC")
    total     = count_md(VAULT)

    log_path = VAULT / "log.md"
    recent = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        recent = [l for l in lines if l.startswith("## [")][-5:]
        recent.reverse()

    jobs = await db.get_all_jobs()
    done_jobs   = sum(1 for j in jobs if j["status"] == "done")
    failed_jobs = sum(1 for j in jobs if j["status"] == "failed")

    # Category breakdown
    category_stats = {}
    sources_dir = VAULT / "Sources"
    if sources_dir.exists():
        for cat_dir in sources_dir.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("."):
                category_stats[cat_dir.name] = len(list(cat_dir.glob("*.md")))

    # Adjacency list stats
    adj_nodes = len(search._adjacency) if search._adjacency else 0
    adj_edges = sum(len(v) for v in search._adjacency.values()) if search._adjacency else 0

    return {
        "wiki": {
            "entities": entities, "concepts": concepts,
            "sources": sources, "mocs": mocs, "total": total,
            "categories": category_stats,
            "adjacency_nodes": adj_nodes, "adjacency_edges": adj_edges,
        },
        "jobs": {"done": done_jobs, "failed": failed_jobs},
        "recent_log": recent,
        "vault_path": str(VAULT.resolve()),
    }


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(CFG["ollama_url"] + "/api/tags", timeout=5.0)
            models = [m["name"] for m in r.json().get("models", [])]
        return {"ollama": "ok", "models": models}
    except Exception as e:
        return {"ollama": "error", "detail": str(e)}


@app.get("/api/config")
def get_config():
    return {
        "vault_path": str(VAULT.resolve()),
        "ocr_model": CFG["ocr_model"],
        "llm_model": CFG["llm_model"],
        "categories": [{"slug": c["slug"], "name": c["name"]} for c in CFG["categories"]],
    }


# ─── Programmatic API endpoints ──────────────────────────────────────────────

@app.post("/api/v1/ingest")
async def api_v1_ingest(request: Request):
    """Push structured content directly into the vault."""
    body = await request.json()
    content  = body.get("content", "").strip()
    title    = body.get("title", "Untitled").strip()
    category = body.get("category", "uncategorized").strip()
    page_type = body.get("type", "source").strip()
    tags     = body.get("tags", [])

    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)

    today = datetime.now().strftime("%Y-%m-%d")
    slug = _to_slug(title)

    if page_type == "entity":
        folder = "Entities"
    elif page_type == "concept":
        folder = "Concepts"
    else:
        folder = f"Sources/{category}"

    fm = (f"---\ntitle: {title}\ntype: {page_type}\ncategory: {category}\n"
          f"tags: {json.dumps(tags)}\nsources: []\n"
          f'created: "{today}"\nupdated: "{today}"\n---\n\n')
    full_content = fm + content

    rel_path = f"{folder}/{slug}.md"
    wiki._write_page(rel_path, full_content)

    # Incremental search update
    search.add_document(rel_path, full_content)
    search.build_adjacency_list()
    search.save_adjacency_list(ADJ_PATH)

    # Record in DB
    entity_id = str(uuid.uuid4())
    await db.upsert_entity(
        file_path=rel_path, source_file="api",
        page_type=page_type, category=category, title=title,
    )

    wiki._append_log(f"## [{today}] api-ingest | {title}\n\nPushed via API to {rel_path}\n")
    wiki._update_index(
        {"title": title, "folder": folder, "filename": f"{slug}.md"} if page_type == "source" else {},
        [{"title": title, "filename": f"{slug}.md"}] if page_type == "entity" else [],
        [{"title": title, "filename": f"{slug}.md"}] if page_type == "concept" else [],
    )

    return {"status": "ok", "path": rel_path, "title": title}


@app.post("/api/v1/search")
async def api_v1_search(request: Request):
    """Search the wiki programmatically."""
    body = await request.json()
    query = body.get("query", "").strip()
    k = body.get("k", 6)
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    results = search.search_with_neighbors(query, k=k)
    return {"results": results}


@app.get("/api/v1/graph")
def api_v1_graph():
    """Get the adjacency list (wikilink graph)."""
    return {"nodes": len(search._adjacency),
            "edges": sum(len(v) for v in search._adjacency.values()),
            "adjacency": search._adjacency}


# ─── Web Search Ingestion ────────────────────────────────────────────────────

@app.post("/api/web-search")
async def web_search_ingest(request: Request, background_tasks: BackgroundTasks):
    """
    Search the web, fetch results, and ingest into the wiki.
    Uses the configured search API.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    num_results = body.get("num", 3)

    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    job_id = str(uuid.uuid4())
    await db.upsert_job(job_id, filename=f"web:{query}", status="running", step="web_search")
    emit(job_id, "status", {"status": "running", "step": "web_search",
                             "detail": f"Searching: {query}"})

    async def _run_web_search():
        try:
            from z_ai_web_dev_sdk import ZAI
            zai = await ZAI.create()
            results = await zai.functions.invoke("web_search", {"query": query, "num": num_results})

            # Save each result as a source file
            for result in results:
                url = result.get("url", "")
                name = result.get("name", "Untitled")
                snippet = result.get("snippet", "")

                # Try to fetch full page content
                try:
                    page_data = await zai.functions.invoke("page_reader", {"url": url})
                    page_text = page_data.get("data", {}).get("html", "") if isinstance(page_data, dict) else ""
                    # Simple HTML to text
                    import re
                    page_text = re.sub(r'<[^>]+>', ' ', page_text)
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    content = page_text[:50000] if page_text else snippet
                except Exception:
                    content = snippet

                if not content:
                    continue

                # Save to attachments and trigger standard processing
                safe_name = re.sub(r'[^\w\s-]', '', name)[:60].strip().replace(' ', '_')
                raw_path = RAW_DIR / f"web_{safe_name}.md"
                today = datetime.now().strftime("%Y-%m-%d")
                md_content = (
                    f"# {name}\n\n"
                    f"> **Source:** [{url}]({url})\n"
                    f"> **Retrieved:** {today}\n\n"
                    f"{content}\n"
                )
                raw_path.write_text(md_content, encoding="utf-8")

                # Process through wiki engine
                emit(job_id, "status", {
                    "status": "running", "step": "web_ingest",
                    "detail": f"Processing: {name}",
                })
                pages = await wiki.ingest(
                    raw_text=md_content,
                    filename=f"web_{safe_name}.md",
                    job_id=job_id,
                    emit_fn=lambda event, data: emit(job_id, event, data),
                )

            await search.async_rebuild()
            search.build_adjacency_list()
            search.save_adjacency_list(ADJ_PATH)

            await db.upsert_job(job_id, status="done", step="done")
            emit(job_id, "done", {"pages_written": 0, "web_results": len(results)})
        except Exception as e:
            log.error(f"Web search job {job_id[:8]} failed: {e}", exc_info=True)
            await db.upsert_job(job_id, status="failed", step="error", error=str(e))
            emit(job_id, "error", {"message": str(e)})

    background_tasks.add_task(_run_web_search)
    return {"job_id": job_id, "status": "started", "query": query}


def _to_slug(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:80] or "untitled"


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=CFG["server"]["host"],
        port=CFG["server"]["port"],
        reload=False,
    )
