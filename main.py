"""
Secondo Cervello v4 — The Ultimate Local AI Knowledge Base

Pipeline:
  Upload -> Smart Route:
    PDFs/Images  -> OCR (glm-ocr, keep_alive=5m, explicit unload) -> raw .md
    Audio        -> Whisper transcription -> raw .md
    Text/Code    -> Read directly (NO OCR, skip vision model entirely)
  -> Qwen loads -> reads existing wiki + SCHEMA.md -> fixes typos
  -> adds [[wikilinks]] -> categorizes -> writes to correct folder
  -> cross-references -> processes pending tasks -> updates index

Folder Watcher:
  Drop files into the configured inbox folder -> auto-processed through
  the same pipeline. Enable with watch_enabled: true in config.yaml.

v4 fixes applied:
  - Fix #2:  progress_cb is async-safe (no coroutines in sync lambdas)
  - Fix #3:  search.add_document() exists (was already there)
  - Fix #4:  Web search rewritten with httpx + DuckDuckGo HTML (no z_ai_web_dev_sdk)
  - Fix #17: Auto-execute pending tasks after every ingest (not just on revise)
  - Fix #1:  emit_fn standardised to 2-arg everywhere
  - NEW: Folder watcher (watchdog) for auto-processing dropped files
  - NEW: Smart routing — text files skip OCR entirely, go directly to Qwen
"""

import asyncio
import hashlib
import json
import logging
import re
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
from watcher import FolderWatcher, ALL_TEXT_EXTENSIONS, ALL_SUPPORTED, get_file_route
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

SUPPORTED_EXTENSIONS = ALL_SUPPORTED


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
    """Full pipeline: OCR -> unload -> Qwen enrichment -> write wiki pages."""
    try:
        await db.upsert_job(job_id, filename=filename, status="running", step="starting")
        emit(job_id, "status", {"status": "running", "step": "starting"})

        suffix = tmp_path.suffix.lower()

        # ── Step 1: Smart routing — OCR / text / audio ──────────────────
        # Text files (.txt, .md, .csv, .json, .html, .py, etc.) SKIP OCR
        # entirely and go directly to Qwen for markdown conversion.
        # Only PDFs and images need the vision model (glm-ocr).
        # Audio files go through Whisper transcription.

        route = get_file_route(suffix)
        log.info(f"Job {job_id[:8]}: routing '{filename}' -> {route}")

        if route == "ocr" and suffix in PDF_EXTENSIONS:
            async def on_page(done, total):
                emit(job_id, "progress", {"step": "ocr", "done": done, "total": total})
                await db.upsert_job(job_id, step="ocr", progress=done, total=total)

            emit(job_id, "status", {"status": "running", "step": "ocr",
                                     "detail": f"OCR processing PDF ({suffix})…"})
            raw_md = await ocr_pdf(tmp_path, progress_cb=lambda d, t: asyncio.ensure_future(on_page(d, t)))

        elif route == "ocr" and suffix in IMAGE_EXTENSIONS:
            emit(job_id, "status", {"status": "running", "step": "ocr",
                                     "detail": f"OCR processing image ({suffix})…"})
            raw_md = await ocr_image(tmp_path, progress_cb=lambda d, t: None)

        elif route == "transcribe":
            emit(job_id, "status", {"status": "running", "step": "transcribe",
                                     "detail": f"Transcribing audio ({suffix})…"})
            raw_md = await transcribe_audio(tmp_path, progress_cb=lambda d, t: None)

        elif route == "text":
            # SMART ROUTE: Text/code files skip OCR entirely!
            # They're already text — no vision model needed.
            # Go directly to Qwen for structuring and wiki conversion.
            raw_md = tmp_path.read_text(encoding="utf-8", errors="replace")
            emit(job_id, "status", {"status": "running", "step": "reading",
                                     "detail": f"Reading text file ({suffix}) — skipping OCR"})
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

        # Fix #1: emit_fn is 2-arg (event, data), job_id captured in closure
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

        # ── Step 3: Rebuild search index (non-blocking) + adjacency list ─
        await search.async_rebuild()
        search.build_adjacency_list()
        search.save_adjacency_list(ADJ_PATH)

        # Fix #17: Auto-execute pending tasks after every ingest
        tasks_done = await wiki._process_pending_tasks(
            emit_fn=lambda event, data: emit(job_id, event, data)
        )
        if tasks_done > 0:
            log.info(f"Auto-executed {tasks_done} pending tasks after ingest")
            # Rebuild search if tasks created new pages
            await search.async_rebuild()
            search.build_adjacency_list()
            search.save_adjacency_list(ADJ_PATH)

        await db.upsert_job(job_id, status="done", step="done", progress=1, total=1)
        emit(job_id, "done", {"pages_written": pages_written, "tasks_executed": tasks_done})
        log.info(f"Job {job_id[:8]} done — {pages_written} pages written, {tasks_done} tasks executed")

    except Exception as e:
        log.error(f"Job {job_id[:8]} failed: {e}", exc_info=True)
        await db.upsert_job(job_id, status="failed", step="error", error=str(e))
        emit(job_id, "error", {"message": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── Folder watcher ────────────────────────────────────────────────────────────
watcher = FolderWatcher(run_pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()
    await search.async_rebuild()
    search.build_adjacency_list()
    search.load_adjacency_list(ADJ_PATH)

    # Start folder watcher if enabled
    watcher.start()

    log.info(f"Vault: {VAULT.resolve()}")
    log.info(f"Ready -> http://localhost:{CFG['server']['port']}")
    yield

    # Cleanup: stop watcher
    watcher.stop()


app = FastAPI(title="Secondo Cervello v4", lifespan=lifespan)
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
        "watch_enabled": CFG.get("watch_enabled", False),
        "watch_folder": str(watcher.inbox.resolve()),
        "watcher_running": watcher.is_running,
        "categories": [{"slug": c["slug"], "name": c["name"]} for c in CFG["categories"]],
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "routing": {
            "ocr": sorted(PDF_EXTENSIONS | IMAGE_EXTENSIONS),
            "transcribe": sorted(AUDIO_EXTENSIONS),
            "text_skip_ocr": sorted(ALL_TEXT_EXTENSIONS),
        },
    }


# ─── Watcher API endpoints ─────────────────────────────────────────────────────

@app.get("/api/watcher/status")
def watcher_status():
    """Get the current status of the folder watcher."""
    inbox = watcher.inbox
    pending_files = []
    if inbox.exists():
        for f in sorted(inbox.iterdir()):
            if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~"):
                route = get_file_route(f.suffix)
                pending_files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "route": route,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })

    return {
        "enabled": CFG.get("watch_enabled", False),
        "running": watcher.is_running,
        "inbox_path": str(inbox.resolve()),
        "pending_files": pending_files,
    }


@app.post("/api/watcher/toggle")
async def watcher_toggle(request: Request):
    """Toggle the folder watcher on/off."""
    body = await request.json() if request.headers.get("content-type") else {}
    enable = body.get("enable", not watcher.is_running)

    if enable and not watcher.is_running:
        watcher.start()
        return {"status": "started", "running": True}
    elif not enable and watcher.is_running:
        watcher.stop()
        return {"status": "stopped", "running": False}
    else:
        return {"status": "unchanged", "running": watcher.is_running}


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

    # Incremental search update (Fix #3: add_document already exists)
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


# ─── Web Search Ingestion (Fix #4: httpx + DuckDuckGo, no z_ai_web_dev_sdk) ─

@app.post("/api/web-search")
async def web_search_ingest(request: Request, background_tasks: BackgroundTasks):
    """
    Search the web using DuckDuckGo HTML, fetch results, and ingest into the wiki.
    Fix #4: Rewritten to use httpx + DuckDuckGo instead of z_ai_web_dev_sdk.
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
            results_found = 0

            # Step 1: Search DuckDuckGo HTML
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                },
                follow_redirects=True,
                timeout=15.0,
            ) as client:
                search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
                r = await client.get(search_url)
                r.raise_for_status()
                html = r.text

                # Parse results from DDG HTML
                # DDG HTML uses <a class="result__a" href="...">Title</a>
                result_links = re.findall(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    html,
                )
                result_snippets = re.findall(
                    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                    html,
                )

                for idx, (url, name_html) in enumerate(result_links[:num_results]):
                    # Clean HTML from title
                    name = re.sub(r'<[^>]+>', '', name_html).strip()
                    snippet = re.sub(r'<[^>]+>', '', result_snippets[idx]).strip() if idx < len(result_snippets) else ""

                    if not name:
                        continue

                    # Try to fetch the page content
                    page_text = ""
                    try:
                        page_r = await client.get(url, timeout=10.0)
                        if page_r.status_code == 200:
                            # Simple HTML to text: strip tags
                            page_text = re.sub(r'<script[^>]*>.*?</script>', ' ', page_r.text, flags=re.DOTALL)
                            page_text = re.sub(r'<style[^>]*>.*?</style>', ' ', page_text, flags=re.DOTALL)
                            page_text = re.sub(r'<[^>]+>', ' ', page_text)
                            page_text = re.sub(r'\s+', ' ', page_text).strip()
                            page_text = page_text[:50000]
                    except Exception:
                        pass

                    content = page_text if page_text else snippet
                    if not content:
                        continue

                    # Save to attachments and process through wiki engine
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
                    results_found += 1

            await search.async_rebuild()
            search.build_adjacency_list()
            search.save_adjacency_list(ADJ_PATH)

            await db.upsert_job(job_id, status="done", step="done")
            emit(job_id, "done", {"pages_written": 0, "web_results": results_found})
        except Exception as e:
            log.error(f"Web search job {job_id[:8]} failed: {e}", exc_info=True)
            await db.upsert_job(job_id, status="failed", step="error", error=str(e))
            emit(job_id, "error", {"message": str(e)})

    background_tasks.add_task(_run_web_search)
    return {"job_id": job_id, "status": "started", "query": query}


def _to_slug(s: str) -> str:
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
