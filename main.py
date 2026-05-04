"""
Secondo Cervello v2 — The Ultimate Local AI Knowledge Base
One process · One port · Upload → OCR → Wiki → Knowledge Graph

Pipeline:
  Upload → OCR (glm-ocr, keep_alive=0) → raw .md
         → Qwen reads raw + existing wiki → fixes typos → adds [[wikilinks]]
         → categorizes → writes to correct vault folder → updates index
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config_loader import CFG
from ocr import ocr_pdf, ocr_image
from wiki import WikiEngine
from search import BM25Search

# ─── Logging ──────────────────────────────────────────────────────────────────
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

for d in [RAW_DIR, UPLOADS_TMP, DB_PATH.parent, TMPL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Global state ─────────────────────────────────────────────────────────────
wiki   = WikiEngine(VAULT, CFG)
search = BM25Search(VAULT)
_sse_clients: list[asyncio.Queue] = []

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def emit(job_id: str, event: str, data: dict):
    """Broadcast SSE event to all connected clients."""
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


# ─── SQLite helpers ───────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id         TEXT PRIMARY KEY,
                filename   TEXT,
                status     TEXT,
                step       TEXT,
                progress   INTEGER DEFAULT 0,
                total      INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                error      TEXT
            )
        """)


def db_write(job_id: str, **kw):
    kw["updated_at"] = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row:
            sets = ", ".join(f"{k}=?" for k in kw)
            con.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*kw.values(), job_id))
        else:
            kw.setdefault("created_at", datetime.now().isoformat())
            kw["id"] = job_id
            cols, vals = ", ".join(kw), ", ".join("?" * len(kw))
            con.execute(f"INSERT INTO jobs ({cols}) VALUES ({vals})", list(kw.values()))


def db_all_jobs() -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Background pipeline ──────────────────────────────────────────────────────
async def run_pipeline(job_id: str, filename: str, tmp_path: Path):
    """
    Full pipeline: OCR → Qwen enrichment → write wiki pages.
    Emits SSE events at every step.
    """
    try:
        db_write(job_id, filename=filename, status="running", step="starting")
        emit(job_id, "status", {"status": "running", "step": "starting"})

        suffix = tmp_path.suffix.lower()

        # ── Step 1: OCR / text extraction ────────────────────────────────────
        if suffix == ".pdf":
            def on_page(done, total):
                db_write(job_id, step="ocr", progress=done, total=total)
                emit(job_id, "progress", {"step": "ocr", "done": done, "total": total})

            emit(job_id, "status", {"status": "running", "step": "ocr",
                                     "detail": "Running OCR with vision model…"})
            raw_md = await ocr_pdf(tmp_path, progress_cb=on_page)

        elif suffix in IMAGE_EXTENSIONS:
            def on_page(done, total):
                db_write(job_id, step="ocr", progress=done, total=total)
                emit(job_id, "progress", {"step": "ocr", "done": done, "total": total})

            emit(job_id, "status", {"status": "running", "step": "ocr",
                                     "detail": "Running OCR on image…"})
            raw_md = await ocr_image(tmp_path, progress_cb=on_page)

        elif suffix in {".md", ".txt"}:
            raw_md = tmp_path.read_text(encoding="utf-8", errors="replace")
            emit(job_id, "status", {"status": "running", "step": "reading",
                                     "detail": f"Reading {suffix} file…"})
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Save raw output to attachments
        raw_md_path = RAW_DIR / (Path(filename).stem + "_raw.md")
        raw_md_path.write_text(raw_md, encoding="utf-8")

        # Copy original to attachments
        shutil.copy2(tmp_path, RAW_DIR / filename)

        # ── Step 2: Qwen enrichment ─────────────────────────────────────────
        db_write(job_id, step="enriching", progress=0, total=1)
        emit(job_id, "status", {"status": "running", "step": "enriching",
                                 "detail": "Qwen is processing the document…"})

        pages_written = await wiki.ingest(
            raw_text=raw_md,
            filename=filename,
            job_id=job_id,
            emit_fn=emit,
        )

        # Rebuild search index
        search.rebuild()

        db_write(job_id, status="done", step="done", progress=1, total=1)
        emit(job_id, "done", {"pages_written": pages_written})
        log.info(f"Job {job_id[:8]} done — {pages_written} pages written")

    except Exception as e:
        log.error(f"Job {job_id[:8]} failed: {e}", exc_info=True)
        db_write(job_id, status="failed", step="error", error=str(e))
        emit(job_id, "error", {"message": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── FastAPI setup ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    search.rebuild()
    log.info(f"Vault: {VAULT.resolve()}")
    log.info(f"Ready → http://localhost:{CFG['server']['port']}")
    yield


app = FastAPI(title="Secondo Cervello v2", lifespan=lifespan)
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
            {"error": f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    tmp_path = UPLOADS_TMP / f"{job_id}_{file.filename}"
    content = await file.read()
    tmp_path.write_bytes(content)
    db_write(job_id, filename=file.filename, status="queued", step="queued")
    background_tasks.add_task(run_pipeline, job_id, file.filename, tmp_path)
    log.info(f"Queued job {job_id[:8]} for {file.filename}")
    return {"job_id": job_id, "filename": file.filename}


@app.get("/api/jobs")
def get_jobs():
    return db_all_jobs()


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

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/wiki/tree")
def wiki_tree():
    def _tree(path: Path, base: Path) -> dict:
        if path.is_file():
            return {"name": path.name, "path": str(path.relative_to(base)),
                    "type": "file"}
        children = sorted(
            [_tree(c, base) for c in path.iterdir()
             if not c.name.startswith(".") and c.name != ".obsidian"],
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
    """Re-run Qwen on an existing page to improve it."""
    body = await request.json()
    page_path = body.get("path", "").strip()
    if not page_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        new_content = await wiki.reprocess(page_path)
        search.rebuild()
        return {"status": "ok", "path": page_path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    save = body.get("save", False)
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)

    results = search.search(question, k=CFG["wiki"]["context_pages"])
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
    search.rebuild()
    return {"report": result}


@app.get("/api/stats")
def stats():
    def count_md(d: Path) -> int:
        return len(list(d.rglob("*.md"))) if d.exists() else 0

    entities  = count_md(VAULT / "Entities")
    concepts  = count_md(VAULT / "Concepts")
    sources   = count_md(VAULT / "Sources")
    mocs      = count_md(VAULT / "MOC")
    total     = count_md(VAULT)

    # Recent log entries
    log_path = VAULT / "log.md"
    recent = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        recent = [l for l in lines if l.startswith("## [")][-5:]
        recent.reverse()

    # Job stats
    jobs = db_all_jobs()
    done_jobs   = sum(1 for j in jobs if j["status"] == "done")
    failed_jobs = sum(1 for j in jobs if j["status"] == "failed")
    running_jobs = sum(1 for j in jobs if j["status"] == "running")

    # Category breakdown
    category_stats = {}
    sources_dir = VAULT / "Sources"
    if sources_dir.exists():
        for cat_dir in sources_dir.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("."):
                category_stats[cat_dir.name] = len(list(cat_dir.glob("*.md")))

    return {
        "wiki": {
            "entities": entities,
            "concepts": concepts,
            "sources": sources,
            "mocs": mocs,
            "total": total,
            "categories": category_stats,
        },
        "jobs": {"done": done_jobs, "failed": failed_jobs, "running": running_jobs},
        "recent_log": recent,
        "vault_path": str(VAULT.resolve()),
    }


@app.get("/api/health")
async def health():
    """Check Ollama connectivity."""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(CFG["ollama_url"] + "/api/tags", timeout=5.0)
            models = [m["name"] for m in r.json().get("models", [])]
        return {"ollama": "ok", "models": models}
    except Exception as e:
        return {"ollama": "error", "detail": str(e)}


@app.get("/api/config")
def get_config():
    """Return non-sensitive config for the UI."""
    return {
        "vault_path": str(VAULT.resolve()),
        "ocr_model": CFG["ocr_model"],
        "llm_model": CFG["llm_model"],
        "categories": [{"slug": c["slug"], "name": c["name"]} for c in CFG["categories"]],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=CFG["server"]["host"],
        port=CFG["server"]["port"],
        reload=False,
    )
