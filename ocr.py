"""
OCR engine: PDF → Markdown via glm-ocr on Ollama.
Async, parallel pages, VRAM-aware semaphore.
"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Callable

import fitz
import httpx
from PIL import Image

from config_loader import CFG

log = logging.getLogger("ocr")

OLLAMA_URL   = CFG["ollama_url"] + "/api/generate"
OCR_MODEL    = CFG["ocr_model"]
MAX_WORKERS  = CFG["ocr"]["max_concurrency"]
DPI          = CFG["ocr"]["dpi"]
MAX_DIM      = CFG["ocr"]["max_image_dimension"]

OCR_PROMPT = """\
You are a precise OCR engine. Extract ALL content from this page into clean Markdown.
Rules:
- Preserve headings (# ## ###), tables, lists, code blocks
- Keep reading order top-to-bottom, left-to-right
- Output ONLY the Markdown content — no preamble, no commentary
- Mark unclear sections with [?]
"""


def _render_page(page: fitz.Page, dpi: int, max_dim: int) -> str:
    """Render a PDF page to base64 PNG. CPU-bound, runs in thread pool."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


async def _ocr_page(
    client: httpx.AsyncClient,
    b64: str,
    page_num: int,
    sem: asyncio.Semaphore,
) -> tuple[int, str]:
    async with sem:
        payload = {
            "model":   OCR_MODEL,
            "prompt":  OCR_PROMPT,
            "images":  [b64],
            "stream":  False,
            "options": {"temperature": 0.0},
        }
        try:
            r = await client.post(OLLAMA_URL, json=payload, timeout=180.0)
            r.raise_for_status()
            return page_num, r.json()["response"].strip()
        except Exception as e:
            log.warning(f"  Page {page_num} OCR error: {e}")
            return page_num, f"> ⚠️ **Page {page_num} OCR failed:** `{e}`"


async def ocr_pdf(
    pdf_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Run OCR on a PDF. Returns assembled Markdown string.
    progress_cb(done, total) called after each page.
    """
    doc   = fitz.open(str(pdf_path))
    total = len(doc)
    log.info(f"OCR: {pdf_path.name} ({total} pages, workers={MAX_WORKERS})")

    # Pre-render all pages in thread pool (CPU-bound)
    pages_b64: list[str] = await asyncio.gather(*[
        asyncio.to_thread(_render_page, doc[i], DPI, MAX_DIM)
        for i in range(total)
    ])
    doc.close()

    sem = asyncio.Semaphore(MAX_WORKERS)
    results: list[tuple[int, str]] = []
    done = 0

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_ocr_page(client, pages_b64[i], i + 1, sem))
            for i in range(total)
        ]
        for fut in asyncio.as_completed(tasks):
            page_num, md = await fut
            results.append((page_num, md))
            done += 1
            if progress_cb:
                progress_cb(done, total)
            log.info(f"  OCR {done}/{total}")

    results.sort(key=lambda x: x[0])

    lines = [f"# {pdf_path.stem.replace('_', ' ')}\n\n"]
    for page_num, md in results:
        lines.append(f"\n<!-- page:{page_num} -->\n{md}\n\n---\n")

    return "".join(lines)
