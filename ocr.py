"""
OCR engine: PDF/Image → Markdown via vision model on Ollama.
Async, parallel pages, VRAM-aware semaphore, immediate model unload.

Supports: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Callable

import httpx

from config_loader import CFG

log = logging.getLogger("ocr")

OLLAMA_URL   = CFG["ollama_url"] + "/api/generate"
OCR_MODEL    = CFG["ocr_model"]
MAX_WORKERS  = CFG["ocr"]["max_concurrency"]
DPI          = CFG["ocr"]["dpi"]
MAX_DIM      = CFG["ocr"]["max_image_dimension"]
KEEP_ALIVE   = CFG["ocr"]["keep_alive"]

# ─── OCR Prompt ──────────────────────────────────────────────────────────────

OCR_PROMPT = """\
You are a precise document OCR and transcription engine. Your task is to convert this document page into clean, comprehensive Markdown.

CRITICAL RULES:
1. Transcribe EVERY piece of text on this page — do not skip, summarize, or paraphrase anything
2. Maintain the original reading order (top-to-bottom, left-to-right)
3. Output ONLY the Markdown transcription — no preamble, no commentary, no explanations

TEXT HANDLING:
- Use headings (# ## ### ####) to match the document's visual hierarchy
- Preserve bold (**text**) and italic (*text*) formatting where visible
- Keep all footnotes, citations, and references exactly as they appear
- Transcribe equations using LaTeX notation: inline $equation$ or block $$equation$$
- Preserve list formatting (ordered and unordered)
- Keep paragraph breaks where they appear in the original

TABLE HANDLING:
- Convert ALL tables to Markdown table format with | separators and alignment
- Include header rows even if visually merged
- For merged cells, place the content in the leftmost/topmost cell and leave others empty
- If a table is too complex for standard markdown, represent it as a structured list:
  > **Table: [Title/Description]**
  > | Column1 | Column2 | Column3 |
  > |---------|---------|---------|
  > | data    | data    | data    |
- Preserve all numerical data exactly — never approximate or round

GRAPH AND CHART HANDLING:
- For each graph or chart, provide a structured description:
  > **Chart: [Type — e.g., Bar Chart / Line Graph / Scatter Plot]: [Title if present]**
  > - X-axis: [description and range if visible]
  > - Y-axis: [description and range if visible]
  > - Data series: [describe each series/legend entry]
  > - Key data points: [list the most significant values visible]
  > - Trend/Observation: [what the data visually shows]
- Extract any visible numerical values from the axes, bars, or data points
- If a trend line or pattern is visible, describe it

INFOGRAPHIC HANDLING:
- For infographics, create a structured description preserving all information:
  > **Infographic: [Title/Subject]**
  > - Section: [heading] — [detailed content]
  > - Section: [heading] — [detailed content]
  > - Key statistics: [list all numbers and percentages]
  > - Visual elements: [describe icons, illustrations, color coding]
- Every number, percentage, and statistic must be transcribed exactly

DIAGRAM AND FLOWCHART HANDLING:
- Describe the structure, nodes, and connections:
  > **Diagram: [Type — e.g., Flowchart / Architecture / Circuit]: [Title]**
  > - Components: [list all nodes/blocks with labels]
  > - Connections: [A] → [B] → [C]
  > - Decision points: [describe any branching logic]
  > - Labels/annotations: [describe any text on arrows or connections]
- If the diagram has a clear flow, represent it with arrow notation
- Preserve all labels, even small ones

IMAGE AND ILLUSTRATION HANDLING:
- For informative images (photos, screenshots, illustrations with content):
  > **Image: [Brief description of what is shown]**
  > - Subject: [what the image depicts]
  > - Key elements: [important visible details, objects, people]
  > - Text in image: [any readable text within the image]
  > - Context: [how it relates to the surrounding content if apparent]
- For purely decorative images, omit them

FORMULA AND EQUATION HANDLING:
- Inline equations: $E = mc^2$
- Block equations:
  $$
  \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
  $$
- Preserve all subscripts, superscripts, Greek letters, and special symbols

UNCLEAR SECTIONS:
- Mark truly illegible text with [?]
- Mark uncertain readings with [?!] followed by your best guess: [?!probably this word]
- Mark partially visible text with [...?] for missing portions

REMEMBER: Your goal is to create a Markdown document that preserves ALL information from the original page. A reader should be able to understand the complete content without ever seeing the original document.
"""

# Simplified prompt for image files (not PDF pages) — more conversational
IMAGE_OCR_PROMPT = """\
You are a precise image transcription engine. Extract ALL content from this image into clean Markdown.

Follow the same rules as the main OCR prompt:
- Transcribe every piece of text
- Convert tables to Markdown tables
- Describe graphs, charts, and infographics in structured format
- Describe diagrams and flowcharts with connections
- Describe informative images with details
- Use LaTeX for equations
- Mark unclear text with [?]
- Output ONLY Markdown — no preamble or commentary

Be thorough. Every detail matters.
"""


def _render_pdf_page(page, dpi: int, max_dim: int) -> str:
    """Render a PDF page to base64 PNG. CPU-bound, runs in thread pool."""
    import fitz  # PyMuPDF — imported here to allow graceful failure if missing
    from PIL import Image

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _load_image_file(path: Path, max_dim: int) -> str:
    """Load an image file, resize if needed, return base64 PNG."""
    from PIL import Image

    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = "PNG"
    img.save(buf, format=fmt, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


async def _ocr_image(
    client: httpx.AsyncClient,
    b64: str,
    page_num: int,
    sem: asyncio.Semaphore,
    prompt: str = OCR_PROMPT,
) -> tuple[int, str]:
    """Send a single image to the OCR model and return the markdown."""
    async with sem:
        payload = {
            "model":      OCR_MODEL,
            "prompt":     prompt,
            "images":     [b64],
            "stream":     False,
            "options":    {"temperature": 0.0},
            "keep_alive": KEEP_ALIVE,
        }
        try:
            r = await client.post(OLLAMA_URL, json=payload, timeout=180.0)
            r.raise_for_status()
            return page_num, r.json()["response"].strip()
        except Exception as e:
            log.warning(f"  Page {page_num} OCR error: {e}")
            return page_num, f"> **Page {page_num} OCR failed:** `{e}`"


async def ocr_pdf(
    pdf_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Run OCR on a PDF. Returns assembled Markdown string.
    progress_cb(done, total) called after each page.
    """
    import fitz

    doc   = fitz.open(str(pdf_path))
    total = len(doc)
    log.info(f"OCR: {pdf_path.name} ({total} pages, workers={MAX_WORKERS})")

    # Pre-render all pages in thread pool (CPU-bound)
    pages_b64: list[str] = await asyncio.gather(*[
        asyncio.to_thread(_render_pdf_page, doc[i], DPI, MAX_DIM)
        for i in range(total)
    ])
    doc.close()

    sem = asyncio.Semaphore(MAX_WORKERS)
    results: list[tuple[int, str]] = []
    done = 0

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_ocr_image(client, pages_b64[i], i + 1, sem, OCR_PROMPT))
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


async def ocr_image(
    image_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Run OCR on a single image file. Returns Markdown string.
    """
    log.info(f"OCR: {image_path.name} (image file)")

    if progress_cb:
        progress_cb(0, 1)

    b64 = await asyncio.to_thread(_load_image_file, image_path, MAX_DIM)

    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient() as client:
        _, md = await _ocr_image(client, b64, 1, sem, IMAGE_OCR_PROMPT)

    if progress_cb:
        progress_cb(1, 1)

    title = image_path.stem.replace("_", " ")
    return f"# {title}\n\n{md}\n"
