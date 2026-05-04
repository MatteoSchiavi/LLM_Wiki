"""
OCR engine: PDF/Image → Markdown via vision model on Ollama.
Audio transcription via local Whisper model (faster_whisper).

VRAM management:
  - During OCR page iteration: keep_alive="5m" (model stays loaded during the loop)
  - After all OCR pages complete: explicitly unload the OCR model by POSTing to
    ollama /api/generate with {"model": OCR_MODEL, "keep_alive": 0}
  - This ensures the OCR model is fully evicted from VRAM before Qwen loads

Smart routing (handled in main.py):
  - PDFs/Images  → this OCR module → raw .md
  - Audio        → this transcription module → raw .md
  - Text/Code    → SKIP this module entirely, go directly to Qwen

Supports:
  PDF:  .pdf
  Images: .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp
  Audio: .mp3, .wav, .m4a, .flac, .ogg, .webm
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

# ─── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_URL  = CFG["ollama_url"] + "/api/generate"
OCR_MODEL   = CFG["ocr_model"]
MAX_WORKERS = CFG["ocr"]["max_concurrency"]
DPI         = CFG["ocr"]["dpi"]
MAX_DIM     = CFG["ocr"]["max_image_dimension"]

# keep_alive during the OCR loop: keep the model loaded for 5 minutes so
# successive pages don't trigger a cold load each time
KEEP_ALIVE_DURING = "5m"

# ─── Supported extensions ───────────────────────────────────────────────────────

PDF_EXTENSIONS    = {".pdf"}
IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS  = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

# ─── OCR Prompt (PDF pages) ────────────────────────────────────────────────────

OCR_PROMPT = """\
You are a precise document OCR and transcription engine. Your task is to convert this document page into clean, comprehensive Markdown.

CRITICAL RULES:
1. Transcribe EVERY piece of text on this page — do not skip, summarize, or paraphrase anything
2. Maintain the original reading order (top-to-bottom, left-to-right)
3. Output ONLY the Markdown transcription — no preamble, no commentary, no explanations
4. If the page is blank or contains no readable content, output: `(blank page)`

TEXT HANDLING:
- Use headings (# ## ### ####) to match the document's visual hierarchy
- Preserve bold (**text**) and italic (*text*) formatting where visible
- Keep all footnotes, citations, and references exactly as they appear
- Transcribe equations using LaTeX notation: inline $equation$ or block $$equation$$
- Preserve list formatting (ordered and unordered)
- Keep paragraph breaks where they appear in the original
- For columns: read top-to-bottom within each column, then left-to-right across columns
- For headers/footers: include them once at the top/bottom if they contain useful info

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
- Include table captions or titles if present
- For multi-page tables that continue from a previous page, add:
  > **Table continued from previous page**

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
- For pie charts: list each slice with its label and percentage

INFOGRAPHIC HANDLING:
- For infographics, create a structured description preserving all information:
  > **Infographic: [Title/Subject]**
  > - Section: [heading] — [detailed content]
  > - Section: [heading] — [detailed content]
  > - Key statistics: [list all numbers and percentages]
  > - Visual elements: [describe icons, illustrations, color coding]
- Every number, percentage, and statistic must be transcribed exactly
- Include any timeline or process flow information

DIAGRAM AND FLOWCHART HANDLING:
- Describe the structure, nodes, and connections:
  > **Diagram: [Type — e.g., Flowchart / Architecture / Circuit / UML]: [Title]**
  > - Components: [list all nodes/blocks with labels]
  > - Connections: [A] → [B] → [C]
  > - Decision points: [describe any branching logic]
  > - Labels/annotations: [describe any text on arrows or connections]
- If the diagram has a clear flow, represent it with arrow notation
- Preserve all labels, even small ones
- For circuit diagrams: list components and their connections
- For UML diagrams: describe classes, methods, and relationships

IMAGE AND ILLUSTRATION HANDLING:
- For informative images (photos, screenshots, illustrations with content):
  > **Image: [Brief description of what is shown]**
  > - Subject: [what the image depicts]
  > - Key elements: [important visible details, objects, people]
  > - Text in image: [any readable text within the image]
  > - Context: [how it relates to the surrounding content if apparent]
- For purely decorative images, omit them
- For maps: describe key locations, boundaries, and labels

FORMULA AND EQUATION HANDLING:
- Inline equations: $E = mc^2$
- Block equations:
  $$
  \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
  $$
- Preserve all subscripts, superscripts, Greek letters, and special symbols
- For chemical formulas: H₂SO₄, NaCl
- For multi-line equations, use aligned blocks

CODE AND MONOSPACED TEXT:
- Wrap code in backticks: `variable_name`
- For code blocks:
  ```python
  def example():
      return 42
  ```
- Preserve indentation exactly

UNCLEAR SECTIONS:
- Mark truly illegible text with [?]
- Mark uncertain readings with [?!] followed by your best guess: [?!probably this word]
- Mark partially visible text with [...?] for missing portions
- If an entire section is unreadable: [illegible section — approx. N lines]

PAGE LAYOUT:
- If the page has multiple columns, transcribe each column fully before moving to the next
- For sidebars/callout boxes: transcribe with > blockquote formatting
- For page numbers: ignore them (they are not content)
- For watermarks: note them as [watermark: text] only if they obscure content

REMEMBER: Your goal is to create a Markdown document that preserves ALL information from the original page. A reader should be able to understand the complete content without ever seeing the original document.
"""

# ─── OCR Prompt (image files) ──────────────────────────────────────────────────

IMAGE_OCR_PROMPT = """\
You are a precise image transcription engine. Extract ALL content from this image into clean Markdown.

Follow the same comprehensive rules as the main OCR prompt:
- Transcribe every piece of text, no matter how small
- Convert tables to Markdown tables with all numerical data preserved exactly
- Describe graphs, charts, and infographics in structured format (axes, data points, trends)
- Describe diagrams and flowcharts with connections and arrow notation
- Describe informative images with details, text overlays, and context
- Use LaTeX for equations and formulas
- Handle code blocks with proper language tags
- Mark unclear text with [?], uncertain readings with [?!guess], missing portions with [...?]
- For blank images or images with no content, output: `(blank image)`
- Output ONLY Markdown — no preamble or commentary

EXTRA IMAGE-SPECIFIC RULES:
- If this is a screenshot: preserve all visible UI text, buttons, menus, and content
- If this is a photo of a document: treat it like a document page
- If this is a diagram: describe all nodes, connections, labels, and arrows
- If this is a chart/graph: extract all visible data points and values
- If text is at an angle or rotated: transcribe it as normal horizontal text
- If there are watermarks: note them but don't let them obscure the main content

Be thorough. Every detail matters.
"""


# ─── VRAM management ───────────────────────────────────────────────────────────

async def unload_ocr_model() -> None:
    """
    Explicitly unload the OCR model from VRAM by sending a POST to Ollama's
    /api/generate endpoint with keep_alive=0.  This ensures the model is
    fully evicted from GPU memory so the next model (e.g. Qwen) can load
    without competing for VRAM.
    """
    log.info(f"Unloading OCR model '{OCR_MODEL}' from VRAM (keep_alive=0)")
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model":      OCR_MODEL,
                "prompt":     "",
                "keep_alive": 0,
            }
            r = await client.post(OLLAMA_URL, json=payload, timeout=30.0)
            r.raise_for_status()
            log.info("OCR model unloaded successfully")
    except Exception as e:
        log.warning(f"Failed to unload OCR model (non-fatal): {e}")


# ─── PDF rendering helpers ─────────────────────────────────────────────────────

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
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Core OCR call ─────────────────────────────────────────────────────────────

async def _ocr_image(
    client: httpx.AsyncClient,
    b64: str,
    page_num: int,
    sem: asyncio.Semaphore,
    prompt: str = OCR_PROMPT,
) -> tuple[int, str]:
    """
    Send a single image to the OCR model and return the markdown.
    Uses keep_alive="5m" so the model stays loaded during the page loop.
    """
    async with sem:
        payload = {
            "model":      OCR_MODEL,
            "prompt":     prompt,
            "images":     [b64],
            "stream":     False,
            "options":    {"temperature": 0.0},
            "keep_alive": KEEP_ALIVE_DURING,
        }
        try:
            r = await client.post(OLLAMA_URL, json=payload, timeout=180.0)
            r.raise_for_status()
            return page_num, r.json()["response"].strip()
        except Exception as e:
            log.warning(f"  Page {page_num} OCR error: {e}")
            return page_num, f"> **Page {page_num} OCR failed:** `{e}`"


# ─── Public API: PDF OCR ──────────────────────────────────────────────────────

async def ocr_pdf(
    pdf_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Run OCR on a PDF. Returns assembled Markdown string.
    progress_cb(done, total) called after each page.

    VRAM management:
      - Pages are OCR'd with keep_alive="5m" so the model stays in VRAM
        across the page loop (avoids repeated cold loads).
      - After ALL pages are done, the OCR model is explicitly unloaded via
        unload_ocr_model() to free VRAM before the next pipeline stage.
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

    try:
        async with httpx.AsyncClient() as client:
            tasks = [
                asyncio.create_task(
                    _ocr_image(client, pages_b64[i], i + 1, sem, OCR_PROMPT)
                )
                for i in range(total)
            ]
            for fut in asyncio.as_completed(tasks):
                page_num, md = await fut
                results.append((page_num, md))
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                log.info(f"  OCR {done}/{total}")
    finally:
        # ── Explicitly unload OCR model from VRAM ─────────────────────────
        # Regardless of success or failure, evict the model so Qwen can load.
        await unload_ocr_model()

    results.sort(key=lambda x: x[0])

    lines = [f"# {pdf_path.stem.replace('_', ' ')}\n\n"]
    for page_num, md in results:
        lines.append(f"\n<!-- page:{page_num} -->\n{md}\n\n---\n")

    return "".join(lines)


# ─── Public API: Image OCR ─────────────────────────────────────────────────────

async def ocr_image(
    image_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Run OCR on a single image file. Returns Markdown string.

    VRAM management:
      - Uses keep_alive="5m" during the call.
      - Unloads the OCR model immediately after to free VRAM.
    """
    log.info(f"OCR: {image_path.name} (image file)")

    if progress_cb:
        progress_cb(0, 1)

    b64 = await asyncio.to_thread(_load_image_file, image_path, MAX_DIM)

    sem = asyncio.Semaphore(1)
    try:
        async with httpx.AsyncClient() as client:
            _, md = await _ocr_image(client, b64, 1, sem, IMAGE_OCR_PROMPT)
    finally:
        # ── Explicitly unload OCR model from VRAM ─────────────────────────
        await unload_ocr_model()

    if progress_cb:
        progress_cb(1, 1)

    title = image_path.stem.replace("_", " ")
    return f"# {title}\n\n{md}\n"


# ─── Public API: Audio transcription ──────────────────────────────────────────

def _detect_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            log.info("CUDA detected — using GPU for Whisper transcription")
            return "cuda"
    except ImportError:
        pass
    log.info("No CUDA detected — using CPU for Whisper transcription")
    return "cpu"


def _transcribe_sync(audio_path: Path) -> str:
    """
    Synchronous transcription using faster_whisper. Runs in a thread pool
    so it doesn't block the event loop.
    """
    from faster_whisper import WhisperModel

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    log.info(f"Loading Whisper model (base, device={device}, compute_type={compute_type})")
    model = WhisperModel("base", device=device, compute_type=compute_type)

    log.info(f"Transcribing: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
    )

    language = info.language
    language_probability = info.language_probability
    log.info(
        f"Detected language: {language} "
        f"(probability: {language_probability:.2%})"
    )

    # Build formatted transcript with timestamps
    transcript_parts: list[str] = []
    for segment in segments:
        start = segment.start
        end = segment.end
        text = segment.text.strip()
        if text:
            # Format timestamp as HH:MM:SS.mmm
            start_fmt = _format_timestamp(start)
            end_fmt = _format_timestamp(end)
            transcript_parts.append(f"[{start_fmt} → {end_fmt}]  {text}")

    return language, language_probability, "\n\n".join(transcript_parts)


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole_secs = int(secs)
    millis = int((secs - whole_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_secs:02d}.{millis:03d}"


async def transcribe_audio(
    audio_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> str:
    """
    Transcribe an audio file using a local Whisper model (faster_whisper).
    Returns a Markdown-formatted transcript with timestamps.

    Supported formats: MP3, WAV, M4A, FLAC, OGG, WEBM.
    Language is detected automatically.

    progress_cb(done, total) called: (0,1) at start, (1,1) at end.
    """
    suffix = audio_path.suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {suffix}. "
            f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    log.info(f"Transcribe: {audio_path.name}")

    if progress_cb:
        progress_cb(0, 1)

    # Run the synchronous transcription in a thread pool
    language, language_probability, transcript = await asyncio.to_thread(
        _transcribe_sync, audio_path
    )

    if progress_cb:
        progress_cb(1, 1)

    # Assemble final Markdown output
    title = audio_path.stem.replace("_", " ")
    parts = [
        f"# {title}\n",
        f"\n> **Audio transcription** — Language: {language} "
        f"({language_probability:.0%} confidence)\n",
        f"> **Source:** `{audio_path.name}`\n",
        "\n---\n",
        f"\n{transcript}\n",
    ]

    return "".join(parts)
