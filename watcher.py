"""
Folder watcher: Monitors an inbox directory for new files and automatically
processes them through the Secondo Cervello pipeline.

Uses watchdog's async observer to detect new files. When a file appears,
it waits for it to be fully written (stable size for 2 seconds), then
pushes it through the same pipeline as the /api/upload endpoint.

Configuration (config.yaml):
  watch_enabled: true
  watch_folder: "D:\\obsidian\\LLM_Wiki_inbox"
  watch_debounce_sec: 2.0    # seconds to wait for file to stabilize
  watch_poll_sec: 1.0        # how often to check file size stability
"""

import asyncio
import hashlib
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config_loader import CFG
from ocr import PDF_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS

log = logging.getLogger("watcher")

# ── Text-based extensions that skip OCR and go directly to Qwen ────────────────
# These are read as plain text and never sent through the vision model.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".htm",
    ".log", ".ini", ".cfg", ".yaml", ".yml", ".toml", ".rst", ".tex",
    ".srt", ".vtt",  # subtitle files
}

# ── Code extensions also treated as text (skipped OCR) ─────────────────────────
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh", ".sql",
    ".r", ".R", ".m", ".swift", ".kt", ".scala", ".lua", ".pl",
}

ALL_TEXT_EXTENSIONS = TEXT_EXTENSIONS | CODE_EXTENSIONS

# ── All supported extensions ───────────────────────────────────────────────────
ALL_SUPPORTED = PDF_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | ALL_TEXT_EXTENSIONS


def get_file_route(suffix: str) -> str:
    """
    Determine the processing route for a file based on its extension.

    Returns:
        "ocr"       — needs OCR (PDFs, images)
        "transcribe" — audio transcription (mp3, wav, etc.)
        "text"      — plain text, skip OCR, go directly to Qwen
        "unsupported" — file type not handled
    """
    suffix = suffix.lower()
    if suffix in PDF_EXTENSIONS or suffix in IMAGE_EXTENSIONS:
        return "ocr"
    elif suffix in AUDIO_EXTENSIONS:
        return "transcribe"
    elif suffix in ALL_TEXT_EXTENSIONS:
        return "text"
    else:
        return "unsupported"


class InboxHandler(FileSystemEventHandler):
    """
    Watchdog handler that detects new files in the inbox folder.

    When a file is created or modified, it schedules processing after
    a debounce period (to ensure the file is fully written).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, pipeline_fn, inbox: Path):
        super().__init__()
        self.loop = loop
        self.pipeline_fn = pipeline_fn  # async callable: pipeline_fn(job_id, filename, tmp_path)
        self.inbox = inbox
        self._pending: dict[str, asyncio.Task] = {}  # path -> debounce task
        self._processed: set[str] = set()  # track already-processed files

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        # Re-schedule if file was modified (resets debounce timer)
        self._schedule_file(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        # File moved into inbox
        self._schedule_file(event.dest_path)

    def _schedule_file(self, file_path: str):
        """Schedule a file for processing after debounce period."""
        path = Path(file_path)

        # Skip hidden files, temp files, and unsupported types
        if path.name.startswith(".") or path.name.startswith("~"):
            return
        if path.suffix.lower() not in ALL_SUPPORTED:
            log.debug(f"Skipping unsupported file: {path.name}")
            return

        # Cancel any existing debounce task for this file
        if file_path in self._pending:
            self._pending[file_path].cancel()

        # Create a new debounce task
        debounce_sec = CFG.get("watch_debounce_sec", 2.0)
        task = self.loop.create_task(self._debounce_and_process(file_path, debounce_sec))
        self._pending[file_path] = task

    async def _debounce_and_process(self, file_path: str, debounce_sec: float):
        """Wait for file to stabilize, then process it."""
        try:
            # Wait for debounce period
            await asyncio.sleep(debounce_sec)

            path = Path(file_path)
            if not path.exists():
                log.debug(f"File disappeared during debounce: {path.name}")
                return

            # Wait for file size to stabilize (not being written to)
            poll_sec = CFG.get("watch_poll_sec", 1.0)
            prev_size = -1
            stable_count = 0
            for _ in range(10):  # max 10 checks = 10 seconds
                try:
                    current_size = path.stat().st_size
                except OSError:
                    await asyncio.sleep(poll_sec)
                    continue

                if current_size == prev_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 2:  # size unchanged for 2 consecutive checks
                        break
                else:
                    stable_count = 0
                prev_size = current_size
                await asyncio.sleep(poll_sec)

            if path.stat().st_size == 0:
                log.warning(f"File is empty, skipping: {path.name}")
                return

            # Check if we already processed this file
            file_key = f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime}"
            if file_key in self._processed:
                return
            self._processed.add(file_key)

            route = get_file_route(path.suffix)
            log.info(f"Watch: new file detected — {path.name} (route: {route})")

            # Copy to tmp (the pipeline deletes tmp after processing)
            job_id = str(uuid.uuid4())
            tmp_dir = Path(CFG["uploads_tmp"])
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"{job_id}_{path.name}"
            shutil.copy2(path, tmp_path)

            # Run the pipeline
            try:
                await self.pipeline_fn(job_id, path.name, tmp_path)
                log.info(f"Watch: processed {path.name} -> job {job_id[:8]}")
            except Exception as e:
                log.error(f"Watch: pipeline failed for {path.name}: {e}")

        except asyncio.CancelledError:
            pass  # debounce was reset, that's fine
        except Exception as e:
            log.error(f"Watch: error processing {file_path}: {e}")
        finally:
            self._pending.pop(file_path, None)


class FolderWatcher:
    """
    Manages the inbox folder watcher lifecycle.

    Usage:
        watcher = FolderWatcher(pipeline_fn)
        watcher.start()   # starts watching
        watcher.stop()    # stops watching
    """

    def __init__(self, pipeline_fn):
        self.pipeline_fn = pipeline_fn
        self.inbox: Path = Path(CFG.get("watch_folder", str(Path(CFG["vault_dir"]) / "inbox")))
        self._observer: Observer | None = None
        self._handler: InboxHandler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def start(self):
        """Start watching the inbox folder."""
        if not CFG.get("watch_enabled", False):
            log.info("Folder watcher is disabled in config (watch_enabled: false)")
            return

        # Create inbox directory if it doesn't exist
        self.inbox.mkdir(parents=True, exist_ok=True)

        # Also create a "processed" subfolder for already-handled files
        (self.inbox / "_processed").mkdir(parents=True, exist_ok=True)

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        self._handler = InboxHandler(self._loop, self.pipeline_fn, self.inbox)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.inbox), recursive=False)
        self._observer.daemon = True
        self._observer.start()

        log.info(f"Folder watcher started — monitoring: {self.inbox}")
        log.info(f"  Route map:")
        log.info(f"    PDFs, Images    -> OCR (glm-ocr)")
        log.info(f"    Audio files     -> Transcribe (Whisper)")
        log.info(f"    Text/Code files -> Direct to Qwen (no OCR)")

        # Process any files already in the inbox on startup
        self._process_existing_files()

    def _process_existing_files(self):
        """Process any files that were already in the inbox before watcher started."""
        if not self.inbox.exists():
            return

        for f in sorted(self.inbox.iterdir()):
            if f.is_file() and not f.name.startswith(".") and not f.name.startswith("~"):
                if f.suffix.lower() in ALL_SUPPORTED:
                    route = get_file_route(f.suffix)
                    log.info(f"Watch: found existing file in inbox — {f.name} (route: {route})")
                    # Schedule via the handler (with debounce)
                    if self._handler:
                        self._handler._schedule_file(str(f))

    def stop(self):
        """Stop watching the inbox folder."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            log.info("Folder watcher stopped")
