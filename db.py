"""Async SQLite database layer for Secondo Cervello using aiosqlite.

Provides connection-per-operation isolation so multiple async tasks never
share a single sqlite3 connection (which is not thread-safe and would
serialize under the GIL anyway).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

log = logging.getLogger("db")

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_RAW_SOURCES = """
CREATE TABLE IF NOT EXISTS raw_sources_log (
    id              TEXT PRIMARY KEY,
    file_path       TEXT,
    sha256          TEXT,
    ocr_status      TEXT,
    ocr_completed_at TEXT,
    wiki_status     TEXT,
    wiki_completed_at TEXT,
    created_at      TEXT,
    updated_at      TEXT
)
"""

_SCHEMA_WIKI_ENTITIES = """
CREATE TABLE IF NOT EXISTS wiki_entities_log (
    id          TEXT PRIMARY KEY,
    file_path   TEXT UNIQUE,
    source_file TEXT,
    page_type   TEXT,
    category    TEXT,
    title       TEXT,
    created_at  TEXT,
    updated_at  TEXT
)
"""

_SCHEMA_JOBS = """
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
"""

_ALL_SCHEMAS = [_SCHEMA_RAW_SOURCES, _SCHEMA_WIKI_ENTITIES, _SCHEMA_JOBS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()


def _row_to_dict(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    """Convert an aiosqlite.Row to a plain dict, or return None."""
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: list[aiosqlite.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    """Async SQLite database with per-operation connections.

    Usage::

        db = Database(".state/jobs.db")
        await db.init()
        await db.upsert_source(...)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._initialized = False

    # ── Connection context manager ────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a fresh connection with row_factory set, then close it."""
        con = await aiosqlite.connect(self.db_path)
        con.row_factory = aiosqlite.Row
        try:
            yield con
            await con.commit()
        except Exception:
            await con.rollback()
            raise
        finally:
            await con.close()

    # ── Initialisation ────────────────────────────────────────────────────

    async def init(self) -> None:
        """Create all tables if they do not exist and ensure parent dirs."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as con:
            for ddl in _ALL_SCHEMAS:
                await con.execute(ddl)
        self._initialized = True
        log.info("Database initialised at %s", self.db_path)

    # ── raw_sources_log ───────────────────────────────────────────────────

    async def upsert_source(
        self,
        file_id: str,
        file_path: str,
        sha256: str,
        **kwargs: Any,
    ) -> None:
        """Insert a new source or update an existing one by id."""
        now = _now()
        async with self._connect() as con:
            row = await con.execute_fetchall(
                "SELECT id FROM raw_sources_log WHERE id = ?", (file_id,)
            )
            if row:
                # Update existing
                sets: dict[str, Any] = {"updated_at": now, **kwargs}
                set_clause = ", ".join(f"{k} = ?" for k in sets)
                values = list(sets.values()) + [file_id]
                await con.execute(
                    f"UPDATE raw_sources_log SET {set_clause} WHERE id = ?",
                    values,
                )
                log.debug("Updated source %s", file_id)
            else:
                # Insert new
                cols = {
                    "id": file_id,
                    "file_path": file_path,
                    "sha256": sha256,
                    "ocr_status": kwargs.get("ocr_status", "pending"),
                    "wiki_status": kwargs.get("wiki_status", "pending"),
                    "created_at": now,
                    "updated_at": now,
                }
                # Merge any extra kwargs
                for k, v in kwargs.items():
                    if k not in cols:
                        cols[k] = v
                    else:
                        cols[k] = v  # allow overrides
                col_names = ", ".join(cols.keys())
                placeholders = ", ".join("?" for _ in cols)
                await con.execute(
                    f"INSERT INTO raw_sources_log ({col_names}) VALUES ({placeholders})",
                    list(cols.values()),
                )
                log.debug("Inserted source %s", file_id)

    async def get_source_by_path(self, file_path: str) -> dict[str, Any] | None:
        """Return a source row matching *file_path*, or None."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM raw_sources_log WHERE file_path = ?", (file_path,)
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)

    async def get_source_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Return a source row matching *sha256*, or None."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM raw_sources_log WHERE sha256 = ?", (sha256,)
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)

    async def get_pending_sources(self) -> list[dict[str, Any]]:
        """Return all sources whose wiki_status is not 'done'."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM raw_sources_log WHERE wiki_status != 'done' "
                "ORDER BY created_at ASC"
            )
            rows = await cursor.fetchall()
        return _rows_to_dicts(rows)

    # ── wiki_entities_log ─────────────────────────────────────────────────

    async def upsert_entity(
        self,
        file_path: str,
        source_file: str,
        page_type: str,
        category: str,
        title: str,
    ) -> None:
        """Insert or update an entity row keyed by file_path (UNIQUE)."""
        now = _now()
        async with self._connect() as con:
            row = await con.execute_fetchall(
                "SELECT id FROM wiki_entities_log WHERE file_path = ?", (file_path,)
            )
            if row:
                await con.execute(
                    """UPDATE wiki_entities_log
                       SET source_file = ?, page_type = ?, category = ?,
                           title = ?, updated_at = ?
                       WHERE file_path = ?""",
                    (source_file, page_type, category, title, now, file_path),
                )
                log.debug("Updated entity %s", file_path)
            else:
                import uuid
                entity_id = str(uuid.uuid4())
                await con.execute(
                    """INSERT INTO wiki_entities_log
                       (id, file_path, source_file, page_type, category,
                        title, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (entity_id, file_path, source_file, page_type, category,
                     title, now, now),
                )
                log.debug("Inserted entity %s", file_path)

    async def get_entity(self, file_path: str) -> dict[str, Any] | None:
        """Return an entity row matching *file_path*, or None."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM wiki_entities_log WHERE file_path = ?", (file_path,)
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)

    async def get_all_entities(self) -> list[dict[str, Any]]:
        """Return all entity rows, newest first."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM wiki_entities_log ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
        return _rows_to_dicts(rows)

    async def delete_entity(self, file_path: str) -> bool:
        """Delete an entity row by file_path. Returns True if a row was deleted."""
        async with self._connect() as con:
            cursor = await con.execute(
                "DELETE FROM wiki_entities_log WHERE file_path = ?", (file_path,)
            )
            deleted = cursor.rowcount
        if deleted:
            log.info("Deleted entity %s", file_path)
        else:
            log.warning("No entity found to delete for %s", file_path)
        return bool(deleted)

    # ── jobs ──────────────────────────────────────────────────────────────

    async def upsert_job(self, job_id: str, **kwargs: Any) -> None:
        """Insert a new job or update an existing one by id."""
        now = _now()
        async with self._connect() as con:
            row = await con.execute_fetchall(
                "SELECT id FROM jobs WHERE id = ?", (job_id,)
            )
            if row:
                sets = {"updated_at": now, **kwargs}
                set_clause = ", ".join(f"{k} = ? " for k in sets)
                values = list(sets.values()) + [job_id]
                await con.execute(
                    f"UPDATE jobs SET {set_clause} WHERE id = ?", values
                )
                log.debug("Updated job %s", job_id)
            else:
                cols = {
                    "id": job_id,
                    "created_at": now,
                    "updated_at": now,
                    **kwargs,
                }
                col_names = ", ".join(cols.keys())
                placeholders = ", ".join("?" for _ in cols)
                await con.execute(
                    f"INSERT INTO jobs ({col_names}) VALUES ({placeholders})",
                    list(cols.values()),
                )
                log.debug("Inserted job %s", job_id)

    async def get_all_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent jobs, newest first."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
        return _rows_to_dicts(rows)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a single job row, or None."""
        async with self._connect() as con:
            cursor = await con.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)
