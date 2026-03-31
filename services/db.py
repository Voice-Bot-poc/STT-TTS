"""
db.py — Async SQLAlchemy + aiomysql database layer for conversation history.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CREATE TABLE SQL (run this once against your MySQL database):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CREATE TABLE IF NOT EXISTS conversation_history (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id      VARCHAR(128)    NOT NULL,
        user_text       TEXT            NOT NULL,
        assistant_text  TEXT            NOT NULL,
        created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session_time (session_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design decisions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- We use SQLAlchemy 2.x async mode with aiomysql as the DBAPI driver.
  Connection string format:  mysql+aiomysql://user:pass@host:port/db
- The engine and session factory are created lazily (on first DB call) so
  that import of this module does NOT immediately attempt a DB connection.
  This allows the service to start even when MySQL is temporarily unreachable.
- Each query function acquires its own session and commits/rolls back
  independently.  There is no shared transaction state between calls.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine + session factory — lazy init
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_connection_url() -> str:
    """Construct the aiomysql connection URL from env vars."""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DB", "voicebot")
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db}"


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = _build_connection_url()
        logger.info("Creating async MySQL engine …")
        _engine = create_async_engine(
            url,
            pool_size=5,          # keep pool small for a POC service
            max_overflow=10,
            pool_pre_ping=True,   # validate connections before use
            echo=False,           # set True to log all SQL for debugging
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # avoids lazy-load errors after commit
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a SQLAlchemy AsyncSession.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(...)

    Automatically commits on success and rolls back on exception.
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

async def fetch_history(session_id: str, limit: int = 5) -> list[dict]:
    """
    Retrieve the last *limit* conversation exchanges for *session_id*.

    Returns a list of dicts ordered oldest → newest, each with keys:
        'user_text', 'assistant_text'

    We fetch DESC (newest first) then reverse so the LLM sees oldest first.
    """
    sql = text(
        """
        SELECT user_text, assistant_text
        FROM conversation_history
        WHERE session_id = :session_id
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    async with get_db_session() as session:
        result = await session.execute(
            sql, {"session_id": session_id, "limit": limit}
        )
        rows = result.fetchall()

    # Reverse so history is chronological (oldest → newest)
    history = [
        {"user_text": row.user_text, "assistant_text": row.assistant_text}
        for row in reversed(rows)
    ]

    logger.info(
        "Fetched %d history rows for session '%s'", len(history), session_id
    )
    return history


async def save_exchange(
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """
    Insert one conversation exchange into the history table.

    Args:
        session_id:     Conversation/session identifier.
        user_text:      The user's transcribed speech.
        assistant_text: The LLM's reply.
    """
    sql = text(
        """
        INSERT INTO conversation_history (session_id, user_text, assistant_text, created_at)
        VALUES (:session_id, :user_text, :assistant_text, :created_at)
        """
    )

    async with get_db_session() as session:
        await session.execute(
            sql,
            {
                "session_id": session_id,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "created_at": datetime.utcnow(),
            },
        )

    logger.info("Saved exchange for session '%s'", session_id)


async def dispose_engine() -> None:
    """
    Gracefully close all pooled connections.  Call on application shutdown
    (e.g. in a FastAPI lifespan handler) to avoid connection leaks.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        logger.info("MySQL engine disposed.")
        _engine = None
