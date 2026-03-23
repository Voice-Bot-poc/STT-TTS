"""
pipeline.py — Orchestrates the full STT → LLM → MySQL → TTS pipeline.

Each stage is executed sequentially:
    1. STT       — transcribe audio bytes with Whisper
    2. DB fetch  — retrieve last 5 conversation turns from MySQL
    3. LLM       — call Anthropic Claude with history + transcript
    4. DB write  — store the new exchange in MySQL
    5. TTS       — synthesise LLM reply to base64 MP3 audio

Latency is measured around each stage with time.perf_counter() (monotonic,
high-resolution; not affected by system clock adjustments).

Error handling strategy:
    - Every stage is individually wrapped in try/except.
    - On failure, an HTTPException is raised with a structured detail dict:
          { "stage": "<stage_name>", "detail": "<error message>" }
    - This prevents any stage's exception from silently corrupting the next
      stage's inputs.  FastAPI serialises the detail dict to JSON automatically.
"""

import logging
import time
from asyncio import get_event_loop
from functools import partial

from fastapi import HTTPException

import db
import llm
import stt
import tts_util
from models import LatencyBreakdown, ProcessResponse

logger = logging.getLogger(__name__)


def _ms(start: float, end: float) -> float:
    """Convert perf_counter interval (seconds) to milliseconds, rounded to 1 dp."""
    return round((end - start) * 1000, 1)


async def run_pipeline(session_id: str, audio_bytes: bytes) -> ProcessResponse:
    """
    Run the full VoiceBot pipeline and return a ProcessResponse.

    Args:
        session_id:  Client-supplied conversation identifier.
        audio_bytes: Raw audio bytes from the uploaded file.

    Returns:
        ProcessResponse with transcript, LLM reply, audio, and latency data.

    Raises:
        HTTPException(500) — with { stage, detail } if any stage fails.
    """
    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------ STT
    t0 = time.perf_counter()
    try:
        transcript = await stt.transcribe(audio_bytes)
    except Exception as exc:
        logger.exception("STT stage failed")
        raise HTTPException(
            status_code=500,
            detail={"stage": "stt", "detail": str(exc)},
        )
    t_stt = _ms(t0, time.perf_counter())
    logger.info("[STT] %.1f ms → %r", t_stt, transcript[:80])

    # ------------------------------------------------------------------ DB FETCH
    t0 = time.perf_counter()
    try:
        history = await db.fetch_history(session_id, limit=5)
    except Exception as exc:
        logger.exception("DB fetch stage failed for session '%s'", session_id)
        raise HTTPException(
            status_code=500,
            detail={"stage": "db", "detail": f"History fetch failed: {exc}"},
        )
    t_db_fetch = _ms(t0, time.perf_counter())
    logger.info("[DB fetch] %.1f ms → %d rows", t_db_fetch, len(history))

    # ------------------------------------------------------------------ LLM
    t0 = time.perf_counter()
    try:
        messages = llm.build_messages(history, transcript)
        response_text = await llm.call_llm(messages)
    except Exception as exc:
        logger.exception("LLM stage failed")
        raise HTTPException(
            status_code=500,
            detail={"stage": "llm", "detail": str(exc)},
        )
    t_llm = _ms(t0, time.perf_counter())
    logger.info("[LLM] %.1f ms → %r", t_llm, response_text[:80])

    # ------------------------------------------------------------------ DB WRITE
    t0 = time.perf_counter()
    try:
        await db.save_exchange(session_id, transcript, response_text)
    except Exception as exc:
        logger.exception("DB write stage failed for session '%s'", session_id)
        raise HTTPException(
            status_code=500,
            detail={"stage": "db", "detail": f"History write failed: {exc}"},
        )
    t_db_write = _ms(t0, time.perf_counter())
    logger.info("[DB write] %.1f ms", t_db_write)

    # ------------------------------------------------------------------ TTS
    t0 = time.perf_counter()
    try:
        # gTTS is synchronous I/O-bound — run in thread pool to avoid blocking
        # the event loop during the Google Translate network call.
        loop = get_event_loop()
        audio_base64, _duration = await loop.run_in_executor(
            None,
            partial(tts_util.synthesise_text, response_text),
        )
    except Exception as exc:
        logger.exception("TTS stage failed")
        raise HTTPException(
            status_code=500,
            detail={"stage": "tts", "detail": str(exc)},
        )
    t_tts = _ms(t0, time.perf_counter())
    logger.info("[TTS] %.1f ms → %d b64 chars", t_tts, len(audio_base64))

    # ------------------------------------------------------------------ TOTALS
    t_total = _ms(pipeline_start, time.perf_counter())
    logger.info(
        "[Pipeline] total=%.1f ms  (stt=%.1f db_fetch=%.1f llm=%.1f db_write=%.1f tts=%.1f)",
        t_total, t_stt, t_db_fetch, t_llm, t_db_write, t_tts,
    )

    return ProcessResponse(
        session_id=session_id,
        transcript=transcript,
        response_text=response_text,
        audio_base64=audio_base64,
        audio_format="mp3",
        latency_ms=LatencyBreakdown(
            stt=t_stt,
            db_fetch=t_db_fetch,
            llm=t_llm,
            db_write=t_db_write,
            tts=t_tts,
            total=t_total,
        ),
    )
