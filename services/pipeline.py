"""
pipeline.py — Orchestrates the full STT → ClinicQueue → TTS pipeline.

Stages:
    1. STT          — transcribe audio bytes with Whisper
    2. ClinicQueue  — send transcript to ClinicQueue voice API, get response text
    3. TTS          — synthesise reply to MP3 audio
"""

import logging
import time
import os
import threading
import httpx
from asyncio import CancelledError

from fastapi import HTTPException

from services import stt
from services import tts_util
from models.models import LatencyBreakdown, ProcessResponse

from services import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ClinicQueue config
# ---------------------------------------------------------------------------

# _CLINICQUEUE_BASE_URL = os.getenv(
#     "CLINICQUEUE_BASE_URL",
#     "https://identified-fill-battery-victorian.trycloudflare.com"
# )
_CLINICQUEUE_BASE_URL = "http://localhost:5000"
_VOICE_CHAT_ENDPOINT = f"{_CLINICQUEUE_BASE_URL}/api/voice/chat"
_LLM_TIMEOUT_SECONDS = float(os.getenv("CLINICQUEUE_TIMEOUT_SECONDS", "15.0"))
_STT_SLOW_THRESHOLD_MS = float(os.getenv("STT_SLOW_THRESHOLD_MS", "30000"))
_PIPELINE_SINGLE_FLIGHT = threading.Lock()


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


async def run_pipeline(session_id: str, audio_bytes: bytes) -> ProcessResponse:
    if not _PIPELINE_SINGLE_FLIGHT.acquire(blocking=False):
        raise HTTPException(status_code=204)

    pipeline_start = time.perf_counter()
    logger.info("[Pipeline] session=%s start", session_id)
    try:
        # ------------------------------------------------------------------ STT
        t0 = time.perf_counter()
        if stt.should_skip_audio(audio_bytes):
            transcript = ""
        else:
            try:
                transcript = await stt.transcribe(audio_bytes)
            except Exception as exc:
                logger.warning("STT stage failed; returning empty transcript for this chunk: %s", exc)
                transcript = ""
        t_stt = _ms(t0, time.perf_counter())

        if t_stt > _STT_SLOW_THRESHOLD_MS:
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s slow-stt %.1f ms -> wait signal", session_id, t_stt)
            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text="Wait",
                audio_bytes=b"",
                audio_format="mp3",
                latency_ms=LatencyBreakdown(
                    stt=int(t_stt),
                    db_fetch=0,
                    llm=0,
                    db_write=0,
                    tts=0,
                    total=int(t_total),
                ),
            )

        if not transcript or not transcript.strip():
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

            return ProcessResponse(
                session_id=session_id,
                transcript="",
                response_text="",
                audio_bytes=b"",
                audio_format="mp3",
                latency_ms=LatencyBreakdown(
                    stt=int(t_stt),
                    db_fetch=0,
                    llm=0,
                    db_write=0,
                    tts=0,
                    total=int(t_total),
                ),
            )

        logger.info("[STT] %.1f ms → %r", t_stt, transcript[:80])

        # ------------------------------------------------------------------ CLINICQUEUE
        t0 = time.perf_counter()
        try:
            response_text, intent, booking_completed = await _call_clinicqueue(
                session_id=session_id,
                transcript=transcript,
            )
        except Exception as exc:
            logger.exception("ClinicQueue stage failed")
            raise HTTPException(
                status_code=500,
                detail={"stage": "clinicqueue", "detail": str(exc)},
            )
        t_clinicqueue = _ms(t0, time.perf_counter())

        if not response_text.strip():
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text="",
                audio_bytes=b"",
                audio_format="mp3",
                latency_ms=LatencyBreakdown(
                    stt=int(t_stt),
                    db_fetch=0,
                    llm=int(t_clinicqueue),
                    db_write=0,
                    tts=0,
                    total=int(t_total),
                ),
            )

        # ------------------------------------------------------------------ TTS
        t0 = time.perf_counter()
        try:
            audio_bytes, _duration = await tts_util.synthesise_text(response_text)
        except Exception as exc:
            logger.exception("TTS stage failed")
            raise HTTPException(
                status_code=500,
                detail={"stage": "tts", "detail": str(exc)},
            )
        t_tts = _ms(t0, time.perf_counter())

        # ------------------------------------------------------------------ TOTALS
        t_total = _ms(pipeline_start, time.perf_counter())
        logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

        return ProcessResponse(
            session_id=session_id,
            transcript=transcript,
            response_text=response_text,
            audio_bytes=audio_bytes,
            audio_format="mp3",
            latency_ms=LatencyBreakdown(
                stt=int(t_stt),
                db_fetch=0,       # no longer used
                llm=int(t_clinicqueue),  # reuse field for ClinicQueue latency
                db_write=0,       # no longer used
                tts=int(t_tts),
                total=int(t_total),
            ),
        )
    finally:
        _PIPELINE_SINGLE_FLIGHT.release()

async def run_chat_pipeline(session_id: str, text: str) -> dict:
    """
    Text-only pipeline (no STT, no TTS)

    Used by /chat endpoint
    """
    t0 = time.perf_counter()

    try:
        response_text, intent, booking_completed = await _call_clinicqueue(
            session_id=session_id,
            transcript=text,
        )
    except Exception as exc:
        logger.exception("Chat pipeline failed")
        raise HTTPException(
            status_code=500,
            detail={"stage": "clinicqueue", "detail": str(exc)},
        )

    latency = _ms(t0, time.perf_counter())

    return {
        "english_output": {
            "reply_message": response_text,
            "intent": intent,
            "extracted_entities": {},
            "missing_info": []
        },
        "final_output": {
            "reply_message": response_text,
            "intent": intent,
            "extracted_entities": {},
            "missing_info": []
        },
        "original_language": "en"
    }

async def _call_clinicqueue(
    session_id: str,
    transcript: str,
) -> tuple[str, str, bool]:
    """
    POST transcript to ClinicQueue /api/voice/chat.

    Returns:
        (response_text, intent, booking_completed)
    """
    payload = {
        "transcript": transcript,
        "sessionId": session_id,
    }

    timeout = httpx.Timeout(
        timeout=_LLM_TIMEOUT_SECONDS,
        connect=min(1.5, _LLM_TIMEOUT_SECONDS),
        read=_LLM_TIMEOUT_SECONDS,
        write=min(1.5, _LLM_TIMEOUT_SECONDS),
        pool=min(1.0, _LLM_TIMEOUT_SECONDS),
    )

    request_start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(_VOICE_CHAT_ENDPOINT, json=payload)
        _ = _ms(request_start, time.perf_counter())

        if response.status_code != 200:
            raise RuntimeError(
                f"ClinicQueue returned {response.status_code}: {response.text}"
            )

        data = response.json()

        response_text = data.get("responseText", "").strip()
        intent = data.get("intent", "Other")
        booking_completed = data.get("bookingCompleted", False)

        if not response_text:
            return "", intent, booking_completed

        return response_text, intent, booking_completed
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, CancelledError):
        return "", "Timeout", False

