"""
pipeline.py — Orchestrates the full STT → ClinicQueue → TTS pipeline.

Stages:
    1. STT          — transcribe audio bytes with Whisper
    2. ClinicQueue  — send transcript to ClinicQueue voice API, get response text
    3. TTS          — synthesise reply to base64 MP3 audio
"""

import logging
import time
import os
import httpx
from asyncio import get_event_loop, CancelledError
from functools import partial

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
_CLINICQUEUE_BASE_URL = "https://unallegorical-lauditorily-elliot.ngrok-free.dev"
_VOICE_CHAT_ENDPOINT = f"{_CLINICQUEUE_BASE_URL}/api/voice/chat"


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


async def run_pipeline(session_id: str, audio_bytes: bytes) -> ProcessResponse:
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

    # ------------------------------------------------------------------ CLINICQUEUE
    t0 = time.perf_counter()
    try:
        print("Calling:", _VOICE_CHAT_ENDPOINT)
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
    logger.info(
        "[ClinicQueue] %.1f ms → intent=%s booking=%s reply=%r",
        t_clinicqueue, intent, booking_completed, response_text[:80]
    )

    # ------------------------------------------------------------------ TTS
    t0 = time.perf_counter()
    try:
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
        "[Pipeline] total=%.1f ms  (stt=%.1f clinicqueue=%.1f tts=%.1f)",
        t_total, t_stt, t_clinicqueue, t_tts,
    )

    return ProcessResponse(
        session_id=session_id,
        transcript=transcript,
        response_text=response_text,
        audio_base64=audio_base64,
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

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            response = await client.post(_VOICE_CHAT_ENDPOINT, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"ClinicQueue returned {response.status_code}: {response.text}"
            )

        data = response.json()

        response_text = data.get("responseText", "").strip()
        intent = data.get("intent", "Other")
        booking_completed = data.get("bookingCompleted", False)

        if not response_text:
            raise RuntimeError("ClinicQueue returned empty responseText")

        return response_text, intent, booking_completed
    except (httpx.ReadTimeout, CancelledError):
        logger.error("ClinicQueue/LLM took too long to respond (>90s)")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Can you repeat that?", "Error", False