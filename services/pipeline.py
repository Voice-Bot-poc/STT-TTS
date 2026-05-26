# """
# pipeline.py — Orchestrates the full STT → ClinicQueue → TTS pipeline.

# Stages:
#     1. STT          — transcribe audio bytes with Whisper
#     2. ClinicQueue  — send transcript to ClinicQueue voice API, get response text
#     3. TTS          — synthesise reply to base64 MP3 audio
# """

# import logging
# import time
# import os
# import httpx
# from asyncio import get_event_loop, CancelledError
# from functools import partial

# from fastapi import HTTPException

# from services import stt
# from services import tts_util
# from models.models import LatencyBreakdown, ProcessResponse

# from services import db

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # ClinicQueue config
# # ---------------------------------------------------------------------------

# # _CLINICQUEUE_BASE_URL = os.getenv(
# #     "CLINICQUEUE_BASE_URL",
# #     "https://identified-fill-battery-victorian.trycloudflare.com"
# # )
# _CLINICQUEUE_BASE_URL = "http://localhost:59711"
# _VOICE_CHAT_ENDPOINT = f"{_CLINICQUEUE_BASE_URL}/api/voice/chat"
# _LLM_TIMEOUT_SECONDS = float(os.getenv("CLINICQUEUE_TIMEOUT_SECONDS", "15.0"))
# _STT_SLOW_THRESHOLD_MS = float(os.getenv("STT_SLOW_THRESHOLD_MS", "30000"))


# def _ms(start: float, end: float) -> float:
#     return round((end - start) * 1000, 1)


# async def run_pipeline(session_id: str, audio_bytes: bytes) -> ProcessResponse:

#     pipeline_start = time.perf_counter()
#     logger.info("[Pipeline] session=%s start", session_id)
#     try:
#         # ------------------------------------------------------------------ STT
#         t0 = time.perf_counter()
#         try:
#             # Debug override: always attempt transcription to bypass RMS/VAD gates.
#             transcript = await stt.transcribe(audio_bytes)
#         except Exception as exc:
#             logger.warning("STT stage failed; returning empty transcript for this chunk: %s", exc)
#             transcript = ""
#         t_stt = _ms(t0, time.perf_counter())

#         if t_stt > _STT_SLOW_THRESHOLD_MS:
#             t_total = _ms(pipeline_start, time.perf_counter())
#             logger.info("[Pipeline] session=%s slow-stt %.1f ms -> wait signal", session_id, t_stt)
#             return ProcessResponse(
#                 session_id=session_id,
#                 transcript=transcript,
#                 response_text="Wait",
#                 audio_base64="",
#                 audio_format="wav",
#                 latency_ms=LatencyBreakdown(
#                     stt=int(t_stt),
#                     db_fetch=0,
#                     llm=0,
#                     db_write=0,
#                     tts=0,
#                     total=int(t_total),
#                 ),
#             )

#         if not transcript or not transcript.strip():
#             t_total = _ms(pipeline_start, time.perf_counter())
#             logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

#             return ProcessResponse(
#                 session_id=session_id,
#                 transcript="",
#                 response_text="",
#                 audio_base64="",
#                 audio_format="wav",
#                 latency_ms=LatencyBreakdown(
#                     stt=int(t_stt),
#                     db_fetch=0,
#                     llm=0,
#                     db_write=0,
#                     tts=0,
#                     total=int(t_total),
#                 ),
#             )

#         logger.info("[STT] %.1f ms → %r", t_stt, transcript[:80])

#         # ------------------------------------------------------------------ CLINICQUEUE
#         t0 = time.perf_counter()
#         try:
#             response_text, intent, booking_completed = await _call_clinicqueue(
#                 session_id=session_id,
#                 transcript=transcript,
#             )
#         except Exception as exc:
#             logger.exception("ClinicQueue stage failed")
#             raise HTTPException(
#                 status_code=500,
#                 detail={"stage": "clinicqueue", "detail": str(exc)},
#             )
#         t_clinicqueue = _ms(t0, time.perf_counter())

#         if not response_text.strip():
#             t_total = _ms(pipeline_start, time.perf_counter())
#             logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

#             return ProcessResponse(
#                 session_id=session_id,
#                 transcript=transcript,
#                 response_text="",
#                 audio_base64="",
#                 audio_format="wav",
#                 latency_ms=LatencyBreakdown(
#                     stt=int(t_stt),
#                     db_fetch=0,
#                     llm=int(t_clinicqueue),
#                     db_write=0,
#                     tts=0,
#                     total=int(t_total),
#                 ),
#             )

#         # ------------------------------------------------------------------ TTS
#         t0 = time.perf_counter()
#         try:
#             loop = get_event_loop()
#             audio_base64, _duration = await loop.run_in_executor(
#                 None,
#                 partial(tts_util.synthesise_text, response_text),
#             )
#         except Exception as exc:
#             logger.exception("TTS stage failed")
#             raise HTTPException(
#                 status_code=500,
#                 detail={"stage": "tts", "detail": str(exc)},
#             )
#         t_tts = _ms(t0, time.perf_counter())

#         # ------------------------------------------------------------------ TOTALS
#         t_total = _ms(pipeline_start, time.perf_counter())
#         logger.info("[Pipeline] session=%s stop total=%.1f ms", session_id, t_total)

#         return ProcessResponse(
#             session_id=session_id,
#             transcript=transcript,
#             response_text=response_text,
#             audio_base64=audio_base64,
#             audio_format="wav",
#             latency_ms=LatencyBreakdown(
#                 stt=int(t_stt),
#                 db_fetch=0,       # no longer used
#                 llm=int(t_clinicqueue),  # reuse field for ClinicQueue latency
#                 db_write=0,       # no longer used
#                 tts=int(t_tts),
#                 total=int(t_total),
#             ),
#         )
#     finally:

# async def run_chat_pipeline(session_id: str, text: str) -> dict:
#     """
#     Text-only pipeline (no STT, no TTS)

#     Used by /chat endpoint
#     """
#     t0 = time.perf_counter()

#     try:
#         response_text, intent, booking_completed = await _call_clinicqueue(
#             session_id=session_id,
#             transcript=text,
#         )
#     except Exception as exc:
#         logger.exception("Chat pipeline failed")
#         raise HTTPException(
#             status_code=500,
#             detail={"stage": "clinicqueue", "detail": str(exc)},
#         )

#     latency = _ms(t0, time.perf_counter())

#     return {
#         "english_output": {
#             "reply_message": response_text,
#             "intent": intent,
#             "extracted_entities": {},
#             "missing_info": []
#         },
#         "final_output": {
#             "reply_message": response_text,
#             "intent": intent,
#             "extracted_entities": {},
#             "missing_info": []
#         },
#         "original_language": "en"
#     }

# async def _call_clinicqueue(
#     session_id: str,
#     transcript: str,
# ) -> tuple[str, str, bool]:
#     """
#     POST transcript to ClinicQueue /api/voice/chat.

#     Returns:
#         (response_text, intent, booking_completed)
#     """
#     payload = {
#         "transcript": transcript,
#         "sessionId": session_id,
#     }

#     timeout = httpx.Timeout(
#         timeout=_LLM_TIMEOUT_SECONDS,
#         connect=min(1.5, _LLM_TIMEOUT_SECONDS),
#         read=_LLM_TIMEOUT_SECONDS,
#         write=min(1.5, _LLM_TIMEOUT_SECONDS),
#         pool=min(1.0, _LLM_TIMEOUT_SECONDS),
#     )

#     request_start = time.perf_counter()

#     try:
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             response = await client.post(_VOICE_CHAT_ENDPOINT, json=payload)
#         _ = _ms(request_start, time.perf_counter())

#         if response.status_code != 200:
#             raise RuntimeError(
#                 f"ClinicQueue returned {response.status_code}: {response.text}"
#             )

#         data = response.json()

#         response_text = (data.get("responseText") or "").strip()
#         response_source = "responseText"
#         if not response_text:
#             english_output = data.get("english_output") or {}
#             final_output = data.get("final_output") or {}
#             english_reply = (english_output.get("reply_message") or "").strip()
#             final_reply = (final_output.get("reply_message") or "").strip()
#             top_reply = (data.get("reply_message") or "").strip()
#             if english_reply:
#                 response_text = english_reply
#                 response_source = "english_output.reply_message"
#             elif final_reply:
#                 response_text = final_reply
#                 response_source = "final_output.reply_message"
#             elif top_reply:
#                 response_text = top_reply
#                 response_source = "reply_message"
#             else:
#                 response_text = ""
#                 response_source = "none"

#         intent = data.get("intent", "Other")
#         if intent == "Other":
#             intent = (
#                 (data.get("english_output") or {}).get("intent")
#                 or (data.get("final_output") or {}).get("intent")
#                 or "Other"
#             )

#         booking_completed = data.get("bookingCompleted", False)

#         logger.debug(
#             "[ClinicQueue] session=%s response_source=%s response_text=%r",
#             session_id,
#             response_source,
#             response_text,
#         )

#         if not response_text:
#             return "", intent, booking_completed

#         return response_text, intent, booking_completed
#     except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, CancelledError):
#         return "", "Timeout", False



"""
pipeline.py — Orchestrates the full STT → ClinicQueue → TTS pipeline.

Stages:
    1. STT          — transcribe audio bytes with Whisper
    2. ClinicQueue  — send transcript to ClinicQueue voice API, get response text
    3. TTS          — synthesise reply to base64 WAV audio
"""

import logging
import time
import os
import asyncio
import httpx
import json
from asyncio import get_event_loop, CancelledError
from functools import partial

from fastapi import HTTPException

from services import stt
from services import tts_util
from models.models import LatencyBreakdown, ProcessResponse

from services import db
from services.runtime_state import mark_completed, mark_tts_end, mark_tts_start

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ClinicQueue config
# ---------------------------------------------------------------------------
_CLINICQUEUE_BASE_URL = "https://ludie-silvicolous-indiscreetly.ngrok-free.dev"

_VOICE_CHAT_ENDPOINT = f"{_CLINICQUEUE_BASE_URL}/api/voice/chat"

_LLM_TIMEOUT_SECONDS = float(os.getenv("CLINICQUEUE_TIMEOUT_SECONDS", "50.0"))

_CLINICQUEUE_TIMEOUT_FALLBACK_TEXT = os.getenv(
    "CLINICQUEUE_TIMEOUT_FALLBACK_TEXT",
    "Sorry, I am still processing that. Please try again.",
)

_STT_SLOW_THRESHOLD_MS = float(os.getenv("STT_SLOW_THRESHOLD_MS", "30000"))
_SESSION_RUNTIME_TTL_SECONDS = float(os.getenv("PIPELINE_SESSION_RUNTIME_TTL_SECONDS", "1800"))
_PIPELINE_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_PIPELINE_SESSION_LAST_SEEN: dict[str, float] = {}
_PIPELINE_SESSION_LOCKS_GUARD = asyncio.Lock()
_CLINICQUEUE_CLIENT: httpx.AsyncClient | None = None

_GOODBYE_INTENTS = {"bye", "goodbye", "thank you", "thanks", "thank you bye"}
_GOODBYE_RESPONSE = os.getenv("GOODBYE_RESPONSE_TEXT", "Thank you. Goodbye.")


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


def _is_goodbye_intent(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().replace(".", "").replace(",", "").split())
    return normalized in _GOODBYE_INTENTS


def _empty_latency(session_id: str, transcript: str, response_text: str, stt_ms: float, total_ms: float) -> ProcessResponse:
    return ProcessResponse(
        session_id=session_id,
        transcript=transcript,
        response_text=response_text,
        audio_base64="",
        audio_format="wav",
        latency_ms=LatencyBreakdown(stt=int(stt_ms), db_fetch=0, llm=0, db_write=0, tts=0, total=int(total_ms)),
    )


def _get_clinicqueue_client() -> httpx.AsyncClient:
    global _CLINICQUEUE_CLIENT
    if _CLINICQUEUE_CLIENT is None or _CLINICQUEUE_CLIENT.is_closed:
        timeout = httpx.Timeout(
            timeout=_LLM_TIMEOUT_SECONDS,
            connect=min(1.0, _LLM_TIMEOUT_SECONDS),
            read=_LLM_TIMEOUT_SECONDS,
            write=min(1.0, _LLM_TIMEOUT_SECONDS),
            pool=min(0.5, _LLM_TIMEOUT_SECONDS),
        )
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=16, keepalive_expiry=30.0)
        _CLINICQUEUE_CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits, http2=False)
        logger.info("[ClinicQueue] persistent AsyncClient initialized keepalive=true timeout=%.1fs", _LLM_TIMEOUT_SECONDS)
    return _CLINICQUEUE_CLIENT


async def close_http_clients() -> None:
    global _CLINICQUEUE_CLIENT
    if _CLINICQUEUE_CLIENT is not None:
        await _CLINICQUEUE_CLIENT.aclose()
        _CLINICQUEUE_CLIENT = None


async def _get_pipeline_session_lock(session_id: str) -> asyncio.Lock:
    now = time.monotonic()
    async with _PIPELINE_SESSION_LOCKS_GUARD:
        stale_session_ids = [
            sid
            for sid, last_seen in _PIPELINE_SESSION_LAST_SEEN.items()
            if now - last_seen > _SESSION_RUNTIME_TTL_SECONDS
        ]
        for sid in stale_session_ids:
            lock = _PIPELINE_SESSION_LOCKS.get(sid)
            if lock is not None and lock.locked():
                continue
            _PIPELINE_SESSION_LOCKS.pop(sid, None)
            _PIPELINE_SESSION_LAST_SEEN.pop(sid, None)

        lock = _PIPELINE_SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _PIPELINE_SESSION_LOCKS[session_id] = lock

        _PIPELINE_SESSION_LAST_SEEN[session_id] = now
        return lock


def _normalize_phone_number(phone_number: str, session_id: str) -> str:
    raw = (phone_number or "").strip()
    if not raw:
        logger.warning(
            "[ClinicQueuePhone] missing phone_number for session_id=%r",
            session_id,
        )
        return ""

    lowered = raw.lower()
    if lowered == "unknown" or lowered.startswith("voice:") or raw == session_id:
        logger.warning(
            "[ClinicQueuePhone] rejected placeholder phone_number=%r for session_id=%r",
            raw,
            session_id,
        )
        return ""

    if any(ch.isalpha() for ch in raw):
        logger.warning(
            "[ClinicQueuePhone] rejected non-phone phone_number=%r for session_id=%r",
            raw,
            session_id,
        )
        return ""

    digits_only = "".join(ch for ch in raw if ch.isdigit())
    if 10 <= len(digits_only) <= 15:
        logger.info(
            "[ClinicQueuePhone] accepted phone_number raw=%r normalized=%r session_id=%r",
            raw,
            digits_only,
            session_id,
        )
        return digits_only

    logger.warning(
        "[ClinicQueuePhone] rejected invalid length phone_number=%r digits=%r session_id=%r",
        raw,
        digits_only,
        session_id,
    )
    return ""


async def run_pipeline(
    session_id: str,
    audio_bytes: bytes,
    phone_number: str = "",
    synthesize_tts: bool = True,
) -> ProcessResponse:
    session_lock = await _get_pipeline_session_lock(session_id)
    await session_lock.acquire()
    pipeline_start = time.perf_counter()
    logger.info("[Pipeline] session=%s start", session_id)
    try:
        # ------------------------------------------------------------------ STT
        t0 = time.perf_counter()
        logger.info("[STT] session=%s start bytes=%d phone_number=%r", session_id, len(audio_bytes), phone_number)
        try:
            transcript = await stt.transcribe(audio_bytes, session_id=session_id)
        except Exception as exc:
            logger.warning("STT stage failed; returning empty transcript: %s", exc)
            transcript = ""
        t_stt = _ms(t0, time.perf_counter())

        logger.info(
            "[STT] session=%s finished in %.1f ms transcript_chars=%d transcript=%r",
            session_id,
            t_stt,
            len(transcript.strip()),
            transcript[:120],
        )

        if t_stt > _STT_SLOW_THRESHOLD_MS:
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s slow-stt %.1f ms -> wait signal", session_id, t_stt)
            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text="Wait",
                audio_base64="",
                audio_format="wav",
                latency_ms=LatencyBreakdown(stt=int(t_stt), db_fetch=0, llm=0, db_write=0, tts=0, total=int(t_total)),
            )

        if not transcript or not transcript.strip():
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s stop after STT total=%.1f ms", session_id, t_total)
            return ProcessResponse(
                session_id=session_id,
                transcript="",
                response_text="",
                audio_base64="",
                audio_format="wav",
                latency_ms=LatencyBreakdown(stt=int(t_stt), db_fetch=0, llm=0, db_write=0, tts=0, total=int(t_total)),
            )

        logger.info("[STT] %.1f ms → %r", t_stt, transcript[:80])

        if _is_goodbye_intent(transcript):
            mark_completed(session_id)
            response_text = _GOODBYE_RESPONSE
            if not synthesize_tts:
                t_total = _ms(pipeline_start, time.perf_counter())
                logger.info("[Pipeline] session=%s goodbye text-only total=%.1f ms", session_id, t_total)
                return ProcessResponse(
                    session_id=session_id,
                    transcript=transcript,
                    response_text=response_text,
                    audio_base64="",
                    audio_format="pcm_stream",
                    latency_ms=LatencyBreakdown(stt=int(t_stt), db_fetch=0, llm=0, db_write=0, tts=0, total=int(t_total)),
                )

            t0 = time.perf_counter()
            try:
                mark_tts_start(session_id)
                loop = get_event_loop()
                audio_base64, _duration = await loop.run_in_executor(
                    None,
                    partial(tts_util.synthesise_text, response_text),
                )
            finally:
                mark_tts_end(session_id)
            t_tts = _ms(t0, time.perf_counter())
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info("[Pipeline] session=%s goodbye complete total=%.1f ms tts=%.1f", session_id, t_total, t_tts)
            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text=response_text,
                audio_base64=audio_base64,
                audio_format="wav",
                latency_ms=LatencyBreakdown(stt=int(t_stt), db_fetch=0, llm=0, db_write=0, tts=int(t_tts), total=int(t_total)),
            )

        # ------------------------------------------------------------------ CLINICQUEUE
        t0 = time.perf_counter()
        try:
            response_text, intent, booking_completed = await _call_clinicqueue(
                session_id=session_id,
                transcript=transcript,
                phone_number=phone_number,
            )
        except Exception as exc:
            logger.exception("ClinicQueue stage failed")
            raise HTTPException(status_code=500, detail={"stage": "clinicqueue", "detail": str(exc)})
        t_clinicqueue = _ms(t0, time.perf_counter())

        # ✅ FIX 1: Log exactly what came back from ClinicQueue so you can see it
        logger.info(
            "[ClinicQueue] %.1f ms → response_text=%r  intent=%r  booking_completed=%s",
            t_clinicqueue, response_text, intent, booking_completed,
        )

        if not response_text or not response_text.strip():
            t_total = _ms(pipeline_start, time.perf_counter())
            # ✅ FIX 2: Was silently returning empty — now logs a clear warning
            logger.warning(
                "[Pipeline] session=%s ClinicQueue returned EMPTY response_text — "
                "check JSON key names in _call_clinicqueue. Stopping pipeline.",
                session_id,
            )
            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text="",
                audio_base64="",
                audio_format="wav",
                latency_ms=LatencyBreakdown(stt=int(t_stt), db_fetch=0, llm=int(t_clinicqueue), db_write=0, tts=0, total=int(t_total)),
            )

        if not synthesize_tts:
            t_total = _ms(pipeline_start, time.perf_counter())
            logger.info(
                "[Pipeline] session=%s COMPLETE text-only total=%.1f ms | stt=%.1f llm=%.1f",
                session_id,
                t_total,
                t_stt,
                t_clinicqueue,
            )
            return ProcessResponse(
                session_id=session_id,
                transcript=transcript,
                response_text=response_text,
                audio_base64="",
                audio_format="pcm_stream",
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
        logger.info("[TTS] starting synthesis for %d chars: %r", len(response_text), response_text[:60])
        try:
            mark_tts_start(session_id)
            loop = get_event_loop()
            # ✅ FIX 3: run_in_executor is correct — but exceptions were silently
            # swallowed before. Now they bubble up with full tracebacks.
            audio_base64, _duration = await loop.run_in_executor(
                None,
                partial(tts_util.synthesise_text, response_text),
            )
        except Exception as exc:
            # Log the full exception so streaming/full-file TTS failures are visible.
            logger.exception("TTS stage failed — response_text was %r", response_text)
            raise HTTPException(status_code=500, detail={"stage": "tts", "detail": str(exc)})
        finally:
            mark_tts_end(session_id)
        t_tts = _ms(t0, time.perf_counter())

        logger.info("[TTS] %.1f ms → %d bytes audio (base64 len=%d)", t_tts, len(audio_base64) * 3 // 4, len(audio_base64))

        # ------------------------------------------------------------------ TOTALS
        t_total = _ms(pipeline_start, time.perf_counter())
        logger.info("[Pipeline] session=%s COMPLETE total=%.1f ms | stt=%.1f llm=%.1f tts=%.1f",
                    session_id, t_total, t_stt, t_clinicqueue, t_tts)

        return ProcessResponse(
            session_id=session_id,
            transcript=transcript,
            response_text=response_text,
            audio_base64=audio_base64,
            audio_format="wav",
            latency_ms=LatencyBreakdown(
                stt=int(t_stt),
                db_fetch=0,
                llm=int(t_clinicqueue),
                db_write=0,
                tts=int(t_tts),
                total=int(t_total),
            ),
        )
    finally:
        _PIPELINE_SESSION_LAST_SEEN[session_id] = time.monotonic()
        session_lock.release()


async def run_chat_pipeline(
    session_id: str,
    text: str,
    phone_number: str = "",
) -> dict:
    """Text-only pipeline (no STT, no TTS) — used by /chat endpoint."""
    t0 = time.perf_counter()
    try:
        response_text, intent, booking_completed = await _call_clinicqueue(
            session_id=session_id,
            transcript=text,
            phone_number=phone_number,
        )
    except Exception as exc:
        logger.exception("Chat pipeline failed")
        raise HTTPException(status_code=500, detail={"stage": "clinicqueue", "detail": str(exc)})

    latency = _ms(t0, time.perf_counter())
    logger.info("[Chat] %.1f ms → %r", latency, response_text[:80])

    return {
        "english_output": {
            "reply_message": response_text,
            "intent": intent,
            "extracted_entities": {},
            "missing_info": [],
        },
        "final_output": {
            "reply_message": response_text,
            "intent": intent,
            "extracted_entities": {},
            "missing_info": [],
        },
        "original_language": "en",
    }


async def _call_clinicqueue(
    session_id: str,
    transcript: str,
    phone_number: str = "",
) -> tuple[str, str, bool]:
    """
    POST transcript to ClinicQueue /api/voice/chat.
    Returns: (response_text, intent, booking_completed)
    """
    clinicqueue_phone_number = _normalize_phone_number(phone_number, session_id)
    if not clinicqueue_phone_number:
        raise ValueError("ClinicQueue voice booking requires a real caller phoneNumber.")

    payload = {
        "transcript": transcript,
        "sessionId": session_id,
        "phoneNumber": clinicqueue_phone_number,
    }
    logger.info(
        "[ClinicQueue] request JSON body: %s",
        json.dumps(payload, ensure_ascii=False),
    )
    logger.info(
        "[ClinicQueue] sending /api/voice/chat sessionId=%r phoneNumber=%r transcript_preview=%r",
        session_id,
        clinicqueue_phone_number,
        transcript[:80],
    )

    request_start = time.perf_counter()

    try:
        client = _get_clinicqueue_client()
        response = await client.post(_VOICE_CHAT_ENDPOINT, json=payload)
        request_ms = _ms(request_start, time.perf_counter())

        logger.info(
            "[ClinicQueue] HTTP %d in %.1f ms — body preview: %s",
            response.status_code,
            request_ms,
            response.text[:300],   # ✅ FIX 5: Log raw response so you can see exact JSON keys
        )

        if response.status_code != 200:
            raise RuntimeError(f"ClinicQueue returned {response.status_code}: {response.text}")

        data = response.json()

        # ✅ FIX 6: Log ALL top-level keys so mismatches are immediately obvious
        logger.info("[ClinicQueue] response keys: %s", list(data.keys()))

        # Try all known key locations in priority order
        response_text = (data.get("responseText") or "").strip()
        response_source = "responseText"

        if not response_text:
            english_output = data.get("english_output") or {}
            final_output = data.get("final_output") or {}
            english_reply = (english_output.get("reply_message") or "").strip()
            final_reply = (final_output.get("reply_message") or "").strip()
            top_reply = (data.get("reply_message") or "").strip()

            if english_reply:
                response_text = english_reply
                response_source = "english_output.reply_message"
            elif final_reply:
                response_text = final_reply
                response_source = "final_output.reply_message"
            elif top_reply:
                response_text = top_reply
                response_source = "reply_message"
            else:
                response_text = ""
                response_source = "none"
                # ✅ FIX 7: Dump the entire response when we can't find text — critical for debugging
                logger.error(
                    "[ClinicQueue] Could NOT extract response_text from any known key! "
                    "Full response JSON: %s",
                    data,
                )

        logger.info("[ClinicQueue] response_source=%s response_text=%r", response_source, response_text[:100])

        intent = data.get("intent", "Other")
        if intent == "Other":
            intent = (
                (data.get("english_output") or {}).get("intent")
                or (data.get("final_output") or {}).get("intent")
                or "Other"
            )

        booking_completed = data.get("bookingCompleted", False)

        if not response_text:
            return "", intent, booking_completed

        return response_text, intent, booking_completed

    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, CancelledError) as exc:
        logger.error("[ClinicQueue] TIMEOUT after %.1f ms: %s", _ms(request_start, time.perf_counter()), exc)
        return _CLINICQUEUE_TIMEOUT_FALLBACK_TEXT, "Timeout", False
