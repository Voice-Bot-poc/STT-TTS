"""
stt.py — Whisper-based Speech-to-Text transcription module.

Uses: openai-whisper (the original OpenAI package — runs locally, no API key needed).
Configure via env var:  WHISPER_MODEL  (default: "base")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASYNC / SYNC design note
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Whisper is a *synchronous*, CPU/GPU-heavy operation.  If we call it
directly inside an `async def` FastAPI handler it would block the entire
event loop for the duration of transcription (often several seconds),
starving all other in-flight requests.

Solution: we offload the blocking work to a thread pool via
    asyncio.get_event_loop().run_in_executor(None, blocking_fn)
`None` means "use the default ThreadPoolExecutor".  This lets FastAPI's
event loop remain responsive while Whisper runs on a worker thread.

The temp-file lifecycle is:
  1. Write audio bytes → NamedTemporaryFile (delete=False so Windows can read it)
  2. Pass file path to Whisper (it opens the file internally)
  3. Delete temp file in a `finally` block — always cleaned up, even on error.
"""

import asyncio
import logging
import os
import tempfile
from functools import partial

import whisper  # openai-whisper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model — loaded once at module import time (expensive: ~500 MB for "base")
# ---------------------------------------------------------------------------

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

logger.info("Loading Whisper model '%s' …", _MODEL_NAME)
_whisper_model = whisper.load_model(_MODEL_NAME)
logger.info("Whisper model '%s' loaded.", _MODEL_NAME)


# ---------------------------------------------------------------------------
# Internal sync helper (runs in thread pool)
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_bytes: bytes) -> str:
    """
    Blocking transcription helper.  Must NOT be called directly from async code.
    Use `transcribe()` instead, which schedules this on a thread pool.
    """
    tmp_path: str | None = None
    try:
        # NamedTemporaryFile with delete=False is required on Windows because
        # Whisper opens the file by path; a file opened by Python cannot be
        # opened again by another process on Windows if delete=True.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="whisper_"
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        logger.info("Transcribing %d bytes from %s …", len(audio_bytes), tmp_path)
        result = _whisper_model.transcribe(tmp_path, fp16=False)
        transcript = result.get("text", "").strip()
        logger.info("Transcription complete: %d chars", len(transcript))
        return transcript

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def transcribe(audio_bytes: bytes) -> str:
    """
    Async entry point for STT.  Offloads blocking Whisper work to a thread.

    Args:
        audio_bytes: Raw audio bytes (any format Whisper supports: wav, mp3, m4a …)

    Returns:
        Transcribed text string (may be empty if audio is silent/inaudible).

    Raises:
        RuntimeError: if Whisper raises any exception during transcription.
    """
    loop = asyncio.get_event_loop()
    # partial() is not strictly needed for a simple call, but makes it explicit
    # that we're passing a bound callable with its argument to run_in_executor.
    transcript = await loop.run_in_executor(
        None,                              # default ThreadPoolExecutor
        partial(_transcribe_sync, audio_bytes),
    )
    return transcript
