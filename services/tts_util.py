"""
tts_util.py — Thin wrapper around the existing gTTS engine in tts_service.py.

Design decision:
  Rather than duplicating gTTS logic here, we import the private helper
  `_synthesise_gtts` directly from tts_service.  This keeps a single source of
  truth for all audio-synthesis code.  If the engine ever changes (e.g. swapping
  gTTS for ElevenLabs), only tts_service.py needs updating.

The pipeline always requests mp3 (no transcoding needed) and uses the default
(normal speed) voice, both of which are fast/safe choices for a voice assistant.
"""

import base64
import logging

from services.tts_service import _synthesise_gtts, _estimate_duration

logger = logging.getLogger(__name__)

# Pipeline always outputs mp3 — simplest, fastest, no ffmpeg dependency needed.
_PIPELINE_FORMAT = "mp3"
_PIPELINE_SLOW = False  # normal speech speed


def synthesise_text(text: str) -> tuple[str, float]:
    """
    Convert *text* to speech using gTTS and return:
        (audio_base64: str, duration_seconds: float)

    Raises:
        RuntimeError — if gTTS or encoding fails (caller maps to HTTP 500).

    NOTE: gTTS makes a network call to translate.google.com.  It is synchronous.
    When called from the async pipeline, wrap with run_in_executor if needed.
    However, because gTTS is I/O-bound and relatively quick, we keep it simple
    here; the pipeline.py caller handles threading if it becomes a bottleneck.
    """
    logger.info("TTS (pipeline): synthesising %d chars", len(text))

    audio_bytes = _synthesise_gtts(text, slow=_PIPELINE_SLOW, fmt=_PIPELINE_FORMAT)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    duration = _estimate_duration(text, slow=_PIPELINE_SLOW)

    logger.info("TTS (pipeline): done — %d bytes, ~%.1fs", len(audio_bytes), duration)
    return audio_b64, duration
