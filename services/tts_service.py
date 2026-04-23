import io
import time
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)

_TARGET_WAV_SAMPLE_RATE = 48000
_TARGET_WAV_CHANNELS = 1
_TARGET_WAV_SAMPLE_WIDTH = 2

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AudioFormat = Literal["mp3", "wav", "ogg"]
VoiceID = Literal["default", "slow"]   # extend when you add more engines


@dataclass
class TTSRequest:
    text: str
    voice: VoiceID = "default"
    format: AudioFormat = "mp3"


@dataclass
class TTSResult:
    audio_base64: str           # base64-encoded audio bytes
    duration: float             # approximate duration in seconds
    format: AudioFormat
    char_count: int             # number of characters synthesised


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------

# Average reading speed: ~150 words per minute → ~12.5 words/second
# Average word length: ~5 chars → ~62.5 chars/second
_CHARS_PER_SECOND_NORMAL = 14.0   # conservative estimate, tunable
_CHARS_PER_SECOND_SLOW   =  9.0


def _estimate_duration(text: str, slow: bool) -> float:
    """
    Rough but instant duration estimate based on character count.
    Replace or supplement with the real audio length once you decode the bytes
    if the engine provides it (e.g. mutagen for mp3, wave module for wav).
    """
    cps = _CHARS_PER_SECOND_SLOW if slow else _CHARS_PER_SECOND_NORMAL
    return round(len(text.strip()) / cps, 2)


# ---------------------------------------------------------------------------
# Engine: gTTS  (pip install gTTS)
# ---------------------------------------------------------------------------

def _synthesise_gtts(text: str, slow: bool, fmt: AudioFormat) -> bytes:
    """
    Use gTTS to produce audio bytes in memory — no temp files, no disk I/O.

    gTTS always produces MP3.  If another format is requested we transcode
    with pydub (pip install pydub) + ffmpeg.  If pydub is unavailable we
    fall back to returning MP3 regardless and log a warning.
    """
    from gtts import gTTS  # lazy import so the module is optional at import time

    tts = gTTS(text=text, lang="en", slow=slow)

    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    mp3_bytes = buf.read()

    if fmt == "mp3":
        return mp3_bytes

    # Transcode to wav / ogg
    try:
        from pydub import AudioSegment  # optional dependency

        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        if fmt == "wav":
            # Force deterministic WebRTC-friendly PCM format.
            audio = (
                audio.set_frame_rate(_TARGET_WAV_SAMPLE_RATE)
                .set_channels(_TARGET_WAV_CHANNELS)
                .set_sample_width(_TARGET_WAV_SAMPLE_WIDTH)
            )
        out_buf = io.BytesIO()
        audio.export(out_buf, format=fmt)
        out_buf.seek(0)
        return out_buf.read()
    except ImportError:
        logger.warning(
            "pydub is not installed — cannot transcode to %s; returning mp3 instead.",
            fmt,
        )
        return mp3_bytes


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class TTSService:
    """
    Stateless TTS service.

    Usage
    -----
        svc = TTSService()
        result = svc.synthesise(TTSRequest(text="Hello world"))
        # result.audio_base64  → base64 string ready for JSON response
        # result.duration      → float seconds
    """

    def synthesise(self, req: TTSRequest) -> TTSResult:
        if not req.text or not req.text.strip():
            raise ValueError("text must be a non-empty string")

        slow = req.voice == "slow"
        fmt: AudioFormat = req.format if req.format in ("mp3", "wav", "ogg") else "mp3"

        logger.info(
            "TTS request: %d chars, voice=%s, format=%s",
            len(req.text), req.voice, fmt,
        )

        audio_bytes = _synthesise_gtts(req.text, slow=slow, fmt=fmt)
        audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")
        duration    = _estimate_duration(req.text, slow=slow)

        return TTSResult(
            audio_base64=audio_b64,
            duration=duration,
            format=fmt,
            char_count=len(req.text),
        )


# ---------------------------------------------------------------------------
# Convenience factory (singleton-ish, no state so just a module-level instance)
# ---------------------------------------------------------------------------

_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Return the shared TTSService instance (lazy init)."""
    global _service
    if _service is None:
        _service = TTSService()
    return _service