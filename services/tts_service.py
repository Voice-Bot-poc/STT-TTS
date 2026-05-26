import base64
import io
import logging
import os
import wave
from dataclasses import dataclass
from typing import Literal, Optional

from services.streaming_tts import (
    DEFAULT_CHUNK_MS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLE_WIDTH,
    get_streaming_tts_service,
)
from services.tts_normalizer import normalize_for_tts

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "wav", "ogg"]
VoiceID = Literal["default", "slow"]


@dataclass
class TTSRequest:
    text: str
    voice: VoiceID = "default"
    format: AudioFormat = "wav"


@dataclass
class TTSResult:
    audio_base64: str
    duration: float
    format: AudioFormat
    char_count: int


def _estimate_duration(text: str, slow: bool) -> float:
    cps = 9.0 if slow else 14.0
    return round(len(text.strip()) / cps, 2)


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def _synthesise_streaming_tts(text: str, speed: float | None = None, fmt: AudioFormat = "wav") -> tuple[bytes, float]:
    """
    Collect the streaming TTS engine into a WAV file for legacy JSON/file APIs.
    Realtime WebRTC playback uses /tts/pcm-stream and never waits for this full
    buffer.
    """
    if fmt != "wav":
        logger.warning("Streaming TTS produces PCM/WAV; requested %s, returning WAV.", fmt)

    chunks: list[bytes] = []
    chunks.extend(get_streaming_tts_service().stream_pcm_sync(text, DEFAULT_CHUNK_MS, session_id=None, speed=speed))

    pcm = b"".join(chunks)
    duration = 0.0
    if pcm:
        duration = len(pcm) / float(TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH)

    return pcm, duration


class TTSService:
    def synthesise(self, req: TTSRequest) -> TTSResult:
        if not req.text or not req.text.strip():
            raise ValueError("text must be a non-empty string")

        normalized_text = normalize_for_tts(req.text)
        if normalized_text != req.text:
            logger.info("TTS normalized text from %d to %d chars", len(req.text), len(normalized_text))

        slow = req.voice == "slow"
        default_speed = float(os.getenv("KOKORO_SPEED", "0.80"))
        slow_speed = float(os.getenv("KOKORO_SLOW_SPEED", "0.78"))
        effective_speed = slow_speed if slow else default_speed
        logger.info(
            "TTS request: %d chars, voice=%s, engine=kokoro speed=%.2f target_rate=%d chunk_ms=%d",
            len(normalized_text),
            req.voice,
            effective_speed,
            TARGET_SAMPLE_RATE,
            DEFAULT_CHUNK_MS,
        )

        pcm, duration = _synthesise_streaming_tts(normalized_text, speed=effective_speed, fmt="wav")
        audio_bytes = _pcm_to_wav(pcm)
        return TTSResult(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            duration=round(duration, 2),
            format="wav",
            char_count=len(normalized_text),
        )


_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    global _service
    if _service is None:
        _service = TTSService()
    return _service
