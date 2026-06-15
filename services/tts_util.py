import base64
import logging
import wave
from io import BytesIO

from services.streaming_tts import (
    DEFAULT_CHUNK_MS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLE_WIDTH,
    get_streaming_tts_service,
)

logger = logging.getLogger(__name__)


def synthesise_text(text: str) -> tuple[str, float]:
    """
    Legacy full-audio helper backed by the streaming TTS engine.
    The realtime WebRTC path uses /tts/pcm-stream and does not call this helper.
    """
    logger.info("TTS (pipeline): collecting streaming synthesis for %d chars", len(text))

    chunks: list[bytes] = []
    chunks.extend(get_streaming_tts_service().stream_pcm_sync(text, DEFAULT_CHUNK_MS))

    pcm = b"".join(chunks)
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(pcm)

    duration = round(len(pcm) / (TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH), 2)
    logger.info("TTS (pipeline): collected %d bytes WAV, ~%.1fs", len(wav_buffer.getvalue()), duration)
    return base64.b64encode(wav_buffer.getvalue()).decode("utf-8"), duration
