import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Literal

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "wav", "ogg"]
VoiceID = Literal["default", "slow"]


@dataclass
class TTSRequest:
    text: str
    voice: VoiceID = "default"
    format: AudioFormat = "mp3"


@dataclass
class TTSResult:
    audio_base64: str
    duration: float
    format: AudioFormat
    char_count: int


_CHARS_PER_SECOND_NORMAL = 14.0
_CHARS_PER_SECOND_SLOW = 9.0
_EDGE_TTS_DEFAULT_VOICE = "en-US-AriaNeural"


def _estimate_duration(text: str, slow: bool) -> float:
    cps = _CHARS_PER_SECOND_SLOW if slow else _CHARS_PER_SECOND_NORMAL
    return round(len(text.strip()) / cps, 2)


async def _synthesise_gtts(text: str, slow: bool, fmt: AudioFormat) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        _EDGE_TTS_DEFAULT_VOICE,
        rate="-20%" if slow else "+0%",
    )

    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])

    return b"".join(chunks)


def _synthesise_sync(text: str, slow: bool, fmt: AudioFormat) -> bytes:
    return asyncio.run(_synthesise_gtts(text, slow=slow, fmt=fmt))


class TTSService:
    def synthesise(self, req: TTSRequest) -> TTSResult:
        if not req.text or not req.text.strip():
            raise ValueError("text must be a non-empty string")

        slow = req.voice == "slow"
        fmt: AudioFormat = "mp3"

        logger.info(
            "TTS request: %d chars, voice=%s, format=%s",
            len(req.text), req.voice, fmt,
        )

        audio_bytes = _synthesise_sync(req.text, slow=slow, fmt=fmt)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        duration = _estimate_duration(req.text, slow=slow)

        return TTSResult(
            audio_base64=audio_b64,
            duration=duration,
            format=fmt,
            char_count=len(req.text),
        )


_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    global _service
    if _service is None:
        _service = TTSService()
    return _service
