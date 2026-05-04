import logging

from services.tts_service import _estimate_duration, _synthesise_gtts

logger = logging.getLogger(__name__)

_PIPELINE_FORMAT = "mp3"
_PIPELINE_SLOW = False


async def synthesise_text(text: str) -> tuple[bytes, float]:
    logger.info("TTS (pipeline): synthesising %d chars", len(text))

    raw_mp3_bytes = await _synthesise_gtts(text, slow=_PIPELINE_SLOW, fmt=_PIPELINE_FORMAT)

    duration = _estimate_duration(text, slow=_PIPELINE_SLOW)

    logger.info(
        "TTS (pipeline): done - %d bytes MP3, ~%.1fs",
        len(raw_mp3_bytes),
        duration,
    )
    return raw_mp3_bytes, duration
