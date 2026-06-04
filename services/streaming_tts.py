"""
streaming_tts.py — PCM streaming TTS backed by edge-tts (Microsoft Neural voices).

Replaces Kokoro with edge-tts for zero-GPU, low-latency (~0.3s) synthesis.
The public interface (StreamingTtsService, get_streaming_tts_service) is
100% unchanged so pipeline.py requires zero modifications.

Dependencies to install:
    pip install edge-tts pydub
    # ffmpeg must be on PATH (used by pydub for MP3 decoding)
    # Ubuntu/Debian: sudo apt install ffmpeg
    # Windows: https://ffmpeg.org/download.html
"""

import asyncio
import io
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Iterator

import numpy as np

from services.runtime_state import cancel_tts, get_session
from services.tts_normalizer import normalize_for_tts

logger = logging.getLogger(__name__)

# ── Audio config (unchanged from Kokoro version) ────────────────────────────
TARGET_SAMPLE_RATE  = int(os.getenv("TTS_TARGET_SAMPLE_RATE", "16000"))
TARGET_CHANNELS     = 1
TARGET_SAMPLE_WIDTH = 2
DEFAULT_CHUNK_MS    = int(os.getenv("STREAMING_TTS_CHUNK_MS", "60"))
MIN_CHUNK_MS        = int(os.getenv("STREAMING_TTS_MIN_CHUNK_MS", "20"))
MAX_CHUNK_MS        = int(os.getenv("STREAMING_TTS_MAX_CHUNK_MS", "200"))

# ── Sentence streaming config (Fix 4 — unchanged) ───────────────────────────
MIN_SENTENCE_CHARS = int(os.getenv("TTS_MIN_SENTENCE_CHARS", "20"))
_HINDI_RE          = re.compile(r"[\u0900-\u097F]")
_SENTENCE_END_RE   = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

# ── edge-tts voice config ────────────────────────────────────────────────────
# Find more voices: python -m edge_tts --list-voices
EDGE_TTS_EN_VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-US-AriaNeural")
EDGE_TTS_HI_VOICE = os.getenv("EDGE_TTS_HI_VOICE", "hi-IN-SwaraNeural")
EDGE_TTS_SPEED    = os.getenv("EDGE_TTS_SPEED", "-5%")   # slight slowdown for telephony clarity


def detect_language(text: str) -> str:
    return "hi" if _HINDI_RE.search(text or "") else "en"


class StreamingTtsUnavailable(RuntimeError):
    pass


# ── Keep old env vars around so nothing crashes if they're still set ─────────
# (They are simply ignored now.)
_KOKORO_SPEED_DEFAULT  = float(os.getenv("KOKORO_SPEED", "0.80"))   # unused
_NORMALIZE_MAX_GAIN    = float(os.getenv("KOKORO_NORMALIZE_MAX_GAIN", "3.5"))  # unused


class EdgeTtsStreamingTts:
    """
    edge-tts backed PCM streaming adapter for real-time voice playback.

    Drop-in replacement for KokoroStreamingTts — exposes the exact same
    public methods:
        stream_pcm()
        stream_pcm_sync()
        stream_pcm_from_text_stream()
        cancel_session()
        validate_startup()
    """

    def __init__(self) -> None:
        self.en_voice    = EDGE_TTS_EN_VOICE
        self.hi_voice    = EDGE_TTS_HI_VOICE
        self.speed       = EDGE_TTS_SPEED

    # ── validate_startup ─────────────────────────────────────────────────────
    def validate_startup(self) -> None:
        """
        Called once at service startup. edge-tts needs no model loading —
        just log that we're ready.
        """
        logger.info(
            "[TTS] EdgeTTS ready | en_voice=%s hi_voice=%s speed=%s target_rate=%d",
            self.en_voice,
            self.hi_voice,
            self.speed,
            TARGET_SAMPLE_RATE,
        )

    # ── primary sync PCM generator ───────────────────────────────────────────
    def stream_pcm(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,   # accepted for API compatibility, ignored
    ) -> Iterator[bytes]:
        """
        Synthesise `text` with edge-tts and yield PCM-16 chunks.

        This is a synchronous generator. It runs the async edge-tts call
        inside a fresh event loop so it can safely be called from a
        background thread (same pattern that Kokoro used).
        """
        if not text or not text.strip():
            return

        text     = normalize_for_tts(text)
        language = detect_language(text)
        voice    = self.hi_voice if language == "hi" else self.en_voice

        logger.info(
            "[TTS] EdgeTTS start | session=%s chars=%d voice=%s chunk_ms=%d",
            session_id or "-", len(text), voice, chunk_ms,
        )
        start = time.perf_counter()

        # ── Step 1: call edge-tts API → MP3 bytes ────────────────────────
        try:
            mp3_bytes = asyncio.run(self._synthesise_to_mp3(text, voice))
        except Exception as exc:
            logger.exception("[TTS] EdgeTTS synthesis failed session=%s: %s", session_id, exc)
            raise StreamingTtsUnavailable(str(exc)) from exc

        if not mp3_bytes:
            logger.warning("[TTS] EdgeTTS returned empty audio session=%s", session_id)
            return

        synth_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[TTS] EdgeTTS synthesis done | session=%s synth_ms=%.1f mp3_bytes=%d",
            session_id or "-", synth_ms, len(mp3_bytes),
        )

        # ── Step 2: MP3 → PCM-16 @ TARGET_SAMPLE_RATE ────────────────────
        try:
            pcm = self._mp3_to_pcm(mp3_bytes)
        except Exception as exc:
            logger.exception("[TTS] MP3→PCM conversion failed session=%s: %s", session_id, exc)
            raise StreamingTtsUnavailable(str(exc)) from exc

        total_ms = (time.perf_counter() - start) * 1000.0
        duration_s = len(pcm) / float(TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH)
        logger.info(
            "[TTS] EdgeTTS complete | session=%s total_ms=%.1f pcm_bytes=%d duration_s=%.2f",
            session_id or "-", total_ms, len(pcm), duration_s,
        )

        # ── Step 3: yield in chunk_ms sized pieces ────────────────────────
        yield from self._chunk_pcm(pcm, chunk_ms)

    # ── stream_pcm_sync (alias — keeps StreamingTtsService unchanged) ────────
    def stream_pcm_sync(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        return self.stream_pcm(text, chunk_ms, session_id=session_id, speed=speed)

    # ── sentence-by-sentence streaming from LLM token iterator (Fix 4) ───────
    def stream_pcm_from_text_stream(
        self,
        text_iterator,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        """
        Accepts a token-by-token iterator (e.g. from an LLM stream).
        As soon as a complete sentence ending in . ! ? is detected it is
        immediately flushed to edge-tts — without waiting for the full
        LLM response. The caller hears the first sentence ~1-3 s sooner.
        """
        buffer         = ""
        sentence_index = 0

        def _flush(sentence: str) -> Iterator[bytes]:
            nonlocal sentence_index
            sentence = sentence.strip()
            if not sentence or len(sentence) < MIN_SENTENCE_CHARS:
                return
            logger.info(
                "[TTS] flushing sentence=%d chars=%d preview=%r session=%s",
                sentence_index, len(sentence), sentence[:60], session_id or "-",
            )
            yield from self.stream_pcm(sentence, chunk_ms, session_id=session_id, speed=speed)

        for token in text_iterator:
            if not token:
                continue
            buffer += token
            parts = _SENTENCE_END_RE.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    yield from _flush(sentence)
                    sentence_index += 1
                buffer = parts[-1]

        # Flush remaining fragment after token stream ends
        if buffer.strip():
            logger.info(
                "[TTS] flushing final fragment chars=%d preview=%r session=%s",
                len(buffer), buffer.strip()[:60], session_id or "-",
            )
            yield from self.stream_pcm(buffer.strip(), chunk_ms, session_id=session_id, speed=speed)

    # ── cancel (barge-in support) ─────────────────────────────────────────────
    def cancel_session(self, session_id: str) -> None:
        cancel_tts(session_id)

    # ── internal: async edge-tts call → raw MP3 bytes ────────────────────────
    async def _synthesise_to_mp3(self, text: str, voice: str) -> bytes:
        """
        Calls the edge-tts API and collects all audio chunks into memory.
        edge-tts streams MP3 data; we buffer it all then convert to PCM once.
        """
        try:
            import edge_tts
        except ImportError as exc:
            raise StreamingTtsUnavailable(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from exc

        communicate  = edge_tts.Communicate(text, voice, rate=self.speed)
        mp3_chunks: list[bytes] = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        return b"".join(mp3_chunks)

    # ── internal: MP3 bytes → raw PCM-16 at TARGET_SAMPLE_RATE ───────────────
    @staticmethod
    def _mp3_to_pcm(mp3_bytes: bytes) -> bytes:
        """
        Decode MP3 → PCM-16 mono at TARGET_SAMPLE_RATE using pydub + ffmpeg.

        Requirements:
            pip install pydub
            ffmpeg on PATH  (apt install ffmpeg  /  choco install ffmpeg)
        """
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise StreamingTtsUnavailable(
                "pydub is not installed. Run: pip install pydub"
            ) from exc

        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        seg = (
            seg.set_frame_rate(TARGET_SAMPLE_RATE)
               .set_channels(TARGET_CHANNELS)
               .set_sample_width(TARGET_SAMPLE_WIDTH)
        )
        return seg.raw_data

    # ── internal: slice PCM bytes into chunk_ms sized pieces ─────────────────
    @staticmethod
    def _chunk_pcm(pcm: bytes, chunk_ms: int) -> Iterator[bytes]:
        clamped_ms = max(MIN_CHUNK_MS, min(MAX_CHUNK_MS, int(chunk_ms)))
        if clamped_ms != chunk_ms:
            logger.warning(
                "[TTS] chunk_ms clamped requested=%d clamped=%d min=%d max=%d",
                chunk_ms, clamped_ms, MIN_CHUNK_MS, MAX_CHUNK_MS,
            )
        chunk_bytes  = max(2, TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH * clamped_ms // 1000)
        chunk_bytes -= chunk_bytes % 2   # keep 16-bit aligned
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset: offset + chunk_bytes]
            if chunk:
                yield chunk


# ── StreamingTtsService ──────────────────────────────────────────────────────
# Public interface used by pipeline.py — completely unchanged.
# Only __init__ swaps KokoroStreamingTts → EdgeTtsStreamingTts.

class StreamingTtsService:
    def __init__(self) -> None:
        self._kokoro = EdgeTtsStreamingTts()   # ← only change from original

    def validate_startup(self) -> None:
        self._kokoro.validate_startup()

    def stream_pcm_sync(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        return self._kokoro.stream_pcm(text, chunk_ms, session_id=session_id, speed=speed)

    async def stream_pcm(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[bytes]:
        stream_start   = time.monotonic()
        samples_yielded = 0
        chunk_index    = 0
        try:
            for chunk in self._kokoro.stream_pcm_sync(text, chunk_ms, session_id=session_id, speed=speed):
                yield chunk
                chunk_index     += 1
                samples_yielded += len(chunk) // TARGET_SAMPLE_WIDTH
                expected_wall    = samples_yielded / float(TARGET_SAMPLE_RATE)
                elapsed          = time.monotonic() - stream_start
                sleep_s          = expected_wall - elapsed
                if sleep_s > 0:
                    if chunk_index == 1 or chunk_index % 25 == 0:
                        logger.info(
                            "[TTS] pacing session=%s chunk=%d chunk_ms=%.1f sleep_ms=%.1f "
                            "lag_ms=%.1f realtime_factor=%.3f",
                            session_id or "-",
                            chunk_index,
                            (len(chunk) / float(TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH)) * 1000.0,
                            sleep_s * 1000.0,
                            max(0.0, -sleep_s) * 1000.0,
                            elapsed / max(expected_wall, 0.001),
                        )
                    await asyncio.sleep(sleep_s)
                else:
                    await asyncio.sleep(0)
        except StreamingTtsUnavailable:
            raise

    async def stream_pcm_from_llm(
        self,
        text_iterator,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Async entry point: LLM token stream → sentence-by-sentence TTS.
        The sync EdgeTTS generator runs in a thread executor so it never
        blocks the FastAPI event loop.
        """
        loop     = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        SENTINEL = object()

        def _run_in_thread():
            try:
                for chunk in self._kokoro.stream_pcm_from_text_stream(
                    text_iterator, chunk_ms, session_id=session_id, speed=speed
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result(timeout=10)
            except Exception as exc:
                logger.exception("[TTS] thread error session=%s: %s", session_id, exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop).result(timeout=5)

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is SENTINEL:
                break
            yield chunk

    def cancel_session(self, session_id: str) -> None:
        self._kokoro.cancel_session(session_id)


# ── singleton ────────────────────────────────────────────────────────────────
_service: StreamingTtsService | None = None


def get_streaming_tts_service() -> StreamingTtsService:
    global _service
    if _service is None:
        _service = StreamingTtsService()
    return _service