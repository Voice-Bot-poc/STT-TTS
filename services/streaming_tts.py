from dotenv import load_dotenv
load_dotenv()

"""
streaming_tts.py — PCM streaming TTS via edge-tts + ffmpeg STDIN streaming.

edge-tts delivers MP3 chunks over the network, which are piped into
ffmpeg stdin in real time — no collect-all-then-decode delay.
Expected first-chunk latency: ~350-500ms.

Voices: Indian female (en-IN-NeerjaNeural / hi-IN-SwaraNeural)
"""

import asyncio
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Iterator

from services.runtime_state import cancel_tts
from services.tts_normalizer import normalize_for_tts

logger = logging.getLogger(__name__)

# ── Audio config ───────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE  = int(os.getenv("TTS_TARGET_SAMPLE_RATE", "16000"))
TARGET_CHANNELS     = 1
TARGET_SAMPLE_WIDTH = 2  # s16le = 2 bytes per sample
DEFAULT_CHUNK_MS    = int(os.getenv("STREAMING_TTS_CHUNK_MS", "20"))
MIN_CHUNK_MS        = int(os.getenv("STREAMING_TTS_MIN_CHUNK_MS", "20"))
MAX_CHUNK_MS        = int(os.getenv("STREAMING_TTS_MAX_CHUNK_MS", "200"))
MIN_SENTENCE_CHARS  = int(os.getenv("TTS_MIN_SENTENCE_CHARS", "20"))

_HINDI_RE        = re.compile(r"[\u0900-\u097F]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?,\:;\u0964\u0965])\s+|(?<=[.!?,\:;\u0964\u0965])$")

# ── edge-tts config ────────────────────────────────────────────────────────────
EDGE_TTS_EN_VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-IN-NeerjaNeural")
EDGE_TTS_HI_VOICE = os.getenv("EDGE_TTS_HI_VOICE", "hi-IN-SwaraNeural")
EDGE_TTS_SPEED    = os.getenv("EDGE_TTS_SPEED", "+0%")
FFMPEG_PATH       = os.getenv("FFMPEG_PATH", "ffmpeg")

VOICE_MAP = {
    "en": os.getenv("EDGE_TTS_EN_VOICE", "en-IN-NeerjaNeural"),
    "hi": os.getenv("EDGE_TTS_HI_VOICE", "hi-IN-SwaraNeural"),
    "mr": os.getenv("EDGE_TTS_MR_VOICE", "mr-IN-AarohiNeural"),
}


def detect_language(text: str) -> str:
    return "hi" if _HINDI_RE.search(text or "") else "en"


class StreamingTtsUnavailable(RuntimeError):
    pass



# ══════════════════════════════════════════════════════════════════════════════
#  edge-tts + ffmpeg STDIN streaming backend
#  Key change: ffmpeg is started FIRST with stdin=PIPE, then MP3 chunks are
#  written into it as edge-tts delivers them over the network.
#  ffmpeg decodes and outputs PCM in real time → first PCM in ~350ms.
# ══════════════════════════════════════════════════════════════════════════════

class EdgeTtsBackend:

    def __init__(self) -> None:
        self.en_voice = EDGE_TTS_EN_VOICE
        self.hi_voice = EDGE_TTS_HI_VOICE
        self.speed    = EDGE_TTS_SPEED

    def validate_startup(self) -> None:
        logger.info(
            "[TTS-ET] EdgeTTS ready | en=%s hi=%s speed=%s",
            self.en_voice, self.hi_voice, self.speed,
        )

    def stream_pcm(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
        language: str = "en",
    ) -> Iterator[bytes]:
        if not text or not text.strip():
            return

        # edge-tts uses its own string-format speed (e.g. "-20%") from EDGE_TTS_SPEED.
        # Ignore any legacy numeric speed overrides meant for Kokoro.
        if speed is not None:
            logger.warning(
                "[TTS-ET] ignoring numeric speed=%.2f — edge-tts uses EDGE_TTS_SPEED=%s",
                speed, self.speed,
            )

        text     = normalize_for_tts(text)

        voice = VOICE_MAP.get(language)
        if voice is None:
            logger.warning("[TTS] No voice for language=%r falling back to en", language)
            voice = VOICE_MAP["en"]

        logger.info(
            "[TTS-ET] start | session=%s chars=%d voice=%s language=%s speed=%s chunk_ms=%d",
            session_id, len(text), voice, language, speed, chunk_ms,
        )
        start = time.perf_counter()

        chunk_bytes = max(2, TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH * chunk_ms // 1000)
        chunk_bytes -= chunk_bytes % 2

        import queue as _queue
        pcm_queue: _queue.Queue = _queue.Queue(maxsize=128)
        error_box: list         = []
        first_logged            = [False]

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._stream_via_ffmpeg_stdin(
                        text, voice, chunk_bytes, session_id,
                        pcm_queue, first_logged, start,
                    )
                )
            except Exception as exc:
                error_box.append(exc)
                logger.exception("[TTS-ET] thread error session=%s: %s", session_id, exc)
            finally:
                pcm_queue.put(None)  # sentinel
                loop.close()

        threading.Thread(target=_run, daemon=True).start()

        while True:
            try:
                chunk = pcm_queue.get(timeout=20)
            except Exception:
                break
            if chunk is None:
                break
            yield chunk

        if error_box:
            raise StreamingTtsUnavailable(str(error_box[0]))

        logger.info(
            "[TTS-ET] complete | session=%s total_ms=%.1f",
            session_id or "-", (time.perf_counter() - start) * 1000.0,
        )

    async def _stream_via_ffmpeg_stdin(
        self,
        text: str,
        voice: str,
        chunk_bytes: int,
        session_id: str | None,
        pcm_queue,
        first_logged: list,
        start: float,
    ):
        """
        TRUE streaming pipeline:
          edge-tts MP3 chunks → ffmpeg stdin → ffmpeg stdout PCM → pcm_queue

        ffmpeg is launched once with stdin=PIPE and stdout=PIPE.
        A writer coroutine feeds MP3 data into ffmpeg stdin as edge-tts
        delivers it over the network.
        A reader coroutine reads PCM from ffmpeg stdout and puts it into
        pcm_queue in chunk_bytes sized pieces.
        Both run concurrently — first PCM arrives ~350ms after call start.
        """
        try:
            import edge_tts
        except ImportError as exc:
            raise StreamingTtsUnavailable("pip install edge-tts") from exc

        # ── Launch ffmpeg with stdin/stdout pipes ──────────────────────────
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-hide_banner", "-loglevel", "error",
            "-f", "mp3",          # input format
            "-i", "pipe:0",       # read from stdin
            "-f", "s16le",        # output raw PCM
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "pipe:1",             # write to stdout
        ]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin  = asyncio.subprocess.PIPE,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.DEVNULL,
        )

        # ── Writer: edge-tts → ffmpeg stdin ───────────────────────────────
        async def _writer():
            max_retries = 3
            try:
                for attempt in range(max_retries):
                    try:
                        communicate = edge_tts.Communicate(text, voice, rate=self.speed)
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                proc.stdin.write(chunk["data"])
                                await proc.stdin.drain()
                        break  # Success, exit retry loop
                    except Exception as exc:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "[TTS-ET] writer attempt %d failed for session=%s: %s. Retrying...",
                                attempt + 1, session_id, exc
                            )
                            await asyncio.sleep(0.2 * (attempt + 1))
                            continue
                        logger.exception("[TTS-ET] writer error session=%s: %s", session_id, exc)
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        # ── Reader: ffmpeg stdout → pcm_queue ─────────────────────────────
        async def _reader():
            buf = b""
            while True:
                data = await proc.stdout.read(4096)
                if not data:
                    break
                buf += data
                while len(buf) >= chunk_bytes:
                    pcm_chunk = buf[:chunk_bytes]
                    buf = buf[chunk_bytes:]
                    pcm_queue.put(pcm_chunk)
                    if not first_logged[0]:
                        first_logged[0] = True
                        logger.info(
                            "[TTS-ET] first PCM chunk | session=%s latency_ms=%.1f",
                            session_id or "-",
                            (time.perf_counter() - start) * 1000.0,
                        )
            # flush remainder
            if buf:
                if len(buf) % 2:
                    buf += b"\x00"
                pcm_queue.put(buf)

        # ── Run writer and reader concurrently ────────────────────────────
        await asyncio.gather(_writer(), _reader())
        await proc.wait()

        edge_done_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[TTS-ET] ffmpeg done | session=%s total_ms=%.1f",
            session_id or "-", edge_done_ms,
        )

    def cancel_session(self, session_id: str) -> None:
        cancel_tts(session_id)


# ══════════════════════════════════════════════════════════════════════════════
#  Sentence-streaming wrapper  (works with any backend)
# ══════════════════════════════════════════════════════════════════════════════

class SentenceStreamingMixin:
    def stream_pcm_from_text_stream(
        self,
        text_iterator,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
        language: str = "en",
    ) -> Iterator[bytes]:
        buffer         = ""
        sentence_index = 0

        def _flush(sentence: str) -> Iterator[bytes]:
            nonlocal sentence_index
            sentence = sentence.strip()
            if not sentence:
                return
            logger.info(
                "[TTS] flush sentence=%d chars=%d preview=%r session=%s",
                sentence_index, len(sentence), sentence[:60], session_id or "-",
            )
            yield from self.stream_pcm(sentence, chunk_ms, session_id=session_id, speed=speed, language=language)

        for token in text_iterator:
            if not token:
                continue
            buffer += token
            
            while True:
                # Find the first punctuation mark that gives us a chunk >= MIN_SENTENCE_CHARS
                split_idx = -1
                for match in _SENTENCE_END_RE.finditer(buffer):
                    if match.end() >= MIN_SENTENCE_CHARS:
                        split_idx = match.end()
                        break
                
                if split_idx != -1:
                    sentence = buffer[:split_idx]
                    yield from _flush(sentence)
                    sentence_index += 1
                    buffer = buffer[split_idx:]
                else:
                    break

        if buffer.strip():
            logger.info(
                "[TTS] flush final chars=%d preview=%r session=%s",
                len(buffer), buffer.strip()[:60], session_id or "-",
            )
            yield from _flush(buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  Concrete backend class
# ══════════════════════════════════════════════════════════════════════════════

class EdgeTtsStreaming(SentenceStreamingMixin, EdgeTtsBackend):
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  StreamingTtsService  — public API
# ══════════════════════════════════════════════════════════════════════════════

class StreamingTtsService:

    def __init__(self) -> None:
        self._backend = EdgeTtsStreaming()
        logger.info("[TTS] Using edge-tts backend (en=%s hi=%s speed=%s)",
                    EDGE_TTS_EN_VOICE, EDGE_TTS_HI_VOICE, EDGE_TTS_SPEED)

    def validate_startup(self) -> None:
        self._backend.validate_startup()

    def stream_pcm_sync(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
        language: str = "en",
    ) -> Iterator[bytes]:
        return self._backend.stream_pcm(
            text, chunk_ms, session_id=session_id, speed=speed, language=language
        )

    async def stream_pcm(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
        language: str = "en",
    ) -> AsyncIterator[bytes]:
        stream_start    = time.monotonic()
        samples_yielded = 0
        chunk_index     = 0
        try:
            for chunk in self._backend.stream_pcm(
                text, chunk_ms, session_id=session_id, speed=speed, language=language
            ):
                yield chunk
                chunk_index     += 1
                samples_yielded += len(chunk) // TARGET_SAMPLE_WIDTH
                expected_wall    = samples_yielded / float(TARGET_SAMPLE_RATE)
                elapsed          = time.monotonic() - stream_start
                sleep_s          = expected_wall - elapsed
                if sleep_s > 0:
                    if chunk_index == 1 or chunk_index % 25 == 0:
                        logger.info(
                            "[TTS] pacing session=%s chunk=%d sleep_ms=%.1f rtf=%.3f",
                            session_id or "-", chunk_index,
                            sleep_s * 1000.0,
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
        language: str = "en",
    ) -> AsyncIterator[bytes]:
        loop     = asyncio.get_event_loop()
        queue    = asyncio.Queue(maxsize=16)
        SENTINEL = object()

        def _run_in_thread():
            try:
                for chunk in self._backend.stream_pcm_from_text_stream(
                    text_iterator, chunk_ms, session_id=session_id, speed=speed, language=language
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(chunk), loop
                    ).result(timeout=10)
            except Exception as exc:
                logger.exception("[TTS] thread error session=%s: %s", session_id, exc)
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(SENTINEL), loop
                ).result(timeout=5)

        threading.Thread(target=_run_in_thread, daemon=True).start()

        while True:
            chunk = await queue.get()
            if chunk is SENTINEL:
                break
            yield chunk

    def cancel_session(self, session_id: str) -> None:
        self._backend.cancel_session(session_id)


# ── singleton ──────────────────────────────────────────────────────────────────
_service: StreamingTtsService | None = None


def get_streaming_tts_service() -> StreamingTtsService:
    global _service
    if _service is None:
        _service = StreamingTtsService()
    return _service
    return _service
