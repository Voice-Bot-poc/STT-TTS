import asyncio
import logging
import math
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Iterator
from scipy.signal import resample_poly as _scipy_resample_poly
 
import numpy as np
 
from services.runtime_state import cancel_tts, get_session
from services.tts_normalizer import normalize_for_tts
 
logger = logging.getLogger(__name__)
 
TARGET_SAMPLE_RATE = int(os.getenv("TTS_TARGET_SAMPLE_RATE", "16000"))
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
DEFAULT_CHUNK_MS = int(os.getenv("STREAMING_TTS_CHUNK_MS", "60"))
MIN_CHUNK_MS = int(os.getenv("STREAMING_TTS_MIN_CHUNK_MS", "20"))
MAX_CHUNK_MS = int(os.getenv("STREAMING_TTS_MAX_CHUNK_MS", "200"))
KOKORO_SAMPLE_RATE = int(os.getenv("KOKORO_SAMPLE_RATE", "16000"))
KOKORO_SPEED_DEFAULT = float(os.getenv("KOKORO_SPEED", "0.80"))
NORMALIZE_MAX_GAIN = float(os.getenv("KOKORO_NORMALIZE_MAX_GAIN", "3.5"))
 
# Fix 4: minimum chars before flushing a sentence to TTS (avoids tiny fragments)
MIN_SENTENCE_CHARS = int(os.getenv("TTS_MIN_SENTENCE_CHARS", "20"))
 
_HINDI_RE = re.compile(r"[\u0900-\u097F]")
# Fix 4: sentence boundary detection — splits on . ! ? followed by space or end
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")
 
 
def detect_language(text: str) -> str:
    return "hi" if _HINDI_RE.search(text or "") else "en"
 
 
def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
 
 
class StreamingTtsUnavailable(RuntimeError):
    pass
 
 
class KokoroStreamingTts:
    """Kokoro-backed PCM streaming adapter for realtime voice playback."""
 
    def __init__(self) -> None:
        self.en_lang_code = os.getenv("KOKORO_EN_LANG_CODE", os.getenv("KOKORO_LANG_CODE", "a")).strip() or "a"
        self.en_voice = os.getenv("KOKORO_EN_VOICE", os.getenv("KOKORO_VOICE", "af_heart")).strip() or "af_heart"
        self.hi_lang_code = os.getenv("KOKORO_HI_LANG_CODE", self.en_lang_code).strip() or self.en_lang_code
        self.hi_voice = os.getenv("KOKORO_HI_VOICE", self.en_voice).strip() or self.en_voice
        self.speed = KOKORO_SPEED_DEFAULT
        self.split_pattern = os.getenv("KOKORO_SPLIT_PATTERN", r"(?<=[\.!\?])\s+|\n+")
        self.device_name = os.getenv("KOKORO_DEVICE", "auto").strip().lower() or "auto"
        self.normalize_audio = _parse_bool(os.getenv("KOKORO_NORMALIZE", "true"), default=True)
        self._pipelines: dict[str, object] = {}
        self._device = "cpu"
        self._lock = threading.Lock()
 
    def validate_startup(self) -> None:
        self._get_pipeline(self.en_lang_code)
 
    def stream_pcm(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        if not text or not text.strip():
            return
 
        language = detect_language(text)
        logger.info("TTS language detected=%s", language)
        if language == "hi":
            voice = self.hi_voice
            lang_code = self.hi_lang_code
        else:
            voice = self.en_voice
            lang_code = self.en_lang_code
 
        logger.info("Kokoro language=%s voice=%s lang_code=%s", language, voice, lang_code)
 
        text = normalize_for_tts(text)
 
        pipeline = self._get_pipeline(lang_code)
        session_generation = self._current_generation(session_id)
        effective_speed = self._resolve_speed(speed)
        logger.info(
            "[TTS] synthesis start session=%s chars=%d voice=%s device=%s speed=%.2f chunk_ms=%d",
            session_id or "-",
            len(text),
            voice,
            self._device,
            effective_speed,
            chunk_ms,
        )
 
        pcm_bytes_total = 0
        first_chunk = True
        start = time.perf_counter()
        try:
            for segment_index, item in enumerate(
                pipeline(text, voice=voice, speed=effective_speed, split_pattern=self.split_pattern)
            ):
                if not self._is_generation_current(session_id, session_generation):
                    logger.info(
                        "[TTS] cancelled by barge-in session=%s generation=%s segment=%d",
                        session_id,
                        session_generation,
                        segment_index,
                    )
                    return
 
                pcm, meta = self._extract_pcm(item)
                if not pcm:
                    continue
 
                if first_chunk:
                    first_chunk = False
                    logger.info(
                        "[TTS] first audio session=%s chars=%d generation=%s first_chunk_ms=%.1f bytes=%d",
                        session_id or "-",
                        len(text),
                        session_generation,
                        (time.perf_counter() - start) * 1000.0,
                        len(pcm),
                    )
                    logger.info(
                        "[TTS] audio meta session=%s source_rate=%d target_rate=%d resampled=%s peak=%.4f gain=%.2f samples=%d",
                        session_id or "-",
                        meta["source_rate"],
                        meta["target_rate"],
                        meta["resampled"],
                        meta["peak"],
                        meta["gain"],
                        meta["samples"],
                    )
 
                for chunk in self._chunk_pcm(pcm, chunk_ms):
                    if not self._is_generation_current(session_id, session_generation):
                        logger.info(
                            "[TTS] cancelled by barge-in session=%s generation=%s",
                            session_id,
                            session_generation,
                        )
                        return
 
                    pcm_bytes_total += len(chunk)
                    logger.info(
                        "[TTS] emitted pcm chunk bytes=%d session=%s generation=%s",
                        len(chunk),
                        session_id or "-",
                        session_generation,
                    )
                    yield chunk
        except Exception as exc:
            logger.exception("[TTS] synthesis failed session=%s generation=%s: %s", session_id, session_generation, exc)
            raise StreamingTtsUnavailable(str(exc)) from exc
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            duration_s = pcm_bytes_total / float(TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH)
            realtime_factor = duration_s / max(elapsed_ms / 1000.0, 0.001)
            logger.info(
                "[TTS] synthesis complete session=%s generation=%s bytes=%d duration_s=%.2f elapsed_ms=%.1f realtime_factor=%.2f",
                session_id or "-",
                session_generation,
                pcm_bytes_total,
                duration_s,
                elapsed_ms,
                realtime_factor,
            )
 
    def stream_pcm_sync(
        self,
        text: str,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        return self.stream_pcm(text, chunk_ms, session_id=session_id, speed=speed)
 
    # ── Fix 4: Accept a token stream from the LLM and begin TTS sentence-by-sentence ──
    def stream_pcm_from_text_stream(
        self,
        text_iterator,           # any iterable of str tokens from LLM
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]:
        """
        Accepts a token-by-token iterator from the LLM.
        As soon as a complete sentence is detected (ends with . ! ?),
        it is immediately flushed to Kokoro for TTS — without waiting
        for the full LLM response.
 
        This means the caller hears the first sentence ~1-3s earlier
        than waiting for the complete reply.
        """
        buffer = ""
        sentence_index = 0
 
        def _flush_sentence(sentence: str) -> Iterator[bytes]:
            sentence = sentence.strip()
            if not sentence or len(sentence) < MIN_SENTENCE_CHARS:
                return
            logger.info(
                "[TTS] Fix4 flushing sentence=%d chars=%d preview=%r session=%s",
                sentence_index,
                len(sentence),
                sentence[:60],
                session_id or "-",
            )
            yield from self.stream_pcm(sentence, chunk_ms, session_id=session_id, speed=speed)
 
        for token in text_iterator:
            if not token:
                continue
            buffer += token
 
            # Check if buffer contains one or more complete sentences
            parts = _SENTENCE_END_RE.split(buffer)
            if len(parts) > 1:
                # All parts except the last are complete sentences — flush them
                for sentence in parts[:-1]:
                    yield from _flush_sentence(sentence)
                    sentence_index += 1
                buffer = parts[-1]  # keep the incomplete trailing fragment
 
        # Flush whatever remains after the LLM token stream ends
        if buffer.strip():
            logger.info(
                "[TTS] Fix4 flushing final fragment chars=%d preview=%r session=%s",
                len(buffer),
                buffer.strip()[:60],
                session_id or "-",
            )
            yield from self.stream_pcm(buffer.strip(), chunk_ms, session_id=session_id, speed=speed)
 
    def cancel_session(self, session_id: str) -> None:
        cancel_tts(session_id)
 
    def _get_pipeline(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline
 
        with self._lock:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is not None:
                return pipeline
 
            try:
                device = self._resolve_device()
 
                from kokoro import KPipeline
 
                load_start = time.perf_counter()
                try:
                    pipeline = KPipeline(lang_code=lang_code, device=device)
                except TypeError:
                    logger.warning("[TTS] Kokoro pipeline does not accept device=; falling back to default constructor")
                    pipeline = KPipeline(lang_code=lang_code)
                    device = self._infer_pipeline_device(pipeline)
 
                self._pipelines[lang_code] = pipeline
                self._device = device
                logger.info(
                    "[TTS] Kokoro initialized device=%s lang_code=%s speed=%.2f load_ms=%.1f",
                    self._device,
                    lang_code,
                    self.speed,
                    (time.perf_counter() - load_start) * 1000.0,
                )
                return pipeline
            except Exception as exc:
                logger.exception("[TTS] Kokoro initialization failed: %s", exc)
                raise StreamingTtsUnavailable(str(exc)) from exc
 
    @staticmethod
    def _detect_language(text: str) -> str:
        return detect_language(text)
 
    def _resolve_device(self) -> str:
        try:
            import torch
        except Exception:
            return "cpu"
 
        if self.device_name != "auto":
            if self.device_name == "cuda" and not torch.cuda.is_available():
                return "cpu"
            if self.device_name == "mps":
                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend is None or not mps_backend.is_available():
                    return "cpu"
            return self.device_name
 
        if torch.cuda.is_available():
            return "cuda"
 
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
 
        return "cpu"
 
    def _infer_pipeline_device(self, pipeline) -> str:
        model = getattr(pipeline, "model", None)
        inferred = getattr(model, "device", None)
        return str(inferred) if inferred is not None else self._device
 
    def _current_generation(self, session_id: str | None) -> int | None:
        if session_id is None:
            return None
        return get_session(session_id).tts_generation
 
    def _is_generation_current(self, session_id: str | None, generation: int | None) -> bool:
        if session_id is None or generation is None:
            return True
        state = get_session(session_id)
        return state.tts_generation == generation and state.tts_active
 
    def _resolve_speed(self, speed: float | None) -> float:
        if speed is None:
            return self.speed
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            return self.speed
        return min(max(speed_value, 0.5), 1.3)
 
    def _extract_pcm(self, item) -> tuple[bytes, dict[str, object]]:
        audio = None
        source_sample_rate = KOKORO_SAMPLE_RATE
 
        if hasattr(item, "output") and hasattr(item.output, "audio"):
            audio = item.output.audio
            source_sample_rate = int(getattr(item.output, "sample_rate", getattr(item.output, "sampling_rate", source_sample_rate)))
        elif isinstance(item, tuple) and len(item) >= 3:
            audio = item[2]
        else:
            audio = item
 
        if audio is None:
            return b"", {
                "source_rate": source_sample_rate,
                "target_rate": TARGET_SAMPLE_RATE,
                "resampled": False,
                "peak": 0.0,
                "gain": 1.0,
                "samples": 0,
            }
 
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
 
        audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        resampled = False
 
        if source_sample_rate != TARGET_SAMPLE_RATE and audio_np.size > 0:
            audio_np = self._resample_poly(audio_np, source_sample_rate, TARGET_SAMPLE_RATE)
            resampled = True
 
        # Fix 5: only normalize when audio actually needs it
        gain = 1.0
        if self.normalize_audio and audio_np.size > 0:
            peak = float(np.max(np.abs(audio_np)))
            if peak < 1e-6 or peak > 0.85:
                audio_np, gain = self._normalize_chunk(audio_np)
 
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype("<i2")
        return pcm16.tobytes(), {
            "source_rate": source_sample_rate,
            "target_rate": TARGET_SAMPLE_RATE,
            "resampled": resampled,
            "peak": peak,
            "gain": gain,
            "samples": int(audio_np.size),
        }
 
    @staticmethod
    def _resample_poly(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate <= 0 or source_rate == target_rate or audio.size == 0:
            return audio
 
        gcd = math.gcd(source_rate, target_rate)
        up = target_rate // gcd
        down = source_rate // gcd
        return _scipy_resample_poly(audio, up, down).astype(np.float32)
 
    @staticmethod
    def _normalize_chunk(audio: np.ndarray) -> tuple[np.ndarray, float]:
        if audio.size == 0:
            return audio, 1.0
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            return audio, 1.0
        gain = min(1.0 / peak, NORMALIZE_MAX_GAIN)
        return audio * gain, gain
 
    @staticmethod
    def _chunk_pcm(pcm: bytes, chunk_ms: int) -> Iterator[bytes]:
        clamped_ms = max(MIN_CHUNK_MS, min(MAX_CHUNK_MS, int(chunk_ms)))
        if clamped_ms != chunk_ms:
            logger.warning(
                "[TTS] chunk_ms adjusted requested=%d clamped=%d min=%d max=%d",
                chunk_ms,
                clamped_ms,
                MIN_CHUNK_MS,
                MAX_CHUNK_MS,
            )
        chunk_bytes = max(2, TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH * clamped_ms // 1000)
        chunk_bytes -= chunk_bytes % 2
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            if chunk:
                yield chunk
 
 
class StreamingTtsService:
    def __init__(self) -> None:
        self._kokoro = KokoroStreamingTts()
 
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
        stream_start = time.monotonic()
        samples_yielded = 0
        chunk_index = 0
        try:
            for chunk in self._kokoro.stream_pcm_sync(text, chunk_ms, session_id=session_id, speed=speed):
                yield chunk
                chunk_index += 1
                samples_yielded += len(chunk) // TARGET_SAMPLE_WIDTH
                expected_wall = samples_yielded / float(TARGET_SAMPLE_RATE)
                elapsed = time.monotonic() - stream_start
                sleep_s = expected_wall - elapsed
                if sleep_s > 0:
                    if chunk_index == 1 or chunk_index % 25 == 0:
                        logger.info(
                            "[TTS] pacing session=%s chunk=%d chunk_ms=%.1f sleep_ms=%.1f lag_ms=%.1f realtime_factor=%.3f",
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
 
    # ── Fix 4: Async wrapper — accepts LLM token stream, yields PCM as sentences arrive ──
    async def stream_pcm_from_llm(
        self,
        text_iterator,           # sync iterable of str tokens from Ollama stream
        chunk_ms: int = DEFAULT_CHUNK_MS,
        session_id: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Async entry point for LLM-token-stream → sentence-by-sentence TTS.
 
        Usage in pipeline.py:
            async for pcm_chunk in tts_service.stream_pcm_from_llm(
                ollama_token_iterator,
                session_id=session_id,
            ):
                yield pcm_chunk
 
        The sync Kokoro generator runs in a thread executor so it never
        blocks the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        SENTINEL = object()
 
        def _run_in_thread():
            try:
                for chunk in self._kokoro.stream_pcm_from_text_stream(
                    text_iterator, chunk_ms, session_id=session_id, speed=speed
                ):
                    # Put each PCM chunk onto the async queue from the thread
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result(timeout=10)
            except Exception as exc:
                logger.exception("[TTS] Fix4 thread error session=%s: %s", session_id, exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop).result(timeout=5)
 
        # Run the blocking Kokoro generator in a background thread
        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
 
        # Yield PCM chunks from the queue as they arrive
        while True:
            chunk = await queue.get()
            if chunk is SENTINEL:
                break
            yield chunk
 
    def cancel_session(self, session_id: str) -> None:
        self._kokoro.cancel_session(session_id)
 
 
_service: StreamingTtsService | None = None
 
 
def get_streaming_tts_service() -> StreamingTtsService:
    global _service
    if _service is None:
        _service = StreamingTtsService()
    return _service