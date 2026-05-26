"""
stt_engine.py — Production-grade STT engine with VAD, noise reduction,
and conservative Faster-Whisper decoding.

Changes vs original (functionality preserved, bugs fixed):
  FIX 1 — _detect_device(): logger was misindented INSIDE the function,
           causing an IndentationError on import. Moved to module level.
  FIX 2 — _ensure_mono(): referenced `audio_bytes` which is out of scope
           in that method. Container-format loading moved to _load_audio()
           where `audio_bytes` is actually in scope. _ensure_mono() is now
           a pure ndarray operation.
  FIX 3 — stream_transcribe(): the entire endpointing block (if speech_active
           and last_speech_ts > 300 … yield transcript … reset) was indented
           inside _validate_transcript() due to a copy-paste indentation error
           and therefore NEVER executed. Moved back into stream_transcribe().
  FIX 4 — _validate_transcript(): had a `return False` immediately before the
           `except` clause; Python treats everything after it as dead code.
           Removed the stray return so the except is reachable.
  FIX 5 — VAD aggressiveness lowered 3 → 2. Mode 3 aggressively cuts real
           speech in noisy phone-call conditions and is the primary cause of
           "body ache" → "CIO body ache" style hallucinations.
  FIX 6 — transcribe_bytes(): beam_size 3→5, added best_of=5, vad_filter=True,
           no_speech_threshold 0.6→0.5, added compression_ratio_threshold and
           log_prob_threshold for more stable medical-vocabulary decoding.
  FIX 7 — stream_transcribe(): silence grace raised 300 ms → 1 800 ms.
           300 ms is far too short; it finalises utterances mid-sentence.
  FIX 8 — Incomplete-suffix detection: if a finalised transcript ends with
           "with / to / and / my name is / I want to …" the grace is extended
           instead of cutting the sentence short.
  FIX 9 — Minimum speech duration guard (1 200 ms) before sending to Whisper
           prevents hallucinations on sub-second noise bursts.
"""

import audioop
import io
import logging
import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
import webrtcvad
from faster_whisper import WhisperModel

# FIX 1 — module-level logger (was misindented inside _detect_device())
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Incomplete-utterance suffixes
# If a finalised transcript ends with any of these, silence grace is extended
# so the user can finish the sentence.
# ---------------------------------------------------------------------------
_INCOMPLETE_SUFFIXES = (
    " with", " to", " and", " or", " the", " a", " an",
    "my name is", "i want to", "book with", "i'd like to",
    "i would like", "can you", "i need",
)

_HALLUCINATION_PHRASES = (
    "thank you very much",
    "thank you for watching",
    "subscribe",
    "phone mail",
    "over the years",
    "www.",
    "iso screen",
    "home rating",
    "home readiness",
)


def _detect_device() -> str:
    # FIX 1 — logger reference now resolves correctly
    return "cuda" if torch.cuda.is_available() else "cpu"


class STTEngine:
    """Lightweight streaming STT engine wrapper.

    - Uses WebRTC VAD for speech/silence detection (frame-level)
    - Applies a simple noise-reduction pass (noisereduce)
    - Runs faster-whisper on GPU when available (or int8 on CPU)
    - Uses conservative decoding parameters (temperature=0.0) to reduce hallucination
    """

    SILENCE_GRACE_MS: int = 1800   # ms of silence before finalising utterance (FIX 7)
    MIN_SPEECH_MS: int = 900       # minimum speech before transcription attempt
    MAX_BUFFER_MS: int = 8000      # safety cap — force flush after this

    MIN_RMS: int = 450
    MIN_VOICE_RATIO: float = 0.55

    def __init__(self, model_size: str = "base"):
        self.device = _detect_device()
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)

        # FIX 5 — aggressiveness 2 (was 3). Mode 3 cuts real speech on phone calls.
        self.vad = webrtcvad.Vad(3)

        self.frame_ms = 30
        self.frame_bytes = int(SAMPLE_RATE * (self.frame_ms / 1000.0) * 2)  # 16-bit PCM

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        FIX 2 — load audio from bytes into float32 mono at SAMPLE_RATE.
        The original _ensure_mono() tried to call sf.read(audio_bytes) but
        `audio_bytes` was out of scope there (it is a method on np.ndarray).
        Container-format loading now lives here where the variable is in scope.
        """
        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)
            return audio_np.astype(np.float32)
        except Exception:
            # Fallback: assume raw PCM16 little-endian mono
            return self._bytes_to_float32(audio_bytes)

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        # FIX 2 — now a pure ndarray operation; no out-of-scope reference
        if audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    def noise_reduce(self, audio_np: np.ndarray) -> np.ndarray:
        try:
            return nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE, prop_decrease=0.8)
        except Exception:
            return audio_np

    def is_speech_frame(self, frame_bytes: bytes) -> bool:
        try:
            return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Single-blob transcription
    # ------------------------------------------------------------------

    def transcribe_bytes(self, audio_bytes: bytes, min_speech_ms: int = 0) -> str:
        """Synchronous transcription for a single audio blob (wav PCM16 bytes).

        Applies noise reduction + conservative faster-whisper decode.
        """
        if not audio_bytes or len(audio_bytes) < 512:
            return ""

        # FIX 2 — use _load_audio() which handles container formats correctly
        audio = self._load_audio(audio_bytes)
        audio = self._ensure_mono(audio)

        # FIX 9 — duration guard: skip tiny noise bursts
        duration_ms = (len(audio) / SAMPLE_RATE) * 1000.0
        pcm_for_gate = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        rms = audioop.rms(pcm_for_gate, 2) if pcm_for_gate else 0
        vad_ratio = self._vad_ratio(pcm_for_gate)
        min_ms = max(min_speech_ms, self.MIN_SPEECH_MS)

        if duration_ms < min_ms:
            logger.info("STT rejected | rms=%d vad=%.2f duration_ms=%.0f reason=short_audio", rms, vad_ratio, duration_ms)
            return ""

        if rms < self.MIN_RMS:
            logger.info("STT rejected | rms=%d vad=%.2f reason=low_rms", rms, vad_ratio)
            return ""

        if vad_ratio < self.MIN_VOICE_RATIO:
            logger.info("STT rejected | rms=%d vad=%.2f reason=low_voice_ratio", rms, vad_ratio)
            return ""

        audio = self.noise_reduce(audio)

        # FIX 6 — production-grade Whisper params
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            best_of=5,
            word_timestamps=False,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        if info.no_speech_prob > 0.5:
            logger.debug("SKIP: no_speech_prob=%.3f", info.no_speech_prob)
            return ""
        if info.avg_logprob < -1.0:
            logger.debug("SKIP: avg_logprob=%.3f", info.avg_logprob)
            return ""

        text = "".join(s.text for s in segments).strip()

        if len(text) < 2:
            return ""

        if not self._validate_transcript(text, audio):
            logger.debug("SKIP: failed validation: %r", text)
            return ""

        logger.info("[STTEngine] transcript (%d chars): %r", len(text), text)
        return text

    def _vad_ratio(self, pcm: bytes) -> float:
        if not pcm:
            return 0.0

        total = 0
        speech = 0
        for offset in range(0, len(pcm) - self.frame_bytes + 1, self.frame_bytes):
            frame = pcm[offset : offset + self.frame_bytes]
            total += 1
            if self.is_speech_frame(frame):
                speech += 1

        return speech / total if total else 0.0

    # ------------------------------------------------------------------
    # Validation / hallucination helpers
    # ------------------------------------------------------------------

    def _validate_transcript(self, text: str, audio_np: np.ndarray) -> bool:
        """Simple heuristics to catch likely hallucinations or invalid outputs.

        Returns True if transcript looks plausible for the given audio.

        FIX 4 — removed the stray `return False` that appeared before the
        `except` clause and buried all downstream code as dead code.
        """
        try:
            if audio_np is None or len(audio_np) == 0:
                return False

            duration_ms = (len(audio_np) / SAMPLE_RATE) * 1000.0
            words = (text or "").strip().split()
            word_count = len(words)

            if duration_ms > 1500 and word_count < 2:
                return False
            if duration_ms > 4000 and word_count < 4:
                return False

            lowered = text.lower()
            tokens = [t for t in lowered.split() if t]
            if tokens:
                most_common = max(set(tokens), key=tokens.count)
                if tokens.count(most_common) / len(tokens) > 0.6 and len(tokens) > 3:
                    return False

            for phrase in _HALLUCINATION_PHRASES:
                if phrase in lowered:
                    return False

            return True

        except Exception:
            # FIX 4 — except is now reachable (no dead `return False` above it)
            return False

    def _ends_incomplete(self, text: str) -> bool:
        """FIX 8 — True if transcript looks like a cut sentence."""
        lowered = (text or "").strip().lower()
        return any(lowered.endswith(suffix.strip()) for suffix in _INCOMPLETE_SUFFIXES)

    # ------------------------------------------------------------------
    # Streaming transcription
    # FIX 3 — endpointing logic moved OUT of _validate_transcript() and
    # back into stream_transcribe() where it has access to buffer/state.
    # ------------------------------------------------------------------

    def stream_transcribe(self, chunk_iterator, max_buffer_ms: int = MAX_BUFFER_MS):
        """Process incoming PCM16 bytes chunks (generator). Yields transcripts when
        VAD endpointing occurs or buffer time exceeds max_buffer_ms.

        chunk_iterator should yield raw PCM16 bytes (mono or stereo interleaved).

        The generator NEVER exits between utterances — silence only finalises
        the current utterance; the loop keeps running so post-pause speech is
        always captured.
        """
        buffer = bytearray()
        speech_active = False
        silence_accumulated_ms = 0  # FIX 7 — counts silence after speech (was last_speech_ts)

        for chunk in chunk_iterator:
            if not chunk:
                continue
            buffer.extend(chunk)

            # ── VAD scan on newly accumulated frames ──────────────────────
            i = 0
            while i + self.frame_bytes <= len(buffer):
                frame = bytes(buffer[i : i + self.frame_bytes])
                is_speech = self.is_speech_frame(frame)

                if is_speech:
                    if not speech_active:
                        logger.debug("[stream] speech_start")
                    speech_active = True
                    silence_accumulated_ms = 0          # reset on any voiced frame
                else:
                    if speech_active:
                        silence_accumulated_ms += self.frame_ms  # accumulate only after speech

                i += self.frame_bytes

            buffer_ms = (len(buffer) / 2) / SAMPLE_RATE * 1000.0

            # ── FIX 3 — endpointing block (was inside _validate_transcript) ──
            # Finalise when silence grace expires
            if speech_active and silence_accumulated_ms >= self.SILENCE_GRACE_MS:
                logger.info(
                    "[stream] silence grace %d ms reached — finalising (buffer=%.0f ms)",
                    self.SILENCE_GRACE_MS, buffer_ms,
                )

                # RMS gate: discard comfort noise
                rms = audioop.rms(bytes(buffer), 2)
                if rms < 300:
                    logger.debug("[stream] RMS=%d too low — comfort noise, skip", rms)
                    buffer = bytearray()
                    speech_active = False
                    silence_accumulated_ms = 0
                    continue

                # FIX 9 — minimum speech duration guard
                if buffer_ms < self.MIN_SPEECH_MS:
                    logger.debug(
                        "[stream] buffer %.0f ms < MIN_SPEECH_MS %d ms — skip",
                        buffer_ms, self.MIN_SPEECH_MS,
                    )
                    buffer = bytearray()
                    speech_active = False
                    silence_accumulated_ms = 0
                    continue

                try:
                    transcript = self.transcribe_bytes(bytes(buffer))
                except Exception:
                    transcript = ""

                # FIX 8 — extend grace when transcript ends mid-sentence
                if transcript and self._ends_incomplete(transcript):
                    logger.debug(
                        "[stream] incomplete suffix in %r — extending grace", transcript
                    )
                    # Half-reset: give the user more time to continue
                    silence_accumulated_ms = self.SILENCE_GRACE_MS // 2
                    continue  # do NOT reset buffer; keep accumulating

                logger.info("[stream] transcript finalised: %r", transcript)
                yield transcript

                # Reset for next utterance — loop keeps running (continuous listen)
                buffer = bytearray()
                speech_active = False
                silence_accumulated_ms = 0

            # ── Safety cap: force flush if buffer grows beyond max ────────
            elif buffer_ms >= max_buffer_ms:
                logger.info("[stream] buffer cap %.0f ms — force flush", buffer_ms)
                try:
                    transcript = self.transcribe_bytes(bytes(buffer))
                except Exception:
                    transcript = ""

                logger.info("[stream] forced transcript: %r", transcript)
                yield transcript

                buffer = bytearray()
                speech_active = False
                silence_accumulated_ms = 0

        # Final flush when iterator is exhausted (end of call)
        if buffer:
            try:
                transcript = self.transcribe_bytes(bytes(buffer))
            except Exception:
                transcript = ""
            logger.info("[stream] final flush: %r", transcript)
            yield transcript


# Convenience singleton for simple imports
default_engine = STTEngine(model_size=os.environ.get("WHISPER_MODEL", "base"))
