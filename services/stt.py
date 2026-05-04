"""
stt.py — Whisper-based Speech-to-Text transcription module.

Uses: openai-whisper (the original OpenAI package — runs locally, no API key needed).
Configure via env var:  WHISPER_MODEL  (default: "base")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASYNC / SYNC design note
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Whisper is a *synchronous*, CPU/GPU-heavy operation.  If we call it
directly inside an `async def` FastAPI handler it would block the entire
event loop for the duration of transcription (often several seconds),
starving all other in-flight requests.

Solution: we offload the blocking work to a thread pool via
    asyncio.get_event_loop().run_in_executor(None, blocking_fn)
`None` means "use the default ThreadPoolExecutor".  This lets FastAPI's
event loop remain responsive while Whisper runs on a worker thread.

The temp-file lifecycle is:
  1. Write audio bytes → NamedTemporaryFile (delete=False so Windows can read it)
  2. Pass file path to Whisper (it opens the file internally)
  3. Delete temp file in a `finally` block — always cleaned up, even on error.
"""

import asyncio
import audioop
import io
import logging
import os
import tempfile
import threading
import wave
from functools import partial

import webrtcvad
import whisper  # openai-whisper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model — loaded once at module import time (expensive: ~500 MB for "base")
# ---------------------------------------------------------------------------

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.75"))
_MIN_AUDIO_BYTES = int(os.getenv("WHISPER_MIN_AUDIO_BYTES", "2000"))
_FAST_NO_SPEECH_SKIP_THRESHOLD = float(os.getenv("WHISPER_FAST_NO_SPEECH_SKIP_THRESHOLD", "0.60"))
_HALLUCINATION_PHRASES = (
    "thank you for watching",
    "subscribe",
    "www.",
    "iso screen",
)

logger.info("Loading Whisper model '%s' …", _MODEL_NAME)
import torch
_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Whisper loading on device: %s", _device)
_whisper_model = whisper.load_model(_MODEL_NAME, device=_device)
logger.info("Whisper model '%s' loaded.", _MODEL_NAME)
_WHISPER_MODEL_LOCK = threading.Lock()
_STT_QUEUE_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Internal sync helper (runs in thread pool)
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_bytes: bytes) -> str:
    """
    Blocking transcription helper.  Must NOT be called directly from async code.
    Use `transcribe()` instead, which schedules this on a thread pool.
    """
    tmp_path: str | None = None
    try:
        if not audio_bytes:
            return ""

        # Skip chunks that are too short for stable decode.
        if len(audio_bytes) < _MIN_AUDIO_BYTES:
            return ""

        # NamedTemporaryFile with delete=False is required on Windows because
        # Whisper opens the file by path; a file opened by Python cannot be
        # opened again by another process on Windows if delete=True.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="whisper_"
        ) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()  # ADD THIS LINE - ensure bytes are written to disk
            tmp_path = tmp.name

        logger.info("Transcribing %d bytes from %s …", len(audio_bytes), tmp_path)
        try:
            with _WHISPER_MODEL_LOCK:
                result = _whisper_model.transcribe(
                    tmp_path,
                    fp16=torch.cuda.is_available(),
                    language="en",
                    condition_on_previous_text=False,
                    initial_prompt=(
                        "Indian names: Raj, Ravi, Priya, Pooja, Amit, Rohit, Neha, Anjali, "
                        "Kavita, Ramesh, Suresh, Kuldeep. "
                        "Surnames: Sharma, Patel, Gupta, Singh, Verma, Joshi, Shah, Mehta, "
                        "Iyer, Nair, Reddy. "
                        "Medical: appointment, OPD, doctor, prescription, blood test, fever, "
                        "BP, diabetes."
                    ),
                )
        except (RuntimeError, KeyError) as exc:
            logger.warning("Whisper decode error suppressed; returning empty transcript: %s", exc)
            return ""
        transcript = result.get("text", "").strip()

        no_speech_prob = _extract_no_speech_prob(result)
        if no_speech_prob >= _NO_SPEECH_THRESHOLD:
            logger.info(
                "Transcription treated as silence (no_speech_prob=%.3f >= %.2f)",
                no_speech_prob,
                _NO_SPEECH_THRESHOLD,
            )
            return ""

        if _is_hallucination(transcript):
            logger.info("Transcription filtered as hallucination: %r", transcript)
            return ""

        if transcript:
            logger.info("Transcription complete: %d chars", len(transcript))
        return transcript

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def transcribe(audio_bytes: bytes) -> str:
    """
    Async entry point for STT.  Offloads blocking Whisper work to a thread.

    Args:
        audio_bytes: Raw audio bytes (any format Whisper supports: wav, mp3, m4a …)

    Returns:
        Transcribed text string (may be empty if audio is silent/inaudible).

    Raises:
        RuntimeError: if Whisper raises any exception during transcription.
    """
    if not audio_bytes:
        return ""

    if should_skip_audio(audio_bytes):
        return ""

    async with _STT_QUEUE_LOCK:
        loop = asyncio.get_running_loop()
        # partial() is not strictly needed for a simple call, but makes it explicit
        # that we're passing a bound callable with its argument to run_in_executor.
        transcript = await loop.run_in_executor(
            None,                              # default ThreadPoolExecutor
            partial(_transcribe_sync, audio_bytes),
        )
        return transcript


def should_skip_audio(audio_bytes: bytes) -> bool:
    if not audio_bytes or len(audio_bytes) < _MIN_AUDIO_BYTES:
        logger.info("SKIP: too short (%d bytes)", len(audio_bytes) if audio_bytes else 0)
        return True

    no_speech_prob = _estimate_no_speech_prob_fast(audio_bytes)
    if no_speech_prob >= _FAST_NO_SPEECH_SKIP_THRESHOLD:
        logger.info("SKIP: RMS gate (no_speech_prob=%.3f)", no_speech_prob)
        return True

    has_voice = _has_voice_webrtcvad(audio_bytes)
    if not has_voice:
        logger.info("SKIP: VAD gate rejected")
        return True

    logger.info("PASS: audio passed all gates")
    return False


def _has_voice_webrtcvad(audio_bytes: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            pcm = wav_file.readframes(wav_file.getnframes())

        logger.info("VAD check: rate=%d ch=%d width=%d pcm_bytes=%d", rate, channels, sampwidth, len(pcm))

        if channels != 1 or sampwidth != 2:
            return False
        if rate not in (8000, 16000, 32000, 48000):
            return False

        vad = webrtcvad.Vad(2)
        frame_duration_ms = 20
        frame_bytes = int(rate * (frame_duration_ms / 1000.0) * sampwidth)
        if frame_bytes <= 0 or len(pcm) < frame_bytes:
            return False

        speech_frames = 0
        total_frames = 0
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[i:i + frame_bytes]
            total_frames += 1
            if vad.is_speech(frame, rate):
                speech_frames += 1

        if total_frames == 0:
            return False

        return (speech_frames / total_frames) >= 0.10
    except (wave.Error, EOFError, ValueError):
        return False


def _estimate_no_speech_prob_fast(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(frame_count)

        logger.info(
            "Audio format: rate=%d ch=%d width=%d frames=%d rms=%d",
            rate,
            channels,
            sample_width,
            frame_count,
            audioop.rms(frames, sample_width) if frames else 0,
        )

        if not frames:
            return 1.0

        rms = audioop.rms(frames, sample_width)
    except (wave.Error, EOFError, audioop.error):
        # Fallback for malformed WAVs: treat as PCM16 payload.
        if len(audio_bytes) < 2:
            return 1.0
        try:
            rms = audioop.rms(audio_bytes, 2)
        except audioop.error:
            return 1.0

    # Map RMS to a rough no-speech probability curve; tuned for fast silence culling.
    # rms <= 0      -> 1.00 (silence)
    # rms around 350 -> ~0.50
    # rms >= 700    -> 0.00 (likely voiced)
    scaled = min(max(rms / 700.0, 0.0), 1.0)
    return 1.0 - scaled


def _extract_no_speech_prob(result: dict) -> float:
    segments = result.get("segments") or []
    if not isinstance(segments, list) or not segments:
        return 0.0

    probs: list[float] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        value = seg.get("no_speech_prob")
        if isinstance(value, (int, float)):
            probs.append(float(value))

    if not probs:
        return 0.0

    return max(probs)


def _is_hallucination(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False

    words = clean.split()
    if len(words) < 2:
        return False

    lowered = clean.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in lowered:
            return True

    return False
