"""
Remote Faster-Whisper STT client for finalized conversational utterances.

The STT model runs on the remote Faster-Whisper service configured by
STT_API_URL. This module keeps transcription low-latency and production-safe by
only sending finalized utterances, rejecting stale/reused audio, applying local
telephony VAD/RMS gates, and filtering common hallucinations before ClinicQueue.
"""

import asyncio
import audioop
import hashlib
import io
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import httpx
import noisereduce as nr
import numpy as np
import webrtcvad

from services.runtime_state import get_session

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

STT_API_URL = os.getenv("STT_API_URL", "http://10.30.1.34:9000/transcribe")
_ENV_STT_TIMEOUT_SECONDS = float(os.getenv("STT_TIMEOUT_SECONDS", "5"))
STT_TIMEOUT_SECONDS = max(_ENV_STT_TIMEOUT_SECONDS, 5.0)
STT_CONNECT_TIMEOUT_SECONDS = float(os.getenv("STT_CONNECT_TIMEOUT_SECONDS", "5"))
STT_MODEL_HINT = os.getenv("REMOTE_WHISPER_MODEL", "distil-large-v3")
MIN_AUDIO_MS = int(os.getenv("STT_MIN_AUDIO_MS", "900"))
MIN_RMS = int(os.getenv("STT_MIN_RMS", "450"))
VAD_RATIO_THRESHOLD = float(os.getenv("STT_VAD_RATIO_THRESHOLD", "0.55"))
NO_SPEECH_THRESHOLD = float(os.getenv("STT_NO_SPEECH_THRESHOLD", "0.65"))
LOG_PROB_THRESHOLD = float(os.getenv("STT_LOG_PROB_THRESHOLD", "-1.15"))
LANG_CONF_THRESHOLD = float(os.getenv("STT_LANG_CONF_THRESHOLD", "0.35"))
SESSION_STATE_TTL_SECONDS = float(os.getenv("STT_SESSION_STATE_TTL_SECONDS", "1800"))

_HALLUCINATION_PHRASES = (
    "thank you very much",
    "thank you for watching",
    "thanks for watching",
    "phone mail",
    "over the years",
    "please subscribe",
    "subscribe",
    "www.",
    "http",
    "iso screen",
    "home rating",
    "home readiness",
    "music",
    "applause",
)

_SHORT_ALLOWED_TRANSCRIPTS = {"yes", "no", "bye", "hi", "hello", "ok", "okay", "confirm"}

_WHITELIST_PHRASES = (
    "reschedule",
    "cancel appointment",
    "book appointment",
    "yes",
    "no",
    "okay",
    "confirm",
)

_MEDICAL_BOOKING_TERMS = (
    "appointment",
    "book",
    "booking",
    "doctor",
    "clinic",
    "hospital",
    "patient",
    "date",
    "time",
    "today",
    "tomorrow",
    "morning",
    "afternoon",
    "evening",
    "pain",
    "fever",
    "cough",
    "headache",
    "body",
    "ache",
    "name",
    "phone",
    "number",
    "yes",
    "no",
    "bye",
    "goodbye",
    "hello",
    "thank",
)

_INCOMPLETE_SUFFIXES = (
    " with", " to", " and", " or", " the", " a", " an",
    "my name is", "i want to", "book with", "i'd like to",
    "i would like", "can you", "i need",
)

_STOPWORDS = {
    "i",
    "me",
    "my",
    "we",
    "you",
    "your",
    "to",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "with",
    "is",
    "am",
    "are",
    "be",
    "it",
    "this",
    "that",
    "please",
    "can",
    "could",
    "would",
    "want",
    "like",
    "need",
    "now",
}

_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_guard = asyncio.Lock()
_session_state: dict[str, "SttSessionState"] = {}
_client: httpx.AsyncClient | None = None


@dataclass
class SttSessionState:
    last_audio_hash: str = ""
    last_transcript: str = ""
    last_seen: float = field(default_factory=time.monotonic)
    recent_hashes: list[str] = field(default_factory=list)
    hallucination_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedAudio:
    wav_bytes: bytes
    pcm: bytes
    duration_ms: int
    rms: int
    vad_ratio: float
    audio_hash: str


def _load_stt_prompt() -> str:
    prompt_file = os.getenv("STT_PROMPT_FILE", "prompts/stt_prompt.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            logger.info("STT initial_prompt loaded from: %s (%d chars)", prompt_file, len(prompt))
            return prompt
    except FileNotFoundError:
        logger.warning("STT_PROMPT_FILE not found: %s; using empty prompt", prompt_file)
        return ""
    except Exception as exc:
        logger.warning("Failed to load STT prompt: %s; using empty prompt", exc)
        return ""


_INITIAL_PROMPT = _load_stt_prompt()

logger.info(
    "Remote STT config: url=%s model_hint=%s timeout=%.1fs min_ms=%d min_rms=%d vad_ratio=%.2f",
    STT_API_URL,
    STT_MODEL_HINT,
    STT_TIMEOUT_SECONDS,
    MIN_AUDIO_MS,
    MIN_RMS,
    VAD_RATIO_THRESHOLD,
)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(
            timeout=STT_TIMEOUT_SECONDS,
            connect=STT_CONNECT_TIMEOUT_SECONDS,
            read=STT_TIMEOUT_SECONDS,
            write=STT_TIMEOUT_SECONDS,
        )
        limits = httpx.Limits(max_connections=32, max_keepalive_connections=8, keepalive_expiry=30.0)
        _client = httpx.AsyncClient(timeout=timeout, limits=limits)
        logger.info("Remote STT persistent AsyncClient initialized keepalive=true")
    return _client


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    now = time.monotonic()
    async with _session_locks_guard:
        stale_session_ids = [
            sid
            for sid, state in _session_state.items()
            if now - state.last_seen > SESSION_STATE_TTL_SECONDS
        ]
        for sid in stale_session_ids:
            lock = _session_locks.get(sid)
            if lock is not None and lock.locked():
                continue
            _session_locks.pop(sid, None)
            _session_state.pop(sid, None)

        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
        return lock


def _get_session_state(session_id: str) -> SttSessionState:
    state = _session_state.get(session_id)
    if state is None:
        state = SttSessionState()
        _session_state[session_id] = state
    state.last_seen = time.monotonic()
    return state


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            pcm = wav.readframes(wav.getnframes())

        if width != SAMPLE_WIDTH:
            raise ValueError(f"unsupported sample width: {width}")

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, rate
    except Exception:
        audio = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        return audio, SAMPLE_RATE


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)

    duration = audio.size / float(source_rate)
    target_size = max(1, int(duration * target_rate))
    source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def _voice_ratio(pcm: bytes) -> float:
    if not pcm:
        return 0.0

    vad = webrtcvad.Vad(3)
    frame_ms = 20
    frame_bytes = SAMPLE_RATE * frame_ms // 1000 * SAMPLE_WIDTH
    total = 0
    speech = 0

    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        total += 1
        try:
            if vad.is_speech(frame, SAMPLE_RATE):
                speech += 1
        except Exception:
            pass

    return speech / total if total else 0.0


def _prepare_audio(audio_bytes: bytes) -> PreparedAudio:
    audio, rate = _decode_audio(audio_bytes)
    audio = _resample_linear(audio, rate, SAMPLE_RATE)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    duration_ms = int(audio.size * 1000 / SAMPLE_RATE)
    pcm = (audio * 32767.0).astype("<i2").tobytes()
    rms = audioop.rms(pcm, SAMPLE_WIDTH) if pcm else 0

    if rms >= MIN_RMS:
        try:
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.45).astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            pcm = (audio * 32767.0).astype("<i2").tobytes()
            rms = audioop.rms(pcm, SAMPLE_WIDTH) if pcm else 0
        except Exception as exc:
            logger.debug("STT noise reduction skipped: %s", exc)

    vad_ratio = _voice_ratio(pcm)

    return PreparedAudio(
        wav_bytes=_pcm_to_wav(pcm),
        pcm=pcm,
        duration_ms=duration_ms,
        rms=rms,
        vad_ratio=vad_ratio,
        audio_hash=hashlib.sha1(pcm).hexdigest(),
    )


def _has_speech(pcm: bytes) -> bool:
    ratio = _voice_ratio(pcm)
    logger.info("STT VAD ratio %.2f threshold=%.2f", ratio, VAD_RATIO_THRESHOLD)
    return ratio >= VAD_RATIO_THRESHOLD


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("[BLANK_AUDIO]", "").split()).strip(" ,")


def _ends_incomplete(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(lowered.endswith(suffix.strip()) for suffix in _INCOMPLETE_SUFFIXES)


def _has_medical_booking_intent(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(term in lowered for term in _MEDICAL_BOOKING_TERMS)


def _matches_whitelist(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(phrase in lowered for phrase in _WHITELIST_PHRASES)


def _meaningful_word_count(text: str) -> int:
    words = [word.strip(" ,.!?") for word in (text or "").lower().split()]
    return sum(1 for word in words if word and word not in _STOPWORDS)


def _segment_confidence_is_low(payload: dict[str, Any]) -> bool:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        return False

    low = 0
    checked = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        checked += 1
        no_speech = _extract_float(segment, "no_speech_prob", "noSpeechProb", default=0.0)
        avg_logprob = _extract_float(segment, "avg_logprob", "avgLogprob", default=0.0)
        if no_speech >= NO_SPEECH_THRESHOLD or (avg_logprob and avg_logprob < LOG_PROB_THRESHOLD):
            low += 1

    return checked > 0 and low / checked >= 0.5


def _compute_confidence(payload: dict[str, Any]) -> float:
    no_speech_prob = _extract_float(payload, "no_speech_prob", "noSpeechProb", default=0.0)
    avg_logprob = _extract_float(payload, "avg_logprob", "avgLogprob", default=-1.0)
    lang_prob = _extract_float(payload, "language_probability", "languageProbability", default=1.0)

    logprob_score = (avg_logprob + 3.0) / 3.0
    logprob_score = max(0.0, min(1.0, logprob_score))
    speech_score = max(0.0, min(1.0, 1.0 - no_speech_prob))
    lang_score = max(0.0, min(1.0, lang_prob))

    return round(0.4 * speech_score + 0.4 * logprob_score + 0.2 * lang_score, 3)


def _looks_like_hallucination(text: str, prepared: PreparedAudio, state: SttSessionState | None = None) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True

    if _matches_whitelist(lowered) or _has_medical_booking_intent(lowered):
        return False

    for phrase in _HALLUCINATION_PHRASES:
        if phrase not in lowered:
            continue
        if state is not None:
            state.hallucination_counts[phrase] = state.hallucination_counts.get(phrase, 0) + 1
            logger.info(
                "STT hallucination pattern matched phrase=%r count=%d text=%r",
                phrase,
                state.hallucination_counts[phrase],
                text,
            )
        return True

    words = lowered.split()
    if len(lowered) < 3 and lowered not in _SHORT_ALLOWED_TRANSCRIPTS:
        return True

    if len(words) >= 4:
        most_common_count = max(words.count(word) for word in set(words))
        if most_common_count / len(words) > 0.55:
            return True

    if prepared.duration_ms < 700 and len(words) > 5:
        return True

    if prepared.duration_ms < 1200 and len(text) > 80:
        return True

    if prepared.rms < MIN_RMS and len(words) > 3:
        return True

    return False


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("transcript", "full_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    segments = payload.get("segments")
    if isinstance(segments, list):
        return " ".join(str(segment.get("text", "")) for segment in segments if isinstance(segment, dict))

    return ""


def _extract_float(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


async def _remote_transcribe(prepared: PreparedAudio) -> dict:
    client = await _get_client()
    files = {"file": ("utterance.wav", prepared.wav_bytes, "audio/wav")}
    data = {
        "language": "en",
        "task": "transcribe",
        "model": STT_MODEL_HINT,
        "initial_prompt": _INITIAL_PROMPT,
        "condition_on_previous_text": "false",
        "temperature": "0",
        "vad_filter": "true",
        "no_speech_threshold": str(NO_SPEECH_THRESHOLD),
        "compression_ratio_threshold": "2.35",
        "log_prob_threshold": str(LOG_PROB_THRESHOLD),
    }

    response = await client.post(STT_API_URL, files=files, data=data)
    response.raise_for_status()
    payload = response.json()
    logger.info("Remote STT raw response keys=%s", list(payload.keys()) if isinstance(payload, dict) else type(payload))
    return payload if isinstance(payload, dict) else {}


def _validate_remote_result(payload: dict[str, Any], prepared: PreparedAudio, state: SttSessionState) -> dict:
    text = _normalize_text(_extract_text(payload))
    no_speech_prob = _extract_float(payload, "no_speech_prob", "noSpeechProb", default=0.0)
    avg_logprob = _extract_float(payload, "avg_logprob", "avgLogprob", default=0.0)
    lang_prob = _extract_float(payload, "language_probability", "languageProbability", default=1.0)
    confidence = _compute_confidence(payload)
    meaningful_words = _meaningful_word_count(text)
    vad = prepared.vad_ratio
    duration_ms = prepared.duration_ms

    if _segment_confidence_is_low(payload):
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f reason=low_segment_confidence text=%r",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            text,
        )
        text = ""
    elif no_speech_prob >= NO_SPEECH_THRESHOLD:
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f no_speech_prob=%.3f reason=no_speech",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            no_speech_prob,
        )
        text = ""
    elif avg_logprob and avg_logprob < LOG_PROB_THRESHOLD:
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f avg_logprob=%.3f reason=low_confidence",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            avg_logprob,
        )
        text = ""
    elif lang_prob < LANG_CONF_THRESHOLD:
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f language_probability=%.3f reason=low_language_confidence",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            lang_prob,
        )
        text = ""
    elif (
        duration_ms >= 700
        and vad >= 0.8
        and meaningful_words >= 2
    ):
        pass
    elif _looks_like_hallucination(text, prepared, state):
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f reason=hallucination text=%r",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            text,
        )
        text = ""
    else:
        logger.info(
            "STT accepted | rms=%d vad=%.2f duration_ms=%d confidence=%.3f words=%d text=%r",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            meaningful_words,
            text,
        )

    return {
        "transcript": text,
        "incomplete": _ends_incomplete(text),
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
        "duration_ms": prepared.duration_ms,
        "rms": prepared.rms,
        "confidence": confidence,
    }


def should_skip_audio(audio_bytes: bytes, last_transcript: str = "") -> bool:
    if not audio_bytes:
        logger.info("STT skip empty audio")
        return True

    prepared = _prepare_audio(audio_bytes)
    if prepared.duration_ms < MIN_AUDIO_MS and not (
        prepared.duration_ms >= 700 and prepared.vad_ratio >= 0.8
    ):
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d reason=short_audio",
            prepared.rms,
            prepared.vad_ratio,
            prepared.duration_ms,
        )
        return True

    if prepared.rms < MIN_RMS:
        logger.info(
            "STT rejected | rms=%d vad=%.2f threshold=%d reason=low_rms",
            prepared.rms,
            prepared.vad_ratio,
            MIN_RMS,
        )
        return True

    if prepared.vad_ratio < VAD_RATIO_THRESHOLD:
        logger.info(
            "STT rejected | rms=%d vad=%.2f threshold=%.2f reason=low_voice_ratio",
            prepared.rms,
            prepared.vad_ratio,
            VAD_RATIO_THRESHOLD,
        )
        return True

    return False


async def transcribe(
    audio_bytes: bytes,
    session_id: str = "default",
    last_transcript: str = "",
) -> str:
    result = await transcribe_with_meta(audio_bytes, session_id=session_id, last_transcript=last_transcript)
    return result["transcript"]


async def transcribe_with_meta(
    audio_bytes: bytes,
    session_id: str = "default",
    last_transcript: str = "",
) -> dict:
    if not audio_bytes:
        return {"transcript": "", "incomplete": False}

    prepared = _prepare_audio(audio_bytes)
    state = _get_session_state(session_id)
    runtime_session = get_session(session_id)
    now = time.monotonic()

    if runtime_session.completed:
        logger.info("STT rejected | session=%s reason=session_completed", session_id)
        return {"transcript": "", "incomplete": False}

    if runtime_session.tts_active:
        logger.info(
            "Dropping STT during active TTS playback session=%s rms=%d vad=%.2f",
            session_id,
            prepared.rms,
            prepared.vad_ratio,
        )
        return {"transcript": "", "incomplete": False}

    if now < runtime_session.tts_cooldown_until:
        logger.info(
            "STT rejected | session=%s rms=%d vad=%.2f reason=tts_cooldown remaining_ms=%.0f",
            session_id,
            prepared.rms,
            prepared.vad_ratio,
            (runtime_session.tts_cooldown_until - now) * 1000.0,
        )
        return {"transcript": "", "incomplete": False}

    if prepared.audio_hash == state.last_audio_hash or prepared.audio_hash in state.recent_hashes:
        logger.info("STT rejected | session=%s hash=%s reason=stale_audio", session_id, prepared.audio_hash[:10])
        return {"transcript": "", "incomplete": False}

    if prepared.duration_ms < MIN_AUDIO_MS and not (
        prepared.duration_ms >= 700 and prepared.vad_ratio >= 0.8
    ):
        logger.info(
            "STT rejected | session=%s rms=%d vad=%.2f duration_ms=%d reason=short_audio",
            session_id,
            prepared.rms,
            prepared.vad_ratio,
            prepared.duration_ms,
        )
        state.recent_hashes.append(prepared.audio_hash)
        state.recent_hashes = state.recent_hashes[-8:]
        return {"transcript": "", "incomplete": False}

    if prepared.rms < MIN_RMS:
        logger.info(
            "STT rejected | session=%s rms=%d vad=%.2f threshold=%d reason=low_rms",
            session_id,
            prepared.rms,
            prepared.vad_ratio,
            MIN_RMS,
        )
        state.recent_hashes.append(prepared.audio_hash)
        state.recent_hashes = state.recent_hashes[-8:]
        return {"transcript": "", "incomplete": False}

    if prepared.vad_ratio < VAD_RATIO_THRESHOLD:
        logger.info(
            "STT rejected | session=%s rms=%d vad=%.2f threshold=%.2f reason=low_voice_ratio",
            session_id,
            prepared.rms,
            prepared.vad_ratio,
            VAD_RATIO_THRESHOLD,
        )
        state.recent_hashes.append(prepared.audio_hash)
        state.recent_hashes = state.recent_hashes[-8:]
        return {"transcript": "", "incomplete": False}

    lock = await _get_session_lock(session_id)
    async with lock:
        for attempt in range(2):
            try:
                payload = await _remote_transcribe(prepared)
                result = _validate_remote_result(payload, prepared, state)
                break
            except asyncio.CancelledError:
                logger.info("Remote STT cancelled session=%s", session_id)
                raise
            except httpx.TimeoutException:
                if attempt == 0:
                    logger.warning("Remote STT timed out session=%s retrying_once=true", session_id)
                    continue
                logger.warning("Remote STT timed out session=%s retrying_once=false", session_id)
                return {"transcript": "", "incomplete": False}
            except httpx.HTTPError as exc:
                logger.warning("Remote STT request failed session=%s: %s", session_id, exc)
                return {"transcript": "", "incomplete": False}
            except Exception as exc:
                logger.exception("Remote STT failed session=%s: %s", session_id, exc)
                return {"transcript": "", "incomplete": False}
        else:
            return {"transcript": "", "incomplete": False}

    transcript = (result.get("transcript") or "").strip()
    state.last_audio_hash = prepared.audio_hash
    state.last_seen = time.monotonic()
    state.recent_hashes.append(prepared.audio_hash)
    state.recent_hashes = state.recent_hashes[-8:]

    if transcript:
        state.last_transcript = transcript
        logger.info(
            "Remote STT transcript session=%s duration_ms=%d rms=%d vad=%.2f confidence=%.3f chars=%d text=%r",
            session_id,
            result.get("duration_ms", prepared.duration_ms),
            result.get("rms", prepared.rms),
            prepared.vad_ratio,
            result.get("confidence", 0.0),
            len(transcript),
            transcript,
        )

    return {
        "transcript": transcript,
        "incomplete": bool(result.get("incomplete", False)),
        "duration_ms": result.get("duration_ms", prepared.duration_ms),
        "rms": result.get("rms", prepared.rms),
        "vad": prepared.vad_ratio,
        "confidence": result.get("confidence", 0.0),
    }
