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
import re
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import httpx
import noisereduce as nr
import numpy as np
import webrtcvad

from services.runtime_state import get_session

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# NEW — replace with these
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
STT_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_TIMEOUT_SECONDS = float(os.getenv("STT_TIMEOUT_SECONDS", "10.0"))
STT_CONNECT_TIMEOUT_SECONDS = float(os.getenv("STT_CONNECT_TIMEOUT_SECONDS", "5.0"))
STT_MODEL_HINT = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
MIN_AUDIO_MS = int(os.getenv("STT_MIN_AUDIO_MS", "600"))
MIN_RMS = int(os.getenv("STT_MIN_RMS", "450"))
VAD_RATIO_THRESHOLD = float(os.getenv("STT_VAD_RATIO_THRESHOLD", "0.55"))
NO_SPEECH_THRESHOLD = float(os.getenv("STT_NO_SPEECH_THRESHOLD", "0.65"))
LOG_PROB_THRESHOLD = float(os.getenv("STT_LOG_PROB_THRESHOLD", "-1.15"))
LANG_CONF_THRESHOLD = float(os.getenv("STT_LANG_CONF_THRESHOLD", "0.35"))
SESSION_STATE_TTL_SECONDS = float(os.getenv("STT_SESSION_STATE_TTL_SECONDS", "1800"))

LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "urdu": "ur",
    "tamil": "ta",
    "telugu": "te",
    "gujarati": "gu",
    "bengali": "bn",
    "kannada": "kn",
    "punjabi": "pa",
    "malayalam": "ml",
}

# Remap languages that sound identical to supported ones on phone calls
LANGUAGE_REMAP = {
    "ur": "hi",
}

# ── Self-detecting Language helpers ──────────────────────────────────────────

def _detect_language_from_script(text: str) -> str | None:
    for char in text:
        block = unicodedata.name(char, "").split()[0] if unicodedata.name(char, "") else ""
        if block == "DEVANAGARI":
            return "hi"
    return None  # couldn't determine from script


ROMAN_HINDI_KEYWORDS = {
    "mujhe", "bukhar", "hai", "nahi", "nahin", "nahee", "theek", "thik", "dard",
    "khansi", "khaansi", "khaasi", "jukham", "jukam", "sardi", "ultee", "ulti",
    "chakar", "chakkar", "kamzori", "kamjori", "pait", "bhai", "namaste", "pranam",
    "achha", "acha", "bimar", "beemar", "ilaj", "dawa", "dawai", "goli", "badan",
    "sar", "takleef", "taklif", "pareshan", "pareshani", "bahut", "bohot", "bhut",
    "aaram", "aaraam", "saans", "sans", "khujli", "dino", "dinon", "hafta", "hafte",
    "mahina", "mahine", "ghante", "ghanta", "mera", "meri", "mere", "aap", "tum",
    "hume", "humein", "aapka", "aapki", "aapke", "kya", "kab", "kaise", "kyun",
    "kyon", "kuch", "hoga", "hogi", "gaya", "gayi", "gaye", "hua", "hui", "hue",
    "tha", "thi", "the", "raha", "rahi", "rahe"
}


def _contains_roman_hindi(text: str) -> bool:
    if not text:
        return False
    # Extract only clean alphabetic words
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    for word in words:
        if word in ROMAN_HINDI_KEYWORDS:
            return True
    return False


HALLUCINATION_PATTERNS = re.compile(
    r"^(the next|thank you|thanks|subscribe|please|uh|um|"
    r"the date of|i don't know|okay|yes|no)(?:[,.\s]|$)",
    re.IGNORECASE
)


def _is_likely_hallucination(text: str, avg_logprob: float) -> bool:
    if avg_logprob < -1.0 and HALLUCINATION_PATTERNS.match(text.strip()):
        return True
    return False


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

_SHORT_ALLOWED_TRANSCRIPTS = {"yes", "no", "bye", "hi", "hello", "ok", "okay", "confirm", "हाँ", "नहीं", "ठीक", "हेलो", "नमस्ते"}

_WHITELIST_PHRASES = (
    "reschedule",
    "cancel appointment",
    "book appointment",
    "yes",
    "no",
    "okay",
    "confirm",
    "अपॉइंटमेंट",
    "बुक",
    "हाँ",
    "नहीं",
    "ठीक है",
    "डॉक्टर",
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
    "अपॉइंटमेंट",
    "बुकिंग",
    "डॉक्टर",
    "अस्पताल",
    "दर्द",
    "बुखार",
    "खांसी",
    "सिरदर्द",
    "बदन दर्द",
    "कल",
    "आज",
    "सुबह",
    "शाम",
    "दोपहर",
    "नाम",
    "नंबर",
    "हाँ",
    "नहीं",
    "ठीक",
    "मुझे",
    "चाहिए",
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


class SttResult(NamedTuple):
    transcript: str
    language: str


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


# NEW
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
        # No default Authorization header — we pass it per-request in _remote_transcribe()
        # so the same client can be reused safely
        _client = httpx.AsyncClient(timeout=timeout, limits=limits)
        logger.info("Groq Whisper STT AsyncClient initialized timeout=%.1fs", STT_TIMEOUT_SECONDS)
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

   # if rms >= MIN_RMS:
   #     try:
   #        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.45).astype(np.float32)
   #         audio = np.clip(audio, -1.0, 1.0)
   #         pcm = (audio * 32767.0).astype("<i2").tobytes()
   #         rms = audioop.rms(pcm, SAMPLE_WIDTH) if pcm else 0
   #     except Exception as exc:
   #         logger.debug("STT noise reduction skipped: %s", exc)

    # Compute VAD exactly once per utterance and reuse it to reduce latency
    vad_ratio = _voice_ratio(pcm)

    return PreparedAudio(
        wav_bytes=_pcm_to_wav(pcm),
        pcm=pcm,
        duration_ms=duration_ms,
        rms=rms,
        vad_ratio=vad_ratio,
        audio_hash=hashlib.sha1(pcm).hexdigest(),
    )


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
    text = (
        payload.get("transcribedText")
        or payload.get("text")
        or ""
    ).strip()
    if text:
        logger.info("STT extracted original transcript=%r", text)
        return text

    segments = payload.get("segments")
    if isinstance(segments, list):
        text = " ".join(str(segment.get("text", "")) for segment in segments if isinstance(segment, dict)).strip()

    logger.info("STT extracted original transcript=%r", text)
    return text


def _extract_float(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


# NEW
async def _remote_transcribe(prepared: PreparedAudio, session_id: str = None, force_language: str = None) -> dict:
    """
    Calls Groq Whisper API instead of remote Faster-Whisper server.
    Groq returns OpenAI-compatible JSON: { "text": "...", "language": "...", ... }
    We normalize it to match the shape the rest of stt.py already expects.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — cannot call Groq Whisper API")

    client = await _get_client()

    files = {"file": ("utterance.wav", prepared.wav_bytes, "audio/wav")}
    data = {
        "model": STT_MODEL_HINT,        # whisper-large-v3-turbo
        "temperature": "0",
        "response_format": "verbose_json",  # gives us language + segments
    }

    saved_language = None
    if session_id:
        try:
            import redis.asyncio as aioredis
            _r = aioredis.from_url("redis://localhost:6379/0")
            val = await _r.get(f"voice:{session_id}:language")
            saved_language = val.decode() if val else None
            await _r.aclose()
        except Exception:
            saved_language = None

    lang = force_language or saved_language or os.getenv("STT_LANGUAGE", "")
    if lang and lang != "en":
        data["language"] = lang

    if _INITIAL_PROMPT:
        prompt_value = _INITIAL_PROMPT
        # Final hard limit guard — never exceed 896 chars regardless of how prompt was built
        if len(prompt_value) > 890:
            prompt_value = prompt_value[:890]
            logger.warning("[STT] Prompt hard-truncated to 890 chars before Groq API call (original=%d)", len(_INITIAL_PROMPT))
        data["prompt"] = prompt_value

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    response = await client.post(STT_API_URL, files=files, data=data, headers=headers)
    if response.status_code != 200:
        logger.error("Groq 400 detail: %s", response.text)
    response.raise_for_status()
    raw = response.json()

    logger.info("Groq Whisper raw response keys=%s text=%r language=%r", list(raw.keys()), raw.get("text",""), raw.get("language",""))

    # Normalize Groq response → shape that _validate_remote_result() expects:
    # { "text": "...", "language_probability": 1.0, "segments": [...] }
    # Groq verbose_json already has "text" and "segments".
    # We map segment fields to the camelCase/snake_case keys _validate_remote_result reads.
    normalized_segments = []
    for seg in raw.get("segments") or []:
        normalized_segments.append({
            "text":           seg.get("text", ""),
            "no_speech_prob": seg.get("no_speech_prob", 0.0),
            "avg_logprob":    seg.get("avg_logprob", 0.0),
        })

    return {
        "text":                 raw.get("text", ""),
        # Groq doesn't return language_probability — default 1.0 (it's already filtered)
        "language_probability": 1.0,
        "language":             raw.get("language", ""),
        "segments":             normalized_segments,
        # top-level confidence proxies (average across segments if present)
        "no_speech_prob":       (
            sum(s.get("no_speech_prob", 0.0) for s in normalized_segments) / len(normalized_segments)
            if normalized_segments else 0.0
        ),
        "avg_logprob":          (
            sum(s.get("avg_logprob", 0.0) for s in normalized_segments) / len(normalized_segments)
            if normalized_segments else 0.0
        ),
    }


def _validate_remote_result(payload: dict[str, Any], prepared: PreparedAudio, state: SttSessionState) -> dict:
    text = _normalize_text(_extract_text(payload))
    logger.info("Extracted transcript=%r", text)

    detected_language_raw = (payload.get("language") or "").strip().lower()
    if len(detected_language_raw) <= 3:
        language_code = detected_language_raw or "en"
    else:
        language_code = LANGUAGE_NAME_TO_CODE.get(detected_language_raw, "en")
    language_code = LANGUAGE_REMAP.get(language_code, language_code)

    script_lang = _detect_language_from_script(text)
    if script_lang:
        language_code = script_lang  # trust script over Groq's label

    logger.info("STT detected language raw=%r mapped=%r", detected_language_raw, language_code)

    no_speech_prob = _extract_float(payload, "no_speech_prob", "noSpeechProb", default=0.0)
    avg_logprob = _extract_float(payload, "avg_logprob", "avgLogprob", default=0.0)
    lang_prob = _extract_float(payload, "language_probability", "languageProbability", default=1.0)
    confidence = _compute_confidence(payload)
    meaningful_words = _meaningful_word_count(text)
    vad = prepared.vad_ratio
    duration_ms = prepared.duration_ms
    forced_lang = os.getenv("STT_LANGUAGE", "")

    is_non_english = language_code != "en"

    if _segment_confidence_is_low(payload) and not is_non_english:
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
    elif avg_logprob and avg_logprob < LOG_PROB_THRESHOLD and not is_non_english:
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f avg_logprob=%.3f reason=low_confidence",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            avg_logprob,
        )
        text = ""
    elif avg_logprob and avg_logprob < -2.0 and is_non_english:
        logger.info(
            "STT rejected | rms=%d vad=%.2f duration_ms=%d confidence=%.3f avg_logprob=%.3f reason=very_low_confidence_non_en lang=%s text=%r",
            prepared.rms,
            vad,
            duration_ms,
            confidence,
            avg_logprob,
            language_code,
            text,
        )
        text = ""
    elif lang_prob < LANG_CONF_THRESHOLD and forced_lang == "en":
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
        "language": language_code,
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
) -> SttResult:
    result = await transcribe_with_meta(audio_bytes, session_id=session_id, last_transcript=last_transcript)
    return SttResult(
        transcript=result["transcript"],
        language=result.get("language", "en"),
    )


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
                payload = await _remote_transcribe(prepared, session_id=session_id)
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

        # Check for hallucination or Romanized Hindi retry within lock
        transcript = (result.get("transcript") or "").strip()
        avg_logprob = result.get("avg_logprob", 0.0)

        should_retry = False
        retry_reason = ""
        if _is_likely_hallucination(transcript, avg_logprob) and prepared.vad_ratio > 0.7:
            should_retry = True
            retry_reason = "hallucination"
        elif _contains_roman_hindi(transcript) and prepared.vad_ratio > 0.7:
            should_retry = True
            retry_reason = "Romanized Hindi"

        if should_retry:
            logger.info("STT %s detected, retrying with language=hi hint session=%s", retry_reason, session_id)
            try:
                payload = await _remote_transcribe(prepared, session_id=session_id, force_language="hi")
                result = _validate_remote_result(payload, prepared, state)
            except Exception as exc:
                logger.exception("STT retry failed session=%s: %s", session_id, exc)

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
        "language": result.get("language", "en"),
    }
