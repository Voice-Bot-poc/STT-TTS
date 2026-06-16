"""
services/filler_service.py
──────────────────────────
Two-phase filler audio service.

Phase 1 — context-aware spoken instruction (keyword → WAV mapping).
Phase 2 — gentle hold tone that loops until the real LLM response is ready.

All WAV files are pre-loaded into memory at startup so the /filler endpoint
adds near-zero latency.
"""

import logging
import os
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

FILLER_ENABLED: bool = os.getenv("FILLER_ENABLED", "true").lower() in ("true", "1", "yes")
FILLER_AUDIO_DIR: Path = Path(os.getenv("FILLER_AUDIO_DIR", "audio/fillers"))
SAMPLE_RATE: int = int(os.getenv("TTS_TARGET_SAMPLE_RATE", "16000"))

# ── In-memory cache ────────────────────────────────────────────────────────
# Maps filename → raw WAV bytes (complete file including header)
_cache: Dict[str, bytes] = {}
# Maps filename → duration in milliseconds
_durations: Dict[str, int] = {}

# ── Keyword → filler mapping ───────────────────────────────────────────────
# Order matters: first match wins.  Each entry is (keywords_set, wav_filename).
ENGLISH_FILLER_MAP: list[Tuple[set[str], str]] = [
    ({"book", "appointment"},           "please_wait.wav"),
    ({"slot", "slots", "available"},    "checking_slots.wav"),
    ({"reschedule"},                    "looking_up.wav"),
    ({"confirm"},                       "booking_now.wav"),
    ({"yes", "ok", "okay", "name"},    "one_moment.wav"),
]

# Unicode escapes are used for Hindi words to ensure reliable file charset detection:
# "\u092c\u0941\u0915" = "बुक"
# "\u0905\u092a\u0949\u0907\u0902\u091f\u092e\u0947\u0902\u091f" = "अपॉइंटमेंट"
# "\u0938\u094d\u0932\u0949\u091f" = "स्लॉट"
# "\u0909\u092a\u0932\u092c\u094d\u0927" = "उपलब्ध"
# "\u0930\u093f\u0936\u0947\u0921\u094d\u092f\u0942\u0932" = "रिशेड्यूल"
# "\u0915\u0928\u094d\u092b\u0930\u094d\u092e" = "कन्फर्म"
# "\u0939\u093e\u0901" = "हाँ"
# "\u0920\u0940\u0915" = "ठीक"
# "\u0928\u093e\u092e" = "नाम"
HINDI_FILLER_MAP: list[Tuple[set[str], str]] = [
    ({"\u092c\u0941\u0915", "\u0905\u092a\u0949\u0907\u0902\u091f\u092e\u0947\u0902\u091f", "appointment", "book"}, "hi_please_wait.wav"),
    ({"\u0938\u094d\u0932\u0949\u091f", "\u0909\u092a\u0932\u092c\u094d\u0927", "slot", "slots", "available"}, "hi_checking.wav"),
    ({"\u0930\u093f\u0936\u0947\u0921\u094d\u092f\u0942\u0932", "reschedule"}, "hi_looking_up.wav"),
    ({"\u0915\u0928\u094d\u092b\u0930\u094d\u092e", "confirm"}, "hi_processing.wav"),
    ({"\u0939\u093e\u0901", "\u0920\u0940\u0915", "\u0928\u093e\u092e", "yes", "ok", "okay", "name"}, "hi_one_moment.wav"),
]

HINDI_FILLER_FILES: list[str] = [
    "hi_please_wait.wav",
    "hi_one_moment.wav",
    "hi_hold_on.wav",
    "hi_checking.wav",
    "hi_processing.wav",
    "hi_looking_up.wav",
    "hi_stay_on_line.wav",
]

_HOLD_TUNE = "hold.wav"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _read_wav(path: Path) -> Tuple[bytes, int]:
    """Read a WAV file and return (raw_file_bytes, duration_ms)."""
    raw = path.read_bytes()
    with wave.open(str(path), "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration_ms = int((frames / rate) * 1000)
    return raw, duration_ms


def _match_instruction(transcript: str, language: str = "en") -> str:
    """Return the best-matching instruction WAV filename for *transcript*."""
    words = set(transcript.lower().split())
    filler_map = HINDI_FILLER_MAP if language == "hi" else ENGLISH_FILLER_MAP
    default_filler = "hi_one_moment.wav" if language == "hi" else "one_moment.wav"
    for keywords, filename in filler_map:
        if words & keywords:  # set intersection — any keyword present
            return filename
    return default_filler


# ── Public API ──────────────────────────────────────────────────────────────

def preload_fillers() -> None:
    """
    Load every WAV in FILLER_AUDIO_DIR into the in-memory cache.

    Call once at application startup.  If FILLER_ENABLED is False the cache
    stays empty and every later call to get_filler_packet_wav returns blanks.
    """
    if not FILLER_ENABLED:
        logger.info("[FILLER] Filler system disabled via FILLER_ENABLED=false")
        return

    if not FILLER_AUDIO_DIR.is_dir():
        logger.warning("[FILLER] Audio directory not found: %s", FILLER_AUDIO_DIR)
        return

    for wav_path in sorted(FILLER_AUDIO_DIR.glob("*.wav")):
        try:
            raw, dur = _read_wav(wav_path)
            _cache[wav_path.name] = raw
            _durations[wav_path.name] = dur
            logger.info(
                "[FILLER] Cached %-25s  %6d bytes  %4d ms",
                wav_path.name, len(raw), dur,
            )
        except Exception:
            logger.exception("[FILLER] Failed to load %s", wav_path)

    # Preload and verify Hindi files explicitly
    for filename in HINDI_FILLER_FILES:
        path = FILLER_AUDIO_DIR / filename
        if path.exists():
            if filename not in _cache:
                try:
                    raw, dur = _read_wav(path)
                    _cache[filename] = raw
                    _durations[filename] = dur
                    logger.info("[FILLER] Cached %s  %d bytes", filename, len(raw))
                except Exception:
                    logger.exception("[FILLER] Failed to load %s", path)
            else:
                logger.info("[FILLER] Cached %s  %d bytes", filename, len(_cache[filename]))
        else:
            logger.warning("[FILLER] Hindi filler file not found: %s — run create_fillers.py", filename)

    logger.info("[FILLER] Preloaded %d filler files", len(_cache))


def get_filler_packet_wav(
    transcript: str,
    language: str = "en",
) -> Dict[str, Optional[bytes | int]]:
    """
    Return a dict with the two-phase filler audio for *transcript*.

    Keys
    ----
    instruction_wav : bytes | None
        Complete WAV file bytes for the spoken instruction.
    instruction_ms  : int
        Duration of the instruction clip in milliseconds.
    tune_wav        : bytes | None
        Complete WAV file bytes for the hold tone.
    tune_ms         : int
        Duration of the hold tune clip in milliseconds.
    """
    empty: Dict[str, Optional[bytes | int]] = {
        "instruction_wav": None,
        "instruction_ms": 0,
        "tune_wav": None,
        "tune_ms": 0,
    }

    if not FILLER_ENABLED or not transcript or not _cache:
        return empty

    # Phase 1 — keyword-matched spoken instruction
    instr_name = _match_instruction(transcript, language)
    logger.info(
        f"[FILLER_SELECTED] "
        f"language={language} "
        f"file={instr_name}"
    )
    instr_wav = _cache.get(instr_name)
    instr_ms = _durations.get(instr_name, 0)

    # Phase 2 — hold tune
    tune_wav = _cache.get(_HOLD_TUNE)
    tune_ms = _durations.get(_HOLD_TUNE, 0)

    return {
        "instruction_wav": instr_wav,
        "instruction_ms": instr_ms,
        "tune_wav": tune_wav,
        "tune_ms": tune_ms,
    }
