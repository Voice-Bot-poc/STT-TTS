import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TTS_COOLDOWN_SECONDS = 0.4
SESSION_TTL_SECONDS = 1800.0


@dataclass
class VoiceSessionState:
    tts_active: bool = False
    tts_cooldown_until: float = 0.0
    tts_generation: int = 0
    completed: bool = False
    last_seen: float = field(default_factory=time.monotonic)


_sessions: dict[str, VoiceSessionState] = {}


def get_session(session_id: str) -> VoiceSessionState:
    now = time.monotonic()
    stale = [sid for sid, state in _sessions.items() if now - state.last_seen > SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)

    sid = session_id or "default"
    state = _sessions.get(sid)
    if state is None:
        state = VoiceSessionState()
        _sessions[sid] = state
    state.last_seen = now
    return state


def mark_tts_start(session_id: str) -> None:
    state = get_session(session_id)
    state.tts_generation += 1
    state.tts_active = True
    state.last_seen = time.monotonic()
    logger.info(
        "[SessionState] TTS start session=%s generation=%d ts=%.3f",
        session_id,
        state.tts_generation,
        state.last_seen,
    )


def mark_tts_end(session_id: str) -> None:
    state = get_session(session_id)
    now = time.monotonic()
    state.tts_active = False
    state.tts_cooldown_until = now + TTS_COOLDOWN_SECONDS
    state.last_seen = now
    logger.info(
        "[SessionState] TTS end session=%s ts=%.3f cooldown_until=%.3f",
        session_id,
        now,
        state.tts_cooldown_until,
    )


def cancel_tts(session_id: str) -> None:
    state = get_session(session_id)
    state.tts_generation += 1
    state.tts_active = False
    state.tts_cooldown_until = 0.0
    state.last_seen = time.monotonic()
    logger.info(
        "[SessionState] TTS cancel session=%s generation=%d ts=%.3f",
        session_id,
        state.tts_generation,
        state.last_seen,
    )


def mark_completed(session_id: str) -> None:
    state = get_session(session_id)
    state.completed = True
    state.tts_active = False
    state.tts_cooldown_until = 0.0
    state.last_seen = time.monotonic()
    logger.info("[SessionState] completed session=%s", session_id)


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id or "default", None)

