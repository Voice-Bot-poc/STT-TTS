"""
llm.py — Anthropic Claude integration for the VoiceBot pipeline.

Model:  claude-haiku-4-5-20251001  (fast, cheap, voice-assistant-appropriate)
Config env vars:
    ANTHROPIC_API_KEY   — required, your Anthropic secret key
    LLM_SYSTEM_PROMPT   — optional, defaults to the voice-assistant prompt below

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASYNC / SYNC design note
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The official `anthropic` Python SDK (v0.25+) ships both a sync client
(anthropic.Anthropic) and an async client (anthropic.AsyncAnthropic).
We use the *async* client so the network call does NOT block the event
loop — no run_in_executor needed here.

Conversation history format (from DB):
    [ {"user_text": "...", "assistant_text": "..."}, ... ]  (oldest first)

Anthropic messages format:
    [ {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
      {"role": "user",      "content": "<current transcript>"}  ]
"""

import asyncio
import logging
import os
from functools import partial

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL = "claude-haiku-4-5-20251001"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep responses concise and spoken-word friendly. "
    "Avoid markdown, bullet points, or formatting — respond in plain, natural speech."
)

_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# Lazy-initialised async client
_async_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return the shared async Anthropic client (initialised on first call)."""
    global _async_client
    if _async_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it in your .env file or shell before starting the service."
            )
        _async_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _async_client


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------

def build_messages(
    history: list[dict],
    transcript: str,
) -> list[dict]:
    """
    Assemble the Anthropic `messages` list from DB history + current user turn.

    Args:
        history:    List of dicts with keys 'user_text' and 'assistant_text',
                    ordered oldest → newest (as returned by db.fetch_history).
        transcript: The current user's transcribed speech.

    Returns:
        List of {'role': 'user'|'assistant', 'content': str} dicts.
    """
    messages: list[dict] = []

    for turn in history:
        messages.append({"role": "user",      "content": turn["user_text"]})
        messages.append({"role": "assistant", "content": turn["assistant_text"]})

    # Current user turn (the freshly transcribed audio)
    messages.append({"role": "user", "content": transcript})

    return messages


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def call_llm(messages: list[dict]) -> str:
    """
    Call the Anthropic API and return the assistant's text reply.

    Args:
        messages: Fully assembled message list (from build_messages()).

    Returns:
        Plain text response from the model.

    Raises:
        RuntimeError: on API errors or missing API key.
    """
    client = _get_client()

    logger.info(
        "Calling LLM (model=%s) with %d messages …", _MODEL, len(messages)
    )

    response = await client.messages.create(
        model=_MODEL,
        max_tokens=512,          # keep TTS output manageable
        system=_SYSTEM_PROMPT,
        messages=messages,
    )

    # Extract the first text content block
    reply = response.content[0].text.strip()
    logger.info("LLM reply: %d chars", len(reply))
    return reply
