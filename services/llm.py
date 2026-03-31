"""
llm.py — LLM integration for the VoiceBot pipeline (Ollama-compatible).

Supports both Anthropic Claude and local Ollama models via OpenAI-compatible API.

Config env vars:
    LLM_PROVIDER      — "ollama" or "anthropic" (default: "ollama")
    LLM_BASE_URL      — Ollama base URL (default: "http://localhost:11434/v1")
    LLM_MODEL         — Model name (default: "llama3.1:8b")
    LLM_API_KEY       — API key (use "ollama" for local, or actual key for Anthropic)
    LLM_SYSTEM_PROMPT — Optional custom system prompt
"""

import logging
import os

# Use OpenAI client for Ollama compatibility
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama doesn't need real key

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep responses concise and spoken-word friendly. "
    "Avoid markdown, bullet points, or formatting — respond in plain, natural speech."
)

_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# Lazy-initialised async client
_async_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return the shared async OpenAI-compatible client."""
    global _async_client
    if _async_client is None:
        logger.info(
            "Initializing LLM client: provider=%s, model=%s, base_url=%s",
            _PROVIDER, _MODEL, _BASE_URL
        )
        _async_client = AsyncOpenAI(
            base_url=_BASE_URL,
            api_key=_API_KEY,  # Ollama accepts any string here
        )
    return _async_client


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------

def build_messages(
    history: list[dict],
    transcript: str,
) -> list[dict]:
    """
    Assemble the messages list from DB history + current user turn.

    Args:
        history:    List of dicts with keys 'user_text' and 'assistant_text',
                    ordered oldest → newest (as returned by db.fetch_history).
        transcript: The current user's transcribed speech.

    Returns:
        List of {'role': 'user'|'assistant', 'content': str} dicts.
    """
    messages: list[dict] = []

    for turn in history:
        messages.append({"role": "user", "content": turn["user_text"]})
        messages.append({"role": "assistant", "content": turn["assistant_text"]})

    # Current user turn (the freshly transcribed audio)
    messages.append({"role": "user", "content": transcript})

    return messages


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def call_llm(messages: list[dict]) -> str:
    """
    Call the LLM API and return the assistant's text reply.

    Args:
        messages: Fully assembled message list (from build_messages()).

    Returns:
        Plain text response from the model.

    Raises:
        RuntimeError: on API errors or connection failures.
    """
    client = _get_client()

    logger.info(
        "Calling LLM (provider=%s, model=%s) with %d messages …",
        _PROVIDER, _MODEL, len(messages)
    )

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                *messages
            ],
            max_tokens=512,
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        logger.info("LLM reply: %d chars", len(reply))
        return reply

    except Exception as e:
        logger.exception("LLM call failed")
        raise RuntimeError(f"LLM API error: {str(e)}")