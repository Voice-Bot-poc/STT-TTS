"""
test.py — Manual smoke tests for the TTS microservice.

Run with:
    python test.py

Make sure the service is running first:
    python main.py
"""

import base64
import json
import sys
import os
import requests

BASE_URL = "http://localhost:8000"
AUDIO_OUTPUT_DIR = "test_outputs"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
INFO = "\033[94m→\033[0m"

failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global failed
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        failed += 1


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def post_tts(payload: dict) -> requests.Response:
    return requests.post(f"{BASE_URL}/tts", json=payload, timeout=30)


def post_tts_stream(payload: dict) -> requests.Response:
    return requests.post(f"{BASE_URL}/tts/stream", json=payload, timeout=30)


def save_audio(filename: str, data: bytes):
    path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    print(f"  {INFO} Saved → {path}  ({len(data):,} bytes)")


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

section("1 / Health check")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("Status 200", r.status_code == 200)
    check("Body is {'status': 'ok'}", r.json() == {"status": "ok"})
except requests.exceptions.ConnectionError:
    print(f"  {FAIL}  Could not connect to {BASE_URL} — is the server running?")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Basic TTS — default voice, mp3
# ---------------------------------------------------------------------------

section("2 / Basic synthesis (default voice, mp3)")
payload = {"text": "Hello! This is a test of the text to speech microservice."}
r = post_tts(payload)

check("Status 200", r.status_code == 200, f"got {r.status_code}")

if r.status_code == 200:
    body = r.json()
    check("audio_base64 present", "audio_base64" in body)
    check("duration present", "duration" in body)
    check("duration > 0", body.get("duration", 0) > 0, f"got {body.get('duration')}")
    check("format is mp3", body.get("format") == "mp3")

    # Decode and save
    audio_bytes = base64.b64decode(body["audio_base64"])
    check("audio_base64 decodes to bytes", len(audio_bytes) > 0)
    save_audio("test_basic.mp3", audio_bytes)
    print(f"  {INFO} duration={body['duration']}s  chars={body.get('char_count')}")


# ---------------------------------------------------------------------------
# 3. Slow voice
# ---------------------------------------------------------------------------

section("3 / Slow voice")
payload = {"text": "This should sound noticeably slower.", "voice": "slow"}
r = post_tts(payload)

check("Status 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    body = r.json()
    check("audio_base64 present", "audio_base64" in body)
    audio_bytes = base64.b64decode(body["audio_base64"])
    save_audio("test_slow.mp3", audio_bytes)
    print(f"  {INFO} duration={body['duration']}s")


# ---------------------------------------------------------------------------
# 4. WAV format
# ---------------------------------------------------------------------------

section("4 / WAV format")
payload = {"text": "Testing WAV output format.", "format": "wav"}
r = post_tts(payload)

check("Status 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    body = r.json()
    audio_bytes = base64.b64decode(body["audio_base64"])
    # WAV files start with "RIFF" magic bytes (or fallback mp3 if pydub missing)
    is_wav = audio_bytes[:4] == b"RIFF"
    is_mp3_fallback = audio_bytes[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    check(
        "Returns wav (or mp3 fallback if pydub missing)",
        is_wav or is_mp3_fallback,
        f"first 4 bytes: {audio_bytes[:4]}",
    )
    ext = "wav" if is_wav else "mp3"
    save_audio(f"test_wav.{ext}", audio_bytes)


# ---------------------------------------------------------------------------
# 5. Raw stream endpoint
# ---------------------------------------------------------------------------

section("5 / Raw stream endpoint (/tts/stream)")
payload = {"text": "This comes back as raw audio bytes, not JSON."}
r = post_tts_stream(payload)

check("Status 200", r.status_code == 200, f"got {r.status_code}")
check(
    "Content-Type is audio/*",
    r.headers.get("content-type", "").startswith("audio/"),
    f"got {r.headers.get('content-type')}",
)
check("X-Audio-Duration header present", "x-audio-duration" in r.headers)
check("Body is non-empty bytes", len(r.content) > 0)
save_audio("test_stream.mp3", r.content)


# ---------------------------------------------------------------------------
# 6. Long text
# ---------------------------------------------------------------------------

section("6 / Long text")
long_text = (
    "The quick brown fox jumps over the lazy dog. " * 20
).strip()
payload = {"text": long_text}
r = post_tts(payload)

check("Status 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    body = r.json()
    audio_bytes = base64.b64decode(body["audio_base64"])
    check("Audio is larger than short clip", len(audio_bytes) > 5000, f"got {len(audio_bytes)} bytes")
    save_audio("test_long.mp3", audio_bytes)
    print(f"  {INFO} {len(long_text)} chars → {len(audio_bytes):,} bytes, duration={body['duration']}s")


# ---------------------------------------------------------------------------
# 7. Edge cases / validation
# ---------------------------------------------------------------------------

section("7 / Edge cases")

# Empty text
r = post_tts({"text": ""})
check("Empty text → 422", r.status_code == 422, f"got {r.status_code}")

# Whitespace-only text
r = post_tts({"text": "   "})
check("Whitespace-only text → 422 or 500", r.status_code in (422, 500), f"got {r.status_code}")

# Missing text field entirely
r = post_tts({})
check("Missing text field → 422", r.status_code == 422, f"got {r.status_code}")

# Unknown voice (should fall back gracefully or 422)
r = post_tts({"text": "Test", "voice": "robot_overlord"})
check("Unknown voice → 422", r.status_code == 422, f"got {r.status_code}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'═' * 55}")
if failed == 0:
    print(f"  \033[92mAll tests passed.\033[0m  Audio files saved to ./{AUDIO_OUTPUT_DIR}/")
else:
    print(f"  \033[91m{failed} test(s) failed.\033[0m")
print(f"{'═' * 55}\n")

sys.exit(0 if failed == 0 else 1)