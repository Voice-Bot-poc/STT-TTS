"""
create_fillers.py
─────────────────
Generates filler audio WAV files for the STT-TTS voice bot.

All outputs are 16 kHz, mono, 16-bit PCM — matching TTS_TARGET_SAMPLE_RATE
from .env.  Uses edge_tts for speech and raw sine-wave synthesis for the
hold tone.

Usage:
    python create_fillers.py
"""

import asyncio
import math
import os
import struct
import wave
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment

# ── Load environment ────────────────────────────────────────────────────────
load_dotenv()

VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-US-AriaNeural")
SAMPLE_RATE = int(os.getenv("TTS_TARGET_SAMPLE_RATE", "16000"))
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

OUTPUT_DIR = Path("audio") / "fillers"

# ── Filler definitions ─────────────────────────────────────────────────────
# (filename, text)  — text=None means programmatic generation
FILLERS: list[tuple[str, str | None]] = [
    ("please_wait.wav",   "Please wait a moment while I pull up your information. It will just take a second..."),
    ("checking_slots.wav", "Let me take a quick look at the available slots for you. Hold on please..."),
    ("one_moment.wav",    "One moment please, I am pulling up those details for you right now..."),
    ("looking_up.wav",    "Let me look that up for you, please hold on for just a brief moment..."),
    ("booking_now.wav",   "Please hold on while I securely confirm your appointment in our system..."),
    ("hold.wav",          None),  # generated hold tone
]

HINDI_VOICE = "hi-IN-SwaraNeural"
HINDI_FILLERS = {
    "hi_please_wait.wav": "बस एक पल रुकिए, मैं आपकी जानकारी देख रही हूँ।",
    "hi_one_moment.wav": "हाँ जी, एक पल, मैं आपकी मदद कर रही हूँ।",
    "hi_hold_on.wav": "जी, थोड़ा रुकिए।",
    "hi_checking.wav": "हाँ जी, मैं अभी देख रही हूँ।",
    "hi_processing.wav": "आपका काम हो रहा है, बस एक पल।",
    "hi_looking_up.wav": "एक पल, मैं जानकारी देख रही हूँ।",
    "hi_stay_on_line.wav": "जी, लाइन पर रहिए, मैं अभी आपकी मदद कर रही हूँ।",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

async def synthesize_speech(text: str, output_path: Path, voice: str = VOICE) -> None:
    """Use edge_tts to synthesize *text* → MP3, then convert to WAV."""
    mp3_path = output_path.with_suffix(".mp3")

    communicate = edge_tts.Communicate(text, voice, rate="+50%")
    await communicate.save(str(mp3_path))

    # Convert MP3 → WAV (16 kHz, mono, 16-bit PCM)
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = (
        audio
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(SAMPLE_WIDTH)
    )
    audio.export(str(output_path), format="wav")

    # Clean up intermediate MP3
    mp3_path.unlink(missing_ok=True)


def generate_hold_tone(output_path: Path, duration_s: float = 15.0) -> None:
    """
    Generate a gentle G4-B4-D5 major chord (sine waves) as a hold tone.

    Frequencies:
        G4 = 392.00 Hz
        B4 = 493.88 Hz
        D5 = 587.33 Hz
    """
    freqs = [392.00, 493.88, 587.33]
    n_samples = int(SAMPLE_RATE * duration_s)
    amplitude = 0.18  # keep it gentle

    samples: list[int] = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Fade-in / fade-out envelope (0.3 s each)
        fade = min(t / 0.3, 1.0) * min((duration_s - t) / 0.3, 1.0)
        value = sum(math.sin(2.0 * math.pi * f * t) for f in freqs) / len(freqs)
        sample = int(value * amplitude * fade * 32767)
        sample = max(-32768, min(32767, sample))
        samples.append(sample)

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def wav_duration(path: Path) -> float:
    """Return the duration of a WAV file in seconds."""
    with wave.open(str(path), "r") as wf:
        return wf.getnframes() / wf.getframerate()


# ── Main ────────────────────────────────────────────────────────────────────

async def create_english_fillers() -> None:
    print(f"English Voice : {VOICE}")
    for filename, text in FILLERS:
        out = OUTPUT_DIR / filename

        if text is None:
            # Programmatic hold tone
            generate_hold_tone(out)
        else:
            await synthesize_speech(text, out, voice=VOICE)

        dur = wav_duration(out)
        print(f"  [OK] {out}  ({dur:.2f}s)")


async def create_hindi_fillers() -> None:
    # Delete existing Hindi filler files before regenerating
    print("Deleting existing Hindi filler files...")
    for filename in HINDI_FILLERS:
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  Deleted: {filename}")
        else:
            print(f"  Not found (skip): {filename}")
    print("Done deleting. Regenerating now...")

    print(f"Hindi Voice : {HINDI_VOICE}")
    for filename, text in HINDI_FILLERS.items():
        out = OUTPUT_DIR / filename
        await synthesize_speech(text, out, voice=HINDI_VOICE)

        dur = wav_duration(out)
        print(f"  [OK] {out}  ({dur:.2f}s)")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Sample: {SAMPLE_RATE} Hz, {CHANNELS}ch, {SAMPLE_WIDTH * 8}-bit PCM")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    print("--- Generating English Fillers ---")
    await create_english_fillers()

    print("\n--- Generating Hindi Fillers ---")
    await create_hindi_fillers()

    print("\nAll filler audio files created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
