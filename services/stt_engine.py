import io
import os
import numpy as np
import torch
import soundfile as sf
import webrtcvad
import noisereduce as nr
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

import logging
import librosa

def _detect_device():
logger = logging.getLogger(__name__)
    return "cuda" if torch.cuda.is_available() else "cpu"


class STTEngine:
    """Lightweight streaming STT engine wrapper.

    - Uses WebRTC VAD for speech/silence detection (frame-level)
    - Applies a simple noise-reduction pass (noisereduce)
    - Runs faster-whisper on GPU when available (or int8 on CPU)
    - Uses conservative decoding parameters (temperature=0.0) to reduce hallucination
    """

    def __init__(self, model_size="base"):
        self.device = _detect_device()
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)

        # WebRTC VAD expects 10/20/30ms frames
        self.vad = webrtcvad.Vad(2)  # 0-3, higher -> more aggressive

        # chunk sizes in ms
        self.frame_ms = 30
        self.frame_bytes = int(SAMPLE_RATE * (self.frame_ms / 1000.0) * 2)  # 16-bit PCM

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            return audio.mean(axis=1)
        # Try reading common container formats (wav, mp3, m4a) via soundfile.
        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_np is None:
                raise RuntimeError("soundfile returned no data")
            # ensure mono
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)
            return audio_np.astype(np.float32)
        except Exception:
            # Fallback: assume raw PCM16 little-endian
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return arr
        return audio

    def noise_reduce(self, audio_np: np.ndarray) -> np.ndarray:
        try:
            return nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
        except Exception:
            return audio_np

    def is_speech_frame(self, frame_bytes: bytes) -> bool:
        try:
            return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def transcribe_bytes(self, audio_bytes: bytes, min_speech_ms: int = 200) -> str:
        """Synchronous transcription for a single audio blob (wav PCM16 bytes).

        This applies noise reduction + conservative faster-whisper decode.
        """
        # Convert to float32 mono
        audio = self._bytes_to_float32(audio_bytes)
        audio = self._ensure_mono(audio)

        # Convert to float32 mono and resample if needed
        audio = self.noise_reduce(audio)

        # faster-whisper expects float32 PCM in range [-1,1]
        segments, info = self.model.transcribe(
            audio,
            beam_size=3,
            word_timestamps=False,
            temperature=0.0,
            vad_filter=False,
        )

        text = "".join([s.text for s in segments]).strip()

        # Basic hallucination guard
        if len(text) == 0:
            return ""

        # reject excessively short outputs
        if len(text) < 2:
            return ""

        return text

    def stream_transcribe(self, chunk_iterator, max_buffer_ms: int = 2000):
        """Process incoming PCM16 bytes chunks (generator). Yields transcripts when
        VAD endpointing occurs or buffer time exceeds max_buffer_ms.

        chunk_iterator should yield raw PCM16 bytes (mono or stereo interleaved).
        """
        buffer = bytearray()
        speech_active = False
        last_speech_ts = 0

        for chunk in chunk_iterator:
            # accumulate
            buffer.extend(chunk)

            # process frames for VAD
            i = 0
            while i + self.frame_bytes <= len(buffer):
                frame = bytes(buffer[i:i + self.frame_bytes])
                is_speech = self.is_speech_frame(frame)
                if is_speech:
                    speech_active = True
                    last_speech_ts = 0
                else:
                    if speech_active:
                        last_speech_ts += self.frame_ms
                i += self.frame_bytes

    def _validate_transcript(self, text: str, audio_np: np.ndarray) -> bool:
        """Simple heuristics to catch likely hallucinations or invalid outputs.

        Returns True if transcript looks plausible for the given audio.
        """
        try:
            if audio_np is None or len(audio_np) == 0:
                return False

            duration_ms = (len(audio_np) / SAMPLE_RATE) * 1000.0
            words = (text or "").strip().split()
            word_count = len(words)

            # Minimum words vs audio duration
            if duration_ms > 1500 and word_count < 2:
                return False
            if duration_ms > 4000 and word_count < 4:
                return False

            # Reject extremely short token repeated patterns
            lowered = text.lower()
            tokens = [t for t in lowered.split() if t]
            if tokens:
                most_common = max(set(tokens), key=tokens.count)
                if tokens.count(most_common) / len(tokens) > 0.6 and len(tokens) > 3:
                    return False

            return True
        except Exception:
            return False
            # simple endpointing: if speech was active and silence > 300ms, finalize
            if speech_active and last_speech_ts > 300:
                # transcribe the buffered audio
                try:
                    transcript = self.transcribe_bytes(bytes(buffer))
                except Exception:
                    transcript = ""

                yield transcript

                # reset
                buffer = bytearray()
                speech_active = False
                last_speech_ts = 0

            # safety: if buffer grows too big, transcribe partial
            buffer_ms = (len(buffer) / 2) / SAMPLE_RATE * 1000.0
            if buffer_ms > max_buffer_ms:
                try:
                    transcript = self.transcribe_bytes(bytes(buffer))
                except Exception:
                    transcript = ""
                yield transcript
                buffer = bytearray()

        # final flush
        if len(buffer) > 0:
            try:
                transcript = self.transcribe_bytes(bytes(buffer))
            except Exception:
                transcript = ""
            yield transcript


# Convenience singleton for simple imports
default_engine = STTEngine(model_size=os.environ.get("WHISPER_MODEL", "base"))
