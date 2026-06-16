import logging
import threading
import wave
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

class GreetingService:
    def __init__(self):
        self._pcm_data: bytes = b""
        self._ready: bool = False
        self._lock = threading.RLock()
        self._greeting_path = Path("assets/greeting_en.wav")
        self._load_greeting()

    def _load_greeting(self) -> None:
        if not self._greeting_path.exists():
            logger.warning(f"Greeting WAV missing at {self._greeting_path}. Run generate_greeting.py to generate greeting WAV")
            return

        with self._lock:
            try:
                with wave.open(str(self._greeting_path), "rb") as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()

                    if channels != 1 or sampwidth != 2 or framerate != 16000:
                        raise ValueError(
                            f"Invalid greeting WAV format: {channels}ch, {sampwidth*8}-bit, {framerate}Hz. "
                            "Expected: 1ch, 16-bit, 16000Hz."
                        )
                    
                    self._pcm_data = wf.readframes(wf.getnframes())
                    self._ready = True
                    logger.info(f"✅ Greeting loaded into memory ({len(self._pcm_data)} bytes)")
            except Exception as e:
                logger.error(f"Failed to load greeting WAV: {e}")

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def stream_pcm(self, chunk_ms: int = 20) -> Iterator[bytes]:
        """
        Sync iterator yielding PCM chunks. Thread-safe because it just reads 
        from the immutable loaded bytes array.
        """
        with self._lock:
            if not self._ready:
                raise RuntimeError("Greeting audio is not loaded.")
            data = self._pcm_data

        # 16000 Hz * 2 bytes/sample * 1 channel * (chunk_ms / 1000)
        bytes_per_ms = 16000 * 2 // 1000
        chunk_size = chunk_ms * bytes_per_ms  # 20ms = 640 bytes

        offset = 0
        total_length = len(data)
        while offset < total_length:
            chunk = data[offset : offset + chunk_size]
            yield chunk
            offset += chunk_size
