import time, os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from services.streaming_tts import get_streaming_tts_service

svc = get_streaming_tts_service()

print('TTS TEST (direct sync call — matches how pipeline.py uses it)')
print('=' * 50)

t0 = time.perf_counter()
chunks = []

# This is exactly how pipeline.py calls it — sync, not async
for chunk in svc.stream_pcm_sync(
    'Your appointment is confirmed for tomorrow at 10 AM with Dr. Sharma.',
    session_id='test_direct'
):
    chunks.append(chunk)

tts_ms = (time.perf_counter() - t0) * 1000
total_pcm  = sum(len(c) for c in chunks)
duration_s = total_pcm / (16000 * 2)

print(f'Latency      : {tts_ms:.0f}ms')
print(f'PCM bytes    : {total_pcm}')
print(f'Audio length : {duration_s:.2f}s')

if tts_ms < 2000:
    print('OK TTS fast - matches expected production latency')
else:
    print(f'Still slow at {tts_ms:.0f}ms')