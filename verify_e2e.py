import asyncio, time, wave, io, os, sys, struct, math
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

# 2 seconds of 440Hz tone simulating speech
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    frames = []
    for i in range(16000 * 2):
        val = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / 16000))
        frames.append(struct.pack('<h', val))
    w.writeframes(b''.join(frames))
wav_bytes = buf.getvalue()

async def test():
    print('=' * 40)
    print('STT TEST (Groq Whisper)')
    print('=' * 40)
    from services import stt
    t0 = time.perf_counter()
    result = await stt.transcribe_with_meta(wav_bytes, session_id='test_e2e')
    stt_ms = (time.perf_counter() - t0) * 1000
    print(f'Latency    : {stt_ms:.0f}ms')
    safe_transcript = result["transcript"].encode("ascii", "backslashreplace").decode("ascii")
    print(f'Transcript : {safe_transcript}')
    if stt_ms < 2000:
        print('OK STT fast (Groq Whisper)')
    else:
        print('SLOW - check GROQ_API_KEY or network')

    print()
    print('=' * 40)
    print('TTS TEST (edge-tts)')
    print('=' * 40)
    from services.streaming_tts import get_streaming_tts_service
    svc = get_streaming_tts_service()
    t0 = time.perf_counter()
    chunks = []
    async for chunk in svc.stream_pcm(
        'Your appointment is confirmed for tomorrow at 10 AM with Dr. Sharma.',
        session_id='test_e2e'
    ):
        chunks.append(chunk)
    tts_ms = (time.perf_counter() - t0) * 1000
    total_pcm = sum(len(c) for c in chunks)
    duration_s = total_pcm / (16000 * 2)
    print(f'Latency      : {tts_ms:.0f}ms')
    print(f'PCM bytes    : {total_pcm}')
    print(f'Audio length : {duration_s:.2f}s')
    if tts_ms < 2000:
        print('OK TTS fast (edge-tts)')
    else:
        print('SLOW - check edge-tts/ffmpeg install')

    print()
    print('=' * 40)
    print('SUMMARY')
    print('=' * 40)
    print(f'STT : {stt_ms:.0f}ms')
    print(f'TTS : {tts_ms:.0f}ms')
    print(f'STT+TTS total : {stt_ms + tts_ms:.0f}ms')
    if stt_ms + tts_ms < 4000:
        print('OK All good - latency target achieved')
    else:
        print('WARNING - combined latency still high')

asyncio.run(test())