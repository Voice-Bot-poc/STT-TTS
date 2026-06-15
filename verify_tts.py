import asyncio, os, io, time
from dotenv import load_dotenv
load_dotenv()

async def test():
    import edge_tts
    from pydub import AudioSegment

    voice = os.getenv('EDGE_TTS_EN_VOICE', 'en-US-AriaNeural')
    speed = os.getenv('EDGE_TTS_SPEED', '-5%')
    rate  = int(os.getenv('TTS_TARGET_SAMPLE_RATE', '16000'))

    print(f'Voice : {voice}')
    print(f'Speed : {speed}')
    print(f'Rate  : {rate}')
    print('Calling edge-tts...')

    t0 = time.perf_counter()
    communicate = edge_tts.Communicate(
        'Your appointment is confirmed for tomorrow at 10 AM.', voice, rate=speed
    )
    mp3_chunks = []
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            mp3_chunks.append(chunk['data'])
    mp3_bytes = b''.join(mp3_chunks)
    tts_ms = (time.perf_counter() - t0) * 1000

    print(f'edge-tts done   : {tts_ms:.0f}ms  mp3_bytes={len(mp3_bytes)}')
    if len(mp3_bytes) == 0:
        print('FAILED: edge-tts returned empty audio')
        return

    print('Converting MP3 to PCM...')
    t0 = time.perf_counter()
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
    seg = seg.set_frame_rate(rate).set_channels(1).set_sample_width(2)
    pcm = seg.raw_data
    conv_ms = (time.perf_counter() - t0) * 1000

    duration_s = len(pcm) / (rate * 2)
    print(f'MP3->PCM done   : {conv_ms:.0f}ms  pcm_bytes={len(pcm)}  audio_duration={duration_s:.2f}s')

    if len(pcm) > 0:
        print('OK edge-tts + pydub working correctly')
    else:
        print('FAILED: PCM conversion returned empty')

asyncio.run(test())