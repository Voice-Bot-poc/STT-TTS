import asyncio, os, wave, io, httpx
from dotenv import load_dotenv
load_dotenv()

# 1 second of silence as test audio
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b'\x00' * 16000 * 2)
wav_bytes = buf.getvalue()

async def test():
    key   = os.getenv('GROQ_API_KEY')
    model = os.getenv('GROQ_WHISPER_MODEL', 'whisper-large-v3-turbo')
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            files={'file': ('test.wav', wav_bytes, 'audio/wav')},
            data={'model': model, 'response_format': 'verbose_json', 'temperature': '0'},
            headers={'Authorization': f'Bearer {key}'},
        )
        print(f'Status: {resp.status_code}')
        if resp.status_code == 200:
            print('OK Groq STT API working')
            print(f'   Response keys: {list(resp.json().keys())}')
        else:
            print(f'FAILED: {resp.text[:200]}')

asyncio.run(test())