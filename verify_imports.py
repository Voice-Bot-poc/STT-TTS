import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

print('--- Importing services ---')

try:
    from services import stt
    print('OK stt imported')
except Exception as e:
    print(f'FAILED stt: {e}')

try:
    from services.streaming_tts import get_streaming_tts_service
    svc = get_streaming_tts_service()
    svc.validate_startup()
    print('OK streaming_tts imported and validated')
except Exception as e:
    print(f'FAILED streaming_tts: {e}')

try:
    from services import pipeline
    print('OK pipeline imported')
    from services.pipeline import _CLINICQUEUE_BASE_URL
    print(f'OK ClinicQueue URL = {_CLINICQUEUE_BASE_URL}')
except Exception as e:
    print(f'FAILED pipeline: {e}')

print()
print('--- Checking EdgeTTS is active (not Kokoro) ---')
try:
    from services.streaming_tts import EdgeTtsStreaming
    print('OK EdgeTtsStreaming class found')
except ImportError:
    print('FAILED EdgeTtsStreaming not found - streaming_tts.py not updated')

try:
    from services.streaming_tts import KokoroStreamingTts
    print('WARNING KokoroStreamingTts still present - old file may still be there')
except ImportError:
    print('OK KokoroStreamingTts removed correctly')