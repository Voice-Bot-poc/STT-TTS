# VoiceBot STT-TTS Microservice

This service provides the speech pipeline used by the VoiceBot stack. It exposes
HTTP endpoints for:

- Text-to-speech (JSON/base64, raw audio file, and streaming PCM).
- End-to-end processing of an audio clip through STT -> ClinicQueue -> TTS.
- A text-only chat endpoint for debugging the ClinicQueue integration.

It is designed to be called by the .NET orchestrator (Voicebot-orchestrator-backend),
but can be run and tested standalone.

## High-level flow

The full pipeline is triggered by POST /process.

1) STT: audio bytes are sent to a remote Faster-Whisper STT service with
	 local VAD/RMS gating and hallucination filtering.
2) ClinicQueue: the transcript is sent to ClinicQueue /api/voice/chat.
3) TTS: the ClinicQueue reply is synthesized with Kokoro TTS and returned
	 as base64 WAV.

If STT is slow, the pipeline returns a short response with response_text=Wait
so the caller can keep the call alive without blocking.

## Repository layout

- main.py
	FastAPI app, HTTP endpoints, app lifecycle.
- services/pipeline.py
	Orchestrates STT -> ClinicQueue -> TTS, session locking, goodbye handling.
- services/stt.py
	Remote STT client with audio prep, VAD, and rejection rules.
- services/stt_backend.py
	Minimal STT-only FastAPI app that wraps services/stt_engine.py.
- services/streaming_tts.py
	Kokoro streaming PCM engine, language detection, chunking.
- services/tts_service.py
	File-based TTS facade for /tts and /tts/stream.
- services/tts_normalizer.py
	Normalizes dates/times/numbers for spoken output.
- services/runtime_state.py
	In-memory session state for TTS suppression and cooldowns.
- services/db.py
	Async MySQL helper for conversation history (currently not used by pipeline.py).
- services/llm.py
	Deprecated, kept for reference. ClinicQueue now handles LLM calls.
- models/models.py
	Pydantic response shapes for /process.

## HTTP endpoints

### GET /health
Liveness probe.

### POST /tts
Text to audio. Returns JSON with base64 audio.

Example:
```bash
curl -X POST http://localhost:8000/tts \
	-H "Content-Type: application/json" \
	-d '{"text":"Hello","voice":"default","format":"wav"}'
```

### POST /tts/stream
Same as /tts, but returns the raw audio file (audio/mpeg, audio/wav, audio/ogg).

### POST /tts/pcm-stream
Returns streaming PCM16 mono audio at the target sample rate for real-time
playback (used by WebRTC in the .NET orchestrator).

Example:
```bash
curl -X POST http://localhost:8000/tts/pcm-stream \
	-H "Content-Type: application/json" \
	-d '{"text":"Hello","chunk_ms":40,"session_id":"demo"}'
```

### POST /process
Multipart form endpoint that runs STT -> ClinicQueue -> TTS.

Fields:
- audio_file (file, required)
- session_id (string, required)
- phone_number (string, required for ClinicQueue)
- synthesize_tts (bool, optional, default true)

Example:
```bash
curl -X POST http://localhost:8000/process \
	-F "audio_file=@sample.wav" \
	-F "session_id=session-123" \
	-F "phone_number=919876543210" \
	-F "synthesize_tts=true"
```

### POST /chat
Text-only path that calls ClinicQueue directly. Useful for testing without audio.

Example:
```bash
curl -X POST http://localhost:8000/chat \
	-H "Content-Type: application/json" \
	-d '{"text":"I want to book appointment","session_id":"s1","phone_number":"919876543210"}'
```

### POST /synthesise
Legacy endpoint for a single base64 WAV reply (used for call greetings).

## Key runtime behavior

- Session locking in services/pipeline.py prevents concurrent requests for the
	same session_id from overlapping.
- services/runtime_state.py tracks TTS activity. STT is suppressed during
	active TTS playback to reduce self-echo.
- Goodbye intent short-circuits the pipeline and returns a goodbye response
	(configurable via GOODBYE_RESPONSE_TEXT).
- ClinicQueue responses are logged in detail to diagnose missing keys.

## Configuration

Set these via environment variables. Defaults are defined in code.

### ClinicQueue
- CLINICQUEUE_BASE_URL (currently hardcoded in services/pipeline.py)
- CLINICQUEUE_TIMEOUT_SECONDS (default 50)
- CLINICQUEUE_TIMEOUT_FALLBACK_TEXT

### Pipeline behavior
- STT_SLOW_THRESHOLD_MS (default 30000)
- PIPELINE_SESSION_RUNTIME_TTL_SECONDS (default 1800)
- GOODBYE_RESPONSE_TEXT

### Remote STT (services/stt.py)
- STT_API_URL
- STT_TIMEOUT_SECONDS
- STT_CONNECT_TIMEOUT_SECONDS
- REMOTE_WHISPER_MODEL
- STT_MIN_AUDIO_MS
- STT_MIN_RMS
- STT_VAD_RATIO_THRESHOLD
- STT_NO_SPEECH_THRESHOLD
- STT_LOG_PROB_THRESHOLD
- STT_LANG_CONF_THRESHOLD
- STT_SESSION_STATE_TTL_SECONDS
- STT_PROMPT_FILE (default prompts/stt_prompt.txt)
- STT_LANGUAGE (optional, force language)

### TTS (services/streaming_tts.py, services/tts_service.py)
- TTS_TARGET_SAMPLE_RATE
- STREAMING_TTS_CHUNK_MS
- STREAMING_TTS_MIN_CHUNK_MS
- STREAMING_TTS_MAX_CHUNK_MS
- KOKORO_SAMPLE_RATE
- KOKORO_SPEED
- KOKORO_SLOW_SPEED
- KOKORO_DEVICE
- KOKORO_NORMALIZE
- KOKORO_NORMALIZE_MAX_GAIN
- KOKORO_SPLIT_PATTERN
- KOKORO_LANG_CODE, KOKORO_VOICE
- KOKORO_EN_LANG_CODE, KOKORO_EN_VOICE
- KOKORO_HI_LANG_CODE, KOKORO_HI_VOICE

### MySQL (services/db.py)
- MYSQL_HOST
- MYSQL_PORT
- MYSQL_USER
- MYSQL_PASSWORD
- MYSQL_DB

## Local setup

1) Create a Python environment.
2) Install dependencies:
	 pip install -r requirements.txt
3) Ensure the remote STT service and ClinicQueue are reachable.
4) Run the service:
	 python main.py

The service listens on port 8000 by default.

## Notes for reviewers

- The STT logic uses multiple safeguards: duration/rms/VAD gating, duplicate
	audio hash detection, and hallucination suppression.
- The TTS pipeline is streaming-first. The JSON/base64 endpoints simply collect
	the stream into a WAV buffer for compatibility.
- llm.py is kept for reference only; ClinicQueue is the real LLM source.
