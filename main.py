from dotenv import load_dotenv

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from functools import partial

import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from services import db as _db  # alias to avoid shadowing any built-in
from models.models import ProcessResponse
from services.pipeline import close_http_clients, run_pipeline
from services.tts_service import get_tts_service, TTSRequest, AudioFormat, VoiceID
from services.streaming_tts import get_streaming_tts_service, StreamingTtsUnavailable, TARGET_SAMPLE_RATE
from fastapi.responses import RedirectResponse

from services.pipeline import run_chat_pipeline
from services.runtime_state import mark_tts_end, mark_tts_start

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Startup / shutdown handler — dispose MySQL connections cleanly on exit."""
    yield
    await close_http_clients()
    await _db.dispose_engine()


app = FastAPI(
    title="VoiceBot STT-TTS Microservice",
    description=(
        "Converts text to audio (/tts, /tts/stream) and runs the full "
        "STT → LLM → MySQL → TTS pipeline (/process)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

tts_service = get_tts_service()
streaming_tts_service = get_streaming_tts_service()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TTSRequestBody(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesise")
    voice: Optional[VoiceID] = Field("default", description="'default' or 'slow'")
    format: Optional[AudioFormat] = Field("mp3", description="Output format: mp3 | wav | ogg")


class TTSResponseBody(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio bytes")
    duration: float = Field(..., description="Estimated audio duration in seconds")
    format: str = Field(..., description="Actual format of the returned audio")
    char_count: int = Field(..., description="Number of characters synthesised")


class SynthesiseRequestBody(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesise for call greeting")


class StreamingTTSRequestBody(BaseModel):
    text: str = Field(..., min_length=1, description="Text to stream as PCM16 audio")
    chunk_ms: int = Field(40, ge=20, le=200, description="PCM chunk cadence in milliseconds")
    session_id: str = Field("default", description="Conversation session id for TTS echo suppression")
    speed: Optional[float] = Field(None, ge=0.5, le=1.3, description="Optional Kokoro speech speed override")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health_check():
    """Quick liveness probe."""
    return {"status": "ok"}

@app.get("/", tags=["Meta"])
def root():
    """Redirect root to health check."""
    return RedirectResponse(url="/health")

@app.post(
    "/tts",
    response_model=TTSResponseBody,
    tags=["TTS"],
    summary="Text → audio (JSON / base64)",
)
def text_to_speech(body: TTSRequestBody):
    """
    Convert text to audio.

    Returns a JSON body with base64-encoded audio and metadata.
    Decode `audio_base64` on the client side to get raw audio bytes.

    **Example**
    ```
    POST /tts
    { "text": "Hello world", "voice": "default", "format": "mp3" }
    ```
    """
    try:
        result = tts_service.synthesise(
            TTSRequest(
                text=body.text,
                voice=body.voice or "default",
                format=body.format or "mp3",
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

    return TTSResponseBody(
        audio_base64=result.audio_base64,
        duration=result.duration,
        format=result.format,
        char_count=result.char_count,
    )


@app.post("/synthesise", tags=["TTS"], summary="Text to base64 wav for call playback")
async def synthesise(body: SynthesiseRequestBody):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(tts_service.synthesise, TTSRequest(text=body.text, voice="default", format="wav")),
    )
    return {"audio_base64": result.audio_base64}


@app.post(
    "/tts/stream",
    tags=["TTS"],
    summary="Text → audio (raw binary)",
    response_class=Response,
)
def text_to_speech_raw(body: TTSRequestBody):
    """
    Same as `/tts` but returns the raw audio file directly
    (Content-Type: audio/mpeg | audio/wav | audio/ogg).

    Useful when you want to pipe the response straight into a player
    instead of parsing JSON.
    """
    try:
        result = tts_service.synthesise(
            TTSRequest(
                text=body.text,
                voice=body.voice or "default",
                format=body.format or "mp3",
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}
    audio_bytes = base64.b64decode(result.audio_base64)

    return Response(
        content=audio_bytes,
        media_type=mime_map.get(result.format, "audio/mpeg"),
        headers={
            "X-Audio-Duration": str(result.duration),
            "X-Char-Count": str(result.char_count),
        },
    )


@app.post(
    "/tts/pcm-stream",
    tags=["TTS"],
    summary="Text to streaming PCM16 mono 24kHz for realtime RTP playback",
)
async def text_to_pcm_stream(body: StreamingTTSRequestBody):
    async def generate():
        mark_tts_start(body.session_id)
        try:
            async for chunk in streaming_tts_service.stream_pcm(
                body.text,
                body.chunk_ms,
                body.session_id,
                speed=body.speed,
            ):
                yield chunk
        except StreamingTtsUnavailable as exc:
            logger.exception("Streaming TTS unavailable")
            raise HTTPException(status_code=503, detail=str(exc))
        except asyncio.CancelledError:
            logger.info("[TTS] cancelled by client session=%s", body.session_id)
            streaming_tts_service.cancel_session(body.session_id)
            raise
        except Exception as exc:
            logger.exception("Streaming TTS failed")
            raise HTTPException(status_code=500, detail=f"Streaming synthesis error: {exc}")
        finally:
            mark_tts_end(body.session_id)

    return StreamingResponse(
        generate(),
        media_type=f"audio/L16;rate={TARGET_SAMPLE_RATE};channels=1",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
            "X-Channels": "1",
        },
    )


# ---------------------------------------------------------------------------
# POST /process — Full pipeline: STT → LLM → MySQL → TTS
# ---------------------------------------------------------------------------

@app.post(
    "/process",
    response_model=ProcessResponse,
    tags=["Pipeline"],
    summary="Audio → transcript → LLM reply → audio (full pipeline)",
)
async def process_audio(
    audio_file: UploadFile = File(
        ...,
        description="Audio file to transcribe (wav, mp3, m4a, …)",
    ),
    session_id: str = Form(
        ...,
        min_length=1,
        description="Unique session/conversation identifier for history lookup",
    ),
    phone_number: str = Form(
        ...,
        description="Real caller phone number from WhatsApp webhook",
    ),
    synthesize_tts: bool = Form(
        True,
        description="When false, return transcript and ClinicQueue text without blocking TTS synthesis.",
    ),
):
    """
    Full VoiceBot pipeline in one call:

    1. **STT** — transcribe the uploaded audio with Whisper (local)
    2. **ClinicQueue** — call `/api/voice/chat` with transcript, session_id, and real phone_number
    3. **TTS** — synthesise the ClinicQueue reply to audio

    Returns the transcript, LLM response, base64 audio, and per-stage latency.

    On failure in any stage, returns `{ "stage": "...", "detail": "..." }` with HTTP 500.

    **Multipart form fields:**
    - `audio_file` — the audio file
    - `session_id` — string identifier for the conversation session
    - `phone_number` — real caller phone number from WhatsApp webhook
    """
    audio_bytes = await audio_file.read()
    logger.info(
        "[VoiceInput] /process received audio bytes=%d filename=%s content_type=%s session_id=%r phone_number=%r synthesize_tts=%s",
        len(audio_bytes),
        audio_file.filename,
        audio_file.content_type,
        session_id,
        phone_number,
        synthesize_tts,
    )
    return await run_pipeline(
        session_id=session_id,
        audio_bytes=audio_bytes,
        phone_number=phone_number,
        synthesize_tts=synthesize_tts,
)



class ChatRequest(BaseModel):
    text: str
    session_id: str
    phone_number: str


@app.post("/chat")
async def chat(req: ChatRequest):
    logger.info(
        "[VoiceInput] /chat received session_id=%r phone_number=%r text_preview=%r",
        req.session_id,
        req.phone_number,
        req.text[:80],
    )
    return await run_chat_pipeline(
        session_id=req.session_id,
        text=req.text,
        phone_number=req.phone_number,
    )

# ---------------------------------------------------------------------------
# Entry point  (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
