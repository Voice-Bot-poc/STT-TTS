from dotenv import load_dotenv

import logging
from contextlib import asynccontextmanager
from typing import Optional

import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services import db as _db  # alias to avoid shadowing any built-in
from models.models import ProcessResponse
from services.pipeline import run_pipeline
from services.tts_service import get_tts_service, TTSRequest, AudioFormat, VoiceID
from fastapi.responses import RedirectResponse

from services.pipeline import run_chat_pipeline

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
    yield  # startup: nothing to do (lazy init in db.py)
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
):
    """
    Full VoiceBot pipeline in one call:

    1. **STT** — transcribe the uploaded audio with Whisper (local)
    2. **DB fetch** — load last 5 conversation turns for `session_id`
    3. **LLM** — call Anthropic Claude with history + transcript
    4. **DB write** — persist the new exchange
    5. **TTS** — synthesise the LLM reply to MP3 audio

    Returns the transcript, LLM response, base64 audio, and per-stage latency.

    On failure in any stage, returns `{ "stage": "...", "detail": "..." }` with HTTP 500.

    **Multipart form fields:**
    - `audio_file` — the audio file
    - `session_id` — string identifier for the conversation session
    """
    audio_bytes = await audio_file.read()
    return await run_pipeline(session_id=session_id, audio_bytes=audio_bytes)

class ChatRequest(BaseModel):
    text: str
    session_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
    return await run_chat_pipeline(
        session_id=req.session_id,
        text=req.text,
    )

# ---------------------------------------------------------------------------
# Entry point  (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)