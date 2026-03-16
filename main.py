import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import base64

from tts_service import get_tts_service, TTSRequest, AudioFormat, VoiceID

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TTS Microservice",
    description="Converts text to audio. Returns base64-encoded audio in JSON or raw audio bytes.",
    version="1.0.0",
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
# Entry point  (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)