from dotenv import load_dotenv

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

from services import db as _db
from services.pipeline import run_chat_pipeline, run_pipeline
from services.tts_service import AudioFormat, get_tts_service, TTSRequest, VoiceID


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app_: FastAPI):
    yield
    await _db.dispose_engine()


app = FastAPI(
    title="VoiceBot STT-TTS Microservice",
    description=(
        "Converts text to audio (/tts, /tts/stream) and runs voice sessions "
        "over WebSocket (/ws/session)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

tts_service = get_tts_service()


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


@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "ok"}


@app.get("/", tags=["Meta"])
def root():
    return RedirectResponse(url="/health")


@app.post(
    "/tts",
    response_model=TTSResponseBody,
    tags=["TTS"],
    summary="Text to audio (JSON / base64)",
)
def text_to_speech(body: TTSRequestBody):
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
    summary="Text to audio (raw binary)",
    response_class=Response,
)
def text_to_speech_raw(body: TTSRequestBody):
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


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid4())
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes") is not None:
                audio_buffer.extend(message["bytes"])
                continue

            text = message.get("text")
            if text is None:
                continue

            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("WebSocket session=%s received invalid JSON control frame", session_id)
                continue

            message_type = control.get("type")

            if message_type == "barge_in":
                logger.info(
                    "WebSocket session=%s barge_in turn_id=%s",
                    session_id,
                    control.get("turn_id"),
                )
                continue

            if message_type != "end_of_utterance":
                logger.warning("WebSocket session=%s received unknown control type=%s", session_id, message_type)
                continue

            if not audio_buffer:
                logger.warning("WebSocket session=%s end_of_utterance with no audio", session_id)
                continue

            turn_id = str(uuid4())
            result = await run_pipeline(session_id=session_id, audio_bytes=bytes(audio_buffer))
            audio_bytes = base64.b64decode(result.audio_base64)

            await websocket.send_text(json.dumps({"type": "turn_id", "turn_id": turn_id}))
            await websocket.send_bytes(audio_bytes)

            audio_buffer.clear()
    except WebSocketDisconnect:
        logger.info("WebSocket session=%s disconnected", session_id)


class ChatRequest(BaseModel):
    text: str
    session_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
    return await run_chat_pipeline(
        session_id=req.session_id,
        text=req.text,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
