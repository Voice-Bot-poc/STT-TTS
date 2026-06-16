from dotenv import load_dotenv
load_dotenv()
import asyncio
import logging
import unicodedata
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
from services.filler_service import preload_fillers, get_filler_packet_wav, FILLER_ENABLED
from services.greeting_service import GreetingService
 
# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
 
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Language detection from text script
# ---------------------------------------------------------------------------
 
def _detect_text_language(text: str) -> str:
    """
    Detect language from the Unicode script of the text itself.
    This is the most reliable method — it doesn't depend on STT language labels
    or ClinicQueue detectedLanguage, both of which can be wrong for Hinglish.
 
    Returns 'hi' if the text contains significant Devanagari characters,
    otherwise returns 'en'.
    """
    devanagari_count = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            try:
                name = unicodedata.name(char, "")
                if "DEVANAGARI" in name:
                    devanagari_count += 1
            except Exception:
                pass
 
    if total_alpha > 0 and (devanagari_count / total_alpha) > 0.3:
        detected = "hi"
    else:
        detected = "en"
 
    logger.info(
        "[TTS] script detection: devanagari=%d total_alpha=%d ratio=%.2f detected=%s text_preview=%r",
        devanagari_count,
        total_alpha,
        (devanagari_count / total_alpha) if total_alpha > 0 else 0.0,
        detected,
        text[:50],
    )
    return detected
 
 
# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
 
@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Startup / shutdown handler — dispose MySQL connections cleanly on exit."""
 
    # ── Pre-load TTS pipeline at startup so first caller pays zero load cost ──
    try:
        import asyncio as _asyncio
        from services.streaming_tts import get_streaming_tts_service
        loop = _asyncio.get_event_loop()
        await loop.run_in_executor(None, get_streaming_tts_service().validate_startup)
        logger.info("[STARTUP] TTS pipeline warmed up")
    except Exception as e:
        logger.warning("[STARTUP] TTS warm-up skipped (non-fatal): %s", e)
 
    # ── Pre-load filler WAVs into memory ─────────────────────────────────────
    try:
        preload_fillers()
        logger.info("[STARTUP] Filler audio preloaded (enabled=%s)", FILLER_ENABLED)
    except Exception as e:
        logger.warning("[STARTUP] Filler preload skipped (non-fatal): %s", e)
 
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
greeting_service = GreetingService()
 
 
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
    chunk_ms: int = Field(20, ge=20, le=200, description="PCM chunk cadence in milliseconds")
    session_id: str = Field("default", description="Conversation session id for TTS echo suppression")
    speed: Optional[float] = Field(None, ge=0.5, le=1.3, description="Unused with edge-tts (speed controlled via EDGE_TTS_SPEED env var)")
 
 
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
 
 
from services.tts_normalizer import normalize_for_tts

@app.post(
    "/tts/pcm-stream",
    tags=["TTS"],
    summary="Text to streaming PCM16 mono 16kHz for realtime RTP playback",
)
async def text_to_pcm_stream(body: StreamingTTSRequestBody):
    # Detect language directly from the response text script.
    # This is the most reliable method:
    # - Senticore already translated the LLM response to Hindi if needed
    # - If the text contains Devanagari → use Hindi voice
    # - If the text is Roman script → use English voice
    # - No dependency on STT language detection or Redis
    tts_language = _detect_text_language(body.text)
 
    # Normalize dates/times/numbers for the detected language
    normalized_text = normalize_for_tts(body.text)
    if normalized_text != body.text:
        logger.info(
            "[TTS] normalized text: %d→%d chars language=%s",
            len(body.text), len(normalized_text), tts_language
        )

    async def generate():
        mark_tts_start(body.session_id)
        try:
            async for chunk in streaming_tts_service.stream_pcm(
                normalized_text,
                body.chunk_ms,
                body.session_id,
                speed=body.speed,
                language=tts_language,
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
 
 
@app.get(
    "/greeting/pcm-stream",
    tags=["Greeting"],
    summary="Get pre-recorded greeting as PCM stream",
)
async def get_greeting_stream():
    """
    Returns the pre-recorded greeting directly from memory for 0ms latency.
    """
    if not greeting_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Greeting audio is not loaded. Run generate_greeting.py."
        )
 
    return StreamingResponse(
        greeting_service.stream_pcm(chunk_ms=20),
        media_type="audio/L16;rate=16000;channels=1",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": "16000",
            "X-Channels": "1",
        },
    )
 
 
# ---------------------------------------------------------------------------
# POST /filler — Two-phase filler audio for dead-air elimination
# ---------------------------------------------------------------------------
 
class FillerRequest(BaseModel):
    transcript: str = Field("", description="User transcript to match filler context")
    language: str = Field("en", description="Language code: en or hi")
    session_id: str = Field("", description="Session ID to look up detected language from Redis")
 
 
@app.post("/filler", tags=["Filler"], summary="Get filler audio for dead-air elimination")
async def get_filler(body: FillerRequest):
    """
    Returns two-phase filler audio (base64 WAV) matched to the user's transcript.
    """
    if not FILLER_ENABLED or not body.transcript.strip():
        return {
            "instruction_b64": "",
            "instruction_ms": 0,
            "tune_b64": "",
            "tune_ms": 0,
            "enabled": False,
        }
 
    # Resolve language — Redis takes priority over passed language field
    resolved_language = body.language or "en"
    stt_language = "en"
    tts_language = "en"
    if body.session_id:
        try:
            import redis.asyncio as aioredis
            _r = aioredis.from_url("redis://localhost:6379/0")
            stt_val = await _r.get(f"voice:{body.session_id}:language")
            if stt_val:
                stt_language = stt_val.decode() if hasattr(stt_val, "decode") else str(stt_val)
                resolved_language = stt_language
            tts_val = await _r.get(f"voice:{body.session_id}:tts_language")
            if tts_val:
                tts_language = tts_val.decode() if hasattr(tts_val, "decode") else str(tts_val)
            await _r.aclose()
        except Exception as e:
            logger.warning("[FILLER] Redis language lookup failed: %s", e)
            resolved_language = body.language or "en"

    # Enforce resolved_language is either "en" or "hi"
    if resolved_language != "hi" and resolved_language != "en":
        resolved_language = "en"

    session_id = body.session_id
    selected_language = resolved_language
    logger.info(
        f"[FILLER_DEBUG] session={session_id} "
        f"stt_lang={stt_language} "
        f"tts_lang={tts_language} "
        f"selected_filler_lang={selected_language}"
    )

    packet = get_filler_packet_wav(body.transcript, language=resolved_language)
 
    return {
        "instruction_b64": base64.b64encode(packet["instruction_wav"]).decode() if packet["instruction_wav"] else "",
        "instruction_ms": packet["instruction_ms"],
        "tune_b64": base64.b64encode(packet["tune_wav"]).decode() if packet["tune_wav"] else "",
        "tune_ms": packet["tune_ms"],
        "enabled": True,
    }
 
 
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
        audio_filename=audio_file.filename,
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