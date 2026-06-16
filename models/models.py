"""
models.py — Pydantic v2 request/response schemas for the /process pipeline endpoint.

All other endpoints (/tts, /tts/stream) keep their own schemas in main.py / tts_service.py.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline request
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    """
    Body for POST /process.

    NOTE: In FastAPI, when an endpoint mixes UploadFile (multipart) with body
    fields, those additional fields must be declared as Form() parameters in the
    route function rather than as a single Pydantic body model.  This model is
    kept here for documentation purposes and for use in pipeline.py's type hints.
    """
    session_id: str = Field(..., min_length=1, description="Unique conversation/session identifier")


# ---------------------------------------------------------------------------
# Latency breakdown
# ---------------------------------------------------------------------------

class LatencyBreakdown(BaseModel):
    """Wall-clock milliseconds for each pipeline stage."""
    stt: int
    db_fetch:int
    llm: int
    db_write: int
    tts: int
    total: int


# ---------------------------------------------------------------------------
# Pipeline response
# ---------------------------------------------------------------------------

class ProcessResponse(BaseModel):
    """Successful response from POST /process."""
    session_id: str = Field(..., description="Echo of the request session_id")
    transcript: str = Field(..., description="Whisper transcription of the uploaded audio")
    response_text: str = Field(..., description="LLM-generated assistant reply")
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio of the reply")
    audio_format: str = Field("mp3", description="Always 'mp3' for the pipeline endpoint")
    latency_ms: LatencyBreakdown = Field(..., description="Per-stage and total latency in milliseconds")


# ---------------------------------------------------------------------------
# Structured error — returned when any stage fails
# ---------------------------------------------------------------------------

class PipelineError(BaseModel):
    """
    Structured error body.  Returned with an appropriate HTTP status code
    (typically 422 or 500) when a pipeline stage raises an exception.

    stage: one of "stt" | "llm" | "db" | "tts"
    detail: human-readable description of the failure
    """
    stage: str = Field(..., description="Pipeline stage that failed: stt | llm | db | tts")
    detail: str = Field(..., description="Human-readable error description")
