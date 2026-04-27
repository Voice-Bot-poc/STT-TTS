from pydantic import BaseModel, Field


class LatencyBreakdown(BaseModel):
    stt: int
    db_fetch: int
    llm: int
    db_write: int
    tts: int
    total: int


class ProcessResponse(BaseModel):
    session_id: str = Field(..., description="Echo of the request session_id")
    transcript: str = Field(..., description="Whisper transcription of the uploaded audio")
    response_text: str = Field(..., description="LLM-generated assistant reply")
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio of the reply")
    audio_format: str = Field("mp3", description="Always 'mp3' for the pipeline")
    latency_ms: LatencyBreakdown = Field(..., description="Per-stage and total latency in milliseconds")
