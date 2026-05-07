from fastapi import FastAPI, UploadFile, File
from .stt_engine import default_engine

app = FastAPI()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Read uploaded PCM16 WAV/bytes and return conservative transcription.

    This endpoint delegates to the new STT engine which applies VAD,
    noise-reduction, and conservative decoding to reduce hallucinations.
    """
    audio_bytes = await file.read()

    # Delegate to engine (synchronous internal call)
    try:
        text = default_engine.transcribe_bytes(audio_bytes)
    except Exception as ex:
        return {"error": str(ex), "transcript": ""}

    return {"transcript": text}
