from fastapi import FastAPI, UploadFile, File
import numpy as np
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer
from whispercpp import Whisper
import json

app = FastAPI()

SAMPLE_RATE = 16000

# -------------------------
# Load Models
# -------------------------

print("Loading Faster Whisper...")
fw_model = WhisperModel("base", compute_type="int8")

print("Loading Vosk...")
vosk_model = Model("models/vosk-model-small-en-us-0.15")

# print("Loading whisper.cpp...")
# wcpp_model = Whisper.from_pretrained("base")


# -------------------------
# Faster Whisper
# -------------------------

def transcribe_faster_whisper(audio_np):

    segments, info = fw_model.transcribe(audio_np,
                                         beam_size=5,vad_filter=False)

    text = ""
    for segment in segments:
        text += segment.text

    return text.strip()


# -------------------------
# Vosk
# -------------------------

def transcribe_vosk(audio_bytes):

    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    if rec.AcceptWaveform(audio_bytes):
        result = json.loads(rec.Result())
    else:
        result = json.loads(rec.FinalResult())

    return result.get("text", "")


# -------------------------
# Whisper.cpp
# -------------------------

# def transcribe_whispercpp(audio_np):

#     result = wcpp_model.transcribe(audio_np)

#     return result.strip()


# -------------------------
# API Endpoint
# -------------------------

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    audio_bytes = await file.read()

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Run all engines
    fw_text = transcribe_faster_whisper(audio_np)
    # vosk_text = transcribe_vosk(audio_bytes)
    # wcpp_text = transcribe_whispercpp(audio_np)

    return {
        "faster_whisper": fw_text,
        # "vosk": vosk_text,
        # "whisper_cpp": wcpp_text
    }
