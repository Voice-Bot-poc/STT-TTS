from fastapi import FastAPI
import sounddevice as sd
import webrtcvad
import numpy as np
import requests
import threading
import time

app = FastAPI()

SAMPLE_RATE = 16000
FRAME_DURATION = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
transcript = []
vad = webrtcvad.Vad(2)

SILENCE_LIMIT = 35

speech_buffer = []
silence_counter = 0
speech_active = False

STT_SERVER = "https://relaxation-tracking-theft-authentication.trycloudflare.com/transcribe"


def send_to_stt(audio_chunk):

    print("Sending chunk to STT...")

    files = {
        "file": ("speech.raw", audio_chunk, "application/octet-stream")
    }

    response = requests.post(STT_SERVER, files=files)
    transcript = response.json()["faster_whisper"]
    with open(f"transcript.txt","a") as f:
        f.write(transcript)
    print("STT Response:", response.json())

def audio_callback(indata, frames, time_info, status):

    global speech_buffer
    global silence_counter
    global speech_active

    audio_frame = indata[:, 0].tobytes()

    is_speech = vad.is_speech(audio_frame, SAMPLE_RATE)

    if is_speech:

        if not speech_active:
            print("\nSpeech started")

        speech_active = True
        silence_counter = 0
        speech_buffer.append(audio_frame)

    else:

        if speech_active:
            silence_counter += 1
            speech_buffer.append(audio_frame)

        if speech_active and silence_counter > SILENCE_LIMIT:

            print("Speech ended")

            audio_chunk = b''.join(speech_buffer)

            duration = len(audio_chunk) / 2 / SAMPLE_RATE
            print("Chunk duration:", round(duration, 2))

            send_to_stt(audio_chunk)
            speech_buffer = []
            silence_counter = 0
            speech_active = False


def start_microphone():

    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=FRAME_SIZE,
            callback=audio_callback):

        print("Microphone listening...")

        while True:
            time.sleep(1)
    

@app.on_event("startup")
def start_audio_thread():

    thread = threading.Thread(target=start_microphone)
    thread.daemon = True
    thread.start()

# uvicorn vad_backend:app --host 0.0.0.0 --port 8000