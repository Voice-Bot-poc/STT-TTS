from fastapi import FastAPI
import sounddevice as sd
import webrtcvad
import numpy as np
import requests
import threading
import time
import queue
from datetime import datetime

app = FastAPI()

SAMPLE_RATE = 16000
FRAME_DURATION = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2)

SILENCE_LIMIT = 20
MAX_CHUNK_DURATION = 10.0   # seconds
OVERLAP_DURATION = 0.8
OVERLAP_BYTES = int(OVERLAP_DURATION * SAMPLE_RATE * 2)
speech_buffer = []
silence_counter = 0
speech_active = False

audio_queue = queue.Queue()

STT_SERVER = "https://mile-laid-compete-vii.trycloudflare.com/transcribe"


# -----------------------------
# STT WORKER (Consumer)
# -----------------------------

def stt_worker():

    while True:

        audio_chunk = audio_queue.get()

        try:

            print("Sending chunk to STT...")

            files = {
                "file": ("speech.raw", audio_chunk, "application/octet-stream")
            }

            response = requests.post(STT_SERVER, files=files)
            timestamp = datetime.now().strftime("%H:%M:%S")
            transcript = response.json()["faster_whisper"]

            with open("transcript.txt", "a") as f:
                f.write(f"[{timestamp}] {transcript}\n")

            print("STT Response:", transcript)

        except Exception as e:
            print("STT Error:", e)

        audio_queue.task_done()


# -----------------------------
# AUDIO CALLBACK (Producer)
# -----------------------------

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

    # Calculate chunk duration
    audio_chunk = b''.join(speech_buffer)
    duration = len(audio_chunk) / 2 / SAMPLE_RATE

    # Condition 1: silence detected
    silence_trigger = speech_active and silence_counter > SILENCE_LIMIT

    # Condition 2: chunk too long
    max_duration_trigger = speech_active and duration > MAX_CHUNK_DURATION

    if silence_trigger or max_duration_trigger:

        if silence_trigger:
            print("Speech ended")

        if max_duration_trigger:
            print("Max chunk duration reached")

        print("Chunk duration:", round(duration, 2))

        # Send chunk to STT
        audio_queue.put(audio_chunk)

        # -------- overlap handling --------
        overlap_audio = audio_chunk[-OVERLAP_BYTES:]

        speech_buffer = [overlap_audio]

        # reset silence counter
        silence_counter = 0

        # keep speech_active True if duration triggered
        if silence_trigger:
            speech_active = False
        else:
            speech_active = True

# -----------------------------
# MICROPHONE
# -----------------------------

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


# -----------------------------
# START THREADS
# -----------------------------

@app.on_event("startup")
def start_threads():

    mic_thread = threading.Thread(target=start_microphone)
    mic_thread.daemon = True
    mic_thread.start()

    worker_thread = threading.Thread(target=stt_worker)
    worker_thread.daemon = True
    worker_thread.start()