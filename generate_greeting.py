import asyncio
import os
import wave
from pathlib import Path

import edge_tts
import miniaudio
from dotenv import load_dotenv

load_dotenv()

VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-US-AriaNeural")
TEXT = "Thank you for calling the clinic. How can I help you today?"
OUTPUT_WAV = Path("assets/greeting_en.wav")

async def generate_greeting():
    print(f"Generating greeting using voice: {VOICE}")
    
    # Ensure assets directory exists
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary MP3 file for edge_tts output
    temp_mp3 = OUTPUT_WAV.with_suffix(".mp3")
    
    try:
        # Synthesize using edge_tts
        communicate = edge_tts.Communicate(TEXT, VOICE)
        await communicate.save(str(temp_mp3))
        
        # Decode mp3 to PCM using miniaudio, forcing 16kHz mono
        decoder = miniaudio.mp3_read_file_f32(str(temp_mp3))
        
        # Wait, miniaudio.mp3_read_file_s16 is better? Yes!
        # Let's re-read with sample rate conversion.
        mp3_bytes = temp_mp3.read_bytes()
        decoded = miniaudio.decode(mp3_bytes, sample_rate=16000, nchannels=1, output_format=miniaudio.SampleFormat.SIGNED16)
        
        # Save as WAV using wave module
        with wave.open(str(OUTPUT_WAV), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(16000)
            wf.writeframes(decoded.samples)
            
        file_size = OUTPUT_WAV.stat().st_size
        print(f"[OK] Success! Greeting saved to {OUTPUT_WAV}")
        print(f"   Size: {file_size} bytes")
        
    finally:
        # Cleanup temp file
        if temp_mp3.exists():
            temp_mp3.unlink()

if __name__ == "__main__":
    asyncio.run(generate_greeting())
