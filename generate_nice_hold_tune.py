import math
import struct
import wave
from pathlib import Path

def generate_engaging_tune(output_path: Path, duration_s: float = 15.0, sample_rate: int = 16000):
    n_samples = int(sample_rate * duration_s)
    samples = [0.0] * n_samples
    
    # C Major Pentatonic: C4, D4, E4, G4, A4, C5
    freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    
    # A pleasant, engaging, bouncy arpeggiated sequence
    pattern = [0, 2, 4, 5, 4, 2, 3, 1] 
    
    bpm = 110
    beats_per_sec = bpm / 60.0
    sec_per_beat = 1.0 / beats_per_sec
    
    # Generate melody notes
    for i in range(int(duration_s / sec_per_beat)):
        note_idx = pattern[i % len(pattern)]
        freq = freqs[note_idx]
        
        start_sample = int(i * sec_per_beat * sample_rate)
        # Add slight overlap for a ringing bell effect
        note_len_samples = int(sec_per_beat * sample_rate * 1.8) 
        
        for j in range(note_len_samples):
            if start_sample + j >= n_samples:
                break
                
            t = j / sample_rate
            
            # ADSR Envelope for a bell/marimba sound
            attack = 0.02
            if t < attack:
                env = t / attack
            else:
                env = math.exp(-(t - attack) * 4.5) # smooth exponential decay
                
            # Mix sine with some harmonics for a warm, engaging bell tone
            val = math.sin(2 * math.pi * freq * t)
            val += 0.4 * math.sin(2 * math.pi * freq * 2 * t)
            val += 0.2 * math.sin(2 * math.pi * freq * 3 * t)
            val += 0.1 * math.sin(2 * math.pi * freq * 4 * t)
            
            val *= env * 0.20 # melody volume
            
            samples[start_sample + j] += val
            
    # Add a very soft, warm ambient pad underneath to make it sound professional
    base_freqs = [261.63, 329.63, 392.00] # C major chord
    for i in range(n_samples):
        t = i / sample_rate
        pad_val = sum(math.sin(2 * math.pi * f * t) for f in base_freqs) / len(base_freqs)
        pad_val *= 0.04 # very quiet
        
        samples[i] += pad_val
        
        # Soft clipping and 16-bit conversion
        s = samples[i]
        sample_int = int(max(-1.0, min(1.0, s)) * 32767)
        samples[i] = sample_int

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *[int(s) for s in samples]))

if __name__ == "__main__":
    out_path = Path("audio/fillers/hold.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_engaging_tune(out_path, 15.0)
    print("Done generating pleasant engaging hold tune.")
