"""
Checks Audio formatting and stats
"""

import librosa
import soundfile as sf
import os

audio_files = [
    "data/audio_samples/test_case_01_cardiology.wav",
    "data/audio_samples/test_case_02_respiratory.wav",
    "data/audio_samples/test_case_03_pediatric.wav"
]

print("Audio File Analysis")
print("="*60)

for audio_path in audio_files:
    print(f"\nFile: {os.path.basename(audio_path)}")
    
    # Load with librosa
    audio, sr = librosa.load(audio_path, sr=None)  # Don't resample
    
    duration = len(audio) / sr
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    print(f"  Min amplitude: {audio.min():.4f}")
    print(f"  Max amplitude: {audio.max():.4f}")
    print(f"  Mean amplitude: {audio.mean():.4f}")
    
    # Check if audio is silent
    if abs(audio.max()) < 0.01:
        print("  ⚠️ WARNING: Audio appears to be silent or very quiet")
    
    # Get info with soundfile
    info = sf.info(audio_path)
    print(f"  Channels: {info.channels}")
    print(f"  Subtype: {info.subtype}")