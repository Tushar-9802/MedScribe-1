"""
Synthetic audio generation
"""

from gtts import gTTS
import os
import librosa
import soundfile as sf

os.makedirs("data/audio_samples/gtts", exist_ok=True)

cases = {
    "test_case_01_cardiology": """
Forty-five year old male with acute onset chest pain for two hours. 
Pain is substernal, pressure-like, radiating to left arm. 
Associated with diaphoresis and shortness of breath. 
Past medical history significant for hypertension and hyperlipidemia. 
Current medications include lisinopril and atorvastatin. 
No known drug allergies. 
Physical exam shows blood pressure one forty-five over ninety-two, 
heart rate ninety-eight, respiratory rate twenty. 
Diaphoretic, mild distress. 
Cardiac exam regular rhythm, no murmurs. 
Lungs clear bilaterally. 
EKG shows ST elevations in leads two, three, and AVF.
""",
    "test_case_02_respiratory": """
Sixty-seven year old female presenting with progressive dyspnea over three days.
Associated with orthopnea, paroxysmal nocturnal dyspnea, and bilateral lower extremity edema.
Past medical history includes congestive heart failure, hypertension, and diabetes mellitus type two.
Medications include furosemide, metoprolol, and metformin.
Vital signs show blood pressure one sixty over ninety-five, heart rate one hundred ten irregular, 
respiratory rate twenty-eight, oxygen saturation eighty-eight percent on room air.
Physical exam reveals jugular venous distension, bilateral pulmonary crackles to mid-lung fields,
S3 gallop on cardiac auscultation, and three plus pitting edema bilaterally to knees.
""",
    "test_case_03_pediatric": """
Eight year old male brought in by parents for fever and cough for three days.
Fever to one hundred two point five Fahrenheit, productive cough with yellow sputum.
Associated symptoms include rhinorrhea, mild sore throat, decreased appetite.
No vomiting, diarrhea, or rash. 
Immunizations up to date per parent report.
Attends elementary school, multiple sick contacts.
No recent travel. No known exposures to COVID-19.
Vital signs: temperature one hundred one point eight, heart rate one hundred, 
respiratory rate twenty-four, oxygen saturation ninety-seven percent on room air.
General appearance: alert, interactive, mild nasal congestion.
HEENT: nasal discharge present, oropharynx mildly erythematous without exudate.
Lungs: scattered rhonchi bilaterally, no wheezing, no retractions.
"""
}

print("Generating audio with Google TTS (requires internet)...")
print("="*60)

for filename, text in cases.items():
    print(f"\nGenerating: {filename}")
    
    # Generate MP3 with gTTS
    temp_mp3 = f"data/audio_samples/gtts/{filename}_temp.mp3"
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(temp_mp3)
    print(f"  ✓ Created MP3")
    
    # Convert MP3 to WAV using librosa (no FFmpeg needed for MP3 reading)
    print(f"  Converting to 16kHz WAV...")
    try:
        # librosa can read MP3 without FFmpeg
        audio, sr = librosa.load(temp_mp3, sr=16000, mono=True)
        
        # Normalize
        if audio.max() > 0:
            audio = audio / abs(audio).max() * 0.95
        
        # Save as WAV
        output_wav = f"data/audio_samples/{filename}.wav"
        sf.write(output_wav, audio, 16000, subtype='PCM_16')
        
        print(f"    Saved: {output_wav}")
        print(f"    Duration: {len(audio)/16000:.2f}s")
        
        # Delete temp MP3
        os.remove(temp_mp3)
        
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  MP3 file saved at: {temp_mp3}")
        print(f"  You can manually convert it later")

print("\n" + "="*60)
print("gTTS audio generation complete!")
print("Location: data/audio_samples/gtts/")