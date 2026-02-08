"""
MedScribe Pipeline
==================
Orchestrates MedASR (speech-to-text) + MedGemma (text-to-SOAP).

Two HAI-DEF models in one pipeline:
- MedASR: 105M param Conformer, 5.2% WER on medical dictation
- MedGemma: 4B param, LoRA fine-tuned for concise SOAP notes

Usage:
    from src.pipeline import MedScribePipeline
    pipe = MedScribePipeline()
    pipe.load_all()
    result = pipe.transcribe_and_generate("audio.wav")
"""
import os
import time
import torch
import librosa
import numpy as np

try:
    from transformers import AutoModelForCTC, AutoProcessor
    MEDASR_AVAILABLE = True
except ImportError:
    MEDASR_AVAILABLE = False

from src.inference import SOAPGenerator


class MedScribePipeline:
    """Full voice-to-SOAP pipeline using HAI-DEF models."""

    MEDASR_MODEL_ID = "google/medasr"
    SAMPLE_RATE = 16000

    def __init__(self, adapter_path="./models/checkpoints/medgemma_v2_soap/final_model"):
        self.soap_gen = SOAPGenerator(adapter_path=adapter_path)
        self.asr_model = None
        self.asr_processor = None
        self._asr_loaded = False

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def load_asr(self, progress_callback=None):
        """Load MedASR model. Runs on GPU if available, else CPU."""
        if self._asr_loaded:
            return

        if not MEDASR_AVAILABLE:
            raise ImportError(
                "MedASR requires transformers>=5.0.0. "
                "Install with: pip install transformers>=5.0.0"
            )

        if progress_callback:
            progress_callback("Loading MedASR speech recognition...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_processor = AutoProcessor.from_pretrained(self.MEDASR_MODEL_ID)
        self.asr_model = AutoModelForCTC.from_pretrained(self.MEDASR_MODEL_ID).to(device)
        self.asr_model.eval()
        self._asr_loaded = True

        if progress_callback:
            progress_callback("MedASR ready.")

    def load_soap(self, progress_callback=None):
        """Load MedGemma SOAP generator."""
        self.soap_gen.load(progress_callback=progress_callback)

    def load_all(self, progress_callback=None):
        """Load both models."""
        self.load_asr(progress_callback=progress_callback)
        self.load_soap(progress_callback=progress_callback)

    # --------------------------------------------------------
    # TRANSCRIPTION
    # --------------------------------------------------------
    def transcribe(self, audio_path):
        """
        Transcribe audio file to text using MedASR.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            dict with keys: transcript, time_s, audio_duration_s
        """
        if not self._asr_loaded:
            raise RuntimeError("MedASR not loaded. Call .load_asr() first.")

        # Load and resample audio to 16kHz mono
        speech, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        audio_duration = len(speech) / self.SAMPLE_RATE

        # Process through MedASR
        device = next(self.asr_model.parameters()).device
        inputs = self.asr_processor(
            speech,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(device)

        start = time.time()
        with torch.inference_mode():
            outputs = self.asr_model.generate(**inputs)
        elapsed = time.time() - start

        transcript = self.asr_processor.batch_decode(outputs)[0]

        return {
            "transcript": transcript.strip(),
            "time_s": round(elapsed, 1),
            "audio_duration_s": round(audio_duration, 1),
        }

    # --------------------------------------------------------
    # SOAP GENERATION
    # --------------------------------------------------------
    def generate_soap(self, transcript):
        """Generate SOAP note from transcript text."""
        return self.soap_gen.generate(transcript)

    # --------------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------------
    def transcribe_and_generate(self, audio_path):
        """
        Full pipeline: audio -> transcript -> SOAP note.

        Returns:
            dict with keys: transcript, soap_note, asr_time_s, soap_time_s,
                           total_time_s, audio_duration_s, tokens, word_count
        """
        # Step 1: Transcribe
        asr_result = self.transcribe(audio_path)

        # Step 2: Generate SOAP
        soap_result = self.generate_soap(asr_result["transcript"])

        return {
            "transcript": asr_result["transcript"],
            "soap_note": soap_result["soap_note"],
            "asr_time_s": asr_result["time_s"],
            "soap_time_s": soap_result["time_s"],
            "total_time_s": round(asr_result["time_s"] + soap_result["time_s"], 1),
            "audio_duration_s": asr_result["audio_duration_s"],
            "tokens": soap_result["tokens"],
            "word_count": soap_result["word_count"],
        }

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------
    @property
    def asr_loaded(self):
        return self._asr_loaded

    @property
    def soap_loaded(self):
        return self.soap_gen.is_loaded

    @property
    def fully_loaded(self):
        return self._asr_loaded and self.soap_gen.is_loaded