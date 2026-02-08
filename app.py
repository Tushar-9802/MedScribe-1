"""
MedScribe — Voice-to-SOAP Clinical Documentation
=================================================
Gradio interface for the MedScribe pipeline.
Two HAI-DEF models: MedASR (speech) + MedGemma (text-to-SOAP).

Launch:
    python app.py
"""
import os
import time
import gradio as gr
from src.pipeline import MedScribePipeline

# ============================================================
# CONFIGURATION
# ============================================================
ADAPTER_PATH = "./models/checkpoints/medgemma_v2_soap/final_model"
APP_TITLE = "MedScribe"
APP_SUBTITLE = "Voice-to-SOAP Clinical Documentation"

EXAMPLE_TRANSCRIPTS = [
    [
        "45-year-old male presents with substernal chest pain for 2 hours, "
        "7/10 severity, radiating to left arm. Associated with diaphoresis "
        "and anxiety. No nausea or vomiting. BP 145/92, HR 98, RR 16. "
        "Anxious appearing. Regular rhythm, no murmurs."
    ],
    [
        "62-year-old female with type 2 diabetes returns for 3-month follow-up. "
        "Reports good compliance with metformin 1000mg twice daily. Occasional "
        "fasting glucose readings of 140-160. No hypoglycemic episodes. Denies "
        "polyuria, polydipsia, blurred vision. BP 132/78, HR 72, BMI 31.2. "
        "A1C today 7.4%, down from 8.1%. Creatinine 1.1, eGFR 68. Foot exam: "
        "intact sensation, no ulcers. Eyes: last retinal exam 6 months ago, "
        "no retinopathy."
    ],
    [
        "4-year-old male brought in by mother for 3 days of runny nose, cough, "
        "and low-grade fever. Max temp 100.4 at home. Eating and drinking well. "
        "No ear pulling. No history of asthma. Temp 99.8, HR 110, RR 22, "
        "SpO2 99%. Alert, playful. TMs clear bilaterally. Throat mildly "
        "erythematous, no exudate. Lungs clear. No lymphadenopathy."
    ],
    [
        "58-year-old male with hypertension and CKD stage 3b, GFR 38. On "
        "lisinopril 20mg daily, amlodipine 5mg daily. BP today 142/88. Labs: "
        "Cr 1.8, BUN 32, K 4.9, bicarb 20, phosphorus 4.8, PTH 98. Urine "
        "albumin-to-creatinine ratio 450. No edema. Denies fatigue, nausea, "
        "pruritus."
    ],
    [
        "34-year-old female presenting with worsening anxiety and depressed mood "
        "over 3 months since job loss. Reports difficulty sleeping, poor appetite, "
        "loss of interest in activities. Denies suicidal ideation, hallucinations, "
        "or substance use. Currently on sertraline 50mg daily started 6 weeks ago "
        "with minimal improvement. PHQ-9 score 14, GAD-7 score 12. Appears "
        "well-groomed, cooperative. Speech normal rate and rhythm. Mood described "
        "as sad. Affect constricted. Thought process linear. Insight and "
        "judgment intact."
    ],
]


# ============================================================
# CUSTOM CSS — clinical/medical professional theme
# ============================================================
CUSTOM_CSS = """
/* ---- Base ---- */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
}

/* ---- Header ---- */
.app-header {
    text-align: center;
    padding: 24px 0 16px 0;
    border-bottom: 2px solid #1a5276;
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #1a5276;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 14px;
    color: #566573;
    margin: 4px 0 0 0;
}

/* ---- Status bar ---- */
.status-bar {
    background: #eaf2f8;
    border: 1px solid #aed6f1;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 13px;
    color: #1a5276;
    margin-bottom: 16px;
}
.status-bar.ready {
    background: #e8f8f5;
    border-color: #a3e4d7;
    color: #0e6655;
}
.status-bar.processing {
    background: #fef9e7;
    border-color: #f9e79f;
    color: #7d6608;
}
.status-bar.error {
    background: #fdedec;
    border-color: #f5b7b1;
    color: #922b21;
}

/* ---- Metrics row ---- */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-top: 12px;
}
.metric-card {
    flex: 1;
    background: #f8f9fa;
    border: 1px solid #d5dbdb;
    border-radius: 4px;
    padding: 10px 14px;
    text-align: center;
}
.metric-card .label {
    font-size: 11px;
    color: #808b96;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 20px;
    font-weight: 600;
    color: #1a5276;
    margin-top: 2px;
}

/* ---- SOAP output formatting ---- */
.soap-output textarea {
    font-family: "IBM Plex Mono", "Consolas", monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

/* ---- Section dividers ---- */
.section-label {
    font-size: 12px;
    font-weight: 600;
    color: #808b96;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 20px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #eaecee;
}

/* ---- Tab styling ---- */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ---- Footer ---- */
.app-footer {
    text-align: center;
    padding: 16px 0;
    margin-top: 24px;
    border-top: 1px solid #eaecee;
    font-size: 12px;
    color: #aab7b8;
}

/* ---- Button overrides ---- */
.primary-btn {
    background: #1a5276 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.primary-btn:hover {
    background: #154360 !important;
}
"""


# ============================================================
# GLOBAL PIPELINE
# ============================================================
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = MedScribePipeline(adapter_path=ADAPTER_PATH)
    return pipeline


# ============================================================
# HANDLER FUNCTIONS
# ============================================================
def load_models(progress=gr.Progress()):
    """Load both MedASR and MedGemma models."""
    pipe = get_pipeline()
    status_parts = []

    try:
        if not pipe.asr_loaded:
            progress(0.1, desc="Loading MedASR...")
            pipe.load_asr()
            status_parts.append("MedASR loaded")

        if not pipe.soap_loaded:
            progress(0.5, desc="Loading MedGemma...")
            pipe.load_soap()
            status_parts.append("MedGemma loaded")

        progress(1.0, desc="Ready")
        return format_status("ready", "Models loaded. Ready for input.")

    except Exception as e:
        return format_status("error", f"Load failed: {str(e)}")


def process_audio(audio_path, progress=gr.Progress()):
    """Full pipeline: audio -> transcript -> SOAP."""
    pipe = get_pipeline()

    if not pipe.fully_loaded:
        return (
            "",
            "",
            format_status("error", "Models not loaded. Click 'Load Models' first."),
            "", "", "",
        )

    if audio_path is None:
        return (
            "",
            "",
            format_status("error", "No audio provided."),
            "", "", "",
        )

    try:
        # Transcribe
        progress(0.2, desc="Transcribing with MedASR...")
        asr_result = pipe.transcribe(audio_path)
        transcript = asr_result["transcript"]

        # Generate SOAP
        progress(0.5, desc="Generating SOAP note...")
        soap_result = pipe.generate_soap(transcript)

        total_time = round(asr_result["time_s"] + soap_result["time_s"], 1)
        progress(1.0, desc="Complete")

        return (
            transcript,
            soap_result["soap_note"],
            format_status("ready", f"Complete in {total_time}s"),
            f'{asr_result["time_s"]}s',
            f'{soap_result["time_s"]}s',
            f'{total_time}s',
        )

    except Exception as e:
        return (
            "",
            "",
            format_status("error", f"Error: {str(e)}"),
            "", "", "",
        )


def process_text(transcript, progress=gr.Progress()):
    """Text-only pipeline: transcript -> SOAP."""
    pipe = get_pipeline()

    if not pipe.soap_loaded:
        return (
            "",
            format_status("error", "MedGemma not loaded. Click 'Load Models' first."),
            "", "",
        )

    if not transcript or not transcript.strip():
        return (
            "",
            format_status("error", "No transcript provided."),
            "", "",
        )

    try:
        progress(0.3, desc="Generating SOAP note...")
        result = pipe.generate_soap(transcript.strip())
        progress(1.0, desc="Complete")

        return (
            result["soap_note"],
            format_status("ready", f'Complete in {result["time_s"]}s'),
            f'{result["time_s"]}s',
            f'{result["word_count"]} words',
        )

    except Exception as e:
        return (
            "",
            format_status("error", f"Error: {str(e)}"),
            "", "",
        )


def format_status(level, message):
    return f'<div class="status-bar {level}">{message}</div>'


# ============================================================
# BUILD UI
# ============================================================
def build_app():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MedScribe",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=["IBM Plex Sans", "system-ui", "sans-serif"],
            font_mono=["IBM Plex Mono", "Consolas", "monospace"],
        ),
    ) as app:

        # Header
        gr.HTML(
            f'<div class="app-header">'
            f"<h1>{APP_TITLE}</h1>"
            f"<p>{APP_SUBTITLE}</p>"
            f"<p style='font-size:12px; color:#aab7b8; margin-top:2px;'>"
            f"MedASR + MedGemma  |  HAI-DEF Pipeline  |  Edge AI</p>"
            f"</div>"
        )

        # Status
        status_html = gr.HTML(
            format_status("", "Models not loaded. Click 'Load Models' to begin."),
        )

        # Load button
        load_btn = gr.Button(
            "Load Models",
            variant="secondary",
            size="sm",
        )
        load_btn.click(fn=load_models, outputs=[status_html])

        # Tabs
        with gr.Tabs():

            # ---- TAB 1: Voice -> SOAP ----
            with gr.TabItem("Voice to SOAP"):
                gr.HTML('<div class="section-label">Audio Input</div>')

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload audio",
                )

                voice_btn = gr.Button(
                    "Transcribe and Generate",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">Transcript (MedASR)</div>')
                voice_transcript = gr.Textbox(
                    label="Transcript",
                    lines=4,
                    interactive=True,
                    placeholder="Transcript will appear here after processing...",
                    show_label=False,
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma)</div>')
                voice_soap = gr.Textbox(
                    label="SOAP Note",
                    lines=16,
                    interactive=False,
                    show_label=False,
                    elem_classes=["soap-output"],
                )

                with gr.Row():
                    voice_asr_time = gr.Textbox(label="ASR Time", interactive=False, scale=1)
                    voice_soap_time = gr.Textbox(label="SOAP Time", interactive=False, scale=1)
                    voice_total_time = gr.Textbox(label="Total Time", interactive=False, scale=1)

                voice_btn.click(
                    fn=process_audio,
                    inputs=[audio_input],
                    outputs=[
                        voice_transcript,
                        voice_soap,
                        status_html,
                        voice_asr_time,
                        voice_soap_time,
                        voice_total_time,
                    ],
                )

            # ---- TAB 2: Text -> SOAP ----
            with gr.TabItem("Text to SOAP"):
                gr.HTML('<div class="section-label">Medical Transcript</div>')

                text_input = gr.Textbox(
                    label="Transcript",
                    lines=6,
                    placeholder="Paste or type a medical encounter transcript here...",
                    show_label=False,
                )

                text_btn = gr.Button(
                    "Generate SOAP Note",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma)</div>')
                text_soap = gr.Textbox(
                    label="SOAP Note",
                    lines=16,
                    interactive=False,
                    show_label=False,
                    elem_classes=["soap-output"],
                )

                with gr.Row():
                    text_gen_time = gr.Textbox(label="Generation Time", interactive=False, scale=1)
                    text_word_count = gr.Textbox(label="Output Size", interactive=False, scale=1)

                text_btn.click(
                    fn=process_text,
                    inputs=[text_input],
                    outputs=[
                        text_soap,
                        status_html,
                        text_gen_time,
                        text_word_count,
                    ],
                )

                # Example buttons
                gr.HTML('<div class="section-label">Example Transcripts</div>')
                gr.Examples(
                    examples=EXAMPLE_TRANSCRIPTS,
                    inputs=[text_input],
                    label="",
                )

            # ---- TAB 3: About ----
            with gr.TabItem("About"):
                gr.Markdown(
                    """
### Architecture

MedScribe is a two-stage pipeline built entirely on Google's
Health AI Developer Foundations (HAI-DEF) ecosystem:

**Stage 1 -- Speech Recognition (MedASR)**
- 105M parameter Conformer model
- Trained on 5,000+ hours of physician dictation
- 5.2% WER on medical dictation (vs 28.2% for Whisper v3 Large)
- Runs on CPU or GPU, processes audio in real time

**Stage 2 -- SOAP Note Generation (MedGemma 1.5 4B)**
- Fine-tuned with LoRA on 712 curated training samples
- Training data generated by GPT-4o with anti-hallucination constraints
- Produces concise, clinically complete SOAP notes (100 words median)
- Zero WNL shortcuts, zero fabricated findings
- Runs on consumer GPU (RTX 5070 Ti, 16GB VRAM)

### Key Metrics

| Metric | Value |
|--------|-------|
| Train loss | 0.828 |
| Val loss | 0.782 |
| Quality score | 90/100 |
| Section completeness | 100% |
| WNL present | 0% |
| Anti-hallucination compliance | 64% use "Not documented" |
| Avg inference time | 24.8s |
| PLAN items | 2-4 per note |

### Clinical Motivation

Existing AI documentation tools generate verbose, textbook-style notes
that physicians must manually edit for brevity. MedScribe generates
the concise shorthand clinicians actually write -- reducing post-generation
editing time while preserving clinical completeness.

### Limitations

- English only (MedASR training data)
- Not validated for clinical use -- research prototype only
- May produce incomplete notes for rare specialties
- Inference speed depends on GPU hardware
- Training data derived from MTSamples (synthetic encounters)

### Competition

MedGemma Impact Challenge -- Kaggle 2026
$100,000 prize pool | Main Track + Edge AI Prize
"""
                )

        # Footer
        gr.HTML(
            '<div class="app-footer">'
            "MedScribe | MedGemma Impact Challenge 2026 | "
            "HAI-DEF: MedASR + MedGemma 1.5 4B | "
            "Research prototype -- not for clinical use"
            "</div>"
        )

    return app


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )