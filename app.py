"""
MedScribe — Clinical Documentation Workstation
================================================
Voice-to-SOAP + Clinical Intelligence + Patient Intake Analysis
powered by Google HAI-DEF models (MedASR + MedGemma).

Tabs:
1. Voice to SOAP      — MedASR transcription + MedGemma SOAP generation
2. Text to SOAP       — Paste transcript, generate SOAP
3. Clinical Tools     — ICD-10, patient summary, completeness, DDx, med check
4. Patient Analysis   — Structured intake → risk/screening/red flags
5. About              — Architecture, metrics, methodology

Launch: python app.py
"""
import os
import time
import html as html_mod
import gradio as gr
from src.pipeline import MedScribePipeline

# ============================================================
# CONFIGURATION
# ============================================================
ADAPTER_PATH = "./models/checkpoints/medgemma_v2_soap/final_model"

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
        "with minimal improvement. PHQ-9 score 14, GAD-7 score 12."
    ],
]


# ============================================================
# CUSTOM CSS — Modern light medical UI
# ============================================================
CUSTOM_CSS = """
/* ── Layout Stability (Anti-Stutter) ── */
html { 
    overflow-y: scroll !important; /* Forces scrollbar to prevent lateral width snapping */
}

/* Fix audio component height to prevent vertical jumping upon upload */
[data-testid="audio"] {
    min-height: 160px !important; 
}

/* ── Base ── */
.gradio-container {
    max-width: 1060px !important;
    margin: 0 auto !important;
    font-family: "Inter", "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
    background: #f8fafb !important;
    transition: none !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 28px 0 18px 0;
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 28px;
    font-weight: 800;
    color: #0f766e;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header .subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 4px 0 0 0;
    font-weight: 400;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-block;
    min-height: 32px;
    max-height: 32px;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    color: #64748b;
    transition: all 0.2s ease !important;
}
.status-pill.ready { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }

/* ── Section Labels ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 8px 0;
}

/* ── SOAP & Tool Output ── */
.soap-card, .tool-output, .analysis-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    color: #1e293b !important;
}
.soap-card {
    padding: 20px 24px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 13px;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* ── Buttons ── */
.primary-btn {
    background: #0d9488 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ══ ANTI-STUTTER: Suppress Loading Overlays ══ */
.pending, .generating, .translucent {
    opacity: 1 !important;
    animation: none !important;
    border: none !important;
}
.eta-bar, .progress-bar, .loader, .svelte-spinner {
    display: none !important;
    height: 0 !important;
}
div[data-testid] .wrap {
    transition: none !important;
    border: none !important;
    box-shadow: none !important;
}
/* ── Analysis result cards ── */
.analysis-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    color: #1e293b !important;
    line-height: 1.65;
    font-size: 13px;
}

.analysis-card-header {
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Triage Colors ── */
.card-risk { border-left: 4px solid #f59e0b; } /* Amber */
.card-risk .analysis-card-header { color: #b45309; }

.card-ddx { border-left: 4px solid #6366f1; } /* Indigo */
.card-ddx .analysis-card-header { color: #4338ca; }

.card-screening { border-left: 4px solid #0d9488; } /* Teal */
.card-screening .analysis-card-header { color: #0f766e; }

.card-redflags { border-left: 4px solid #ef4444; } /* Red */
.card-redflags .analysis-card-header { color: #dc2626; }

.card-questions { border-left: 4px solid #8b5cf6; } /* Purple */
.card-questions .analysis-card-header { color: #7c3aed; }

/* ── Force light mode on all Gradio elements ── */
.gradio-container, .gradio-container *,
.main, .contain, .tabs, .tabitem {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafb !important;
    --block-background-fill: #ffffff !important;
    --input-background-fill: #ffffff !important;
    --body-background-fill: #f8fafb !important;
    --color-accent: #0d9488 !important;
}
/* Force text inputs to be light */
textarea, input[type="text"], .wrap {
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
}
/* Force dropdown/select to be light */
.secondary-wrap, .wrap-inner {
    background: #ffffff !important;
}
/* Ensure label text is visible */
label, .label-wrap, span.svelte-1gfkn6j {
    color: #475569 !important;
}
/* Audio component light background */
.audio-container, [data-testid="audio"] {
    background: #ffffff !important;
}
/* ── Nuclear light mode — override OS dark preference at browser level ── */
:root, html, body {
    color-scheme: light only !important;
}
.dark, .dark * {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafb !important;
    --block-background-fill: #ffffff !important;
    --input-background-fill: #ffffff !important;
    --body-background-fill: #f8fafb !important;
    --color-accent: #0d9488 !important;
    --body-text-color: #1e293b !important;
    --block-label-text-color: #475569 !important;
    --block-title-text-color: #1e293b !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    color: #1e293b !important;
    background-color: #ffffff !important;
}
.dark textarea, .dark input[type="text"], .dark .wrap {
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
}
.dark label, .dark .label-wrap {
    color: #475569 !important;
}
.dark .tab-nav button { color: #475569 !important; }
.dark .tab-nav button.selected { color: #0f766e !important; }
.dark .tab-nav { border-bottom-color: #e2e8f0 !important; }
.dark .app-footer { color: #94a3b8 !important; border-top-color: #e2e8f0 !important; }

"""


# ============================================================
# GLOBAL STATE
# ============================================================
pipeline = None
last_soap_note = ""


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = MedScribePipeline(adapter_path=ADAPTER_PATH)
    return pipeline


def _status(level, msg):
    return f'<div class="status-pill {level}">{msg}</div>'


def _tool_html(text):
    """Wrap clinical tool output in styled card."""
    if not text:
        return ""
    safe = html_mod.escape(text)
    safe = safe.replace('\n', '<br>')
    return f'<div class="tool-output">{safe}</div>'


def _soap_html(soap_text):
    """
    Convert SOAP text to styled HTML with section headers.
    """
    if not soap_text:
        return ""

    import re

    sections = []
    section_names = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]

    # Split by section headers
    pattern = r'(\*\*(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):\*\*|(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):)'
    parts = re.split(pattern, soap_text)

    # Parse into (header, content) pairs
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        # Check if this part is a header
        clean_header = part.replace("**", "").replace(":", "").strip().upper()
        if clean_header in section_names:
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((clean_header, content))
            i += 2
        else:
            # Leading text before first section (rare)
            if part and not sections:
                sections.append(("", part))
            i += 1

    # Build HTML
    html_parts = []
    for header, content in sections:
        safe_content = html_mod.escape(content).replace('\n', '<br>')
        # Style numbered list items
        safe_content = re.sub(
            r'^(\d+)\.\s',
            r'<span style="color:#0f766e; font-weight:600;">\1.</span> ',
            safe_content,
            flags=re.MULTILINE,
        )
        if header:
            html_parts.append(
                f'<div class="soap-section">'
                f'<div class="soap-section-header">{header}</div>'
                f'{safe_content}'
                f'</div>'
            )
        else:
            html_parts.append(f'<div class="soap-section">{safe_content}</div>')

    inner = "\n".join(html_parts)
    return f'<div class="soap-card">{inner}</div>'


# ============================================================
# HANDLER: Load Models
# ============================================================
def load_models():
    pipe = get_pipeline()
    try:
        if not pipe.asr_loaded:
            pipe.load_asr()
        if not pipe.soap_loaded:
            pipe.load_soap()
        return _status("ready", "✓ Models loaded — ready for input")
    except Exception as e:
        return _status("error", f"Load failed: {str(e)}")


# ============================================================
# HANDLER: Voice to SOAP
# ============================================================
def process_audio(audio_path):
    global last_soap_note
    pipe = get_pipeline()

    if not pipe.fully_loaded:
        return ("", "", _status("error", "Load models first"), "", "", "")
    if audio_path is None:
        return ("", "", _status("error", "No audio provided"), "", "", "")

    try:
        asr_result = pipe.transcribe(audio_path)
        transcript = asr_result["transcript"]

        soap_result = pipe.generate_soap(transcript)
        last_soap_note = soap_result["soap_note"]
        total = round(asr_result["time_s"] + soap_result["time_s"], 1)

        return (
            transcript,
            _soap_html(soap_result["soap_note"]),
            _status("ready", f"✓ Done in {total}s — ASR {asr_result['time_s']}s · SOAP {soap_result['time_s']}s"),
            f'{asr_result["time_s"]}s',
            f'{soap_result["time_s"]}s',
            f'{total}s',
        )
    except Exception as e:
        return ("", "", _status("error", f"Error: {str(e)}"), "", "", "")


# ============================================================
# HANDLER: Text to SOAP
# ============================================================
def process_text(transcript):
    global last_soap_note
    pipe = get_pipeline()

    if not pipe.soap_loaded:
        return ("", _status("error", "Load models first"), "", "")
    if not transcript or not transcript.strip():
        return ("", _status("error", "No transcript provided"), "", "")

    try:
        result = pipe.generate_soap(transcript.strip())
        last_soap_note = result["soap_note"]

        return (
            _soap_html(result["soap_note"]),
            _status("ready", f'✓ Done in {result["time_s"]}s · {result["word_count"]} words'),
            f'{result["time_s"]}s',
            f'{result["word_count"]} words',
        )
    except Exception as e:
        return ("", _status("error", f"Error: {str(e)}"), "", "")


# ============================================================
# HANDLER: Clinical Tools (generic wrapper)
# ============================================================
def _run_tool(tool_fn, tool_name, soap_note_input):
    """Generic wrapper for all clinical tools."""
    pipe = get_pipeline()
    note = soap_note_input.strip() if soap_note_input else last_soap_note
    if not note:
        return _status("error", "No SOAP note available — generate one first"), ""
    if not pipe.soap_loaded:
        return _status("error", "Model not loaded"), ""
    try:
        result = tool_fn(note)
        return (
            _status("ready", f'✓ {tool_name} — {result["time_s"]}s'),
            _tool_html(result["text"]),
        )
    except Exception as e:
        return _status("error", str(e)), ""


def run_icd10(soap_input):
    return _run_tool(get_pipeline().suggest_icd10, "ICD-10 coding", soap_input)


def run_patient_summary(soap_input):
    return _run_tool(get_pipeline().patient_summary, "Patient summary", soap_input)


def run_completeness(soap_input):
    return _run_tool(get_pipeline().completeness_check, "Documentation review", soap_input)


def run_differential(soap_input):
    return _run_tool(get_pipeline().differential_diagnosis, "Differential diagnosis", soap_input)


def run_med_check(soap_input):
    return _run_tool(get_pipeline().medication_check, "Medication review", soap_input)


# ============================================================
# HANDLER: Patient Intake Analysis
# ============================================================
def _format_analysis_html(raw_text):
    """Parse analysis output into color-coded cards."""
    if not raw_text:
        return ""

    sections = {
        "RISK ASSESSMENT": ("card-risk", "Risk Assessment"),
        "DIFFERENTIAL CONSIDERATIONS": ("card-ddx", "Differential Considerations"),
        "RECOMMENDED SCREENINGS": ("card-screening", "Recommended Screenings"),
        "RED FLAGS": ("card-redflags", "Red Flags"),
        "CLINICAL QUESTIONS": ("card-questions", "Questions for Clinician"),
    }

    # Try to split into sections
    import re
    cards = []
    remaining = raw_text

    for key, (css_class, display_name) in sections.items():
        pattern = rf'{key}:?\s*\n?(.*?)(?=(?:{"|".join(sections.keys())}):?\s*\n|$)'
        match = re.search(pattern, remaining, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content:
                safe = html_mod.escape(content).replace('\n', '<br>')
                cards.append(
                    f'<div class="analysis-card {css_class}">'
                    f'<div class="analysis-card-header">{display_name}</div>'
                    f'{safe}'
                    f'</div>'
                )

    if cards:
        return "\n".join(cards)

    # Fallback: single card if parsing fails
    safe = html_mod.escape(raw_text).replace('\n', '<br>')
    return f'<div class="analysis-card card-risk">{safe}</div>'


def run_patient_analysis(age, sex, ethnicity, chief_complaint, duration,
                         medical_history, surgical_history, family_history,
                         allergies, current_meds, smoking, alcohol,
                         exercise, occupation, recent_labs):
    pipe = get_pipeline()
    if not pipe.soap_loaded:
        return _status("error", "Load models first"), ""

    # Validate minimum input
    if not chief_complaint or not chief_complaint.strip():
        return _status("error", "Chief complaint is required"), ""

    # Build structured patient profile
    profile_parts = []
    if age:
        profile_parts.append(f"Age: {age}")
    if sex:
        profile_parts.append(f"Sex: {sex}")
    if ethnicity:
        profile_parts.append(f"Ethnicity: {ethnicity}")
    profile_parts.append(f"Chief Complaint: {chief_complaint.strip()}")
    if duration:
        profile_parts.append(f"Duration: {duration}")
    if medical_history:
        profile_parts.append(f"Medical History: {medical_history.strip()}")
    if surgical_history:
        profile_parts.append(f"Surgical History: {surgical_history.strip()}")
    if family_history:
        profile_parts.append(f"Family History: {family_history.strip()}")
    if allergies:
        profile_parts.append(f"Allergies: {allergies.strip()}")
    if current_meds:
        profile_parts.append(f"Current Medications: {current_meds.strip()}")
    if smoking:
        profile_parts.append(f"Smoking: {smoking}")
    if alcohol:
        profile_parts.append(f"Alcohol: {alcohol}")
    if exercise:
        profile_parts.append(f"Exercise: {exercise}")
    if occupation:
        profile_parts.append(f"Occupation: {occupation}")
    if recent_labs:
        profile_parts.append(f"Recent Labs: {recent_labs.strip()}")

    patient_profile = "\n".join(profile_parts)

    prompt = (
        "You are a clinical decision support assistant. A patient presents with the "
        "following profile. Analyze this comprehensively and provide insights a busy "
        "clinician might miss.\n\n"
        f"PATIENT PROFILE:\n{patient_profile}\n\n"
        "Be concise. Use 2-3 bullet points per section."
        "Provide your analysis in exactly these 5 sections:\n\n"
        "RISK ASSESSMENT:\n"
        "Based on demographics, history, and family history, list conditions this "
        "patient is predisposed to. Consider hereditary patterns and age/sex-specific risks.\n\n"
        "DIFFERENTIAL CONSIDERATIONS:\n"
        "For the chief complaint, list 3-5 differential diagnoses ranked by likelihood. "
        "Include at least one uncommon condition that could be missed ('zebra'). "
        "For each, note supporting and refuting evidence from the profile.\n\n"
        "RECOMMENDED SCREENINGS:\n"
        "Based on age, sex, history, and risk factors, list preventive screenings "
        "this patient should have. Note if any are overdue or especially urgent.\n\n"
        "RED FLAGS:\n"
        "Identify any combinations of symptoms, history, or risk factors that "
        "warrant urgent attention. Flag drug-disease interactions if medications listed.\n\n"
        "CLINICAL QUESTIONS:\n"
        "List 3-5 specific questions the clinician should ask this patient to "
        "narrow the differential or uncover hidden risks."
    )

    try:
        result = pipe.soap_gen.generate_freeform(prompt, max_new_tokens=500)
        return (
            _status("ready", f'✓ Analysis complete — {result["time_s"]}s'),
            _format_analysis_html(result["text"]),
        )
    except Exception as e:
        return _status("error", str(e)), ""


# ============================================================
# BUILD UI
# ============================================================
def build_app():
    with gr.Blocks(title="MedScribe") as app:
        # Force light mode — prevent Gradio from ever switching to dark
        gr.HTML(
            '<script>'
            'document.documentElement.classList.remove("dark");'
            'document.body.classList.remove("dark");'
            'new MutationObserver(function(m){'
            '  document.documentElement.classList.remove("dark");'
            '  document.body.classList.remove("dark");'
            '}).observe(document.documentElement, {attributes:true, attributeFilter:["class"]});'
            '</script>',
            visible=False,
        )
        # ── Header ──
        gr.HTML(
            '<div class="app-header">'
            '<h1>⚕ MedScribe</h1>'
            '<p class="subtitle">Clinical Documentation Workstation</p>'
            '<p class="tagline">MedASR + MedGemma · HAI-DEF Pipeline · Voice-to-SOAP + Clinical Intelligence</p>'
            '</div>'
        )

        # ── Status + Load ──
        status_html = gr.HTML(_status("", "Click Load Models to begin"))
        load_btn = gr.Button("Load Models", variant="secondary", size="sm")
        load_btn.click(fn=load_models, outputs=[status_html], show_progress="hidden")

        with gr.Tabs():

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: Voice to SOAP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🎙 Voice to SOAP"):
                gr.HTML('<div class="section-label">Audio Input</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload clinical dictation",
                    container = False
                )
                voice_btn = gr.Button(
                    "Transcribe & Generate SOAP",
                    variant="primary", elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">Transcript (MedASR)</div>')
                voice_transcript = gr.Textbox(
                    lines=4, interactive=True, show_label=False,
                    placeholder="Transcript appears here — editable before regeneration",
                )
                voice_regen_btn = gr.Button(
                    "↻ Regenerate from Edited Transcript",
                    variant="secondary", size="sm",
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma · Fine-tuned)</div>')
                voice_soap = gr.HTML()

                with gr.Row():
                    voice_asr_time = gr.Textbox(label="ASR", interactive=False, scale=1)
                    voice_soap_time = gr.Textbox(label="SOAP", interactive=False, scale=1)
                    voice_total_time = gr.Textbox(label="Total", interactive=False, scale=1)

                voice_btn.click(
                    fn=process_audio,
                    inputs=[audio_input],
                    outputs=[voice_transcript, voice_soap, status_html,
                             voice_asr_time, voice_soap_time, voice_total_time],
                    show_progress="hidden",
                )

                def regen_from_transcript(transcript):
                    result = process_text(transcript)
                    return (result[0], result[1], "", result[2], "")

                voice_regen_btn.click(
                    fn=regen_from_transcript,
                    inputs=[voice_transcript],
                    outputs=[voice_soap, status_html,
                             voice_asr_time, voice_soap_time, voice_total_time],
                    show_progress="hidden",
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: Text to SOAP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📝 Text to SOAP"):
                gr.HTML('<div class="section-label">Medical Transcript</div>')
                text_input = gr.Textbox(
                    lines=6, show_label=False,
                    placeholder="Paste or type a medical encounter transcript...",
                )
                text_btn = gr.Button(
                    "Generate SOAP Note",
                    variant="primary", elem_classes=["primary-btn"],
                )

                gr.HTML('<div class="section-label">SOAP Note (MedGemma · Fine-tuned)</div>')
                text_soap = gr.HTML()

                with gr.Row():
                    text_gen_time = gr.Textbox(label="Generation Time", interactive=False, scale=1)
                    text_word_count = gr.Textbox(label="Word Count", interactive=False, scale=1)

                text_btn.click(
                    fn=process_text,
                    inputs=[text_input],
                    outputs=[text_soap, status_html, text_gen_time, text_word_count],
                    show_progress="hidden",
                )

                gr.HTML('<div class="section-label">Example Transcripts</div>')
                gr.Examples(examples=EXAMPLE_TRANSCRIPTS, inputs=[text_input], label="")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: Clinical Tools
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🔬 Clinical Tools"):
                gr.HTML(
                    '<div class="section-label">Clinical Intelligence (Base MedGemma)</div>'
                    '<p style="font-size:12px; color:#64748b; margin-bottom:14px;">'
                    'Analyze SOAP notes with MedGemma\'s instruction-following capability. '
                    'Generate a note first, or paste one below.</p>'
                )

                tools_soap_input = gr.Textbox(
                    lines=8, show_label=False,
                    placeholder="Paste a SOAP note here, or generate one in Voice/Text tabs. "
                    "The last generated note is used if this is empty.",
                )

                with gr.Row():
                    icd_btn = gr.Button("ICD-10 Codes", variant="secondary", scale=1)
                    summary_btn = gr.Button("Patient Summary", variant="secondary", scale=1)
                    complete_btn = gr.Button("Completeness Check", variant="secondary", scale=1)

                with gr.Row():
                    ddx_btn = gr.Button("Differential Dx", variant="secondary", scale=1)
                    med_btn = gr.Button("Medication Check", variant="secondary", scale=1)

                tools_status = gr.HTML(_status("", "Select a tool above"))
                gr.HTML('<div class="section-label">Result</div>')
                tools_output = gr.HTML()

                icd_btn.click(fn=run_icd10, inputs=[tools_soap_input],
                              outputs=[tools_status, tools_output], show_progress="hidden")
                summary_btn.click(fn=run_patient_summary, inputs=[tools_soap_input],
                                  outputs=[tools_status, tools_output], show_progress="hidden")
                complete_btn.click(fn=run_completeness, inputs=[tools_soap_input],
                                   outputs=[tools_status, tools_output], show_progress="hidden")
                ddx_btn.click(fn=run_differential, inputs=[tools_soap_input],
                              outputs=[tools_status, tools_output], show_progress="hidden")
                med_btn.click(fn=run_med_check, inputs=[tools_soap_input],
                              outputs=[tools_status, tools_output], show_progress="hidden")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 4: Patient Analysis
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🩺 Patient Analysis"):
                gr.HTML(
                    '<div class="section-label">Patient Intake Analysis (Base MedGemma)</div>'
                    '<p style="font-size:12px; color:#64748b; margin-bottom:14px;">'
                    'Enter patient details for comprehensive risk assessment, differential '
                    'diagnosis, screening recommendations, and red flag detection. '
                    'MedGemma analyzes the complete profile to surface insights a busy '
                    'clinician might miss.</p>'
                )

                # Demographics row
                gr.HTML('<div class="section-label">Demographics</div>')
                with gr.Row():
                    pa_age = gr.Textbox(label="Age", placeholder="e.g. 54", value="58", scale=1)
                    pa_sex = gr.Dropdown(
                        label="Sex", choices=["", "Male", "Female", "Other"],
                        value="Male", scale=1,
                    )
                    pa_ethnicity = gr.Textbox(
                        label="Ethnicity", placeholder="e.g. South Asian", value="South Asian", scale=1,
                    )

                # Current visit
                gr.HTML('<div class="section-label">Current Visit</div>')
                pa_complaint = gr.Textbox(
                    label="Chief Complaint",
                    placeholder="e.g. Persistent fatigue and unexplained weight loss for 3 months",
                    value="Persistent fatigue, unintentional 15-lb weight loss over 4 months, increased thirst and frequent urination",
                    lines=2,
                )
                pa_duration = gr.Textbox(
                    label="Duration / Onset",
                    value="4 months, gradual onset, worsening over past 6 weeks",
                )

                # History
                gr.HTML('<div class="section-label">Medical & Surgical History</div>')
                pa_medical = gr.Textbox(
                    label="Medical History",
                    placeholder="e.g. Type 2 diabetes (10 yrs), hypertension, GERD",
                    value="Hypertension (12 yrs), hyperlipidemia, prediabetes (diagnosed 2 yrs ago), GERD, obesity (BMI 31)",
                    lines=2,
                )
                pa_surgical = gr.Textbox(
                    label="Surgical History",
                    value="Cholecystectomy (2018), right inguinal hernia repair (2021)",
                )

                # Family history
                gr.HTML('<div class="section-label">Family History</div>')
                pa_family = gr.Textbox(
                    label="Family History",
                    placeholder="e.g. Father: MI at 52, T2DM. Mother: breast cancer at 61. "
                    "Sister: Hashimoto's thyroiditis. Paternal grandfather: colon cancer.",
                    value="Father: MI at 55, T2DM. Mother: T2DM, CKD stage 4. Brother: T2DM diagnosed at 42. Paternal uncle: pancreatic cancer at 63.",
                    lines=3,
                )

                # Medications & allergies
                gr.HTML('<div class="section-label">Medications & Allergies</div>')
                with gr.Row():
                    pa_meds = gr.Textbox(
                        label="Current Medications",
                        placeholder="e.g. Metformin 1000mg BID, Lisinopril 20mg daily, Omeprazole 20mg",
                        value="Lisinopril 20mg daily, Atorvastatin 40mg daily, Omeprazole 20mg daily, Metformin 500mg BID (started 6 months ago)",
                        lines=2, scale=2,
                    )
                    pa_allergies = gr.Textbox(
                        label="Allergies",
                        placeholder="e.g. Penicillin (rash), Sulfa (anaphylaxis)",
                        value="ACE inhibitor cough (switched from enalapril), Sulfa (rash)",
                        lines=2, scale=1,
                    )

                # Lifestyle
                gr.HTML('<div class="section-label">Lifestyle & Social</div>')
                with gr.Row():
                    pa_smoking = gr.Dropdown(
                        label="Smoking",
                        choices=["", "Never", "Former", "Current — light", "Current — heavy"],
                        value="Former", scale=1,
                    )
                    pa_alcohol = gr.Dropdown(
                        label="Alcohol",
                        choices=["", "None", "Social", "Moderate", "Heavy"],
                        value="Social", scale=1,
                    )
                    pa_exercise = gr.Dropdown(
                        label="Exercise",
                        choices=["", "Sedentary", "Light", "Moderate", "Active"],
                        value="Sedentary", scale=1,
                    )
                pa_occupation = gr.Textbox(
                    label="Occupation", placeholder="e.g. Office worker, construction",
                    value="Long-haul truck driver",
                )

                # Labs
                gr.HTML('<div class="section-label">Recent Labs (Optional)</div>')
                pa_labs = gr.Textbox(
                    label="Recent Lab Results",
                    placeholder="e.g. A1C 7.4%, Cr 1.1, eGFR 68, TSH 6.8, CBC normal",
                    value="A1C 9.2% (was 6.3% one year ago), FBG 243 mg/dL, Cr 1.4, eGFR 52, microalbumin/Cr ratio 85, LDL 142, triglycerides 310, TSH 5.8, CBC wnl",
                    lines=2,
                )

                # Run button
                analyze_btn = gr.Button(
                    "⚡ Run Comprehensive Analysis",
                    variant="primary", elem_classes=["primary-btn"],
                )

                analysis_status = gr.HTML(_status("", "Fill in patient details and click Analyze"))
                gr.HTML('<div class="section-label">Analysis Results</div>')
                analysis_output = gr.HTML()

                analyze_btn.click(
                    fn=run_patient_analysis,
                    inputs=[pa_age, pa_sex, pa_ethnicity, pa_complaint, pa_duration,
                            pa_medical, pa_surgical, pa_family,
                            pa_allergies, pa_meds, pa_smoking, pa_alcohol,
                            pa_exercise, pa_occupation, pa_labs],
                    outputs=[analysis_status, analysis_output],
                    show_progress="hidden",
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 5: About
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("ℹ About"):
                gr.Markdown("""
### MedScribe — Clinical Documentation Workstation

**Problem**: AI documentation tools generate verbose, textbook-style notes.
A nephrologist reports: *"More often than not I have to go and edit the notes
and shorten them, because they read like textbook lexicon rather than shorthand
designed to deliver efficient summaries with alacrity."*

**Solution**: MedScribe generates concise clinical shorthand using a fine-tuned
MedGemma model, then provides clinical intelligence tools and patient risk
analysis for the complete documentation workflow.

---

### Architecture — Three HAI-DEF Models, One Pipeline

| Component | Model | Role |
|-----------|-------|------|
| Speech Recognition | MedASR (105M, Conformer) | Medical dictation → text, 5.2% WER |
| SOAP Generation | MedGemma 1.5 4B (LoRA) | Concise structured notes (~100 words) |
| Clinical Intelligence | MedGemma 1.5 4B (base) | ICD-10, DDx, risk analysis, screening |

### Clinical Intelligence — 6 Tools

| Tool | Function |
|------|----------|
| ICD-10 Coding | Billing codes from SOAP documentation |
| Patient Summary | Plain-language visit summary |
| Completeness Check | Documentation gap detection |
| Differential Diagnosis | Ranked DDx with evidence |
| Medication Check | Interactions, contraindications, dosage |
| **Patient Intake Analysis** | **Risk stratification, red flags, screening, zebra detection** |

### Training

- 712 curated samples via GPT-4o with anti-hallucination constraints
- LoRA fine-tuning (rank 16, alpha 32) on MedGemma 1.5 4B
- Validation loss 0.782 < Train loss 0.828 (no overfitting)

### Metrics

| Metric | Value |
|--------|-------|
| Quality score | 90/100 |
| Section completeness | 100% (S/O/A/P) |
| Hallucinated findings | 0% |
| Avg word count | 104 words |
| Avg inference | ~25s |

### Deployment

- Consumer GPU (RTX 5070 Ti, 16GB VRAM)
- 4-bit quantization (NF4) — fully offline
- MedASR + MedGemma coexist in 16GB VRAM

### Limitations

- English only · Research prototype · Synthetic training data
- Clinical tools use base model instruction-following
- Not validated for clinical use

---

MedGemma Impact Challenge 2026 | Main Track + Novel Task Prize
""")

        # ── Footer ──
        gr.HTML(
            '<div class="app-footer">'
            "MedScribe · MedGemma Impact Challenge 2026 · "
            "HAI-DEF: MedASR + MedGemma 1.5 4B (fine-tuned + base) · "
            "Research prototype — not for clinical use"
            "</div>"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.slate,
        ),
    )