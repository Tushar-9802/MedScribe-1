
# MedScribe -- Clinical Documentation Workstation

Google MedGemma Impact Challenge 2026 | Main Track + Novel Task Prize

---

## The Problem

Physicians spend 40% of patient encounters on documentation. Electronic Health
Records promised efficiency but delivered the opposite -- clinicians now spend
more time typing than examining patients.

AI documentation tools addressed the transcription problem but introduced a new
one: AI-generated notes are bloated, formulaic, and clinically imprecise. A
45-second encounter produces a 200-word wall of text that the physician must
then pare down to the 20 words that actually matter.

A practicing nephrologist who uses Dragon Copilot daily reports:

> "More often than not I have to go and edit the notes and shorten them,
> because they read like textbook lexicon rather than shorthand designed to
> deliver efficient summaries with alacrity."

Editing AI-generated notes often takes longer than writing from scratch. The
tool meant to save time becomes a source of frustration.

## The Solution

MedScribe is a clinical documentation workstation that generates concise,
clinician-ready SOAP notes from voice or text input. Unlike existing tools that
produce verbose textbook prose, MedScribe generates the same tight shorthand
that experienced clinicians write themselves -- abbreviations, focused
assessments, and actionable plans.

The system then layers on six clinical intelligence tools that transform a
static note into an actionable clinical artifact: ICD-10 coding, differential
diagnosis, medication safety checks, patient summaries, documentation
completeness review, and comprehensive patient intake analysis.

MedScribe runs fully offline on consumer hardware. Patient data never leaves
the device.

---

## Features

### Voice to SOAP

Record audio or upload a file of a clinical encounter. MedASR (Google's medical
speech recognition model) transcribes the dictation, and the fine-tuned MedGemma
model converts it to a structured SOAP note in clinical shorthand.

The transcript is displayed in an editable text field. Clinicians can correct
any transcription errors before triggering SOAP generation, ensuring the final
note reflects what was actually said.

Timing metrics are displayed for both the ASR and SOAP generation stages.

### Text to SOAP

Paste any medical encounter transcript and generate a structured SOAP note.
Useful for:

* Converting existing unstructured documentation into SOAP format
* Processing typed dictation or copy-pasted encounter notes
* Demonstrating the model without audio input
* Testing with the five included example transcripts covering cardiology,
  endocrinology, pediatrics, nephrology, and psychiatry

### Clinical Tools

Five post-generation tools powered by MedGemma's instruction-following
capability. These operate on the most recently generated SOAP note, or on any
note pasted into the input field.

**ICD-10 Coding** -- Suggests ICD-10-CM billing codes supported by the
documentation. Each code includes a description and the documentation evidence
that supports it. Reduces coding time and improves reimbursement accuracy.

**Patient Summary** -- Generates a plain-language visit summary suitable for
patient portals, discharge instructions, or after-visit summaries. Translates
medical jargon into clear, understandable language while preserving clinical
accuracy.

**Completeness Check** -- Reviews the SOAP note for documentation gaps:
subjective complaints without corresponding objective findings, diagnoses
without supporting evidence, missing follow-up or monitoring in the plan,
medications without dosage or frequency, and missing allergy reconciliation.

**Differential Diagnosis** -- Produces a ranked differential diagnosis list
with 3-5 conditions. For each diagnosis, provides supporting evidence from the
note and findings that argue against it. Includes at least one uncommon
condition ("zebra") that could be missed in a busy clinical setting.

**Medication Check** -- Reviews all medications mentioned in the note for
drug-drug interactions, contraindications based on documented conditions, and
dosage or frequency concerns. Flags safety issues that may be missed during a
time-pressured encounter.

### Patient Intake Analysis

A structured intake form that accepts:

* Demographics (age, sex, ethnicity)
* Chief complaint and duration
* Medical and surgical history
* Family history
* Current medications and allergies
* Lifestyle factors (smoking, alcohol, exercise, occupation)
* Recent lab results

MedGemma analyzes the complete profile and produces five sections:

* **Risk Assessment** -- Conditions the patient is predisposed to based on
  demographics, history, and hereditary patterns
* **Differential Considerations** -- 3-5 ranked diagnoses for the chief
  complaint, including supporting/refuting evidence and at least one zebra
* **Recommended Screenings** -- Age, sex, and risk-factor appropriate
  preventive screenings, flagging any that are overdue or urgent
* **Red Flags** -- Combinations of symptoms, history, or risk factors that
  warrant urgent attention, including drug-disease interactions
* **Clinical Questions** -- 3-5 specific questions the clinician should ask
  to narrow the differential or uncover hidden risks

The form comes pre-filled with a realistic demonstration case (58-year-old
South Asian male with rapid-onset diabetes, strong family history, and declining
renal function) that showcases the analysis capabilities.

### Model Comparison

Side-by-side comparison of base MedGemma output (LoRA adapter disabled) versus
the fine-tuned model (adapter enabled) on the same transcript. Demonstrates:

* Word count reduction (typically 40-50% shorter)
* Clinical shorthand vs textbook prose
* Focused PLAN items vs over-specified action lists
* Consistent SOAP structure (100% section completeness)

Both outputs display word count, token count, and generation time. The
comparison uses the same model weights, same prompt, same hardware -- the only
difference is the LoRA adapter trained on 712 curated samples.

---

## Quick Start

### Prerequisites

* Python 3.10+
* CUDA 12.1+ compatible GPU (8GB+ VRAM minimum, 16GB recommended)
* HuggingFace account with access to `google/medgemma-4b-it` and `google/medasr`

### Installation

```bash
git clone https://github.com/Tushar-9802/MedScribe-1.git
cd MedScribe-1

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Running

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860`.

1. Click **Load Models** to load MedASR and MedGemma into VRAM
2. Navigate to any tab to begin
3. For Voice to SOAP: record audio or upload a file, click Transcribe & Generate
4. For Text to SOAP: paste a transcript or click an example, click Generate
5. For Clinical Tools: generate a SOAP note first, then click any tool button
6. For Patient Analysis: fill in the form (or use the pre-filled demo), click Run

---

## Quality Metrics

| Metric                | Value                                 |
| --------------------- | ------------------------------------- |
| Quality score         | 90/100                                |
| Section completeness  | 100% (S/O/A/P always present)         |
| Hallucinated findings | 0%                                    |
| WNL shortcuts         | 0% (every finding explicitly stated)  |
| Avg word count        | 104 words (vs 200+ for verbose tools) |
| Avg SOAP inference    | ~25s (RTX 5070 Ti)                    |
| PLAN items per note   | 2-4 (focused, actionable)             |

## Hardware Requirements

| Configuration | GPU VRAM | Notes                              |
| ------------- | -------- | ---------------------------------- |
| Minimum       | 8GB      | 4-bit quantized, MedGemma only     |
| Recommended   | 16GB     | Both models loaded, full pipeline  |
| Development   | 16GB     | RTX 5070 Ti, Windows 11, CUDA 12.8 |

MedASR (~400MB) and MedGemma 4B (4-bit ~3GB) coexist in 16GB VRAM with
headroom for KV cache during generation.

## Limitations

* **English only** -- MedASR was trained on English medical speech
* **Research prototype** -- not validated for clinical use in any jurisdiction
* **Synthetic training data** -- 712 samples generated via GPT-4o, not from
  real clinical encounters
* **Audio quality dependent** -- MedASR CTC output quality varies with
  recording quality, accent, and speaking speed
* **Hardware dependent** -- inference speed scales with GPU capability;
  ~25s on RTX 5070 Ti, slower on less powerful hardware
* **Clinical tools not fine-tuned** -- ICD-10, DDx, and other tools use base
  MedGemma instruction-following, not domain-specific fine-tuning

---

## Project Structure

```
MedScribe-1/
  app.py                          # Gradio UI (main entry point)
  requirements.txt                # Runtime dependencies
  README.md                       # This file
  TECHNICAL.md                    # Technical deep-dive
  src/
    __init__.py
    pipeline.py                   # MedASR + MedGemma orchestration
    inference.py                  # MedGemma loading, generation, stopping criteria
  models/
    checkpoints/
      medgemma_v2_soap/
        final_model/              # LoRA adapter weights
  data/
    processed/                    # Training data
  train_v2.py                     # LoRA fine-tuning script
  evaluate_v2.py                  # Quality evaluation
  generate_soap_gpt4o.py          # Synthetic training data generation
```

## License

MIT License (code). Model weights subject to Google's terms of use.

## Contact

GitHub: [@Tushar-9802](https://github.com/Tushar-9802)
