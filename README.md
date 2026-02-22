
# MedScribe — Clinical Documentation Workstation

Google MedGemma Impact Challenge 2026 | Main Track + Novel Task Prize

---

## Why MedScribe Exists

A physician sees 20 patients a day. For each encounter, they spend an average
of 16 minutes on documentation — 40% of every patient interaction lost to
typing, clicking, and formatting. Across a career, that is roughly 24,000 hours
spent on paperwork instead of patient care.

AI documentation tools were supposed to fix this. They didn't. They replaced
one problem with another: instead of writing notes from scratch, physicians now
spend their time *editing AI-generated notes* that are too long, too formal,
and clinically imprecise. A 45-second encounter produces a 200-word wall of
textbook prose that the physician must pare down to the 20 words that actually
matter.

A practicing nephrologist who uses Dragon Copilot daily:

> "More often than not I have to go and edit the notes and shorten them,
> because they read like textbook lexicon rather than shorthand designed to
> deliver efficient summaries with alacrity."

MedScribe was built to eliminate that editing step entirely.

---

## What MedScribe Changes

### The Note Itself

MedScribe generates SOAP notes in the same concise shorthand that experienced
clinicians write themselves. Not textbook prose. Not verbose summaries.
Clinical shorthand — abbreviations, focused assessments, actionable plans.

| What existing tools produce                                                                                                                                                                           | What MedScribe produces                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| "The patient is a 54-year-old female who presents to the emergency department with a chief complaint of shortness of breath."                                                                         | "54 yo F c/o SOB."                                                                   |
| "Assessment: After careful review of the imaging findings and clinical presentation, the most likely diagnosis is acute pulmonary embolism involving the segmental branches of the right lower lobe." | "Acute segmental PE, right lower lobe."                                              |
| "Plan: It is recommended to initiate anticoagulation therapy. The patient's respiratory status should be closely monitored."                                                                          | "1. Initiate anticoagulation. 2. Monitor resp status and O2 sat. 3. Pulm follow-up." |

A 200-word note becomes 104 words. No clinical information is lost. The
physician reads it, confirms it, and moves on.

### The Workflow

Current AI scribes stop at transcription. MedScribe continues. After generating
the SOAP note, six clinical intelligence tools transform the static note into
an actionable artifact:

* **Billable Diagnoses** — extracts diagnoses supported by documentation for
  ICD-10 coding, with evidence mapping
* **Patient Summary** — plain-language visit summary for patient portals
* **Completeness Check** — flags documentation gaps before the note is signed
* **Differential Diagnosis** — ranked DDx with supporting evidence
* **Medication Check** — interaction and contraindication screening
* **Patient Intake Analysis** — risk stratification, red flags, and screening
  recommendations from structured intake data

Each tool takes 10-18 seconds and operates on the generated SOAP note. The
clinician runs only the tools they need.

### The Privacy Model

MedScribe runs fully offline on consumer hardware. Patient data never leaves
the device. No cloud API calls. No data logging. No third-party access.
In healthcare, this isn't a feature — it's a requirement.

---

## Quantified Impact

### Per Encounter

| Metric                           | Before MedScribe | With MedScribe              | Savings  |
| -------------------------------- | ---------------- | --------------------------- | -------- |
| Documentation time               | 16 min           | ~5 min (review + confirm)   | 11 min   |
| Note editing after AI generation | 3-5 min          | 0 min (note is chart-ready) | 3-5 min  |
| ICD-10 code lookup               | 2-3 min          | 10 sec (auto-extracted)     | ~2.5 min |
| Completeness check (mental)      | 1-2 min          | 15 sec (automated)          | ~1.5 min |
| Patient summary for portal       | 2-3 min          | 18 sec (auto-generated)     | ~2.5 min |

### Per Physician Per Year (20 patients/day, 250 working days)

| Metric                                       | Value                        |
| -------------------------------------------- | ---------------------------- |
| Encounters per year                          | 5,000                        |
| Documentation time saved per encounter       | ~11 min                      |
| Annual documentation time saved              | ~917 hours                   |
| Additional patient encounters possible       | ~1,100 (at 50 min/encounter) |
| Reduced after-hours charting ("pajama time") | Estimated 2-3 hrs/week       |

These are projections based on published documentation burden data (Sinsky et
al., Annals of Internal Medicine, 2016) and the measured difference between
MedScribe's chart-ready notes and the editing time required for verbose AI
output. Actual impact will vary by specialty, EHR system, and practice setting.

### Quality Metrics (Measured)

| Metric                 | Value                                       |
| ---------------------- | ------------------------------------------- |
| Quality score          | 90/100                                      |
| Section completeness   | 100% (S/O/A/P always present)               |
| Hallucinated findings  | 0%                                          |
| WNL shortcuts          | 0% (every finding explicitly stated)        |
| Average word count     | 104 words (vs 200+ for verbose tools)       |
| Average SOAP inference | ~25s (RTX 5070 Ti)                          |
| Anti-hallucination     | "Not documented in source" for missing data |

---

## Architecture

Three models from Google's HAI-DEF ecosystem in a single pipeline:

| Component             | Model                    | Role                             |
| --------------------- | ------------------------ | -------------------------------- |
| Speech Recognition    | MedASR (105M, Conformer) | Medical dictation → text        |
| SOAP Generation       | MedGemma 1.5 4B + LoRA   | Transcript → concise SOAP note  |
| Clinical Intelligence | MedGemma 1.5 4B (base)   | 6 post-generation analysis tools |

MedASR and MedGemma coexist in 16GB VRAM (4-bit quantization). The LoRA
adapter adds 4.2M trainable parameters (0.1% of base) and was fine-tuned on
712 curated samples with anti-hallucination constraints. Total training cost:
$1.28 via GPT-4o API.

For complete technical details — stopping criteria, CTC decoding pipeline,
generation parameters, training configuration, and UI implementation — see
**[**TECHNICAL.md**](**TECHNICAL.md**)**

---

## Features

### Voice to SOAP

Record or upload audio. MedASR transcribes, MedGemma converts to structured
SOAP. The transcript is editable before generation — clinicians can correct
transcription errors before the note is produced.

### Text to SOAP

Paste any medical encounter transcript. Five example transcripts included
covering cardiology, endocrinology, pediatrics, nephrology, and psychiatry.

### Clinical Tools

Five tools operating on any generated or pasted SOAP note:

| Tool                   | What It Does                                     | Time |
| ---------------------- | ------------------------------------------------ | ---- |
| Billable Diagnoses     | Extracts ICD-10-codeable diagnoses with evidence | ~10s |
| Patient Summary        | Plain-language summary for patients              | ~18s |
| Completeness Check     | Documentation gap detection                      | ~15s |
| Differential Diagnosis | Ranked DDx with evidence                         | ~18s |
| Medication Check       | Interaction/contraindication screening           | ~18s |

### Patient Intake Analysis

Structured form accepting demographics, history, medications, lifestyle, and
labs. MedGemma produces risk assessments, differential considerations,
recommended screenings, red flags, and clinical questions. Pre-filled with a
demonstration case (58-year-old South Asian male with rapid-onset diabetes and
declining renal function).

### Model Comparison

Side-by-side base MedGemma vs fine-tuned output on the same transcript.
Demonstrates the concrete value of LoRA fine-tuning: 46% shorter notes,
clinical shorthand, zero hallucination, 100% section completeness.

---

## Quick Start

### Prerequisites

* Python 3.10+
* CUDA 12.1+ compatible GPU (8GB minimum, 16GB recommended)
* HuggingFace access to `google/medgemma-4b-it` and `google/medasr`

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

Opens at `http://localhost:7860`. Click **Load Models** to load MedASR and
MedGemma into VRAM, then navigate to any tab.

---

## Limitations

* **English only** — MedASR training constraint
* **Research prototype** — not validated for clinical use
* **Synthetic training data** — 712 samples, not real clinical encounters
* **Hardware dependent** — ~25s inference on RTX 5070 Ti
* **Clinical tools not fine-tuned** — use base model instruction-following

---

## Project Structure

```
MedScribe-1/
  app.py                          # Gradio UI (main entry point)
  requirements.txt                # Runtime dependencies
  README.md                       # This file
  TECHNICAL.md                    # Architecture and implementation deep-dive
  src/
    __init__.py
    pipeline.py                   # MedASR + MedGemma orchestration
    inference.py                  # Model loading, generation, stopping criteria
  models/checkpoints/
    medgemma_v2_soap/final_model/ # LoRA adapter weights
  data/processed/                 # Training data
  train_v2.py                     # LoRA fine-tuning script
  evaluate_v2.py                  # Quality evaluation
  generate_soap_gpt4o.py          # Training data generation
```

## License

MIT License (code). Model weights subject to Google's terms of use.

## Documentation and LInks

#### Adaptors/Safetensors: [Huggingface](https://huggingface.co/Tushar9802/MedScribe-soap-lora)

#### Dataset: [Kaggle](https://www.kaggle.com/datasets/tusharjaju/MedScribe-soap-training-data-712-curated-samples)

#### Technical Documentation: [TECHNICAL.md](TECHNICAL.md)

## Contact

GitHub: [@Tushar-9802](https://github.com/Tushar-9802)
