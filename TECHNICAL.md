
# MedScribe — Technical Documentation

Architecture, model details, training methodology, inference pipeline, and
implementation decisions. For clinical context and user-facing documentation,
see [README.md](README.md)

---

## Architecture Overview

MedScribe uses three distinct capabilities from Google's HAI-DEF ecosystem
in a single pipeline:

```
[Audio Input]
      │
[MedASR — 105M Conformer, CTC decode]
      │
[Manual CTC Collapse + 10-step Post-processing]
      │
[Clean Transcript — editable by clinician]
      │
[MedGemma 1.5 4B + LoRA adapter — 4-bit NF4]
      │
[SOAP Stopping Criteria + Post-processing]
      │
[Structured SOAP Note]
      │
[MedGemma 1.5 4B — base instruction-following]
      │
[Freeform Stopping Criteria + Post-processing]
      │
[Clinical Intelligence Tools (6)]
```

| Component             | Model                  | Parameters | VRAM     | Role                                     |
| --------------------- | ---------------------- | ---------- | -------- | ---------------------------------------- |
| Speech Recognition    | MedASR (Conformer)     | 105M       | ~400MB   | Medical dictation to text, 5.2% WER      |
| SOAP Generation       | MedGemma 1.5 4B (LoRA) | 4B + 4.2M  | ~3GB     | Concise structured notes (~100 words)    |
| Clinical Intelligence | MedGemma 1.5 4B (base) | 4B         | (shared) | DDx, risk analysis, screening, summaries |

The SOAP generation and clinical intelligence tools share the same loaded model
instance. The LoRA adapter is active for all operations. Clinical tools use the
model's preserved instruction-following capability through the LoRA layers.
For the Model Comparison feature, the adapter is temporarily disabled via
PEFT's `disable_adapter()` context manager.

---

## MedASR Pipeline

### Model

MedASR is a 105M parameter Conformer-based CTC model from Google's HAI-DEF
collection, trained on medical speech with a reported 5.2% word error rate on
medical dictation.

Model ID: `google/medasr`

### CTC Decoding

MedASR uses CTC (Connectionist Temporal Classification) output, which requires
post-decoding collapse. The pipeline uses manual CTC decoding rather than the
processor's built-in decoder for reliability:

1. Forward pass produces logit matrix (time steps × vocabulary)
2. Argmax across vocabulary dimension gives predicted token IDs per time step
3. Manual collapse: remove consecutive duplicate IDs, then remove blank tokens
4. Map remaining token IDs to characters using the inverse vocabulary

```python
def _manual_ctc_decode(token_ids, blank_id, id_to_token):
    result = []
    prev_id = None
    for token_id in token_ids:
        if token_id != prev_id and token_id != blank_id:
            result.append(id_to_token.get(token_id, ""))
        prev_id = token_id
    return "".join(result)
```

### Transcript Post-processing

Raw CTC output requires extensive cleanup. The pipeline applies a 10-step
post-processing chain:

1. **CTC stutter-killer** — Collapse 3+ identical consecutive characters to 1.
   Preserves valid doubles (ll, ee, ss) but eliminates CTC stutters.
   Regex: `(.)\1{2,}` → `\1`
2. **SentencePiece boundary** — Replace `▁` (U+2581) with space.
3. **Special token removal** — Strip `<s>`, `</s>`, `<epsilon>`, `<pad>`, `<unk>`.
4. **Punctuation token mapping** — Convert `{period}` → `.`, `{comma}` → `,`,
   `{new paragraph}` → `\n`, etc. Handles stuttered variants like `{{period}}`.
5. **Bracket marker removal** — Strip `[EXAM TYPE]`, `[INDICATION]`, etc.
6. **Brace cleanup** — Remove remaining `{` and `}` characters.
7. **Punctuation spacing** — Fix `word .` → `word.` and `word.Next` → `word. Next`.
8. **Whitespace collapse** — Normalize multiple spaces and newlines.
9. **Artifact trimming** — Remove leading/trailing punctuation artifacts.
10. **Sentence capitalization** — Capitalize first character and post-sentence characters.

### Audio Preprocessing

* Loaded via librosa at 16kHz mono (explicit `sr=16000, mono=True`)
* Amplitude normalization via `librosa.util.normalize()`
* No additional noise reduction or VAD applied

---

## MedGemma SOAP Generation

### Model Loading

Base model: `google/medgemma-4b-it` (instruction-tuned variant)

Quantization:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Reduces VRAM from ~8GB (FP16) to ~3GB. The LoRA adapter is loaded via PEFT,
and a warmup generation (10 tokens) runs at load time to compile CUDA kernels.

### LoRA Adapter

| Parameter        | Value                |
| ---------------- | -------------------- |
| Rank             | 16                   |
| Alpha            | 32                   |
| Dropout          | 0.1                  |
| Target modules   | All attention layers |
| Trainable params | ~4.2M (0.1% of base) |
| Adapter size     | ~17MB on disk        |

### Prompt Template

```
You are a clinical documentation assistant. Convert the following medical
text into a structured SOAP note.

MEDICAL TEXT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

Write a complete PLAN (treatments, monitoring, follow-up). End with a full
sentence.
SOAP NOTE:
```

### Generation Parameters

| Parameter      | Value | Rationale                                    |
| -------------- | ----- | -------------------------------------------- |
| max_new_tokens | 400   | Upper bound; stopping criteria trigger first |
| min_new_tokens | 150   | Prevents premature truncation                |
| do_sample      | False | Greedy decoding for deterministic output     |
| temperature    | None  | Disabled (greedy)                            |
| top_p          | None  | Disabled (greedy)                            |
| use_cache      | True  | KV cache for autoregressive efficiency       |

### SOAP Stopping Criteria

Two custom stopping criteria run during SOAP generation:

**SOAP Completion Criteria** (`_SOAPCompletionCriteria`):
Triggers stop when ALL conditions are met:

1. Token count exceeds `min_tokens` (150)
2. All four section headers present (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN)
3. PLAN section ≥ 150 characters
4. PLAN ends with sentence-terminal (`.`, `).`, `?`)
5. PLAN contains ≥ 2 clinical action verbs from curated list:
   `order, start, continue, follow, monitor, check, obtain, repeat, schedule, refer, prescribe, recommend, counsel, return, admit, discharge, consider, evaluate`

**Repetition Detector** (`_StopOnRepetition`):
Triggers if a second "SUBJECTIVE" appears or prompt leakage markers are detected.

### SOAP Post-processing

`_clean_soap()` applies: prompt echo stripping, preamble trimming, duplicate
SOAP note removal, prompt leakage removal, and trailing incomplete fragment
trimming (truncates at last complete sentence).

---

## Freeform Generation (Clinical Tools)

All clinical tools use `generate_freeform()`, which has its own stopping
criteria and post-processing chain separate from SOAP generation.

### Freeform Stopping Criteria

**`_FreeformStopCriteria`** — Monitors generated text and stops on:

1. **SOAP leakage** — both "SUBJECTIVE:" and "OBJECTIVE:" detected (model
   reverting to SOAP generation mode)
2. **Prompt/header leakage** — any prompt marker appears after the first 20
   characters of output (indicates the model is re-emitting its prompt):
   `SOAP NOTE:`, `MEDICAL TEXT:`, `PATIENT PROFILE:`, `BILLABLE DIAGNOSES:`,
   `DOCUMENTATION REVIEW:`, `DIFFERENTIAL DIAGNOSIS:`, `MEDICATION REVIEW:`,
   `GAPS:`, and others
3. **Analysis section repetition** — any of the five analysis headers
   (RISK ASSESSMENT, DIFFERENTIAL CONSIDERATIONS, RECOMMENDED SCREENINGS,
   RED FLAGS, CLINICAL QUESTIONS) appears a second time

### Freeform Post-processing

After generation, three sequential cleanup stages run:

 **Stage 1 — Marker truncation** : If any known section header appears after the
first 20 characters, text is truncated at that point.

 **Stage 2 — Analysis header deduplication** : For each of the five analysis
headers, if a second occurrence exists, text is truncated at that point.

 **Stage 3 — Line deduplication** : Lines are compared after normalization
(lowercase, stripped). If a line longer than 20 characters has been seen
before, all subsequent content is discarded. This catches numbered-list
degeneration where the model repeats items verbatim.

 **Stage 4 — Tail cleanup** : Trailing artifacts like `**RANKED`, `**NOTE`,
`**SUMMARY`, orphaned `**` markers are stripped.

### Tool Output Post-processing

`_clean_tool_output()` runs on all tool outputs before display:

* Strips "Final Answer:" and "The final answer is" blocks
* Strips second+ "Rationale:" sections
* Removes LaTeX `$\boxed{...}$` artifacts from base model instruction-tuning
* **Strips lines containing "not documented in source" or "not specified in
  source"** — removes unsupported differential diagnoses and padding entries
* Strips trailing incomplete headers

---

## Clinical Intelligence Tools

| Tool                    | max_new_tokens | Strategy                                                                                  |
| ----------------------- | -------------- | ----------------------------------------------------------------------------------------- |
| Billable Diagnoses      | 200            | Diagnosis extraction in plain English; no ICD-10 code numbers (avoids hallucinated codes) |
| Patient Summary         | 300            | Plain language, no jargon                                                                 |
| Completeness Check      | 250            | Gap detection checklist                                                                   |
| Differential Diagnosis  | 300            | Ranked list with evidence; unsupported entries auto-stripped                              |
| Medication Check        | 300            | Pre-check: skips model call if no medications detected in note                            |
| Patient Intake Analysis | 800            | 5-section structured analysis with defense-in-depth truncation                            |

### Billable Diagnoses (formerly ICD-10 Codes)

Early versions attempted to generate ICD-10-CM code numbers directly. Testing
revealed that MedGemma cannot reliably recall correct ICD-10 codes — it
generates plausible-looking alphanumeric strings that map to wrong descriptions
(e.g., outputting I50.9 labeled as "pulmonary embolism" when I50.9 is actually
"heart failure, unspecified"). The correct PE code is I26.99.

The tool was redesigned to extract billable diagnoses in plain English with
documentation evidence, which is what coders actually need as a starting point.
This approach produces zero hallucinated codes and faster inference (200 tokens
vs 300).

### Medication Check Pre-screening

Before calling the model, the medication check scans the SOAP note for
medication indicators (drug names, dosage units like "mg", "daily", "bid",
frequency terms, and common medication names). If none are found, it returns
immediately with "No specific medications documented in this note" — saving
~18 seconds of inference on notes without medication data.

### Patient Intake Analysis

Accepts 15 structured fields and generates five sections: Risk Assessment,
Differential Considerations, Recommended Screenings, Red Flags, and Clinical
Questions. The prompt enforces strict constraints:

* Maximum 3 bullet points per section
* Do NOT repeat any section
* Stop after CLINICAL QUESTIONS

 **Defense-in-depth truncation** : After `generate_freeform()` returns, the
`run_patient_analysis` handler applies a second round of section-header
deduplication before passing text to `_format_analysis_html()`. This catches
any repetition that survived the freeform post-processing.

 **Card filtering** : `_format_analysis_html()` skips cards with fewer than 20
characters of content, preventing broken/empty cards from rendering in the UI.

 **Sparse output handling** : `_run_tool()` appends a contextual note when tool
output is under 80 characters, informing the user that the source note may
lack sufficient clinical detail for comprehensive analysis.

---

## Training

### Data Generation

| Parameter | Value                                             |
| --------- | ------------------------------------------------- |
| Source    | GPT-4o API                                   |
| Samples   | 712 curated transcript-SOAP pairs                 |
| Cost      | $1.28 total                                       |
| Format    | Medical encounter transcript → concise SOAP note |

Anti-hallucination constraints enforced during data generation:

* "Not documented in source" for any finding not in the input transcript
* Zero WNL (Within Normal Limits) shortcuts
* Concise clinical shorthand, not verbose prose
* PLAN must contain specific, actionable items

The training data teaches MedGemma a specific output style and safety behavior,
not clinical knowledge — the base model's medical knowledge is preserved.

### Training Configuration

| Parameter      | Value                                                |
| -------------- | ---------------------------------------------------- |
| Base model     | MedGemma 1.5 4B                                      |
| Method         | LoRA (rank 16, alpha 32, dropout 0.05)               |
| Target modules | All attention layers                                 |
| Batch size     | 2 per device, 8 gradient accumulation (effective 16) |
| Learning rate  | 2e-5                                                 |
| Epochs         | 5 (early stopping patience: 2)                       |
| Quantization   | 4-bit NF4 during training                            |

### Results

| Metric          | Value              |
| --------------- | ------------------ |
| Training loss   | 0.828              |
| Validation loss | 0.782              |
| Overfitting     | None (val < train) |

Validation loss below training loss indicates good generalization, likely
due to LoRA's implicit regularization and the small adapter parameter count
(4.2M vs 4B base).

### Fine-tuning Impact

| Metric                | Base MedGemma        | Fine-tuned (LoRA) | Change          |
| --------------------- | -------------------- | ----------------- | --------------- |
| Avg word count        | ~200+ words          | 104 words         | 46% shorter     |
| Section completeness  | 85-95%               | 100%              | Always complete |
| Hallucinated findings | 5-10%                | 0%                | Eliminated      |
| WNL shortcuts         | Present              | 0%                | Eliminated      |
| Clinical style        | Textbook prose       | Shorthand         | Clinician-ready |
| PLAN items            | 4-8 (over-specified) | 2-4 (focused)     | Actionable      |

---

## UI Implementation

### Framework

Gradio 5+ with custom CSS. Soft theme with teal primary and slate neutral hues.

### Tabs

1. **Voice to SOAP** — Audio input + MedASR + SOAP generation
2. **Text to SOAP** — Transcript input + SOAP generation + examples
3. **Clinical Tools** — 5 tool buttons operating on SOAP notes
4. **Patient Analysis** — Structured intake form + 5-section analysis
5. **Model Comparison** — Side-by-side base vs fine-tuned output
6. **About MedScribe** — Clinical context, feature descriptions, limitations
7. **Technical Details** — Architecture, training, metrics

### Dynamic Status Pills

All long-running handlers are Python generators that `yield` intermediate
status updates. Status pills have four visual states: idle (gray),
processing (blue, pulsing animation), ready (green), and error (red).

### Anti-Stutter CSS

Gradio's loading animations cause visible layout shifts. MedScribe suppresses
these through:

* `.pending, .generating, .translucent` — opacity forced to 1, animations disabled
* `.eta-bar, .progress-bar, .loader` — hidden with `display: none`
* `html { overflow-y: scroll }` — permanent scrollbar prevents lateral snapping
* `min-height` on audio and output components prevents vertical collapse
* `container=False` on Audio component removes reflow-causing wrapper
* `show_progress="hidden"` on all click handlers

### Light Mode Enforcement

Three independent mechanisms force light mode regardless of OS preference:

1. **Browser** : `color-scheme: light only !important` on root elements
2. **DOM** : MutationObserver removes `.dark` class during Gradio hydration
3. **CSS** : `.dark, .dark *` selector overrides all Gradio dark-mode variables

---

## Deployment

| Spec         | Value                                           |
| ------------ | ----------------------------------------------- |
| GPU          | RTX 5070 Ti (16GB VRAM)                         |
| Quantization | 4-bit NF4 with double quantization              |
| Total VRAM   | ~3.4GB (MedASR ~400MB + MedGemma ~3GB)          |
| Framework    | Gradio 5+                                       |
| Inference    | Greedy decoding, KV cache, torch.inference_mode |
| Privacy      | Fully offline — no network calls at runtime    |
| OS           | Windows 11 (developed), Linux compatible        |
| Python       | 3.10+                                           |
| CUDA         | 12.8 (PyTorch nightly for Blackwell SM 12.0)    |

### Known Platform Constraints

* **Windows + Flash Attention** : Installation fails due to long path
  limitations. Not required — standard attention is sufficient for
  single-request inference.
* **bitsandbytes on Windows** : Requires recent versions with native Windows
  support or the `bitsandbytes-windows` package.

---

## Dependencies

| Package       | Purpose                                  |
| ------------- | ---------------------------------------- |
| torch         | Model inference (nightly for Blackwell)  |
| torchaudio    | Audio processing support                 |
| transformers  | Model loading and generation             |
| accelerate    | Device mapping and memory management     |
| peft          | LoRA adapter loading and management      |
| bitsandbytes  | 4-bit NF4 quantization                   |
| librosa       | Audio loading, resampling, normalization |
| soundfile     | Audio file I/O backend                   |
| gradio        | Web UI framework                         |
| sentencepiece | Tokenizer support                        |

---

## File Reference

| File                       | Lines | Purpose                                                     |
| -------------------------- | ----- | ----------------------------------------------------------- |
| `app.py`                 | ~1200 | Gradio UI, handlers, CSS, HTML formatting                   |
| `src/inference.py`       | ~400  | SOAPGenerator, stopping criteria (SOAP + freeform), prompts |
| `src/pipeline.py`        | ~400  | MedScribePipeline, MedASR, CTC decode, clinical tools       |
| `train.py`            | ~400  | LoRA fine-tuning script                                     |
| `evaluate.py`         | ~300  | Quality evaluation metrics                                  |
| `generate_soap_gpt4o.py` | ~400  | Training data generation via OpenAI API                     |

## Documentation and LInks

#### Adaptors/Safetensors: [Huggingface](https://huggingface.co/Tushar9802/MedScribe-soap-lora)

#### Dataset: [Kaggle](https://www.kaggle.com/datasets/tusharjaju/MedScribe-soap-training-data-712-curated-samples)
