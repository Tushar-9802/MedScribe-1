# MedScribe -- Technical Documentation

This document covers architecture, model details, training methodology,
inference pipeline, and implementation decisions. For clinical context and
user-facing documentation, see [README.md](https://claude.ai/chat/README.md).

---

## Architecture Overview

MedScribe uses three distinct capabilities from Google's HAI-DEF ecosystem
in a single pipeline:

```
[Audio Input]
      |
[MedASR -- 105M Conformer, CTC decode]
      |
[Manual CTC Collapse + Post-processing]
      |
[Clean Transcript -- editable by clinician]
      |
[MedGemma 1.5 4B + LoRA adapter -- 4-bit NF4]
      |
[Custom Stopping Criteria + Post-processing]
      |
[Structured SOAP Note]
      |
[MedGemma 1.5 4B -- base instruction-following]
      |
[Clinical Intelligence Tools (6)]
```

| Component             | Model                  | Parameters | VRAM     | Role                                  |
| --------------------- | ---------------------- | ---------- | -------- | ------------------------------------- |
| Speech Recognition    | MedASR (Conformer)     | 105M       | ~400MB   | Medical dictation to text, 5.2% WER   |
| SOAP Generation       | MedGemma 1.5 4B (LoRA) | 4B + 4.2M  | ~3GB     | Concise structured notes (~100 words) |
| Clinical Intelligence | MedGemma 1.5 4B (base) | 4B         | (shared) | ICD-10, DDx, risk analysis, screening |

The SOAP generation and clinical intelligence tools share the same loaded model
instance. The LoRA adapter is active for SOAP generation and can be temporarily
disabled (via PEFT's `disable_adapter()` context manager) for the Model
Comparison feature. Clinical tools use the model with the adapter active -- the
base MedGemma instruction-following capability is preserved through the LoRA
layers.

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

1. Forward pass produces logit matrix (time steps x vocabulary)
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

The inverse vocabulary is pre-built at model load time to avoid rebuilding per
inference call.

### Transcript Post-processing

Raw CTC output requires extensive cleanup. The pipeline applies a 10-step
post-processing chain:

1. **CTC stutter-killer** -- Collapse 3+ identical consecutive characters to 1.
   Preserves valid doubles (ll, ee, ss) but eliminates CTC stutters (CCT->CT,
   TTT->T). Regex: `(.)\1{2,}` -> `\1`
2. **SentencePiece boundary** -- Replace `▁` (U+2581) with space. MedASR uses
   SentencePiece tokenization that marks word boundaries with this character.
3. **Special token removal** -- Strip `<s>`, `</s>`, `<epsilon>`, `<pad>`,
   `<unk>` tokens that may survive decoding.
4. **Punctuation token mapping** -- Convert formatting tokens to punctuation:
   `{period}` -> `.`, `{comma}` -> `,`, `{colon}` -> `:`, `{semicolon}` -> `;`,
   `{new paragraph}` -> `\n`, `{question mark}` -> `?`, `{exclamation}` -> `!`.
   Regex handles stuttered variants like `{{period}}`.
5. **Bracket marker removal** -- Strip `[EXAM TYPE]`, `[INDICATION]`, and
   similar metadata brackets.
6. **Brace cleanup** -- Remove any remaining `{` and `}` characters.
7. **Punctuation spacing** -- Fix `word .` -> `word.` and `word.Next` ->
   `word. Next`.
8. **Whitespace collapse** -- Normalize multiple spaces and newlines.
9. **Artifact trimming** -- Remove leading/trailing punctuation artifacts.
10. **Sentence capitalization** -- Capitalize first character and
    post-sentence characters.

### Audio Preprocessing

* Loaded via librosa at 16kHz mono (explicit `sr=16000, mono=True`)
* Amplitude normalization via `librosa.util.normalize()` to handle varying
  float32 ranges from Gradio's audio component
* No additional noise reduction or VAD applied

---

## MedGemma SOAP Generation

### Model Loading

Base model: `google/medgemma-4b-it` (instruction-tuned variant)

Quantization configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

The base model is loaded with 4-bit NF4 quantization and double quantization
enabled, reducing VRAM from ~8GB (FP16) to ~3GB. Compute dtype is bfloat16
for the dequantized operations.

The LoRA adapter is loaded on top via PEFT:

```python
self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
self.model.eval()
self.model.config.use_cache = True
```

A warmup generation (10 tokens) runs at load time to compile CUDA kernels
and populate caches.

### LoRA Adapter

| Parameter        | Value                |
| ---------------- | -------------------- |
| Rank             | 16                   |
| Alpha            | 32                   |
| Dropout          | 0.05                 |
| Target modules   | All attention layers |
| Trainable params | ~4.2M (0.1% of base) |
| Adapter size     | ~17MB on disk        |

Adapter path: `./models/checkpoints/medgemma_v2_soap/final_model`

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

### Custom Stopping Criteria

Two custom stopping criteria run during generation:

**SOAP Completion Criteria** (`_SOAPCompletionCriteria`):
Monitors generated text and triggers stop when ALL conditions are met:

1. Generated token count exceeds `min_tokens` (150)
2. All four section headers (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN) are present
3. PLAN section contains at least 150 characters
4. PLAN ends with a sentence-terminal character (`.`, `).`, or `?`)
5. PLAN contains at least 2 clinical action verbs from a curated list:
   `order, start, continue, follow, monitor, check, obtain, repeat, schedule, refer, prescribe, administer, recommend, counsel, return, admit, discharge, daily, bid, tid, prn, consider, evaluate`

This ensures the PLAN section is clinically complete and doesn't truncate
mid-sentence or mid-instruction.

**Repetition Detector** (`_StopOnRepetition`):
Triggers stop if:

1. A second occurrence of "SUBJECTIVE" appears (model is repeating the note)
2. Any prompt leakage markers appear in the output: "Generate a SOAP note",
   "MEDICAL TEXT:", "You are a clinical documentation", "Convert the following",
   or a `---` separator

### SOAP Post-processing

The `_clean_soap()` function applies sequential cleanup:

1. Strip content before "SOAP NOTE:" if present (prompt echo)
2. Locate first SUBJECTIVE header and trim any preamble
3. Detect and remove duplicate SOAP notes (second SUBJECTIVE occurrence)
4. Remove prompt leakage markers
5. Trim trailing incomplete fragments (sentences ending with dangling
   conjunctions like "and", "or", "with", "for", etc.) by truncating to the
   last complete sentence

### Base Model Comparison

The `generate_base()` method temporarily disables the LoRA adapter using PEFT's
context manager:

```python
with self.model.disable_adapter():
    outputs = self.model.generate(...)
```

This produces output from the base MedGemma model without any fine-tuning,
using the same prompt, same stopping criteria, and same post-processing. The
adapter is automatically re-enabled when the context manager exits, even if an
exception occurs (fixed in PEFT >= 0.5.0).

---

## Clinical Intelligence Tools

All tools share the same `generate_freeform()` method, which generates text
without the SOAP-specific stopping criteria. Each tool provides a specialized
prompt template.

| Tool                    | max_new_tokens | Prompt strategy                      |
| ----------------------- | -------------- | ------------------------------------ |
| ICD-10 Coding           | 250            | Numbered list, evidence-backed codes |
| Patient Summary         | 300            | Plain language, no jargon            |
| Completeness Check      | 250            | Gap detection checklist              |
| Differential Diagnosis  | 300            | Ranked list with evidence            |
| Medication Check        | 300            | Interaction/contraindication review  |
| Patient Intake Analysis | 500            | 5-section structured analysis        |

The Patient Intake Analysis tool constructs a patient profile string from the
structured form inputs and uses a specialized prompt requesting exactly five
sections (Risk Assessment, Differential Considerations, Recommended Screenings,
Red Flags, Clinical Questions).

---

## Training

### Data Generation

Training data was generated using the OpenAI API (GPT-4o Mini) with
anti-hallucination constraints:

* 712 curated transcript-SOAP pairs
* Total cost: $1.28 via GPT-4o Mini API
* Each sample consists of a medical encounter transcript and the target
  concise SOAP note

Anti-hallucination constraints enforced during generation:

* "Not documented in source" for any clinical finding not present in the
  input transcript
* Zero WNL (Within Normal Limits) shortcuts -- every finding must be
  explicitly stated
* Concise clinical shorthand style rather than verbose prose
* PLAN section must contain specific, actionable items

### Training Configuration

| Parameter                    | Value                |
| ---------------------------- | -------------------- |
| Base model                   | MedGemma 1.5 4B      |
| Method                       | LoRA                 |
| Rank                         | 16                   |
| Alpha                        | 32                   |
| Dropout                      | 0.05                 |
| Target modules               | All attention layers |
| Batch size (per device)      | 8                    |
| Gradient accumulation        | 4 (effective: 32)    |
| Learning rate                | 2e-4                 |
| Epochs                       | 3                    |
| Precision                    | BFloat16             |
| Quantization during training | 4-bit NF4            |

### Training Results

| Metric          | Value              |
| --------------- | ------------------ |
| Training loss   | 0.828              |
| Validation loss | 0.782              |
| Overfitting     | None (val < train) |

The validation loss being lower than training loss indicates good
generalization without overfitting, likely due to LoRA's implicit
regularization and the relatively small adapter parameter count (4.2M vs 4B
base parameters).

### Evaluation Results

Evaluated on held-out test set:

| Metric                | Value          |
| --------------------- | -------------- |
| Quality score         | 90/100         |
| Section completeness  | 100% (S/O/A/P) |
| Hallucinated findings | 0%             |
| WNL shortcuts         | 0%             |
| Avg word count        | 104 words      |
| PLAN items per note   | 2-4            |

### Fine-tuning Impact (Base vs LoRA)

| Metric                | Base MedGemma  | Fine-tuned (LoRA) | Change          |
| --------------------- | -------------- | ----------------- | --------------- |
| Avg word count        | ~200+ words    | 104 words         | 46% shorter     |
| Section completeness  | 85-95%         | 100%              | Always complete |
| Hallucinated findings | 5-10%          | 0%                | Eliminated      |
| WNL shortcuts         | Present        | 0%                | Eliminated      |
| Clinical style        | Textbook prose | Shorthand         | Clinician-ready |
| PLAN items            | 4-8            | 2-4               | Focused         |

---

## UI Implementation

### Framework

Gradio 5+ with custom CSS theme. Soft theme with teal primary and slate
neutral hues.

### Tabs

1. **Voice to SOAP** -- Audio input (microphone/upload) + MedASR + SOAP generation
2. **Text to SOAP** -- Transcript input + SOAP generation + example transcripts
3. **Clinical Tools** -- 5 tool buttons operating on SOAP notes
4. **Patient Analysis** -- Structured form with 15 fields + analysis output
5. **Model Comparison** -- Side-by-side base vs fine-tuned output
6. **About MedScribe** -- Clinical context and feature descriptions
7. **Technical Details** -- Architecture, training, and metrics

### Dynamic Status Pills

All long-running handlers are Python generators that `yield` intermediate
status updates to the UI. This provides real-time feedback during the 20-30
second inference window:

* Upload audio -> "Transcribing audio (MedASR)..."
* MedASR completes -> "Generating SOAP note (MedGemma)... ASR took Xs"
* SOAP completes -> "Done in Xs -- ASR Xs + SOAP Xs"

Status pills have three visual states:

* Default (gray): idle/informational
* Processing (blue, pulsing): operation in progress
* Ready (green): operation complete
* Error (red): operation failed

### Anti-Stutter CSS

Gradio's default loading animations cause visible UI jerks (layout shifts)
when components update. MedScribe suppresses these through CSS:

```css
.pending, .generating, .translucent {
    opacity: 1 !important;
    animation: none !important;
    border: none !important;
}
.eta-bar, .progress-bar, .loader, .svelte-spinner {
    display: none !important;
    height: 0 !important;
}
```

Additional stability measures:

* `html { overflow-y: scroll !important; }` -- forces permanent scrollbar to
  prevent lateral width snapping
* `[data-testid="audio"] { min-height: 160px !important; }` -- prevents
  vertical jump on audio upload
* `container=False` on Audio component -- removes Gradio wrapper container
  that caused reflow on file upload
* `.soap-output { min-height: 140px !important; }` -- prevents HTML output
  components from collapsing during intermediate yield states
* `show_progress="hidden"` on all click handlers -- disables Gradio's
  built-in progress indicators

### Light Mode Enforcement

MedScribe forces light mode regardless of OS dark mode preference through
three independent mechanisms:

1. **Browser level** : `color-scheme: light only !important` on `:root, html, body`
2. **DOM level** : JavaScript MutationObserver removes `.dark` class from
   `document.documentElement` whenever Gradio adds it during hydration
3. **CSS level** : `.dark, .dark *` selector overrides all Gradio CSS variables
   to light-mode values as a fallback

---

## Deployment

| Spec           | Value                                        |
| -------------- | -------------------------------------------- |
| GPU            | RTX 5070 Ti (16GB VRAM)                      |
| Quantization   | 4-bit NF4 with double quantization           |
| MedASR VRAM    | ~400MB                                       |
| MedGemma VRAM  | ~3GB (4-bit)                                 |
| Total VRAM     | ~3.4GB (both models loaded)                  |
| Framework      | Gradio 5+                                    |
| Inference mode | Greedy decoding, KV cache, inference_mode    |
| Privacy        | Fully offline -- no network calls at runtime |
| OS             | Windows 11 (developed), Linux compatible     |
| Python         | 3.10+                                        |
| CUDA           | 12.8 (PyTorch nightly for Blackwell SM 12.0) |

### Known Platform Issues

* **Windows + Flash Attention** : Installation fails due to long path
  limitations on Windows. Not used; standard attention is sufficient for
  single-request inference.
* **Windows + multiprocessing** : Some PyTorch multiprocessing features are
  incompatible with Windows. Not applicable for single-GPU inference.
* **bitsandbytes on Windows** : Requires `bitsandbytes-windows` package or
  recent bitsandbytes versions with native Windows support.

---

## Dependencies

Core runtime dependencies (see `requirements.txt` for pinned versions):

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

| File                       | Lines | Purpose                                         |
| -------------------------- | ----- | ----------------------------------------------- |
| `app.py`                 | ~1100 | Gradio UI, handlers, CSS, HTML formatting       |
| `src/inference.py`       | ~360  | SOAPGenerator class, stopping criteria, prompts |
| `src/pipeline.py`        | ~310  | MedScribePipeline, MedASR, CTC decode, tools    |
| `train_v2.py`            | ~400  | LoRA fine-tuning script                         |
| `evaluate_v2.py`         | ~300  | Quality evaluation metrics                      |
| `generate_soap_gpt4o.py` | ~400  | Training data generation via OpenAI API         |
