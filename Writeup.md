
### Project Name

**MedScribe** — Clinical Documentation Workstation

### Your Team

**Tushar Jaju** — Solo developer. Designed the architecture, fine-tuned MedGemma, built the pipeline, and developed the Gradio interface.

 **Dr. Manthan Pritin Patel** , MBBS, MD Anaesthesia — Clinical advisor. Validated the conciseness-first approach against real documentation workflows and confirmed that generated shorthand matches abbreviation patterns used in daily clinical practice.

### Problem Statement

Physicians spend 16 minutes per patient encounter on documentation — 40% of every clinical interaction. AI scribes were supposed to fix this, but they create two new problems: **verbosity** (200+ words of textbook prose for a 45-second conversation) and **hallucination** (invented findings, fabricated ICD-10 codes, "Within Normal Limits" for systems never examined). The tool meant to save time becomes a liability physicians have to audit line by line.

Dr. Patel reports: *"More often than not I have to go and edit the notes and shorten them, because they read like textbook lexicon rather than shorthand designed to deliver efficient summaries with alacrity."*

**Impact potential:** 917 hours saved per physician per year — nearly five months of documentation time returned to patient care. Based on Sinsky et al. (Annals of Internal Medicine, 2016) burden data and MedScribe's measured 46% word count reduction.

### Overall Solution

MedScribe chains **three HAI-DEF models** into a single offline pipeline:

| Component             | Model                             | Role                                     |
| --------------------- | --------------------------------- | ---------------------------------------- |
| Speech Recognition    | **MedASR**(105M, Conformer) | Medical dictation → text, 5.2% WER      |
| SOAP Generation       | **MedGemma 1.5 4B + LoRA**  | Transcript → concise clinical shorthand |
| Clinical Intelligence | **MedGemma 1.5 4B**(base)   | 6 post-generation analysis tools         |

**The core innovation** is a LoRA adapter (rank 16, 4.2M parameters — 0.1% of base) trained on 712 curated samples to generate the shorthand clinicians actually write. Training cost:  **$1.28** .

**Same transcript, same model — adapter off vs on:**

| Section    | Base MedGemma                                                                                                                      | Fine-tuned                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| SUBJECTIVE | "The patient is a 54-year-old female who presents to the emergency department with a chief complaint of shortness of breath."      | "54 yo F c/o SOB."                                                                   |
| ASSESSMENT | "After careful review of the imaging findings and clinical presentation, the most likely diagnosis is acute pulmonary embolism..." | "Acute segmental PE, right lower lobe."                                              |
| PLAN       | "It is recommended to initiate anticoagulation therapy. The patient's respiratory status should be closely monitored."             | "1. Initiate anticoagulation. 2. Monitor resp status and O2 sat. 3. Pulm follow-up." |

Verifiable in MedScribe's Model Comparison tab — runs both versions side-by-side on the same input.

**Six clinical intelligence tools** operate on any generated SOAP note using base MedGemma's instruction-following capability: billable diagnosis extraction, patient summaries, completeness checks, differential diagnosis, medication safety review, and patient intake analysis.

**Everything runs fully offline on consumer hardware. Patient data never leaves the device.**

### Technical Details

**Anti-Hallucination by Design:** Training data enforces a strict contract — if a finding isn't in the input transcript, the model writes "Not documented in source." No plausible guesses, no "Within Normal Limits," no silence. Early versions generated ICD-10 codes directly; testing revealed MedGemma hallucinated plausible-looking codes mapped to wrong descriptions (e.g., I50.9 labeled "pulmonary embolism" — actually heart failure). We removed code generation entirely and replaced it with plain-English diagnosis extraction. We chose to ship fewer features correctly rather than more features dangerously.

**Custom Stopping Criteria:** Clinical completion detector verifies all four SOAP section headers present, PLAN contains ≥150 characters with a complete sentence, and ≥2 clinical action verbs. Separate criteria for repetition detection, prompt leakage, and section re-emission.

**Deployment:** 4-bit NF4 quantization. MedASR (~400MB) + MedGemma (~3GB) coexist in 16GB VRAM. Greedy decoding for deterministic output. Fully offline.

**Results:**

| Metric                | Base MedGemma | Fine-tuned (LoRA) | Change                |
| --------------------- | ------------- | ----------------- | --------------------- |
| Avg word count        | ~200+ words   | 104 words         | **46% shorter** |
| Hallucinated findings | 5-10%         | 0%                | **Eliminated**  |
| Section completeness  | 85-95%        | 100%              | Always complete       |
| Quality score         | —            | 90/100            | Automated rubric      |

**Training efficiency:** 712 samples, $1.28 cost, 17MB adapter, validation loss 0.782 (no overfitting).

**Limitations:** English only; research prototype (no HIPAA/IRB); synthetic training data; 25-second inference (acceptable for outpatient, too slow for ED); clinical tools produce thin output on sparse notes; clinical tools not fine-tuned.

**Resources:**

| Resource         | Link                                                                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Source Code      | [GitHub: Tushar-9802/MedScribe-1](https://github.com/Tushar-9802/MedScribe-1)                                             |
| LoRA Adapter     | [HuggingFace: Tushar9802/MedScribe-soap-lora](https://huggingface.co/Tushar9802/MedScribe-soap-lora)                      |
| Training Dataset | [Kaggle: medscribe-soap-712](https://www.kaggle.com/datasets/tusharjaju/medscribe-soap-training-data-712-curated-samples) |
| Demo Video       | [YouTube: MedScribe Demo](https://youtu.be/pFbt-tCfmis)                                                                   |
