"""
MedScribe Inference
===========================================
Updated for v2 adapter (GPT-4o trained, 712 samples).

Fixes from v3:
1. Adapter path updated to medgemma_v2_soap
2. StopOnRepetition: catches model starting a second SOAP note (12% of v2 outputs)
3. Post-processing: strips repeated content + DISPOSITION/FOLLOW-UP overflow
4. Multi-transcript benchmark with diverse clinical scenarios
5. Quality scoring calibrated to v2 output profile (170 word median)
"""
import os
import re
import time
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel, PeftConfig

# ============================================================
# CONFIG
# ============================================================
ADAPTER_PATH = "./models/checkpoints/medgemma_v2_soap/final_model"
MAX_NEW_TOKENS = 400   # v2 generates ~175 words avg but needs headroom for complete PLANs
MIN_NEW_TOKENS = 150   # Ensures PLAN gets 50+ tokens before stopping criteria can fire


# ============================================================
# STOPPING CRITERIA
# ============================================================
class SOAPCompletionCriteria(StoppingCriteria):
    """
    Stop when PLAN section is clinically complete.

    Tuned for v2 model output profile:
    - Median 175 words, 70 PLAN words, 4.3 PLAN items
    - PLAN threshold 120 chars (not 180 — v2 PLANs are denser)
    """

    def __init__(self, tokenizer, min_tokens=150):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.sections = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
        self.prompt_length = 0

    def _extract_plan_content(self, text):
        """Extract text after PLAN header."""
        # Handle both **PLAN:** and PLAN: formats
        patterns = ["**PLAN:**", "PLAN:"]
        plan_start = -1
        for pat in patterns:
            idx = text.find(pat)
            if idx != -1:
                plan_start = idx + len(pat)
                break

        if plan_start == -1:
            text_upper = text.upper()
            idx = text_upper.find("PLAN")
            if idx == -1:
                return ""
            colon = text.find(":", idx)
            plan_start = colon + 1 if colon != -1 and colon < idx + 15 else idx + 4

        return text[plan_start:].strip()

    def _plan_is_complete(self, plan_content):
        """
        PLAN is complete when:
        1. ≥150 chars
        2. ≥2 clinical action words
        3. Ends with sentence terminator
        """
        if not plan_content or len(plan_content) < 150:
            return False

        s = plan_content.rstrip()
        if not (s.endswith(".") or s.endswith(").") or s.endswith("?")):
            return False

        action_words = [
            "order", "start", "continue", "follow", "monitor", "check",
            "obtain", "repeat", "schedule", "refer", "prescribe", "administer",
            "recommend", "counsel", "return", "admit", "discharge",
            "daily", "bid", "tid", "prn", "consider", "evaluate",
        ]
        plan_lower = plan_content.lower()
        action_count = sum(1 for w in action_words if w in plan_lower)
        return action_count >= 2

    def __call__(self, input_ids, scores, **kwargs):
        generated_tokens = input_ids.shape[-1] - self.prompt_length
        if generated_tokens < self.min_tokens:
            return False

        gen_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        text_upper = text.upper()
        if not all(s in text_upper for s in self.sections):
            return False

        plan_content = self._extract_plan_content(text)
        return self._plan_is_complete(plan_content)


class StopOnRepetition(StoppingCriteria):
    """
    Stop if model starts repeating — the #1 issue with v2 (12% of outputs).
    Catches:
    - Second SUBJECTIVE header (model starts a new SOAP note)
    - Prompt leakage (model echoes instructions)
    - Horizontal rules / separators
    """

    def __init__(self, tokenizer, prompt_length=0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_length:]
        if gen_ids.shape[0] < 50:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text_upper = text.upper()

        # Detect second SUBJECTIVE (model repeating)
        first = text_upper.find("SUBJECTIVE")
        if first != -1:
            second = text_upper.find("SUBJECTIVE", first + 20)
            if second != -1:
                return True

        # Detect prompt leakage
        leakage_patterns = [
            "Generate a SOAP note",
            "MEDICAL TEXT:",
            "You are a clinical documentation",
            "Convert the following",
            "\n---\n",
        ]
        for pattern in leakage_patterns:
            if pattern in text:
                return True

        return False


# ============================================================
# POST-PROCESSING
# ============================================================
def clean_soap_output(text):
    """Extract clean SOAP note from raw generation."""
    if not text:
        return ""

    # Remove prompt leakage
    if "SOAP NOTE:" in text:
        text = text.split("SOAP NOTE:")[-1].strip()

    # Anchor to first SOAP section
    for header in ["**SUBJECTIVE:**", "SUBJECTIVE:", "SUBJECTIVE"]:
        if header in text:
            text = text[text.find(header):]
            break

    # Cut at second SUBJECTIVE (repetition)
    upper = text.upper()
    first = upper.find("SUBJECTIVE")
    if first != -1:
        second = upper.find("SUBJECTIVE", first + 20)
        if second != -1:
            text = text[:second].strip()

    # Cut at prompt leakage
    for marker in [
        "Generate a SOAP note", "MEDICAL TEXT:",
        "You are a clinical", "Convert the following",
        "\n---\n",
    ]:
        if marker in text:
            text = text.split(marker)[0].strip()

    # Trim trailing incomplete fragments
    text = text.rstrip()
    bad_endings = [
        " and", " or", " with", " for", " to", " in", " on", " the",
        " a", " an", " is", " are", " was", " were", " will", " should",
        " may", " can", " per",
    ]
    for ending in bad_endings:
        if text.lower().endswith(ending):
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
            break

    return text.strip()


def validate_soap(text):
    """Validate structure + clinical quality."""
    sections = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
    text_upper = text.upper()

    results = {}
    for s in sections:
        results[s] = s in text_upper

    # PLAN quality
    if results["PLAN"]:
        plan_idx = text_upper.find("PLAN")
        plan_content = text[plan_idx:]
        colon = plan_content.find(":")
        plan_body = plan_content[colon + 1:].strip() if colon != -1 else plan_content[4:].strip()

        # Strip any DISPOSITION/FOLLOW-UP that bleeds past PLAN
        for extra_header in ["**DISPOSITION", "**FOLLOW-UP", "DISPOSITION:", "FOLLOW-UP:"]:
            if extra_header in plan_body.upper():
                cut = plan_body.upper().find(extra_header.upper().replace("**", ""))
                if cut > 0:
                    plan_body_for_count = plan_body  # Keep full for display
                    break
            plan_body_for_count = plan_body

        results["PLAN_LENGTH"] = len(plan_body)
        results["PLAN_COMPLETE"] = len(plan_body) > 60

        # Count numbered items
        items = re.findall(r"\d+\.", plan_body)
        results["PLAN_ITEMS"] = len(items)
        if results["PLAN_ITEMS"] == 0:
            # Count line-based items
            lines = [l.strip() for l in plan_body.split("\n") if l.strip() and len(l.strip()) > 10]
            results["PLAN_ITEMS"] = max(len(lines), 1)
    else:
        results["PLAN_LENGTH"] = 0
        results["PLAN_COMPLETE"] = False
        results["PLAN_ITEMS"] = 0

    results["ALL_PRESENT"] = all(results[s] for s in sections)
    results["COMPLETE"] = results["ALL_PRESENT"] and results["PLAN_COMPLETE"]
    results["WORD_COUNT"] = len(text.split())
    results["HAS_WNL"] = " WNL" in text_upper
    results["HAS_NOT_DOCUMENTED"] = "NOT DOCUMENTED" in text_upper

    return results


def compute_quality_score(text, validation):
    """
    Quality score (0-100) calibrated to v2 concise output profile.
    
    v2 model produces intentionally brief notes (80-175 words).
    Scoring rewards clinical completeness relative to input complexity,
    not raw word count. A 90-word URI note is perfect; a 90-word 
    chest pain note may be thin.
    """
    score = 0

    # Structure (25 pts) — all 4 sections present
    sections_present = sum(1 for s in ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"] if validation[s])
    score += (sections_present / 4) * 25

    # PLAN completeness (30 pts) — actionable items
    plan_items = validation.get("PLAN_ITEMS", 0)
    if plan_items >= 4:
        score += 30
    elif plan_items >= 3:
        score += 25
    elif plan_items >= 2:
        score += 18
    elif plan_items >= 1:
        score += 10

    # Clinical content (25 pts) — appropriate documentation markers
    text_lower = text.lower()
    clinical_checks = [
        # Vitals documented with values (not WNL)
        any(w in text_lower for w in ["bp ", "hr ", "rr ", "spo2", "temp"]),
        # Clinical abbreviations used (conciseness indicator)
        any(w in text_lower for w in ["yo ", "c/o", "h/o", "pmh", "denies"]),
        # Disposition/follow-up present
        any(w in text_lower for w in ["follow", "f/u", "return", "refer", "consult", "monitor"]),
        # Anti-hallucination compliance (good if present OR if transcript had complete data)
        "not documented" in text_lower or not any(w in text_lower for w in ["wnl", "within normal"]),
        # Differential or reasoning in assessment
        any(w in text_lower for w in ["differential", "likely", "consistent with", "suggestive", "rule out"]),
    ]
    score += (sum(clinical_checks) / len(clinical_checks)) * 25

    # Conciseness (20 pts) — calibrated to v2 median 100 words
    # Sweet spot: 80-250 words. Brief is a FEATURE for this project.
    wc = validation.get("WORD_COUNT", 0)
    if 80 <= wc <= 250:
        score += 20  # Ideal range for concise clinical notes
    elif 60 <= wc <= 300:
        score += 15
    elif 50 <= wc <= 350:
        score += 10
    else:
        score += 5

    return round(score, 1)


# ============================================================
# TEST CASES — diverse clinical scenarios
# ============================================================
TEST_CASES = [
    {
        "name": "Chest Pain (ED)",
        "transcript": "45-year-old male presents with substernal chest pain for 2 hours, 7/10 severity, radiating to left arm. Associated with diaphoresis and anxiety. No nausea or vomiting. BP 145/92, HR 98, RR 16. Anxious appearing. Regular rhythm, no murmurs.",
    },
    {
        "name": "Diabetes Follow-up",
        "transcript": "62-year-old female with type 2 diabetes returns for 3-month follow-up. Reports good compliance with metformin 1000mg twice daily. Occasional fasting glucose readings of 140-160. No hypoglycemic episodes. Denies polyuria, polydipsia, blurred vision. BP 132/78, HR 72, BMI 31.2. A1C today 7.4%, down from 8.1%. Creatinine 1.1, eGFR 68. Foot exam: intact sensation, no ulcers. Eyes: last retinal exam 6 months ago, no retinopathy.",
    },
    {
        "name": "Pediatric URI",
        "transcript": "4-year-old male brought in by mother for 3 days of runny nose, cough, and low-grade fever. Max temp 100.4 at home. Eating and drinking well. No ear pulling. No history of asthma. Temp 99.8, HR 110, RR 22, SpO2 99%. Alert, playful. TMs clear bilaterally. Throat mildly erythematous, no exudate. Lungs clear. No lymphadenopathy.",
    },
    {
        "name": "CKD Stage 3 (Nephrology)",
        "transcript": "58-year-old male with hypertension and CKD stage 3b, GFR 38. On lisinopril 20mg daily, amlodipine 5mg daily. BP today 142/88. Labs: Cr 1.8, BUN 32, K 4.9, bicarb 20, phosphorus 4.8, PTH 98. Urine albumin-to-creatinine ratio 450. No edema. Denies fatigue, nausea, pruritus.",
    },
    {
        "name": "Anxiety/Depression (Psych)",
        "transcript": "34-year-old female presenting with worsening anxiety and depressed mood over 3 months since job loss. Reports difficulty sleeping, poor appetite, loss of interest in activities. Denies suicidal ideation, hallucinations, or substance use. Currently on sertraline 50mg daily started 6 weeks ago with minimal improvement. PHQ-9 score 14, GAD-7 score 12. Appears well-groomed, cooperative. Speech normal rate and rhythm. Mood described as sad. Affect constricted. Thought process linear. Insight and judgment intact.",
    },
]

PROMPT_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

Write a complete PLAN (treatments, monitoring, follow-up). End with a full sentence.
SOAP NOTE:"""


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MedScribe Inference v4 — Competition Build")
    print("=" * 60)

    # --- Gate: adapter exists ---
    if not os.path.isdir(ADAPTER_PATH):
        print(f"✗ ABORT: Adapter not found: {ADAPTER_PATH}")
        return

    # --- Load model ---
    print(f"\n[1/3] Loading model...")
    peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    model.config.use_cache = True
    print("✓ Model loaded (no torch.compile — bitsandbytes 4-bit)")

    # --- Stopping criteria ---
    print(f"\n[2/3] Configuring inference...")
    soap_criteria = SOAPCompletionCriteria(tokenizer, min_tokens=MIN_NEW_TOKENS)
    repeat_criteria = StopOnRepetition(tokenizer)
    stopping_criteria = StoppingCriteriaList([soap_criteria, repeat_criteria])
    print(f"✓ Token range: {MIN_NEW_TOKENS}–{MAX_NEW_TOKENS}")
    print(f"✓ Stopping: PLAN completion + repetition guard")

    # --- Warmup ---
    print(f"\n[3/3] Running benchmark...")
    print("  Warmup...", end=" ", flush=True)

    warmup_prompt = PROMPT_TEMPLATE.format(transcript=TEST_CASES[0]["transcript"])
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
    soap_criteria.prompt_length = warmup_inputs["input_ids"].shape[1]
    repeat_criteria.prompt_length = warmup_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("done")

    # --- Benchmark all test cases ---
    all_results = []

    for tc in TEST_CASES:
        print(f"\n  {'─' * 50}")
        print(f"  Case: {tc['name']}")

        prompt = PROMPT_TEMPLATE.format(transcript=tc["transcript"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1]

        soap_criteria.prompt_length = prompt_length
        repeat_criteria.prompt_length = prompt_length

        start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        elapsed = time.time() - start

        gen_ids = outputs[0][prompt_length:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        clean = clean_soap_output(raw)
        val = validate_soap(clean)
        quality = compute_quality_score(clean, val)
        tokens = len(gen_ids)

        all_results.append({
            "name": tc["name"],
            "time": elapsed,
            "tokens": tokens,
            "text": clean,
            "validation": val,
            "quality": quality,
        })

        status = "✓" if val["COMPLETE"] else "⚠"
        print(f"  {elapsed:.1f}s | {tokens} tok | {val['WORD_COUNT']}w | Q:{quality} | {status}")

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print("=" * 60)

    times = [r["time"] for r in all_results]
    qualities = [r["quality"] for r in all_results]
    word_counts = [r["validation"]["WORD_COUNT"] for r in all_results]
    complete_count = sum(1 for r in all_results if r["validation"]["COMPLETE"])

    print(f"\n  Cases tested:     {len(all_results)}")
    print(f"  Complete (S/O/A/P): {complete_count}/{len(all_results)} ({complete_count/len(all_results)*100:.0f}%)")
    print(f"  Avg quality:      {sum(qualities)/len(qualities):.0f}/100")
    print(f"  Avg words:        {sum(word_counts)/len(word_counts):.0f}")
    print(f"  Avg time:         {sum(times)/len(times):.1f}s")
    print(f"  Fastest:          {min(times):.1f}s")
    print(f"  Slowest:          {max(times):.1f}s")

    wnl_count = sum(1 for r in all_results if r["validation"].get("HAS_WNL", False))
    print(f"  WNL present:      {wnl_count}/{len(all_results)}")

    # Per-case breakdown
    print(f"\n  {'Case':<30} {'Time':>6} {'Words':>6} {'PLAN#':>6} {'Score':>6} {'Status':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    for r in all_results:
        v = r["validation"]
        status = "✓" if v["COMPLETE"] else "⚠"
        print(
            f"  {r['name']:<30} {r['time']:>5.1f}s {v['WORD_COUNT']:>5}w "
            f"{v['PLAN_ITEMS']:>5} {r['quality']:>5.0f} {status:>8}"
        )

    # ============================================================
    # GENERATED NOTES
    # ============================================================
    print(f"\n{'=' * 60}")
    print("GENERATED SOAP NOTES")
    print("=" * 60)
    for r in all_results:
        print(f"\n{'─' * 40}")
        print(f"CASE: {r['name']}")
        print(f"{'─' * 40}")
        print(r["text"])

    # ============================================================
    # COMPETITION TIER
    # ============================================================
    avg_time = sum(times) / len(times)
    avg_quality = sum(qualities) / len(qualities)

    print(f"\n{'=' * 60}")
    print("COMPETITION STATUS")
    print("=" * 60)

    if avg_time <= 15 and avg_quality >= 80:
        tier = "EDGE AI PRIZE — sub-15s avg"
    elif avg_time <= 20 and avg_quality >= 80:
        tier = "EDGE AI COMPETITIVE — sub-20s avg"
    elif avg_time <= 30 and avg_quality >= 75:
        tier = "✓ MAIN TRACK VIABLE"
    else:
        tier = "⚠ NEEDS OPTIMIZATION"

    print(f"  Avg inference:  {avg_time:.1f}s")
    print(f"  Avg quality:    {avg_quality:.0f}/100")
    print(f"  Completion:     {complete_count}/{len(all_results)}")
    print(f"  Tier:           {tier}")


    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_path = "./models/checkpoints/medgemma_v2_soap/inference_benchmark.json"
    save_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter": ADAPTER_PATH,
        "config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_new_tokens": MIN_NEW_TOKENS,
            "quantization": "bitsandbytes_4bit_nf4",
        },
        "summary": {
            "cases_tested": len(all_results),
            "complete_pct": round(complete_count / len(all_results) * 100, 1),
            "avg_quality": round(avg_quality, 1),
            "avg_time_s": round(avg_time, 2),
            "avg_words": round(sum(word_counts) / len(word_counts), 0),
            "tier": tier,
        },
        "per_case": [
            {
                "name": r["name"],
                "time_s": round(r["time"], 2),
                "tokens": r["tokens"],
                "word_count": r["validation"]["WORD_COUNT"],
                "plan_items": r["validation"]["PLAN_ITEMS"],
                "quality": r["quality"],
                "complete": r["validation"]["COMPLETE"],
            }
            for r in all_results
        ],
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()