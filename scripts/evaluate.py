"""
MedScribe v2 — Standalone Evaluation
=====================================
Computes eval_loss + quality metrics on the saved adapter.
No retraining required.

Outputs:
- models/checkpoints/medgemma_v2_soap/eval_results.json
- Console summary for writeup
"""
import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
ADAPTER_PATH = "./models/checkpoints/medgemma_v2_soap/final_model"
VAL_FILE = "./data/processed/val.jsonl"
TEST_FILE = "./data/processed/test.jsonl"
OUTPUT_FILE = "./models/checkpoints/medgemma_v2_soap/eval_results.json"

# Generation config
MAX_NEW_TOKENS = 350
MAX_SEQ_LENGTH = 1024

# Match training prompt EXACTLY
INFERENCE_PROMPT_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

Write a complete PLAN (treatments, monitoring, follow-up). End with a full sentence.
SOAP NOTE:"""

print("=" * 60)
print("MedScribe v2 — Evaluation")
print("=" * 60)

# ============================================================
# GATE 0: Validate files
# ============================================================
for path, name in [(ADAPTER_PATH, "adapter"), (VAL_FILE, "val data")]:
    if not os.path.exists(path):
        print(f"✗ ABORT: {name} not found: {path}")
        sys.exit(1)

if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print(f"✗ ABORT: adapter_config.json not found in {ADAPTER_PATH}")
    sys.exit(1)

print(f"✓ Adapter: {ADAPTER_PATH}")
print(f"✓ Val data: {VAL_FILE}")

# ============================================================
# LOAD MODEL
# ============================================================
print(f"\n[1/4] Loading model + adapter...")

cfg = PeftConfig.from_pretrained(ADAPTER_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    cfg.base_model_name_or_path, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print(f"✓ Model loaded")

# ============================================================
# LOAD VAL DATA
# ============================================================
print(f"\n[2/4] Loading validation data...")

val_dataset = load_dataset("json", data_files={"val": VAL_FILE})["val"]
print(f"  Val samples: {len(val_dataset)}")

# Also load test if available
test_dataset = None
if os.path.exists(TEST_FILE):
    test_dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]
    print(f"  Test samples: {len(test_dataset)}")

# ============================================================
# PART A: COMPUTE EVAL LOSS (cross-entropy, same metric as Trainer)
# ============================================================
print(f"\n[3/4] Computing eval loss...")


def compute_loss_for_sample(messages):
    """Compute cross-entropy loss for a single sample, matching training tokenization."""
    user_text = messages[0]["content"]
    assistant_text = messages[1]["content"]

    # Tokenize prompt
    prompt_tokens = tokenizer(
        user_text, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LENGTH
    )
    # Tokenize response
    response_tokens = tokenizer(
        assistant_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH // 2,
    )

    # Truncate prompt to keep full response (this is the fix for NaN)
    max_prompt_len = MAX_SEQ_LENGTH - len(response_tokens["input_ids"])
    if max_prompt_len < 10:
        return None  # Skip if response alone exceeds limit

    prompt_ids = prompt_tokens["input_ids"][:max_prompt_len]

    # Concatenate
    input_ids = prompt_ids + response_tokens["input_ids"]
    token_type_ids = [0] * len(input_ids)

    # Labels: -100 for prompt, actual tokens for response
    labels = [-100] * len(prompt_ids) + response_tokens["input_ids"]

    # Check we have response tokens
    response_label_count = sum(1 for l in labels if l != -100)
    if response_label_count < 5:
        return None

    # To tensors
    input_ids_t = torch.tensor([input_ids], device=model.device)
    token_type_ids_t = torch.tensor([token_type_ids], device=model.device)
    labels_t = torch.tensor([labels], device=model.device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids_t,
            token_type_ids=token_type_ids_t,
            labels=labels_t,
        )

    return outputs.loss.item()


losses = []
skipped = 0

for i, sample in enumerate(val_dataset):
    loss = compute_loss_for_sample(sample["messages"])
    if loss is None or np.isnan(loss):
        skipped += 1
        continue
    losses.append(loss)
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(val_dataset)}] running avg loss: {np.mean(losses):.4f}")

eval_loss = np.mean(losses) if losses else float("nan")
eval_loss_std = np.std(losses) if losses else 0.0
print(f"\n  Val loss:  {eval_loss:.4f} ± {eval_loss_std:.4f}")
print(f"  Samples:   {len(losses)} evaluated, {skipped} skipped")

# ============================================================
# PART B: GENERATION QUALITY METRICS
# ============================================================
print(f"\n[4/4] Running generation quality evaluation...")

REQUIRED_SECTIONS = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
PLAN_ACTION_WORDS = [
    "obtain", "order", "administer", "prescribe", "start", "continue",
    "monitor", "refer", "schedule", "follow", "counsel", "advise",
    "discontinue", "increase", "decrease", "check", "evaluate",
    "recommend", "consider", "discharge", "admit",
]


def extract_first_soap(text):
    """Extract only the first SOAP note if model repeats."""
    upper = text.upper()
    # Find second occurrence of SUBJECTIVE (repetition)
    first = upper.find("SUBJECTIVE")
    if first == -1:
        return text
    second = upper.find("SUBJECTIVE", first + 20)
    if second != -1:
        return text[:second].strip()
    return text


def evaluate_soap(text):
    """Score a generated SOAP note on multiple quality dimensions."""
    result = {
        "has_all_sections": True,
        "missing_sections": [],
        "word_count": 0,
        "plan_words": 0,
        "plan_items": 0,
        "plan_has_actions": False,
        "has_wnl": False,
        "has_not_documented": False,
        "repeats": False,
        "quality_score": 0,
    }

    if not text or len(text.strip()) < 20:
        result["has_all_sections"] = False
        result["missing_sections"] = REQUIRED_SECTIONS
        return result

    # Check for repetition
    upper = text.upper()
    first = upper.find("SUBJECTIVE")
    if first != -1:
        second = upper.find("SUBJECTIVE", first + 20)
        if second != -1:
            result["repeats"] = True

    # Extract first note only
    clean = extract_first_soap(text)
    clean_upper = clean.upper()

    # Section presence
    for section in REQUIRED_SECTIONS:
        if section not in clean_upper:
            result["has_all_sections"] = False
            result["missing_sections"].append(section)

    result["word_count"] = len(clean.split())
    result["has_wnl"] = " WNL" in clean_upper or clean_upper.startswith("WNL")
    result["has_not_documented"] = "NOT DOCUMENTED" in clean_upper

    # PLAN analysis
    plan_idx = clean_upper.find("PLAN")
    if plan_idx != -1:
        plan_text = clean[plan_idx:]
        result["plan_words"] = len(plan_text.split())
        # Count numbered items
        import re
        items = re.findall(r"\d+\.", plan_text)
        result["plan_items"] = len(items)
        # Check for action words
        plan_lower = plan_text.lower()
        actions_found = [w for w in PLAN_ACTION_WORDS if w in plan_lower]
        result["plan_has_actions"] = len(actions_found) >= 2

    # Quality score (0-100)
    score = 0
    if result["has_all_sections"]:
        score += 30
    else:
        score += max(0, 30 - len(result["missing_sections"]) * 8)
    if 100 <= result["word_count"] <= 300:
        score += 20
    elif 80 <= result["word_count"] <= 350:
        score += 10
    if result["plan_words"] >= 20:
        score += 15
    elif result["plan_words"] >= 10:
        score += 8
    if result["plan_has_actions"]:
        score += 15
    if not result["has_wnl"]:
        score += 10
    if not result["repeats"]:
        score += 10
    result["quality_score"] = score

    return result


# Generate on a subset of val samples (all 89 — fast enough)
gen_results = []
gen_times = []
num_to_eval = len(val_dataset)

print(f"  Generating SOAP notes for {num_to_eval} val samples...")

for i, sample in enumerate(val_dataset):
    messages = sample["messages"]
    user_text = messages[0]["content"]
    reference = messages[1]["content"]

    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - start
    gen_times.append(gen_time)

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    quality = evaluate_soap(generated)
    quality["gen_time"] = gen_time
    quality["reference_words"] = len(reference.split())
    gen_results.append(quality)

    if (i + 1) % 20 == 0:
        avg_score = np.mean([r["quality_score"] for r in gen_results])
        avg_time = np.mean(gen_times)
        print(f"  [{i+1}/{num_to_eval}] avg quality: {avg_score:.0f}/100, avg time: {avg_time:.1f}s")

# ============================================================
# AGGREGATE RESULTS
# ============================================================
n = len(gen_results)
all_sections_pct = sum(1 for r in gen_results if r["has_all_sections"]) / n * 100
avg_quality = np.mean([r["quality_score"] for r in gen_results])
avg_words = np.mean([r["word_count"] for r in gen_results])
median_words = np.median([r["word_count"] for r in gen_results])
avg_plan_words = np.mean([r["plan_words"] for r in gen_results])
avg_plan_items = np.mean([r["plan_items"] for r in gen_results])
plan_actions_pct = sum(1 for r in gen_results if r["plan_has_actions"]) / n * 100
wnl_pct = sum(1 for r in gen_results if r["has_wnl"]) / n * 100
repeats_pct = sum(1 for r in gen_results if r["repeats"]) / n * 100
not_doc_pct = sum(1 for r in gen_results if r["has_not_documented"]) / n * 100
avg_time = np.mean(gen_times)
median_time = np.median(gen_times)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "timestamp": datetime.now().isoformat(),
    "adapter": ADAPTER_PATH,
    "val_samples": len(val_dataset),

    # Loss
    "eval_loss": round(eval_loss, 4),
    "eval_loss_std": round(eval_loss_std, 4),
    "eval_loss_samples": len(losses),
    "eval_loss_skipped": skipped,

    # Train loss (from training log)
    "train_loss": 0.8286,  # From training_log.json

    # Generation quality
    "quality": {
        "avg_score": round(avg_quality, 1),
        "all_sections_pct": round(all_sections_pct, 1),
        "avg_word_count": round(avg_words, 0),
        "median_word_count": round(median_words, 0),
        "avg_plan_words": round(avg_plan_words, 1),
        "avg_plan_items": round(avg_plan_items, 1),
        "plan_has_actions_pct": round(plan_actions_pct, 1),
        "wnl_pct": round(wnl_pct, 1),
        "repeats_pct": round(repeats_pct, 1),
        "not_documented_pct": round(not_doc_pct, 1),
    },

    # Speed
    "inference": {
        "avg_time_s": round(avg_time, 2),
        "median_time_s": round(median_time, 2),
        "min_time_s": round(min(gen_times), 2),
        "max_time_s": round(max(gen_times), 2),
    },
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved: {OUTPUT_FILE}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\n  Loss:")
print(f"    Train loss:  {results['train_loss']:.4f}")
print(f"    Val loss:    {eval_loss:.4f} ± {eval_loss_std:.4f}")
gap = results["train_loss"] - eval_loss
if abs(gap) < 0.2:
    print(f"    Gap:         {gap:.4f} — ✓ Good generalization")
elif gap < -0.3:
    print(f"    Gap:         {gap:.4f} — ⚠ Possible overfitting")
else:
    print(f"    Gap:         {gap:.4f}")

print(f"\n  Quality ({n} samples):")
print(f"    Avg score:       {avg_quality:.0f}/100")
print(f"    All 4 sections:  {all_sections_pct:.0f}%")
print(f"    Avg words:       {avg_words:.0f} (median {median_words:.0f})")
print(f"    Avg PLAN words:  {avg_plan_words:.0f}")
print(f"    Avg PLAN items:  {avg_plan_items:.1f}")
print(f"    PLAN has actions: {plan_actions_pct:.0f}%")
print(f"    WNL present:     {wnl_pct:.0f}%")
print(f"    Repeats:         {repeats_pct:.0f}%")
print(f"    'Not documented': {not_doc_pct:.0f}%")

print(f"\n  Inference speed:")
print(f"    Avg:    {avg_time:.1f}s")
print(f"    Median: {median_time:.1f}s")
print(f"    Range:  {min(gen_times):.1f}s — {max(gen_times):.1f}s")

print(f"\n{'=' * 60}")
print(f"✓ EVALUATION COMPLETE")
print(f"  Results: {OUTPUT_FILE}")
print("=" * 60)