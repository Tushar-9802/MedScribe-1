"""
Generate SOAP notes from filtered MTSamples using GPT-4o.

Safety features:
- Validates input file before spending money
- Per-sample SOAP validation
- Batch quality monitoring — aborts if failure rate exceeds threshold
- Checkpoint every batch (resume-safe)
- Cost tracking with budget cap
- Dry-run mode (first 5 samples) before full generation

Input:  data/processed/filtered_for_generation.csv
Output: data/processed/generated_soap.csv
"""
import os
import json
import time
import sys
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "data/processed/filtered_for_generation.csv"
OUTPUT_FILE = "data/processed/generated_soap.csv"
CHECKPOINT_FILE = "data/processed/generation_checkpoint.json"
BATCH_SIZE = 25  # Save checkpoint every N samples
MODEL = "gpt-4o"
TEMPERATURE = 0.2
MAX_TOKENS = 500

# Safety thresholds
MAX_BUDGET = 15.0           # Abort if cost exceeds $15
FAILURE_RATE_ABORT = 0.30   # Abort if >30% of samples fail validation
MIN_BATCH_FOR_CHECK = 50    # Start checking failure rate after this many

DRY_RUN = "--dry-run" in sys.argv  # Run first 5 only with --dry-run flag

SYSTEM_PROMPT = """You are a clinical documentation specialist. You write SOAP notes that are concise, clinically complete, and use standard medical abbreviations. You document ONLY what is explicitly stated in the source text. You never infer, fabricate, or assume clinical findings not present in the source."""

USER_PROMPT_TEMPLATE = """Convert the following medical transcript into a structured SOAP note.

RULES:
- Document ONLY findings explicitly present in the transcript. Do NOT infer or fabricate any clinical data.
- Use standard abbreviations: yo, M/F, c/o, h/o, BP, HR, RR, SpO2, Temp, WBC, Hgb, Cr, BUN, etc.
- Write in the concise but complete style of an experienced physician's progress note. Use abbreviations where standard, but every clinical finding and order must be explicitly documented.
- SUBJECTIVE: Chief complaint, HPI, relevant PMH/meds/allergies, ROS. 2-4 lines.
- OBJECTIVE: Vitals with actual values (never "WNL"), physical exam findings, lab/imaging results if mentioned. 2-4 lines.
- ASSESSMENT: Primary diagnosis with reasoning, differentials if applicable. 1-3 lines.
- PLAN: Must include ALL applicable: (1) diagnostics ordered with specifics, (2) medications with dose/route/frequency, (3) monitoring parameters, (4) disposition/follow-up. 3-6 lines.
- If the transcript does not contain enough information for a section, write "Not documented in source" for that section.
- Target total length: 200-300 words.

TRANSCRIPT:
{transcript}

SOAP NOTE:"""

# ============================================================
# VALIDATION
# ============================================================
REQUIRED_SECTIONS = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]


def validate_soap(text):
    """Validate generated SOAP note quality. Returns dict with valid flag and issues."""
    result = {"valid": True, "issues": []}

    if not text or not isinstance(text, str):
        return {"valid": False, "issues": ["Empty or non-string output"]}

    text_upper = text.upper()

    # Check all sections present
    missing = []
    for section in REQUIRED_SECTIONS:
        if section + ":" not in text_upper and section not in text_upper:
            missing.append(section)
    if missing:
        result["valid"] = False
        result["issues"].append(f"Missing: {', '.join(missing)}")

    # Check for "WNL" (lazy documentation)
    if " WNL" in text_upper or text_upper.startswith("WNL"):
        result["issues"].append("Contains WNL — lazy documentation")

    # Check minimum word count
    word_count = len(text.split())
    if word_count < 80:
        result["valid"] = False
        result["issues"].append(f"Too short: {word_count} words (<80)")
    elif word_count < 100:
        result["issues"].append(f"Short: {word_count} words (80-100)")

    # Check maximum word count
    if word_count > 400:
        result["issues"].append(f"Verbose: {word_count} words (>400)")

    # Check PLAN has substance
    plan_idx = text_upper.find("PLAN")
    if plan_idx != -1:
        plan_text = text[plan_idx:]
        plan_words = len(plan_text.split())
        if plan_words < 15:
            result["valid"] = False
            result["issues"].append(f"PLAN too thin: {plan_words} words")
    elif "PLAN" not in text_upper:
        pass  # Already caught by missing sections check

    # Check for hallucination markers (model adding caveats)
    hallucination_flags = [
        "not documented in the transcript but",
        "although not explicitly mentioned",
        "it can be inferred",
        "likely",  # Only flag if in OBJECTIVE section (inferring findings)
    ]
    obj_idx = text_upper.find("OBJECTIVE")
    assess_idx = text_upper.find("ASSESSMENT")
    if obj_idx != -1 and assess_idx != -1:
        obj_section = text[obj_idx:assess_idx].lower()
        if "likely" in obj_section or "probable" in obj_section:
            result["issues"].append("OBJECTIVE contains inferential language")

    return result


def count_tokens_approx(text):
    """Rough token estimate (1 token ~ 4 chars)."""
    return len(text) // 4


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"processed": 0, "successful": 0, "failed": 0, "invalid": 0, "total_cost": 0.0}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"SOAP Generation — {MODEL}" + (" [DRY RUN]" if DRY_RUN else ""))
    print("=" * 60)

    # ============================================================
    # GATE 0: Input validation
    # ============================================================
    if not os.path.exists(INPUT_FILE):
        print(f"\n✗ ABORT: Input not found: {INPUT_FILE}")
        print(f"  Run filter_mtsamples.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nSamples to process: {len(df)}")

    if len(df) < 100:
        print(f"\n✗ ABORT: Only {len(df)} input samples. Expected 500+.")
        sys.exit(1)

    # Validate input has required columns
    for col in ["transcription", "medical_specialty"]:
        if col not in df.columns:
            print(f"\n✗ ABORT: Missing column '{col}' in input CSV.")
            sys.exit(1)

    # Check transcriptions are non-empty
    empty_count = (df["transcription"].isna() | (df["transcription"].str.len() < 50)).sum()
    if empty_count > len(df) * 0.1:
        print(f"\n✗ ABORT: {empty_count}/{len(df)} transcriptions are empty or too short.")
        sys.exit(1)
    print(f"✓ Input validated: {len(df)} samples, {empty_count} empty/short")

    # ============================================================
    # GATE 1: API key exists
    # ============================================================
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"\n✗ ABORT: OPENAI_API_KEY not found in environment or .env file.")
        sys.exit(1)

    client = OpenAI()

    # Quick API test
    try:
        test_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"✓ API connection verified: {MODEL}")
    except Exception as e:
        print(f"\n✗ ABORT: API test failed: {e}")
        sys.exit(1)

    # ============================================================
    # LOAD CHECKPOINT / EXISTING RESULTS
    # ============================================================
    cp = load_checkpoint()
    start_idx = cp["processed"]

    if os.path.exists(OUTPUT_FILE) and start_idx > 0:
        results_df = pd.read_csv(OUTPUT_FILE)
        results = results_df.to_dict("records")
        print(f"Resuming from sample {start_idx} ({cp['successful']} successful, ${cp['total_cost']:.4f} spent)")
    else:
        results = []

    # Cost tracking (GPT-4o pricing)
    INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000

    # Dry run limit
    total = min(5, len(df)) if DRY_RUN else len(df)
    if DRY_RUN:
        print(f"\n🔍 DRY RUN: Processing first {total} samples only")
        print(f"   Run without --dry-run for full generation\n")

    # ============================================================
    # GENERATION LOOP
    # ============================================================
    batch_count = 0

    for idx in range(start_idx, total):
        row = df.iloc[idx]
        transcript = row["transcription"]
        specialty = row.get("medical_specialty", "Unknown")
        sample_name = row.get("sample_name", "")

        # Skip empty
        if not isinstance(transcript, str) or len(transcript.strip()) < 50:
            cp["processed"] += 1
            cp["failed"] += 1
            continue

        # Generate
        prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)
        input_tokens = count_tokens_approx(SYSTEM_PROMPT + prompt)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            soap = response.choices[0].message.content.strip()
            output_tokens = count_tokens_approx(soap)

            # Cost
            cost = (input_tokens * INPUT_COST_PER_TOKEN) + (
                output_tokens * OUTPUT_COST_PER_TOKEN
            )
            cp["total_cost"] += cost

            # Validate
            validation = validate_soap(soap)
            if not validation["valid"]:
                cp["invalid"] += 1

            results.append({
                "medical_specialty": specialty,
                "sample_name": sample_name,
                "transcription": transcript,
                "soap_note": soap,
                "valid": validation["valid"],
                "issues": "; ".join(validation["issues"]) if validation["issues"] else "",
                "word_count": len(soap.split()),
            })

            cp["successful"] += 1
            status = "✓" if validation["valid"] else f"⚠ {'; '.join(validation['issues'])}"

        except Exception as e:
            results.append({
                "medical_specialty": specialty,
                "sample_name": sample_name,
                "transcription": transcript,
                "soap_note": "",
                "valid": False,
                "issues": f"API error: {str(e)[:100]}",
                "word_count": 0,
            })
            cp["failed"] += 1
            status = f"✗ {str(e)[:50]}"
            time.sleep(3)  # Back off on error

        cp["processed"] += 1
        batch_count += 1

        # Progress
        pct = cp["processed"] / total * 100
        print(
            f"  [{cp['processed']}/{total}] ({pct:.1f}%) "
            f"${cp['total_cost']:.3f} "
            f"[{specialty[:20]}] {status}"
        )

        # ============================================================
        # SAFETY CHECKS — run every batch
        # ============================================================

        # Budget cap
        if cp["total_cost"] > MAX_BUDGET:
            print(f"\n✗ ABORT: Budget exceeded (${cp['total_cost']:.2f} > ${MAX_BUDGET})")
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            save_checkpoint(cp)
            sys.exit(1)

        # Failure rate check (after enough samples)
        if cp["processed"] >= MIN_BATCH_FOR_CHECK:
            total_attempted = cp["successful"] + cp["failed"]
            if total_attempted > 0:
                api_fail_rate = cp["failed"] / total_attempted
                if api_fail_rate > FAILURE_RATE_ABORT:
                    print(f"\n✗ ABORT: API failure rate {api_fail_rate:.0%} exceeds {FAILURE_RATE_ABORT:.0%}")
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                    save_checkpoint(cp)
                    sys.exit(1)

            # Quality check (invalid SOAP rate)
            if cp["successful"] > 0:
                invalid_rate = cp["invalid"] / cp["successful"]
                if invalid_rate > FAILURE_RATE_ABORT:
                    print(f"\n⚠ WARNING: {invalid_rate:.0%} of generated notes are invalid")
                    if invalid_rate > 0.5:
                        print(f"✗ ABORT: Quality too low ({invalid_rate:.0%} invalid). Check prompt.")
                        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                        save_checkpoint(cp)
                        sys.exit(1)

        # Checkpoint save
        if batch_count >= BATCH_SIZE:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            save_checkpoint(cp)
            batch_count = 0
            valid_so_far = sum(1 for r in results if r["valid"])
            print(
                f"    💾 Saved. {valid_so_far} valid / {len(results)} total. "
                f"Cost: ${cp['total_cost']:.4f}"
            )

        # Rate limiting
        time.sleep(0.15)

    # ============================================================
    # FINAL SAVE
    # ============================================================
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False)
    save_checkpoint(cp)

    # ============================================================
    # VERIFY SAVED FILE
    # ============================================================
    verify_df = pd.read_csv(OUTPUT_FILE)
    if len(verify_df) != len(results):
        print(f"\n⚠ Save mismatch: {len(results)} results, {len(verify_df)} rows in file")

    # ============================================================
    # SUMMARY
    # ============================================================
    valid_count = final_df["valid"].sum() if len(final_df) > 0 else 0
    invalid_count = len(final_df) - valid_count

    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Total processed:  {cp['processed']}")
    print(f"  API successful:   {cp['successful']}")
    print(f"  API failed:       {cp['failed']}")
    print(f"  Valid SOAP notes: {valid_count}")
    print(f"  Invalid SOAP:     {invalid_count}")
    print(f"  Total cost:       ${cp['total_cost']:.4f}")

    if valid_count > 0:
        valid_df = final_df[final_df["valid"] == True]
        print(f"\n  Word count stats (valid only):")
        print(f"    Mean:   {valid_df['word_count'].mean():.0f}")
        print(f"    Median: {valid_df['word_count'].median():.0f}")
        print(f"    Min:    {valid_df['word_count'].min():.0f}")
        print(f"    Max:    {valid_df['word_count'].max():.0f}")

    # Issue distribution
    issues_col = final_df[final_df["issues"] != ""]["issues"]
    if len(issues_col) > 0:
        print(f"\n  Issue distribution:")
        all_issues = []
        for i in issues_col:
            if isinstance(i, str):
                all_issues.extend(i.split("; "))
        from collections import Counter
        for issue, count in Counter(all_issues).most_common(10):
            print(f"    {issue}: {count}")

    # Quality gate for next step
    if valid_count < 200:
        print(f"\n⚠ WARNING: Only {valid_count} valid samples. May not be enough for training.")
        print(f"  Consider: adjusting prompt, reviewing failures, or adding data sources.")
    else:
        print(f"\n✓ READY for next step: python scripts/prepare_training_data.py")

    if DRY_RUN:
        print(f"\n🔍 This was a DRY RUN. Review output above.")
        print(f"   If quality looks good, run again without --dry-run")
        print(f"\n   Sample output (first valid note):")
        first_valid = final_df[final_df["valid"] == True].head(1)
        if len(first_valid) > 0:
            print(f"\n{first_valid.iloc[0]['soap_note']}")

    print("=" * 60)


if __name__ == "__main__":
    main()