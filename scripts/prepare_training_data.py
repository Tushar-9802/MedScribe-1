"""
Prepare training data from generated SOAP notes.

1. Validate input file
2. Filter to valid-only + quality gate
3. Stratified 80/10/10 split by specialty
4. Format to conversational JSONL (matches inference prompt EXACTLY)
5. Verify output integrity

Input:  data/processed/generated_soap.csv
Output: data/processed/train.jsonl, val.jsonl, test.jsonl
"""
import json
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("Prepare Training Data")
print("=" * 60)

# ============================================================
# GATE 0: Input file exists
# ============================================================
INPUT_FILE = "data/processed/generated_soap.csv"

if not os.path.exists(INPUT_FILE):
    print(f"\n✗ ABORT: Input not found: {INPUT_FILE}")
    print(f"  Run generate_soap_gpt4o.py first.")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"\nTotal generated samples: {len(df)}")

# ============================================================
# GATE 1: Required columns exist
# ============================================================
REQUIRED_COLS = ["transcription", "soap_note", "valid", "word_count", "medical_specialty"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    print(f"\n✗ ABORT: Missing columns: {missing}")
    print(f"  Available: {df.columns.tolist()}")
    sys.exit(1)
print(f"✓ Required columns present")

# ============================================================
# GATE 2: Enough valid samples exist
# ============================================================
valid_count = df["valid"].sum()
total_count = len(df)
valid_rate = valid_count / total_count if total_count > 0 else 0

print(f"\nQuality overview:")
print(f"  Valid:   {valid_count} ({valid_rate:.0%})")
print(f"  Invalid: {total_count - valid_count}")

if valid_count < 200:
    print(f"\n✗ ABORT: Only {valid_count} valid samples. Need 200+ for training.")
    print(f"  Check generation quality or add more source data.")
    sys.exit(1)

# ============================================================
# FILTER TO VALID + QUALITY RANGE
# ============================================================
df = df[df["valid"] == True].copy()
print(f"\nAfter valid filter: {len(df)}")

# Word count range (match training target)
before = len(df)
df = df[(df["word_count"] >= 100) & (df["word_count"] <= 350)]
print(f"After word count filter (100-350): {len(df)} (removed {before - len(df)})")

# Drop empty SOAP notes
before = len(df)
df = df[df["soap_note"].str.strip().str.len() > 50]
print(f"After empty filter: {len(df)} (removed {before - len(df)})")

# Drop empty transcriptions
before = len(df)
df = df[df["transcription"].str.strip().str.len() > 50]
print(f"After transcription check: {len(df)} (removed {before - len(df)})")

# ============================================================
# GATE 3: Still enough after filtering
# ============================================================
if len(df) < 200:
    print(f"\n✗ ABORT: Only {len(df)} samples after quality filtering. Not enough.")
    sys.exit(1)

if len(df) < 500:
    print(f"\n⚠ WARNING: {len(df)} samples is workable but limited. 800+ recommended.")

# ============================================================
# CONTENT VALIDATION — spot check for quality
# ============================================================
print(f"\nContent spot check (5 random samples):")
spot_check = df.sample(min(5, len(df)), random_state=42)
issues_found = 0

for _, row in spot_check.iterrows():
    soap = row["soap_note"].upper()
    problems = []

    # All 4 sections present?
    for section in ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]:
        if section not in soap:
            problems.append(f"missing {section}")

    # PLAN has substance?
    plan_idx = soap.find("PLAN")
    if plan_idx != -1:
        plan_text = row["soap_note"][plan_idx:]
        if len(plan_text.split()) < 15:
            problems.append("thin PLAN")

    # WNL check
    if "WNL" in soap:
        problems.append("contains WNL")

    status = "✓" if not problems else f"⚠ {', '.join(problems)}"
    if problems:
        issues_found += 1
    print(f"  [{row['medical_specialty'][:25]}] {row['word_count']}w — {status}")

if issues_found > 3:
    print(f"\n⚠ WARNING: {issues_found}/5 spot check samples have issues.")
    print(f"  Consider reviewing generation quality before training.")

# ============================================================
# STRATIFIED SPLIT
# ============================================================
print(f"\nSplitting 80/10/10 stratified by specialty...")

specialty_counts = df["medical_specialty"].value_counts()
MIN_FOR_STRATIFY = 5


def map_rare(spec):
    return spec if specialty_counts.get(spec, 0) >= MIN_FOR_STRATIFY else "Other"


df["strat_group"] = df["medical_specialty"].apply(map_rare)

# First split: 80/20
try:
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["strat_group"]
    )
except ValueError as e:
    print(f"  Stratification failed ({e}), using random split")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50/50 of temp
try:
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["strat_group"]
    )
except ValueError:
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")
print(f"  Total: {len(train_df) + len(val_df) + len(test_df)}")

# ============================================================
# GATE 4: No overlap between splits
# ============================================================
train_idx = set(train_df.index)
val_idx = set(val_df.index)
test_idx = set(test_df.index)

overlap_tv = train_idx & val_idx
overlap_tt = train_idx & test_idx
overlap_vt = val_idx & test_idx

if overlap_tv or overlap_tt or overlap_vt:
    print(f"\n✗ ABORT: Data leakage detected!")
    print(f"  Train/Val overlap: {len(overlap_tv)}")
    print(f"  Train/Test overlap: {len(overlap_tt)}")
    print(f"  Val/Test overlap: {len(overlap_vt)}")
    sys.exit(1)
print(f"✓ No overlap between splits — verified")

# ============================================================
# FORMAT TO JSONL
# ============================================================
# CRITICAL: This prompt MUST EXACTLY match the inference prompt.
# Any mismatch between training and inference degrades quality silently.

TRAINING_PROMPT_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

Write a complete PLAN (treatments, monitoring, follow-up). End with a full sentence.
SOAP NOTE:"""


def row_to_jsonl(row):
    """Convert a row to conversational JSONL format."""
    transcript = row["transcription"].strip()
    soap = row["soap_note"].strip()

    # Final safety: skip if either is empty
    if len(transcript) < 50 or len(soap) < 50:
        return None

    prompt = TRAINING_PROMPT_TEMPLATE.format(transcript=transcript)
    return json.dumps({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": soap},
        ]
    }, ensure_ascii=False)


def save_jsonl(dataframe, filepath):
    """Save dataframe as JSONL with per-line validation."""
    written = 0
    skipped = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            line = row_to_jsonl(row)
            if line is None:
                skipped += 1
                continue
            f.write(line + "\n")
            written += 1
    return written, skipped


os.makedirs("data/processed", exist_ok=True)

train_path = "data/processed/train.jsonl"
val_path = "data/processed/val.jsonl"
test_path = "data/processed/test.jsonl"

print(f"\nWriting JSONL files...")
train_written, train_skipped = save_jsonl(train_df, train_path)
val_written, val_skipped = save_jsonl(val_df, val_path)
test_written, test_skipped = save_jsonl(test_df, test_path)

print(f"  train.jsonl: {train_written} written, {train_skipped} skipped")
print(f"  val.jsonl:   {val_written} written, {val_skipped} skipped")
print(f"  test.jsonl:  {test_written} written, {test_skipped} skipped")

total_skipped = train_skipped + val_skipped + test_skipped
if total_skipped > 0:
    print(f"  ⚠ {total_skipped} samples skipped due to empty content")

# ============================================================
# GATE 5: Verify JSONL integrity
# ============================================================
print(f"\nJSONL integrity verification:")
all_good = True

for name, path, expected in [
    ("train", train_path, train_written),
    ("val", val_path, val_written),
    ("test", test_path, test_written),
]:
    valid_lines = 0
    invalid_lines = 0
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                obj = json.loads(line)
                # Structural checks
                assert "messages" in obj, "missing 'messages' key"
                assert len(obj["messages"]) == 2, f"expected 2 messages, got {len(obj['messages'])}"
                assert obj["messages"][0]["role"] == "user", "first message not user"
                assert obj["messages"][1]["role"] == "assistant", "second message not assistant"
                assert len(obj["messages"][0]["content"]) > 50, "user content too short"
                assert len(obj["messages"][1]["content"]) > 50, "assistant content too short"
                # Content checks
                assert "SOAP NOTE:" in obj["messages"][0]["content"], "missing SOAP NOTE prompt marker"
                valid_lines += 1
            except (json.JSONDecodeError, AssertionError) as e:
                invalid_lines += 1
                if invalid_lines <= 3:  # Show first 3 errors
                    print(f"    ✗ {name} line {line_num}: {e}")

    if total_lines != expected:
        print(f"  ✗ {name}: expected {expected} lines, found {total_lines}")
        all_good = False
    elif invalid_lines > 0:
        print(f"  ⚠ {name}: {valid_lines}/{total_lines} valid ({invalid_lines} invalid)")
        if invalid_lines > total_lines * 0.05:
            all_good = False
    else:
        print(f"  ✓ {name}: {valid_lines}/{total_lines} valid")

if not all_good:
    print(f"\n⚠ WARNING: Some integrity issues detected. Review before training.")
else:
    print(f"\n✓ All JSONL files verified")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("TRAINING DATA SUMMARY")
print("=" * 60)
total_written = train_written + val_written + test_written
print(f"  Total training samples: {total_written}")
print(f"    Train: {train_written}")
print(f"    Val:   {val_written}")
print(f"    Test:  {test_written}")

print(f"\n  Specialty distribution (train):")
for spec, count in train_df["medical_specialty"].value_counts().head(10).items():
    print(f"    {spec}: {count}")

print(f"\n  Word count stats (train):")
wc = train_df["word_count"]
print(f"    Mean:   {wc.mean():.0f}")
print(f"    Median: {wc.median():.0f}")
print(f"    Min:    {wc.min():.0f}")
print(f"    Max:    {wc.max():.0f}")

print(f"\n  Files:")
for path in [train_path, val_path, test_path]:
    size_kb = os.path.getsize(path) / 1024
    print(f"    {path}: {size_kb:.1f} KB")

print(f"\n{'=' * 60}")
if all_good and total_written >= 200:
    print(f"✓ READY for training: python src/training/train.py")
else:
    print(f"⚠ Review issues above before training")
print("=" * 60)