"""
Filter MTSamples to encounter-type notes suitable for SOAP generation.

Two-pass filter:
1. Discard non-encounter specialties (Surgery, Radiology, etc.)
2. Within kept specialties, discard procedure/operative notes by sample_name keywords

Validation gates at every stage — aborts early on bad data.

Output: data/processed/filtered_for_generation.csv
"""
import pandas as pd
import os
import sys

print("=" * 60)
print("MTSamples Filtering — Encounter Notes Only")
print("=" * 60)

# ============================================================
# GATE 0: Input file exists and is readable
# ============================================================
INPUT_FILE = "data/raw/mtsamples.csv"

if not os.path.exists(INPUT_FILE):
    print(f"\n✗ ABORT: Input file not found: {INPUT_FILE}")
    sys.exit(1)

try:
    df = pd.read_csv(INPUT_FILE)
except Exception as e:
    print(f"\n✗ ABORT: Failed to read CSV: {e}")
    sys.exit(1)

print(f"\nRaw samples loaded: {len(df)}")

# ============================================================
# GATE 1: Required columns exist
# ============================================================
REQUIRED_COLS = ["medical_specialty", "sample_name", "transcription"]
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    print(f"\n✗ ABORT: Missing required columns: {missing_cols}")
    print(f"  Available columns: {df.columns.tolist()}")
    sys.exit(1)
print(f"✓ Required columns present: {REQUIRED_COLS}")

# ============================================================
# GATE 2: Data isn't empty or corrupt
# ============================================================
if len(df) < 100:
    print(f"\n✗ ABORT: Only {len(df)} rows — expected ~5000. File may be corrupt.")
    sys.exit(1)

non_null_transcriptions = df["transcription"].notna().sum()
if non_null_transcriptions < 100:
    print(f"\n✗ ABORT: Only {non_null_transcriptions} non-null transcriptions. Data corrupt.")
    sys.exit(1)
print(f"✓ Data integrity: {len(df)} rows, {non_null_transcriptions} non-null transcriptions")

# ============================================================
# CLEAN COLUMNS
# ============================================================
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown").str.strip()
df["sample_name"] = df["sample_name"].fillna("").str.strip()
df["description"] = df["description"].fillna("").str.strip() if "description" in df.columns else ""
df["transcription"] = df["transcription"].fillna("")

# Drop rows with empty transcriptions
before = len(df)
df = df[df["transcription"].str.strip().str.len() > 0]
print(f"\nRemoved {before - len(df)} empty transcriptions. Remaining: {len(df)}")

# ============================================================
# PASS 1: Specialty filter
# ============================================================
KEEP_SPECIALTIES = [
    "Consult - History and Phy.",
    "Cardiovascular / Pulmonary",
    "General Medicine",
    "Gastroenterology",
    "Neurology",
    "SOAP / Chart / Progress Notes",
    "Urology",
    "Nephrology",
    "Emergency Room Reports",
    "Hematology - Oncology",
    "Pain Management",
    "Psychiatry / Psychology",
    "Office Notes",
    "Pediatrics - Neonatal",
    "Endocrinology",
    "Rheumatology",
    "Allergy / Immunology",
    "Physical Medicine - Rehab",
    "Dermatology",
]

# Validate that at least some keep-specialties exist in data
found_specialties = [s for s in KEEP_SPECIALTIES if s in df["medical_specialty"].values]
if len(found_specialties) == 0:
    print(f"\n✗ ABORT: None of the expected specialties found in data.")
    print(f"  Expected: {KEEP_SPECIALTIES[:5]}...")
    print(f"  Found: {df['medical_specialty'].unique()[:10].tolist()}")
    sys.exit(1)

missing_specialties = [s for s in KEEP_SPECIALTIES if s not in df["medical_specialty"].values]
if missing_specialties:
    print(f"\n⚠ Warning: {len(missing_specialties)} expected specialties not in data:")
    for s in missing_specialties:
        print(f"    - {s}")

before = len(df)
df = df[df["medical_specialty"].isin(KEEP_SPECIALTIES)]
discarded_specialty = before - len(df)
print(f"\nPass 1 — Specialty filter:")
print(f"  Kept {len(df)}/{before} samples ({len(found_specialties)} specialties matched)")
print(f"  Discarded {discarded_specialty} non-encounter specialty samples")

# GATE: Enough samples after specialty filter
if len(df) < 200:
    print(f"\n✗ ABORT: Only {len(df)} samples after specialty filter. Expected 1000+.")
    print(f"  Check if specialty names have changed in dataset.")
    sys.exit(1)

# ============================================================
# PASS 2: Discard procedure/operative notes by sample_name
# ============================================================
PROCEDURE_KEYWORDS = [
    # Operative/surgical
    "operative", "surgery", "surgical", "excision", "incision",
    "repair", "removal", "insertion", "implant", "replacement",
    "arthroplasty", "arthroscopy", "laparoscop", "thoracotomy",
    "craniotomy", "laminectomy", "fusion", "fixation", "graft",
    "resection", "amputation", "debridement", "reconstruction",
    # Scopy/imaging procedures
    "colonoscopy", "endoscopy", "bronchoscopy", "cystoscopy",
    "laryngoscopy", "esophagoscopy", "sigmoidoscopy",
    "echocardiogram", "echocardiography", "doppler",
    "ultrasound", "angiogram", "angiography", "angioplasty",
    "catheterization", "cardiac cath", "catheter insertion",
    "treadmill", "stress test",
    # Biopsy/pathology
    "biopsy", "aspiration",
    # Specific procedures
    "stent", "pacemaker", "defibrillator", "ablation",
    "tracheostomy", "intubation", "ventilator",
    "dialysis", "hemodialysis", "peritoneal",
    "injection", "nerve block", "epidural",
    "transfusion", "infusion",
    # Imaging reads
    "x-ray", "xray", "ct scan", "mri", "pet scan",
    "mammogram", "carotid", "venous duplex",
]


def is_procedure(sample_name):
    """Check if sample_name indicates a procedure note."""
    name_lower = sample_name.lower()
    return any(kw in name_lower for kw in PROCEDURE_KEYWORDS)


before = len(df)
procedure_mask = df["sample_name"].apply(is_procedure)
removed_procedures = df[procedure_mask]
df = df[~procedure_mask]

print(f"\nPass 2 — Procedure keyword filter:")
print(f"  Removed {len(removed_procedures)} procedure/report notes")
print(f"  Remaining: {len(df)}")

if len(removed_procedures) > 0:
    print(f"\n  Sample REMOVED (first 10):")
    for _, row in removed_procedures.head(10).iterrows():
        print(f"    [{row['medical_specialty']}] {row['sample_name']}")

print(f"\n  Sample KEPT (first 10):")
for _, row in df.head(10).iterrows():
    print(f"    [{row['medical_specialty']}] {row['sample_name']}")

# GATE: Enough samples after procedure filter
if len(df) < 100:
    print(f"\n✗ ABORT: Only {len(df)} samples after procedure filter. Keyword list too aggressive.")
    sys.exit(1)

# ============================================================
# PASS 3: Length filter
# ============================================================
before = len(df)
too_short_mask = df["transcription"].str.len() < 200
too_long_mask = df["transcription"].str.len() > 5000
too_short = too_short_mask.sum()
too_long = too_long_mask.sum()

df = df[~too_short_mask & ~too_long_mask]

print(f"\nPass 3 — Length filter (200-5000 chars):")
print(f"  Removed {too_short} too short (<200 chars)")
print(f"  Removed {too_long} too long (>5000 chars)")
print(f"  Remaining: {len(df)}")

# ============================================================
# PASS 4: Drop duplicates
# ============================================================
before = len(df)
df = df.drop_duplicates(subset=["transcription"])
dupes = before - len(df)
print(f"\nPass 4 — Deduplication:")
print(f"  Removed {dupes} duplicate transcriptions")
print(f"  Remaining: {len(df)}")

# ============================================================
# FINAL GATE: Enough data for training
# ============================================================
if len(df) < 200:
    print(f"\n✗ ABORT: Only {len(df)} samples remaining. Not enough for LoRA training.")
    print(f"  Need minimum ~500 for viable fine-tuning.")
    sys.exit(1)

if len(df) < 500:
    print(f"\n⚠ WARNING: Only {len(df)} samples. Training will work but results may be limited.")
    print(f"  Recommended: 800+ samples for robust LoRA fine-tuning.")

# ============================================================
# SAVE
# ============================================================
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/filtered_for_generation.csv"
df.to_csv(output_path, index=False)

# ============================================================
# VERIFY SAVED FILE
# ============================================================
verify_df = pd.read_csv(output_path)
if len(verify_df) != len(df):
    print(f"\n✗ ABORT: Save verification failed. Wrote {len(df)} rows, read back {len(verify_df)}.")
    sys.exit(1)
print(f"\n✓ Save verified: {len(verify_df)} rows written and confirmed")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("FILTERING SUMMARY")
print("=" * 60)
print(f"  Input:     4999 raw samples")
print(f"  Output:    {len(df)} encounter notes")
print(f"  Saved:     {output_path}")
print(f"  Discarded: {4999 - len(df)} total")
print(f"    - Wrong specialty:    {discarded_specialty}")
print(f"    - Procedure notes:    {len(removed_procedures)}")
print(f"    - Too short:          {too_short}")
print(f"    - Too long:           {too_long}")
print(f"    - Duplicates:         {dupes}")

print(f"\nSpecialty distribution ({df['medical_specialty'].nunique()} specialties):")
for spec, count in df["medical_specialty"].value_counts().items():
    print(f"  {spec}: {count}")

print(f"\nTranscription length stats:")
lengths = df["transcription"].str.len()
print(f"  Mean:   {lengths.mean():.0f} chars")
print(f"  Median: {lengths.median():.0f} chars")
print(f"  Min:    {lengths.min():.0f} chars")
print(f"  Max:    {lengths.max():.0f} chars")

print(f"\n{'=' * 60}")
print(f"✓ READY for next step: python scripts/generate_soap_gpt4o.py")
print(f"  Estimated GPT-4o cost: ${len(df) * 0.003:.2f}")
print(f"  Estimated time: {len(df) * 2 / 60:.0f} minutes")
print("=" * 60)