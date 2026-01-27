"""
Split structured SOAP samples
These will be distributed across train/val/test in final dataset
Structured samples provide quality baseline for evaluation
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("="*60)
print("Split Structured SOAP Samples")
print("="*60)

# Load cleaned structured samples
print("\n[1/4] Loading structured samples...")
input_file = "data/processed/mtsamples_cleaned.csv"
df = pd.read_csv(input_file)

print(f"Total structured samples: {len(df)}")

# Display specialty distribution
print("\nSpecialty distribution:")
print(df['medical_specialty'].value_counts().head(10))

# Prepare for stratified split
print("\n[2/4] Preparing stratified split...")

# For stratification to work with 80/10/10 split:
# Need at least 10 samples per group (to get at least 1 in each split)
specialty_counts = df['medical_specialty'].value_counts()
min_samples_for_stratify = 10  # Increased from 3

def map_specialty(specialty):
    """Map rare specialties to 'Other' for stratification"""
    if specialty_counts[specialty] >= min_samples_for_stratify:
        return specialty
    else:
        return 'Other'

df['specialty_group'] = df['medical_specialty'].apply(map_specialty)

print(f"Specialty groups for stratification: {df['specialty_group'].nunique()}")
print("\nGrouped distribution:")
grouped_counts = df['specialty_group'].value_counts()
print(grouped_counts)

# Verify minimum group size
min_group_size = grouped_counts.min()
print(f"\nSmallest group size: {min_group_size}")
if min_group_size < 10:
    print(f"WARNING: Group '{grouped_counts.idxmin()}' has only {min_group_size} samples")

# Split: 80% train, 10% val, 10% test
print("\n[3/4] Performing stratified split...")

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['specialty_group']
)

print(f"After first split:")
print(f"  Training: {len(train_df)}")
print(f"  Temporary: {len(temp_df)}")

# Second split: 50/50 of temp = 10% val, 10% test
# For very small groups in temp, we may need to disable stratification
try:
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['specialty_group']
    )
    print(f"Stratified second split successful")
except ValueError as e:
    print(f"Stratification failed on second split, using random split")
    print(f"Reason: {e}")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )
    print(f"Random second split successful")

print(f"\nFinal split sizes:")
print(f"  Training samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Total: {len(train_df) + len(val_df) + len(test_df)}")

# Verify no overlap
assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val overlap detected"
assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test overlap detected"
assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test overlap detected"
print("\nNo overlap between splits - verified")

# Remove temporary column
train_df = train_df.drop('specialty_group', axis=1)
val_df = val_df.drop('specialty_group', axis=1)
test_df = test_df.drop('specialty_group', axis=1)

# Save splits
print("\n[4/4] Saving structured splits...")
os.makedirs("data/processed/structured", exist_ok=True)

train_file = "data/processed/structured/train_structured.csv"
val_file = "data/processed/structured/val_structured.csv"
test_file = "data/processed/structured/test_structured.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"  Saved: {train_file}")
print(f"  Saved: {val_file}")
print(f"  Saved: {test_file}")

# Summary statistics
print("\n" + "="*60)
print("Split Summary")
print("="*60)

print("\nTraining set:")
print(f"  Samples: {len(train_df)}")
print(f"  Mean length: {train_df['transcription'].str.len().mean():.0f} chars")
print(f"  Top specialties:")
for spec, count in train_df['medical_specialty'].value_counts().head(5).items():
    print(f"    {spec}: {count}")

print("\nValidation set:")
print(f"  Samples: {len(val_df)}")
print(f"  Mean length: {val_df['transcription'].str.len().mean():.0f} chars")
print(f"  Top specialties:")
for spec, count in val_df['medical_specialty'].value_counts().head(3).items():
    print(f"    {spec}: {count}")

print("\nTest set:")
print(f"  Samples: {len(test_df)}")
print(f"  Mean length: {test_df['transcription'].str.len().mean():.0f} chars")
print(f"  Top specialties:")
for spec, count in test_df['medical_specialty'].value_counts().head(3).items():
    print(f"    {spec}: {count}")

print("\n" + "="*60)
print("Structured samples split complete")
print("="*60)
