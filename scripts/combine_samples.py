"""
Combine structured and generated SOAP samples
Create final unified dataset for training
"""

import pandas as pd
import glob
import os

print("="*60)
print("Combine All SOAP Samples")
print("="*60)

# Load structured samples
print("\n[1/4] Loading structured samples...")
train_structured = pd.read_csv("data/processed/structured/train_structured.csv")
val_structured = pd.read_csv("data/processed/structured/val_structured.csv")
test_structured = pd.read_csv("data/processed/structured/test_structured.csv")

print(f"Structured samples:")
print(f"  Train: {len(train_structured)}")
print(f"  Val: {len(val_structured)}")
print(f"  Test: {len(test_structured)}")
print(f"  Total: {len(train_structured) + len(val_structured) + len(test_structured)}")

# Load generated samples
print("\n[2/4] Loading generated samples...")
batch_files = sorted(glob.glob("data/processed/generated_soap/batch_*.csv"))
print(f"Found {len(batch_files)} batch files")

all_generated = []
for batch_file in batch_files:
    batch_df = pd.read_csv(batch_file)
    # Keep only successful generations with SOAP structure
    batch_df = batch_df[batch_df['success'] == True]
    batch_df = batch_df[batch_df['has_structure'] == True]
    all_generated.append(batch_df)

generated_df = pd.concat(all_generated, ignore_index=True)
print(f"Generated samples: {len(generated_df)}")
print(f"Quality filtered: {len(generated_df)} samples with SOAP structure")

# Prepare generated samples for merging
print("\n[3/4] Preparing datasets for merge...")

# Generated samples need same columns as structured
# Keep: original_text (as transcription), generated_soap, medical_specialty
generated_clean = generated_df[['original_text', 'generated_soap', 'sample_id']].copy()
generated_clean = generated_clean.rename(columns={'original_text': 'original_transcription'})

# For structured samples, transcription IS the SOAP note already
# We'll use transcription as both input and target for consistency check

# Split generated samples: 80/10/10
from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
train_gen, temp_gen = train_test_split(
    generated_clean,
    test_size=0.2,
    random_state=42
)

# Second split: 50/50 of temp = 10% val, 10% test
val_gen, test_gen = train_test_split(
    temp_gen,
    test_size=0.5,
    random_state=42
)

print(f"Generated split:")
print(f"  Train: {len(train_gen)}")
print(f"  Val: {len(val_gen)}")
print(f"  Test: {len(test_gen)}")

# Prepare structured samples
structured_train = train_structured[['transcription']].copy()
structured_train['soap_note'] = structured_train['transcription']
structured_train['source'] = 'structured'

structured_val = val_structured[['transcription']].copy()
structured_val['soap_note'] = structured_val['transcription']
structured_val['source'] = 'structured'

structured_test = test_structured[['transcription']].copy()
structured_test['soap_note'] = structured_test['transcription']
structured_test['source'] = 'structured'

# Prepare generated samples
train_gen['transcription'] = train_gen['original_transcription']
train_gen['soap_note'] = train_gen['generated_soap']
train_gen['source'] = 'generated'
train_gen = train_gen[['transcription', 'soap_note', 'source']]

val_gen['transcription'] = val_gen['original_transcription']
val_gen['soap_note'] = val_gen['generated_soap']
val_gen['source'] = 'generated'
val_gen = val_gen[['transcription', 'soap_note', 'source']]

test_gen['transcription'] = test_gen['original_transcription']
test_gen['soap_note'] = test_gen['generated_soap']
test_gen['source'] = 'generated'
test_gen = test_gen[['transcription', 'soap_note', 'source']]

# Combine
print("\n[4/4] Combining datasets...")
final_train = pd.concat([structured_train, train_gen], ignore_index=True)
final_val = pd.concat([structured_val, val_gen], ignore_index=True)
final_test = pd.concat([structured_test, test_gen], ignore_index=True)

# Shuffle
final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
final_val = final_val.sample(frac=1, random_state=42).reset_index(drop=True)
final_test = final_test.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset:")
print(f"  Train: {len(final_train)} ({len(structured_train)} structured + {len(train_gen)} generated)")
print(f"  Val: {len(final_val)} ({len(structured_val)} structured + {len(val_gen)} generated)")
print(f"  Test: {len(final_test)} ({len(structured_test)} structured + {len(test_gen)} generated)")
print(f"  Total: {len(final_train) + len(final_val) + len(final_test)}")

# Save combined datasets
print("\nSaving combined datasets...")
os.makedirs("data/processed/combined", exist_ok=True)

final_train.to_csv("data/processed/combined/train_combined.csv", index=False)
final_val.to_csv("data/processed/combined/val_combined.csv", index=False)
final_test.to_csv("data/processed/combined/test_combined.csv", index=False)

print("  Saved: data/processed/combined/train_combined.csv")
print("  Saved: data/processed/combined/val_combined.csv")
print("  Saved: data/processed/combined/test_combined.csv")

# Statistics
print("\n" + "="*60)
print("Dataset Statistics")
print("="*60)

print("\nLength statistics (characters):")
for name, df in [("Train", final_train), ("Val", final_val), ("Test", final_test)]:
    lens = df['soap_note'].str.len()
    print(f"{name}:")
    print(f"  Mean: {lens.mean():.0f}")
    print(f"  Median: {lens.median():.0f}")
    print(f"  Min: {lens.min():.0f}")
    print(f"  Max: {lens.max():.0f}")

print("\nSource distribution:")
for name, df in [("Train", final_train), ("Val", final_val), ("Test", final_test)]:
    print(f"{name}:")
    print(df['source'].value_counts())

print("\n" + "="*60)
print("Datasets combined successfully")
print("="*60)