"""
Prepare unstructured samples for batch SOAP generation
Load samples without SOAP structure and prepare metadata for processing
"""

import pandas as pd
import os

print("="*60)
print("Prepare Unstructured Samples for Batch Processing")
print("="*60)

# Load original cleaned data
print("\n[1/5] Loading all cleaned samples...")
all_samples = pd.read_csv("data/processed/mtsamples_cleaned.csv")
print(f"Total cleaned samples: {len(all_samples)}")

# Load structured samples (already split)
print("\n[2/5] Loading structured samples to exclude...")
train_structured = pd.read_csv("data/processed/structured/train_structured.csv")
val_structured = pd.read_csv("data/processed/structured/val_structured.csv")
test_structured = pd.read_csv("data/processed/structured/test_structured.csv")

structured_indices = set(train_structured.index) | set(val_structured.index) | set(test_structured.index)
print(f"Structured samples: {len(structured_indices)}")

# Get unstructured samples (everything NOT in structured set)
print("\n[3/5] Identifying unstructured samples...")

# The structured samples were filtered from cleaned data
# So we need to identify which original indices correspond to unstructured
# We'll use a different approach: reload original and identify unstructured

# Reload original data before SOAP filtering
original_df = pd.read_csv("data/raw/mtsamples.csv")

# Apply same initial cleaning
original_df = original_df.dropna(subset=['transcription'])
original_df = original_df[original_df['transcription'].str.len() >= 100]
original_df = original_df[original_df['transcription'].str.len() <= 8000]

print(f"Samples after basic cleaning: {len(original_df)}")

# Clean transcriptions
import re

def clean_text(text):
    """Clean common formatting issues"""
    text = text.strip()
    text = re.sub(r'(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):\s*,', r'\1:', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('[**', '').replace('**]', '')
    text = text.replace('___', '')
    return text

original_df['transcription'] = original_df['transcription'].apply(clean_text)

# Identify which ones DON'T have SOAP structure
def has_soap_structure(text):
    """Check if text contains SOAP sections"""
    text_upper = text.upper()
    has_subjective = 'SUBJECTIVE:' in text_upper
    has_objective = 'OBJECTIVE:' in text_upper
    has_assessment = 'ASSESSMENT:' in text_upper
    has_plan = 'PLAN:' in text_upper
    section_count = sum([has_subjective, has_objective, has_assessment, has_plan])
    return section_count >= 3

original_df['has_structure'] = original_df['transcription'].apply(has_soap_structure)

# Unstructured = does NOT have SOAP structure
unstructured_df = original_df[original_df['has_structure'] == False].copy()
print(f"Unstructured samples (no SOAP format): {len(unstructured_df)}")

# Remove temporary column
unstructured_df = unstructured_df.drop('has_structure', axis=1)

# Add batch metadata for processing
print("\n[4/5] Adding batch metadata...")

# Calculate batches (for checkpoint management)
batch_size = 127  # Checkpoint every 127 samples (~1.4 hours)
unstructured_df['batch_id'] = (unstructured_df.index // batch_size).astype(int)
unstructured_df['sample_id'] = range(len(unstructured_df))

num_batches = unstructured_df['batch_id'].max() + 1
print(f"Total batches: {num_batches} (batch size: {batch_size})")
print(f"Processing time estimate: {len(unstructured_df) * 40 / 3600:.1f} hours")

# Save unstructured samples
print("\n[5/5] Saving unstructured samples for processing...")
output_file = "data/processed/unstructured_for_generation.csv"
unstructured_df.to_csv(output_file, index=False)

print(f"  Saved: {output_file}")

# Summary
print("\n" + "="*60)
print("Preparation Summary")
print("="*60)
print(f"Total samples for SOAP generation: {len(unstructured_df)}")
print(f"Batches: {num_batches}")
print(f"Estimated processing time: {len(unstructured_df) * 40 / 3600:.1f} hours")
print(f"Samples per night (14 hours): {14 * 3600 / 40:.0f}")
print(f"Nights needed: {len(unstructured_df) * 40 / (14 * 3600):.1f}")

print("\nLength statistics:")
lengths = unstructured_df['transcription'].str.len()
print(f"  Mean: {lengths.mean():.0f} chars")
print(f"  Median: {lengths.median():.0f} chars")
print(f"  Min: {lengths.min():.0f} chars")
print(f"  Max: {lengths.max():.0f} chars")

print("\nSpecialty distribution (top 10):")
print(unstructured_df['medical_specialty'].value_counts().head(10))

print("\nBatch distribution:")
print(f"  Full batches ({batch_size} samples): {num_batches - 1}")
print(f"  Final batch: {len(unstructured_df) - (num_batches - 1) * batch_size} samples")

print("\n" + "="*60)
print("Unstructured samples ready for generation")
print("="*60)