"""
Clean MTSamples dataset
Remove invalid samples and fix formatting
"""

import pandas as pd
import re
import os

print("="*60)
print("MTSamples Data Cleaning")
print("="*60)

# Load raw data
print("\n[1/5] Loading raw dataset...")
input_file = "data/raw/mtsamples.csv"
df = pd.read_csv(input_file)

print(f"Original samples: {len(df)}")

# Remove samples with missing transcriptions
print("\n[2/5] Removing invalid samples...")
original_count = len(df)

# Drop rows with NaN transcriptions
df = df.dropna(subset=['transcription'])
missing_count = original_count - len(df)
print(f"  Removed {missing_count} samples with missing transcriptions")

# Remove samples that are too short (<100 characters)
df = df[df['transcription'].str.len() >= 100]
too_short = original_count - missing_count - len(df)
print(f"  Removed {too_short} samples too short (<100 chars)")

# Remove samples that are too long (>8000 characters)
# These won't fit in context window with prompt
df = df[df['transcription'].str.len() <= 8000]
too_long = original_count - missing_count - too_short - len(df)
print(f"  Removed {too_long} samples too long (>8000 chars)")

print(f"  Remaining samples: {len(df)}")

# Clean transcription text
print("\n[3/5] Cleaning transcription text...")

def clean_text(text):
    """Clean common formatting issues in transcriptions"""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Fix common issues with SOAP headers
    # "SUBJECTIVE:," -> "SUBJECTIVE:"
    text = re.sub(r'(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):\s*,', r'\1:', text)
    
    # Standardize section headers (capitalize)
    text = re.sub(r'\bsubjective:', 'SUBJECTIVE:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bobjective:', 'OBJECTIVE:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bassessment:', 'ASSESSMENT:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bplan:', 'PLAN:', text, flags=re.IGNORECASE)
    
    # Also handle common variations
    text = re.sub(r'\bCHIEF COMPLAINT:', 'SUBJECTIVE:', text)
    text = re.sub(r'\bHISTORY OF PRESENT ILLNESS:', 'SUBJECTIVE:', text)
    text = re.sub(r'\bPHYSICAL EXAMINATION:', 'OBJECTIVE:', text)
    text = re.sub(r'\bEXAM:', 'OBJECTIVE:', text)
    text = re.sub(r'\bDIAGNOSIS:', 'ASSESSMENT:', text)
    text = re.sub(r'\bIMPRESSION:', 'ASSESSMENT:', text)
    text = re.sub(r'\bRECOMMENDATIONS:', 'PLAN:', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines -> double
    text = re.sub(r' +', ' ', text)  # Multiple spaces -> single
    
    # Remove common transcription artifacts
    text = text.replace('[**', '').replace('**]', '')  # De-identification markers
    text = text.replace('___', '')  # Blank fields
    
    return text

df['transcription'] = df['transcription'].apply(clean_text)
print("  Text cleaning complete")

# Verify SOAP structure presence
print("\n[4/5] Validating SOAP structure...")

def has_soap_structure(text):
    """Check if text contains SOAP sections"""
    has_subjective = 'SUBJECTIVE:' in text.upper()
    has_objective = 'OBJECTIVE:' in text.upper()
    has_assessment = 'ASSESSMENT:' in text.upper()
    has_plan = 'PLAN:' in text.upper()
    
    # At least 3 of 4 sections should be present
    section_count = sum([has_subjective, has_objective, has_assessment, has_plan])
    return section_count >= 3

df['has_structure'] = df['transcription'].apply(has_soap_structure)
structured_count = df['has_structure'].sum()
print(f"  Samples with SOAP structure: {structured_count}/{len(df)} ({structured_count/len(df)*100:.1f}%)")

# Keep only structured samples
df = df[df['has_structure'] == True]
print(f"  Filtered to structured samples: {len(df)}")

# Remove the temporary column
df = df.drop('has_structure', axis=1)

# Save cleaned dataset
print("\n[5/5] Saving cleaned dataset...")
os.makedirs("data/processed", exist_ok=True)
output_file = "data/processed/mtsamples_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"  Saved to: {output_file}")

# Summary statistics
print("\n" + "="*60)
print("Cleaning Summary")
print("="*60)
print(f"Original samples: {original_count}")
print(f"Removed:")
print(f"  Missing transcriptions: {missing_count}")
print(f"  Too short: {too_short}")
print(f"  Too long: {too_long}")
print(f"  No SOAP structure: {original_count - len(df) - missing_count - too_short - too_long}")
print(f"Final samples: {len(df)}")
print(f"Retention rate: {len(df)/original_count*100:.1f}%")

print("\nLength statistics (cleaned):")
lengths = df['transcription'].str.len()
print(f"  Mean: {lengths.mean():.0f} chars")
print(f"  Median: {lengths.median():.0f} chars")
print(f"  Min: {lengths.min():.0f} chars")
print(f"  Max: {lengths.max():.0f} chars")

print("\nSpecialty distribution (top 10):")
print(df['medical_specialty'].value_counts().head(10))

print("\n" + "="*60)
print("Data cleaning complete")
print("="*60)
