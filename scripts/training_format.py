"""
Format combined datasets into conversational JSONL for training
Creates instruction-tuning format for MedGemma fine-tuning
"""

import pandas as pd
import json
import os

print("="*60)
print("Format Datasets for Training")
print("="*60)

# Prompt template
INSTRUCTION_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{input_text}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

SOAP NOTE:"""

def create_conversation(row):
    """Convert row to conversational format"""
    
    # For structured samples: transcription is already SOAP formatted
    # For generated samples: transcription is raw, soap_note is structured
    
    input_text = row['transcription']
    output_text = row['soap_note']
    
    conversation = {
        "messages": [
            {
                "role": "user",
                "content": INSTRUCTION_TEMPLATE.format(input_text=input_text)
            },
            {
                "role": "assistant",
                "content": output_text
            }
        ]
    }
    
    return conversation

# Process each split
print("\n[1/3] Loading combined datasets...")
train_df = pd.read_csv("data/processed/combined/train_combined.csv")
val_df = pd.read_csv("data/processed/combined/val_combined.csv")
test_df = pd.read_csv("data/processed/combined/test_combined.csv")

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# Convert to JSONL
print("\n[2/3] Converting to conversational format...")

def save_jsonl(df, output_file):
    """Save dataframe as JSONL"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            conversation = create_conversation(row)
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

train_file = os.path.join(output_dir, "train.jsonl")
val_file = os.path.join(output_dir, "val.jsonl")
test_file = os.path.join(output_dir, "test.jsonl")

print("Saving train.jsonl...")
save_jsonl(train_df, train_file)

print("Saving val.jsonl...")
save_jsonl(val_df, val_file)

print("Saving test.jsonl...")
save_jsonl(test_df, test_file)

# Verification
print("\n[3/3] Verification...")

def verify_jsonl(file_path):
    """Verify JSONL format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Check first line
    first = json.loads(lines[0])
    assert 'messages' in first, "Missing 'messages' key"
    assert len(first['messages']) == 2, "Should have 2 messages"
    assert first['messages'][0]['role'] == 'user', "First role should be 'user'"
    assert first['messages'][1]['role'] == 'assistant', "Second role should be 'assistant'"
    
    # Check content lengths
    user_content = first['messages'][0]['content']
    assistant_content = first['messages'][1]['content']
    
    return len(lines), len(user_content), len(assistant_content)

print(f"\nTrain verification:")
train_lines, train_user_len, train_asst_len = verify_jsonl(train_file)
print(f"  Lines: {train_lines}")
print(f"  Avg user content: ~{train_user_len} chars")
print(f"  Avg assistant content: ~{train_asst_len} chars")

print(f"\nVal verification:")
val_lines, val_user_len, val_asst_len = verify_jsonl(val_file)
print(f"  Lines: {val_lines}")

print(f"\nTest verification:")
test_lines, test_user_len, test_asst_len = verify_jsonl(test_file)
print(f"  Lines: {test_lines}")

# File sizes
import os
train_size = os.path.getsize(train_file) / 1024 / 1024
val_size = os.path.getsize(val_file) / 1024 / 1024
test_size = os.path.getsize(test_file) / 1024 / 1024

print("\nFile sizes:")
print(f"  train.jsonl: {train_size:.2f} MB")
print(f"  val.jsonl: {val_size:.2f} MB")
print(f"  test.jsonl: {test_size:.2f} MB")
print(f"  Total: {train_size + val_size + test_size:.2f} MB")

# Sample preview
print("\n" + "="*60)
print("Sample Training Example")
print("="*60)

with open(train_file, 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())

print("\nUser message (first 300 chars):")
print(sample['messages'][0]['content'][:300] + "...")

print("\nAssistant message (first 300 chars):")
print(sample['messages'][1]['content'][:300] + "...")

print("\n" + "="*60)
print("Dataset Formatting Complete")
print("="*60)
print("\nFinal training files:")
print(f"  {train_file}")
print(f"  {val_file}")
print(f"  {test_file}")