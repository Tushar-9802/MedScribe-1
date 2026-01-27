"""
Batch SOAP generation - Optimized version
Better logging, GPU monitoring, quality checks
"""

import pandas as pd
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
import os
import json

print("="*60)
print("Batch SOAP Generation (Optimized)")
print("="*60)

# Configuration
INPUT_FILE = "data/processed/unstructured_for_generation.csv"
OUTPUT_DIR = "data/processed/generated_soap"
CHECKPOINT_DIR = "data/processed/checkpoints"
CHECKPOINT_INTERVAL = 127
LOG_INTERVAL = 5  # Log every 5 samples instead of 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load samples
print("\n[1/6] Loading unstructured samples...")
df = pd.read_csv(INPUT_FILE)
print(f"Total samples: {len(df)}")

# Check for existing progress
print("\n[2/6] Checking for previous progress...")
progress_file = os.path.join(CHECKPOINT_DIR, "generation_progress.json")

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_idx = progress['last_completed_index'] + 1
    print(f"RESUMING from sample {start_idx}")
    print(f"Already completed: {start_idx} samples")
else:
    start_idx = 0
    progress = {'last_completed_index': -1, 'completed_batches': []}
    print("Starting fresh")

remaining = len(df) - start_idx
print(f"Remaining: {remaining} samples")
print(f"Estimated: {remaining * 40 / 3600:.1f} hours")

# Load MedGemma
print("\n[3/6] Loading MedGemma 4B...")
load_start = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/medgemma",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "./models/medgemma",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {time.time() - load_start:.1f}s")

# GPU info
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_total:.1f} GB total")

# Prompt template
prompt_template = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{text}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

SOAP NOTE:"""

# Process samples
print("\n[4/6] Generating SOAP notes...")
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

results = []
batch_start_time = time.time()
total_start_time = time.time()
generation_times = []

for idx in range(start_idx, len(df)):
    row = df.iloc[idx]
    sample_id = row['sample_id']
    batch_id = row['batch_id']
    original_text = row['transcription']
    
    # Frequent progress updates
    if idx % LOG_INTERVAL == 0 and idx > start_idx:
        elapsed = time.time() - total_start_time
        processed = idx - start_idx
        avg_time = elapsed / processed
        remaining_samples = len(df) - idx
        eta_seconds = remaining_samples * avg_time
        eta_hours = eta_seconds / 3600
        
        # GPU utilization
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_percent = (vram_used / vram_total) * 100
            gpu_util = f"VRAM: {vram_used:.1f}GB ({vram_percent:.0f}%)"
        else:
            gpu_util = "VRAM: N/A"
        
        # Recent generation speed
        recent_avg = sum(generation_times[-20:]) / len(generation_times[-20:]) if generation_times else 0
        
        print(f"\n{'='*60}")
        print(f"Progress: [{idx}/{len(df)}] Batch {batch_id}")
        print(f"Completed: {processed} samples")
        print(f"Speed: {avg_time:.1f}s/sample (recent: {recent_avg:.1f}s)")
        print(f"ETA: {eta_hours:.1f}h ({int(eta_hours*60)}min)")
        print(f"{gpu_util}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
    
    # Create prompt (limit input to fit context)
    prompt = prompt_template.format(text=original_text[:3000])
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    try:
        gen_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SOAP
        if "SOAP NOTE:" in full_output:
            soap_note = full_output.split("SOAP NOTE:")[-1].strip()
        else:
            soap_note = full_output.strip()
        
        # Quality check (basic)
        has_structure = any(section in soap_note.upper() for section in ['SUBJECTIVE', 'OBJECTIVE', 'ASSESSMENT', 'PLAN'])
        
        results.append({
            'sample_id': sample_id,
            'batch_id': batch_id,
            'original_text': original_text,
            'generated_soap': soap_note,
            'generation_time': gen_time,
            'has_structure': has_structure,
            'success': True
        })
        
        # Sample quality check every 50 samples
        if idx % 50 == 0 and idx > start_idx:
            recent_quality = sum(r.get('has_structure', False) for r in results[-50:]) / min(50, len(results))
            print(f"  Quality check: {recent_quality*100:.0f}% have SOAP structure")
        
    except Exception as e:
        print(f"\nERROR at sample {idx}: {str(e)[:100]}")
        results.append({
            'sample_id': sample_id,
            'batch_id': batch_id,
            'original_text': original_text,
            'generated_soap': '',
            'generation_time': 0,
            'has_structure': False,
            'success': False,
            'error': str(e)[:200]
        })
    
    # Checkpoint save
    if (idx + 1) % CHECKPOINT_INTERVAL == 0 or idx == len(df) - 1:
        ckpt_start = time.time()
        
        # Save batch
        batch_df = pd.DataFrame(results)
        batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_id:03d}.csv")
        batch_df.to_csv(batch_file, index=False)
        
        # Update progress
        progress['last_completed_index'] = idx
        if batch_id not in progress['completed_batches']:
            progress['completed_batches'].append(batch_id)
        progress['last_checkpoint_time'] = datetime.now().isoformat()
        progress['total_samples'] = len(df)
        progress['percent_complete'] = ((idx + 1) / len(df)) * 100
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Batch stats
        batch_quality = sum(r.get('has_structure', False) for r in results) / len(results) * 100
        batch_time = time.time() - batch_start_time
        
        print(f"\n{'*'*60}")
        print(f"CHECKPOINT: Batch {batch_id} saved")
        print(f"Samples in batch: {len(results)}")
        print(f"Quality: {batch_quality:.0f}% have structure")
        print(f"Batch time: {batch_time/60:.1f} min")
        print(f"Progress: {progress['percent_complete']:.1f}%")
        print(f"{'*'*60}\n")
        
        results = []
        batch_start_time = time.time()

# Final summary
print("\n" + "="*60)
print("[5/6] Generation Complete")
print("="*60)

total_time = time.time() - total_start_time
samples_processed = len(df) - start_idx

print(f"Samples processed: {samples_processed}")
print(f"Total time: {total_time / 3600:.2f} hours")
print(f"Avg speed: {total_time / samples_processed:.1f}s per sample")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Combine batches
print("\n[6/6] Combining batches...")
all_batches = []
for batch_id in sorted(set(progress['completed_batches'])):
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_id:03d}.csv")
    if os.path.exists(batch_file):
        batch_df = pd.read_csv(batch_file)
        all_batches.append(batch_df)

if all_batches:
    combined_df = pd.concat(all_batches, ignore_index=True)
    final_file = "data/processed/all_generated_soap.csv"
    combined_df.to_csv(final_file, index=False)
    
    # Final stats
    total_generated = len(combined_df)
    success_count = combined_df['success'].sum()
    structure_count = combined_df.get('has_structure', pd.Series([False]*len(combined_df))).sum()
    
    print(f"\nFinal results:")
    print(f"  Total: {total_generated} samples")
    print(f"  Success: {success_count} ({success_count/total_generated*100:.1f}%)")
    print(f"  With structure: {structure_count} ({structure_count/total_generated*100:.1f}%)")
    print(f"  Saved: {final_file}")

print("\n" + "="*60)
print("Batch generation complete")
print("="*60)
print("\nNext: Run scripts/combine_all_samples.py")
print("(After all nights complete)")