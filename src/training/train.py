"""
MedGemma LoRA Fine-Tuning - Full Training Run
Maximum 10-hour runtime with checkpoint resume capability
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import time
from datetime import datetime, timedelta
import yaml

print("="*60)
print("MedGemma LoRA Fine-Tuning")
print("="*60)

# Load configuration
print("\n[1/7] Loading configuration...")
with open("configs/training.yaml", 'r') as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded from configs/training.yaml")

# Extract key parameters
MODEL_NAME = "./models/medgemma"
TRAIN_FILE = config['data']['train_file']
VAL_FILE = config['data']['validation_file']
OUTPUT_DIR = config['training']['output_dir']
BATCH_SIZE = config['training']['per_device_train_batch_size']
GRAD_ACCUM = config['training']['gradient_accumulation_steps']
LEARNING_RATE = config['training']['learning_rate']
EPOCHS = config['training']['num_train_epochs']

# 10-hour time limit
MAX_RUNTIME_HOURS = 10
start_time = time.time()
max_end_time = start_time + (MAX_RUNTIME_HOURS * 3600)

print(f"\nTraining parameters:")
print(f"  Model: {MODEL_NAME}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRAD_ACCUM}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Max runtime: {MAX_RUNTIME_HOURS} hours")
print(f"  End time: {datetime.fromtimestamp(max_end_time).strftime('%Y-%m-%d %H:%M:%S')}")

# Load dataset
print("\n[2/7] Loading dataset...")
dataset = load_dataset('json', data_files={
    'train': TRAIN_FILE,
    'validation': VAL_FILE
})

print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Calculate steps
steps_per_epoch = len(dataset['train']) // (BATCH_SIZE * GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")

# Estimate time
est_time_hours = (total_steps * 45) / 3600  # 45s per step from dry run
print(f"Estimated time: {est_time_hours:.1f} hours")

if est_time_hours > MAX_RUNTIME_HOURS:
    print(f"\nWARNING: Estimated time ({est_time_hours:.1f}h) exceeds limit ({MAX_RUNTIME_HOURS}h)")
    print("Training will checkpoint and can be resumed")

# Load model
print("\n[3/7] Loading MedGemma 4B...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.use_cache = False
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully")

# Prepare for LoRA
print("\n[4/7] Configuring LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=config['lora']['r'],
    lora_alpha=config['lora']['lora_alpha'],
    lora_dropout=config['lora']['lora_dropout'],
    bias=config['lora']['bias'],
    task_type=config['lora']['task_type'],
    target_modules=config['lora']['target_modules']
)

model = get_peft_model(model, lora_config)
print("\nTrainable parameters:")
model.print_trainable_parameters()

# Tokenize dataset
print("\n[5/7] Tokenizing dataset...")

INSTRUCTION_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{input_text}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

SOAP NOTE:"""

def tokenize_function(examples):
    """Tokenize conversations"""
    texts = []
    for messages in examples['messages']:
        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']
        text = user_msg + "\n\n" + assistant_msg
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding='max_length'
    )
    
    result = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].copy()
    }
    
    if 'token_type_ids' not in tokenized:
        result['token_type_ids'] = [[0] * len(ids) for ids in tokenized['input_ids']]
    else:
        result['token_type_ids'] = tokenized['token_type_ids']
    
    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)

print("Tokenization complete")

# Training arguments
print("\n[6/7] Setting up training...")

# Calculate checkpoint interval (every hour)
checkpoint_steps = int(3600 / 45)  # Save every hour (45s per step)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=config['training']['warmup_ratio'],
    lr_scheduler_type=config['training']['lr_scheduler_type'],
    
    # Precision
    bf16=config['training']['bf16'],
    tf32=config['training']['tf32'],
    
    # Logging
    logging_dir=config['logging']['logging_dir'],
    logging_steps=config['training']['logging_steps'],
    logging_strategy="steps",
    
    # Checkpointing (every hour)
    save_strategy="steps",
    save_steps=checkpoint_steps,
    save_total_limit=config['training']['save_total_limit'],
    
    # Evaluation (every 2 hours)
    eval_strategy="steps",
    eval_steps=checkpoint_steps * 2,
    
    # Model selection
    load_best_model_at_end=False,  # Save memory
    metric_for_best_model="eval_loss",
    
    # Performance
    gradient_checkpointing=config['training']['gradient_checkpointing'],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=config['training']['optim'],
    max_grad_norm=config['training']['max_grad_norm'],
    
    # Reporting
    report_to=config['logging']['report_to'],
    
    # Resume
    resume_from_checkpoint=True,  # Auto-resume if checkpoint exists
)

print(f"Checkpoint every: {checkpoint_steps} steps (~1 hour)")
print(f"Evaluation every: {checkpoint_steps * 2} steps (~2 hours)")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train
print("\n[7/7] Starting training...")
print("="*60)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target end: {datetime.fromtimestamp(max_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

try:
    trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    
    # Save final model
    final_output = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_output)
    print(f"Final model saved: {final_output}")
    
    # Training stats
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed/3600:.2f} hours")
    
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {vram:.2f} GB")
    
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("Training Interrupted")
    print("="*60)
    print("Latest checkpoint saved - can resume with same command")
    
except Exception as e:
    print("\n" + "="*60)
    print("Training Failed")
    print("="*60)
    print(f"Error: {str(e)}")
    raise