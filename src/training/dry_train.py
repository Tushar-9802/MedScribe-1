"""
Training dry run - validate setup with 10 steps
Ensures no errors before overnight training
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

print("="*60)
print("Training Dry Run (10 steps)")
print("="*60)

# Configuration
print("\n[1/6] Loading configuration...")
config = {
    'model_name': './models/medgemma',
    'train_file': './data/processed/train.jsonl',
    'val_file': './data/processed/val.jsonl',
    'output_dir': './models/checkpoints/dry_run',
    'max_steps': 10,
    'batch_size': 2,
    'grad_accum': 16,
    'learning_rate': 2e-4,
}

print("Config loaded:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Load dataset
print("\n[2/6] Loading dataset...")
dataset = load_dataset('json', data_files={
    'train': config['train_file'],
    'validation': config['val_file']
})

print(f"Train samples: {len(dataset['train'])}")
print(f"Val samples: {len(dataset['validation'])}")

# Take small subset for dry run
dataset['train'] = dataset['train'].select(range(40))  # 40 samples = 10 steps with batch 8, grad_accum 4
dataset['validation'] = dataset['validation'].select(range(20))

print(f"Dry run subset - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Load model
print("\n[3/6] Loading MedGemma 4B...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded")

# Prepare for training
print("\n[4/6] Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize function
def tokenize_function(examples):
    """Tokenize conversations for training"""
    texts = []
    for messages in examples['messages']:
        # Concatenate user and assistant messages
        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']
        text = user_msg + "\n\n" + assistant_msg
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding='max_length',
        return_tensors=None,  # Return lists, not tensors
        return_token_type_ids=True,  # Add this
        return_attention_mask=True,   # Add this
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized['labels'] = []
    for input_ids in tokenized['input_ids']:
        # Convert to list if needed
        labels = list(input_ids) if not isinstance(input_ids, list) else input_ids
        tokenized['labels'].append(labels)
    
    return tokenized
print("\n[5/6] Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)

print("Tokenization complete")

# Training arguments
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    max_steps=config['max_steps'],
    per_device_train_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['grad_accum'],
    learning_rate=config['learning_rate'],
    bf16=True,
    logging_steps=1,
    load_best_model_at_end=False,
    report_to="none",
    gradient_checkpointing=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train
print("\n[6/6] Running training dry run...")
print("="*60)

try:
    trainer.train()
    print("\n" + "="*60)
    print("Dry Run SUCCESSFUL")
    print("="*60)
    print("\nTraining completed 10 steps without errors")
    print("VRAM usage:")
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak: {vram:.2f} GB")
    
    print("\nReady for full training run!")
    
except Exception as e:
    print("\n" + "="*60)
    print("Dry Run FAILED")
    print("="*60)
    print(f"Error: {str(e)}")
    print("\nFix the error before running full training")