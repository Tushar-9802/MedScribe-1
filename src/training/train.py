"""
MedScribe v2 — MedGemma LoRA Fine-Tuning
=========================================
Optimized for:
- RTX 5070 Ti (16GB VRAM), Windows
- 712 high-quality training samples (GPT-4o generated)
- Clinical SOAP note generation

Key changes from v1:
- Reduced effective batch size (712 samples ≠ 3200 — need more updates per epoch)
- 5 epochs instead of 3 (smaller dataset needs more passes)
- Lower learning rate (2e-5 → avoids overfitting on small dataset)
- Eval every epoch (not every 250 steps)
- Early stopping on val loss plateau
- Proper max_seq_length for new data profile
- Save best model by val loss
"""
import os
import sys
import time
import json
import torch
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================
# Paths
MODEL_NAME = "./models/medgemma"
TRAIN_FILE = "./data/processed/train.jsonl"
VAL_FILE = "./data/processed/val.jsonl"
OUTPUT_DIR = "./models/checkpoints/medgemma_v2_soap"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

# Training — tuned for 712 samples on 16GB VRAM
BATCH_SIZE = 2              # OOM at 4 with 16GB — 2 is safe
GRAD_ACCUM = 8              # Effective batch = 16 (same: 2×8 = 16)
LEARNING_RATE = 2e-5        # Lower than v1's 2e-4 — small dataset overfits fast at high LR
EPOCHS = 5                  # More passes for small dataset (was 3)
WARMUP_RATIO = 0.1          # 10% warmup
MAX_SEQ_LENGTH = 1024       # Reduced from 1536 — fits 16GB, covers 95%+ of samples
MAX_RUNTIME_HOURS = 8       # Safety cap

# LoRA
LORA_R = 16
LORA_ALPHA = 32             # Alpha = 2*r (stronger adaptation for small dataset)
LORA_DROPOUT = 0.1          # Higher dropout (was 0.05) — regularization for small dataset
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Eval & checkpointing
EVAL_STRATEGY = "epoch"     # Eval every epoch (not steps — 712 samples = few steps per epoch)
SAVE_STRATEGY = "epoch"
EARLY_STOPPING_PATIENCE = 2 # Stop if val loss doesn't improve for 2 epochs

print("=" * 60)
print("MedScribe v2 — LoRA Fine-Tuning")
print("=" * 60)

# ============================================================
# GATE 0: Validate environment
# ============================================================
if not torch.cuda.is_available():
    print("✗ ABORT: No CUDA GPU detected.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")

if gpu_mem < 14:
    print(f"⚠ WARNING: Only {gpu_mem:.1f} GB VRAM. May OOM with current config.")
    print(f"  Consider reducing BATCH_SIZE to 2 or MAX_SEQ_LENGTH to 1024.")

# ============================================================
# GATE 1: Validate data files
# ============================================================
for path, name in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
    if not os.path.exists(path):
        print(f"✗ ABORT: {name} file not found: {path}")
        sys.exit(1)

# Quick line count and integrity check
for path, name in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    valid = 0
    for line in lines:
        try:
            obj = json.loads(line)
            assert "messages" in obj
            assert len(obj["messages"]) == 2
            valid += 1
        except Exception:
            pass
    if valid < len(lines) * 0.9:
        print(f"✗ ABORT: {name} has {valid}/{len(lines)} valid lines. Data corrupt.")
        sys.exit(1)
    print(f"✓ {name}: {valid} valid samples")

# ============================================================
# GATE 2: Validate model exists
# ============================================================
if not os.path.isdir(MODEL_NAME):
    print(f"✗ ABORT: Model not found: {MODEL_NAME}")
    print(f"  Download MedGemma first.")
    sys.exit(1)
print(f"✓ Model: {MODEL_NAME}")

# ============================================================
# LOAD MODEL
# ============================================================
print(f"\n[1/6] Loading base model (4-bit quantized)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
print(f"✓ Base model loaded")

# ============================================================
# ATTACH LoRA
# ============================================================
print(f"\n[2/6] Attaching LoRA adapter...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

model = get_peft_model(model, lora_config)

# Count parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✓ LoRA attached")
print(f"  Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
print(f"  Total:     {total:,}")
print(f"  Config:    r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

# ============================================================
# LOAD + TOKENIZE DATASET
# ============================================================
print(f"\n[3/6] Loading and tokenizing dataset...")

dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
})

print(f"  Train: {len(dataset['train'])} samples")
print(f"  Val:   {len(dataset['validation'])} samples")


def tokenize_function(examples):
    """Tokenize chat-format messages into input_ids + labels + token_type_ids.
    
    Gemma 3 / MedGemma 1.5 requires token_type_ids during training.
    token_type_ids=0 for all text tokens (single-turn, no image).
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_token_type_ids = []

    for messages in examples["messages"]:
        # Build full text: user prompt + assistant response
        user_text = messages[0]["content"]
        assistant_text = messages[1]["content"]

        # Tokenize prompt (user) — these tokens get label=-100 (not trained on)
        prompt_tokens = tokenizer(
            user_text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        # Tokenize response (assistant) — these tokens are the training target
        response_tokens = tokenizer(
            assistant_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH // 2,  # Response is shorter than prompt
        )

        # Concatenate
        input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        # token_type_ids: 0 for all tokens (text-only, no image tokens)
        token_type_ids = [0] * len(input_ids)

        # Labels: -100 for prompt tokens (don't compute loss on input)
        labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

        # Truncate to max length
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            attention_mask = attention_mask[:MAX_SEQ_LENGTH]
            token_type_ids = token_type_ids[:MAX_SEQ_LENGTH]
            labels = labels[:MAX_SEQ_LENGTH]

        # Pad to max length
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [0] * padding_length
            labels = labels + [-100] * padding_length

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "token_type_ids": all_token_type_ids,
        "labels": all_labels,
    }


tokenized = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=50,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
)

# Verify tokenization
sample = tokenized["train"][0]
prompt_tokens = sum(1 for l in sample["labels"] if l == -100)
response_tokens = sum(1 for l in sample["labels"] if l != -100)
print(f"✓ Tokenization complete")
print(f"  Sample: {prompt_tokens} prompt tokens + {response_tokens} response tokens")
print(f"  Total sequence length: {len(sample['input_ids'])}")

# GATE: Verify response tokens exist (not all -100)
if response_tokens < 10:
    print(f"✗ ABORT: Only {response_tokens} response tokens. Tokenization broken.")
    sys.exit(1)

# ============================================================
# TRAINING ARGUMENTS
# ============================================================
print(f"\n[4/6] Configuring training...")

steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
est_time_hours = (total_steps * 30) / 3600  # ~30s per step estimate

print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Steps per epoch:     {steps_per_epoch}")
print(f"  Total steps:         {total_steps}")
print(f"  Estimated time:      {est_time_hours:.1f} hours")

if est_time_hours > MAX_RUNTIME_HOURS:
    print(f"⚠ WARNING: Estimated {est_time_hours:.1f}h exceeds {MAX_RUNTIME_HOURS}h limit.")
    print(f"  Training will be stopped at {MAX_RUNTIME_HOURS}h.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batch
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Optimizer
    optim="adamw_torch",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    # Duration
    num_train_epochs=EPOCHS,

    # Precision
    bf16=True,
    tf32=True,

    # Eval + Save (every epoch)
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    logging_strategy="steps",
    report_to="none",

    # Memory optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Windows compatibility
    dataloader_num_workers=0,
    dataloader_pin_memory=False,

    # Resume
    resume_from_checkpoint=True,

    # Seed
    seed=42,
)

# ============================================================
# TRAINER
# ============================================================
print(f"\n[5/6] Creating trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print(f"✓ Trainer ready")
print(f"  Early stopping: patience={EARLY_STOPPING_PATIENCE} epochs")

# ============================================================
# TRAIN
# ============================================================
print(f"\n[6/6] Starting training...")
print("=" * 60)
start_time = time.time()
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Max end: {(datetime.now() + timedelta(hours=MAX_RUNTIME_HOURS)).strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

try:
    result = trainer.train(resume_from_checkpoint=True if os.listdir(OUTPUT_DIR) else False)
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user. Saving current state...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"✓ Model saved to {FINAL_MODEL_DIR}")
    sys.exit(0)

elapsed = time.time() - start_time
elapsed_hours = elapsed / 3600

# ============================================================
# SAVE BEST MODEL
# ============================================================
print(f"\n{'=' * 60}")
print("SAVING FINAL MODEL")
print("=" * 60)

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Verify save
if os.path.exists(os.path.join(FINAL_MODEL_DIR, "adapter_config.json")):
    print(f"✓ Model saved: {FINAL_MODEL_DIR}")
else:
    print(f"✗ Save may have failed — adapter_config.json not found")

# ============================================================
# FINAL EVAL
# ============================================================
print(f"\nRunning final evaluation...")
eval_results = trainer.evaluate()

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'=' * 60}")
print("TRAINING RESULTS")
print("=" * 60)
print(f"  Duration:        {elapsed_hours:.2f} hours")
print(f"  Train loss:      {result.training_loss:.4f}")
print(f"  Val loss:        {eval_results['eval_loss']:.4f}")
print(f"  Total steps:     {result.global_step}")
print(f"  Epochs completed:{result.num_train_epochs if hasattr(result, 'num_train_epochs') else 'N/A'}")

# Overfitting check
gap = result.training_loss - eval_results["eval_loss"]
if gap < -0.3:
    print(f"  ⚠ Overfitting detected (train - val = {gap:.3f})")
    print(f"    Consider: reduce epochs, increase dropout, or add data")
elif abs(gap) < 0.1:
    print(f"  ✓ Good generalization (train ≈ val, gap = {gap:.3f})")
else:
    print(f"  ✓ Train/val gap: {gap:.3f}")

# Save training log
log_path = os.path.join(OUTPUT_DIR, "training_log.json")
log_data = {
    "timestamp": datetime.now().isoformat(),
    "duration_hours": elapsed_hours,
    "train_loss": result.training_loss,
    "eval_loss": eval_results["eval_loss"],
    "total_steps": result.global_step,
    "config": {
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
    },
    "gpu": gpu_name,
    "gpu_mem_gb": gpu_mem,
}

with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)
print(f"\n✓ Training log: {log_path}")

print(f"\n{'=' * 60}")
print(f"✓ TRAINING COMPLETE")
print(f"  Model: {FINAL_MODEL_DIR}")
print(f"  Next:  python test_inference_v3.py")
print("=" * 60)