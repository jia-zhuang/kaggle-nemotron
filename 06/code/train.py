import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, stat, shutil, gc, zipfile
import re
from pathlib import Path
import polars as pl
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# === Hyperparameters ===
SUBSAMPLE_SIZE = 600

LORA_RANK = 32
MAX_SEQ_LEN = 8192

NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4        
LR = 2e-4

COMPLETION_ONLY_LOSS=False

OUTPUT_DIR = "./adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= Resume helper =========
def find_latest_checkpoint(output_dir: str):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    checkpoints = []
    for p in output_dir.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"checkpoint-(\d+)", p.name)
            if m:
                checkpoints.append((int(m.group(1)), str(p)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]

# 用环境变量控制恢复行为：
#   auto  -> 自动找最新 checkpoint
#   none  -> 强制从头训练
#   具体路径 -> 从指定 checkpoint 恢复
RESUME_MODE = os.environ.get("RESUME_FROM_CHECKPOINT", "auto")

if RESUME_MODE.lower() == "auto":
    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
elif RESUME_MODE.lower() in {"none", "false", "0", ""}:
    resume_checkpoint = None
else:
    resume_checkpoint = RESUME_MODE  # 用户手动指定路径


# === Data ===
MODEL_PATH = "/root/code/Nemotron-3-Nano-30B-A3B/"
train_df = pd.read_json('./train_manual_reason_all_add_guess_opt.jsonl', lines=True, dtype=str)
print('NUM_EXAMPLES:', train_df.shape[0])

hf_dataset = Dataset.from_pandas(train_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_training_text(example):
    prompt = example["prompt"]
    answer = example["extracted_answer"]

    think = example["llm_think"]
    response = example.get("llm_response")

    label = example['label']

    # if 'bitwise and binary transformation tasks' in label:
    #     cot = think
    # else:
    #     cot = response

    cot = think

    assert isinstance(cot, str)
    assert len(cot) > 10

    user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
    
    # Combine the CoT with the final answer
    # if 'textual cipher and string transformation' in label:
    #     assistant_msg = cot
    # else:
    #     assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"

    # if 'bitwise and binary transformation tasks' in label:
    #     assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"
    # else:
    #     assistant_msg = cot
    
    # assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
    assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"
    
    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}

# Apply mapping
hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)

print('>===== TRAINING EXAMPLE =====<')
print(hf_dataset[0]['text'])
print('<===== ################ =====>')

#hf_dataset = load_dataset(
#    "json",
#    data_files="final_Nemotron_training_data_sample600_completions.jsonl",
#    split="train",
#)



# === Model ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 配合 checkpointing，训练时建议关掉 cache 
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.gradient_checkpointing_enable()

#for name, mod in sys.modules.items():
#    if "modeling_nemotron_h" in name:
#        mod.is_fast_path_available = False
#        print(f"Patched {name}: is_fast_path_available = False")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# === Training ===
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE, # Optimized
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    completion_only_loss=COMPLETION_ONLY_LOSS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ignore_data_skip=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("Starting training...")

if resume_checkpoint is not None:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch.")
    trainer.train()


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR) # Ensure tokenizer config is saved
