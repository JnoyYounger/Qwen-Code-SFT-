import os
import torch
import torch.distributed as dist # <--- 导入分布式库
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import argparse

# ✅ 1. 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./Qwen1.5-1.8B-Chat", help="Path to base model directory")
parser.add_argument("--data_path", type=str, default="data/Opencode.jsonl", help="Path to processed dataset")
parser.add_argument("--output_dir", type=str, default="outputs_2500", help="Path to save the fine-tuned model")
parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization")
parser.add_argument("--max_steps", type=int, default=1000, help="Total number of training steps")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device (Set to 1 for <11GB GPUs)")
parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--num_proc", type=int, default=16, help="Number of CPU cores for tokenization")
args = parser.parse_args()


# -------------------------------
# 加载数据
# -------------------------------
print(f"✅ Loading dataset from {args.data_path}")
dataset = load_dataset(
    "json", 
    data_files=args.data_path,
    cache_dir="./hf_cache"  # 磁盘空间修复
)
train_data = dataset["train"]

# -------------------------------
# 加载 tokenizer
# -------------------------------
print(f"✅ Loading tokenizer from {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token


# <-- 重写 tokenize_function 以支持批量 (batched=True)
def tokenize_function(examples): # 'example' -> 'examples'
    texts = [
        prompt + "\n" + response 
        for prompt, response in zip(examples["prompt"], examples["response"])
    ]
    # 添加 eos token (Qwen1.5-Chat 格式推荐)
    texts = [t + tokenizer.eos_token for t in texts]
    
    return tokenizer(
        texts,
        truncation=True,
        padding=False, 
        max_length=args.max_seq_length,
    )

# -------------------------------
# 👇 关键修复: 同步数据处理
# -------------------------------
# 我们必须从 os.environ 中显式读取 "LOCAL_RANK"
# `accelerate launch` 会为每个进程设置这个变量 (0, 1, 2, 3...)
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# 检查我们是否处于 DDP (分布式) 模式
is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1

# 如果是 DDP 并且 PyTorch 分布式还没初始化，则初始化
# (Accelerate 启动时通常已经为我们做了，但这是个安全保障)
if is_ddp and not dist.is_initialized():
    dist.init_process_group(backend='nccl')

print(f"✅ Tokenizing dataset with max_length={args.max_seq_length} using {args.num_proc} cores... [Rank {local_rank}]")

if is_ddp and local_rank != 0:
    # -------------------------------
    # 👇 非主进程在此等待
    # -------------------------------
    print(f"[Rank {local_rank}] Waiting for main process (rank 0) to tokenize data...")
    dist.barrier() # 等待 rank 0 完成

tokenized_datasets = train_data.map(
    tokenize_function, 
    batched=True,      # 开启批量处理
    num_proc=args.num_proc,  # 启用多核处理
    remove_columns=train_data.column_names
)

if is_ddp and local_rank == 0:
    # -------------------------------
    # 👇 Rank 0 (主进程) 完成后，通知其他进程
    # -------------------------------
    print(f"[Rank 0] Tokenization complete. Signaling other processes...")
    dist.barrier() # 通知其他进程 "缓存已准备好"

if is_ddp:
     print(f"[Rank {local_rank}] Proceeding after tokenization barrier.")
# -------------------------------
# 👆 修复结束
# -------------------------------


# -------------------------------
# 加载 Qwen 模型（QLoRA 配置）
# -------------------------------
print("✅ Loading base model with 4-bit quantization (QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"Loading model on local_rank: {local_rank} (handled by Accelerate)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=bnb_config,
    # -------------------------------
    # 👇 关键修复:
    # 使用 local_rank 整数，而不是 torch.cuda.current_device()
    # -------------------------------
    device_map={'': local_rank}, 
    low_cpu_mem_usage=True,
    local_files_only=True,
    torch_dtype=torch.float16 # <--- 修复加载 OOM
)

# -------------------------------
# 关键修复：为 k-bit 训练 + 梯度检查点准备模型
# -------------------------------
model = prepare_model_for_kbit_training(model) # 修复 'no grad_fn' 错误


# -------------------------------
# 设置 LoRA 配置
# -------------------------------
print("✅ Setting up optimized LoRA config...")
lora_config = LoraConfig(
    r=32,                  
    lora_alpha=64,           
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj"
    ],                      
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------
# 训练参数设置
# -------------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,    
    gradient_accumulation_steps=args.grad_accum,    
    learning_rate=2e-4,
    
    max_steps=args.max_steps,        
    #num_train_epochs=1,     # <-- 添加这一行 (或者 2, 3)
    fp16=True,
    gradient_checkpointing=True,    # 节省显存
    
    # <-- ✅ MODIFIED: 添加这一行来修复 DDP + GC 冲突
    gradient_checkpointing_kwargs={'use_reentrant': False},
    
    save_total_limit=3,
    logging_steps=10,
    save_steps=200,
    eval_strategy="no",
    
    lr_scheduler_type="cosine",    
    warmup_steps=100,              
    
    report_to="none"
)

# -------------------------------
# Data Collator (动态 Padding)
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------------
# 开始训练
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Starting QLoRA fine-tuning...")
trainer.train()
print("✅ Training complete! Saving model...")
trainer.save_model(args.output_dir)
print(f"✅ Model saved to {args.output_dir}")