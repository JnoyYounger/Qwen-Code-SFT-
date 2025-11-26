# data_utils.py
from transformers import AutoTokenizer
from datasets import load_dataset

def get_tokenizer(model_name_or_path, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
    # for some code models, ensure eos token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def preprocess(dataset_path, tokenizer, max_length=1024):
    ds = load_dataset("json", data_files=dataset_path)["train"]
    def tokenize_fn(example):
        text = example["prompt"] + example["response"]
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        # label masking: we want the model to predict the response only.
        # Simplest approach: labels = input_ids but set -100 for prompt tokens.
        input_ids = tokens["input_ids"]
        # compute prompt length:
        prompt_tokens = tokenizer(example["prompt"], truncation=True, max_length=max_length)["input_ids"]
        labels = [-100]*len(prompt_tokens) + input_ids[len(prompt_tokens):]
        # handle length mismatch
        labels = labels[:len(input_ids)]
        return {"input_ids": input_ids, "attention_mask": tokens["attention_mask"], "labels": labels}

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=False)
    return tokenized
