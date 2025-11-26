# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="bigcode/starcoderbase-1b")
parser.add_argument("--peft_model", type=str, required=True, help="path to peft output dir")
parser.add_argument("--prompt", type=str, default="Write factorial in Python.")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# wrap with peft
model = PeftModel.from_pretrained(model, args.peft_model)
model.eval()

input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=0.0)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
