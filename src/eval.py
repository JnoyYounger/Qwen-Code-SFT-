import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os


def chat(model, tokenizer, prompt):
    """
    Use model to generate text for a chat-style prompt.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_evaluation_data(filepath="data/Opencode.jsonl", limit=100):
    """
    Load evaluation pairs (prompt, reference) from a .jsonl file.
    """
    PROMPT_KEY_NAME = "prompt"
    REFERENCE_KEY_NAME = "response"

    if not os.path.exists(filepath):
        print(f"ERROR: evaluation file not found: {filepath}")
        return []

    data_pairs = []
    print(f"Loading first {limit} items from {filepath} ...")

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                record = json.loads(line)
                prompt_text = record.get(PROMPT_KEY_NAME)
                reference_text = record.get(REFERENCE_KEY_NAME)

                if prompt_text and reference_text is not None:
                    if not isinstance(reference_text, str):
                        reference_text = str(reference_text)

                    data_pairs.append({
                        "prompt": prompt_text,
                        "reference": reference_text,
                    })
                else:
                    if not prompt_text:
                        print(f"WARN: skip line {i+1}, missing '{PROMPT_KEY_NAME}'")
                    if reference_text is None:
                        print(f"WARN: skip line {i+1}, missing '{REFERENCE_KEY_NAME}'")

            except json.JSONDecodeError:
                print(f"WARN: skip line {i+1}, JSON decode error")

    print(f"Loaded {len(data_pairs)} evaluation items.")
    return data_pairs


def _lcs_length(X, Y):
    """
    Longest Common Subsequence length for token lists X and Y.
    """
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]


def calculate_rouge_l(prediction, reference):
    """
    Compute ROUGE-L F1-Score using simple whitespace tokenization.
    """
    pred_tokens = prediction.strip().split()
    ref_tokens = reference.strip().split()

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if pred_len == 0 or ref_len == 0:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    if lcs_len == 0:
        return 0.0

    R_lcs = lcs_len / ref_len
    P_lcs = lcs_len / pred_len
    F1_lcs = 2 * (P_lcs * R_lcs) / (P_lcs + R_lcs)
    return F1_lcs


def main():
    print("Loading base model ./Qwen1.5-1.8B-Chat")
    base_tokenizer = AutoTokenizer.from_pretrained(
        "./Qwen1.5-1.8B-Chat",
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "./Qwen1.5-1.8B-Chat",
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True,
    )

    print("Loading fine-tuned checkpoint ./outputs_2500/checkpoint-1000")
    checkpoint_dir = "./outputs_2500/checkpoint-1000"
    try:
        ft_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )
    except Exception as e:
        print(f"WARN: tokenizer not in {checkpoint_dir}, fallback to base tokenizer. Error: {e}")
        ft_tokenizer = base_tokenizer

    print("Loading fine-tuned base ...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        "./Qwen1.5-1.8B-Chat",
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True,
    )

    print("Applying LoRA adapter from ./outputs_2500 ...")
    ft_model = PeftModel.from_pretrained(ft_model, "./outputs_2500", local_files_only=True)

    print("Models ready.")

    # Load evaluation data
    evaluation_pairs = load_evaluation_data(filepath="data/1.jsonl", limit=100)
    if not evaluation_pairs:
        print("No evaluation data found. Exit.")
        return

    print("Start batch evaluation")

    base_rouge_l_scores = []
    ft_rouge_l_scores = []

    output_filename = "evaluation_results.txt"
    with open(output_filename, "w", encoding="utf-8") as f_out:
        for i, pair in enumerate(evaluation_pairs):
            print(f"Evaluating {i+1}/{len(evaluation_pairs)} ...")

            prompt = pair["prompt"]
            reference = pair["reference"]

            messages = [{"role": "user", "content": prompt}]
            base_prompt_formatted = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ft_prompt_formatted = ft_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            f_out.write(f"\n\n{'='*60} Item {i+1}/{len(evaluation_pairs)} {'='*60}\n")
            f_out.write(f"Prompt:\n{prompt}\n")
            f_out.write(f"\nReference:\n")
            f_out.write(f"{reference}\n")

            # Base model
            base_ans = chat(base_model, base_tokenizer, base_prompt_formatted)
            base_ans_only = base_ans[len(base_prompt_formatted):].strip()
            f_out.write("\n--- Base model answer ---\n")
            f_out.write(base_ans_only)
            base_f1 = calculate_rouge_l(base_ans_only, reference)
            base_rouge_l_scores.append(base_f1)
            f_out.write(f"\n(ROUGE-L F1: {base_f1:.4f})")

            # Fine-tuned model
            ft_ans = chat(ft_model, ft_tokenizer, ft_prompt_formatted)
            ft_ans_only = ft_ans[len(ft_prompt_formatted):].strip()
            f_out.write("\n--- FT model answer ---\n")
            f_out.write(ft_ans_only)
            ft_f1 = calculate_rouge_l(ft_ans_only, reference)
            ft_rouge_l_scores.append(ft_f1)
            f_out.write(f"\n(ROUGE-L F1: {ft_f1:.4f})")

            better = "FT better than Base" if ft_f1 > base_f1 else ("Base better than FT" if base_f1 > ft_f1 else "Tie")
            f_out.write(f"\nConclusion: {better}\n")

        # Summary
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        base_avg = _avg(base_rouge_l_scores)
        ft_avg = _avg(ft_rouge_l_scores)
        gain = ft_avg - base_avg

        f_out.write("\n\n" + "="*60 + "\n")
        f_out.write(f"Base average ROUGE-L F1: {base_avg:.4f}\n")
        f_out.write(f"FT average ROUGE-L F1: {ft_avg:.4f}\n")
        f_out.write(f"Gain: {gain:.4f}\n")

    print("Evaluation finished. Results written to evaluation_results.txt")


if __name__ == "__main__":
    main()