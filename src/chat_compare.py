import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------------------------------------------
# 👇 修复重点：chat 函数增加了 apply_chat_template 和切片解码
# -----------------------------------------------------------------
def chat(model, tokenizer, prompt):
    """
    使用模型进行对话生成 (修复版)。
    """
    # 1. 构建符合 Qwen 格式的对话消息
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 2. 应用对话模板 (Chat Template)
    # 这会自动添加 <|im_start|>user...<|im_end|><|im_start|>assistant 等特殊标记
    # add_generation_prompt=True 会告诉模型"现在轮到你说话了"
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 转为 Tensor 并移动到设备
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,  # 👈 建议改小一点，防止废话太多，通常 512 够用了
            do_sample=True,
            temperature=0.7,     # 温度，太高容易胡话，太低容易死板
            top_p=0.9,
            repetition_penalty=1.1, # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id # 防止报警
        )

    # 4. 关键步骤：只解码新生成的 token，去掉输入的 prompt 部分
    # model.generate 返回的是 [输入+输出]，我们只需要 [输出]
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()
# -----------------------------------------------------------------


def main():
    base_path = "./Qwen1.5-1.8B-Chat"
    adapter_path = "./outputs_10000/checkpoint-10000" # 或者直接 ./outputs

    print(f"🔹 加载基础模型 {base_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_path, 
        local_files_only=True,
        trust_remote_code=True # Qwen 有时需要这个
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    print(f"🔹 加载微调模型 {adapter_path}")
    # 注意：微调模型的 Tokenizer 通常和基座一样，除非你改了词表，否则可以直接复用 base_tokenizer
    # 这里为了保险起见还是加载一遍，但要注意 adapter 路径里有没有 tokenizer.json
    try:
        ft_tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    except:
        print("⚠️ Adapter 路径未找到 tokenizer，复用基座 tokenizer")
        ft_tokenizer = base_tokenizer

    # (A) 加载用于微调演示的基座模型 (为了对比，我们需要两个独立的模型实例)
    # 显存如果不够，这一步会报错。如果显存不够，建议每次只加载一个模型测试。
    print("... 正在加载微调模型的基座...")
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    # (B) 应用 LoRA
    print("... 正在应用 LoRA adapter ...")
    ft_model = PeftModel.from_pretrained(ft_base_model, adapter_path, local_files_only=True)

    print("\n✅ 模型加载完成，开始实时对话测试！")
    print("输入 'exit' 退出。")

    while True:
        prompt = input("\n🧩 请输入测试内容：")
        if prompt.strip().lower() in ["exit", "quit"]:
            print("👋 已退出。")
            break

        print("\n--- 🧠 基础模型回答 ---")
        # 基础模型也要用 apply_chat_template，否则效果也会变差
        base_ans = chat(base_model, base_tokenizer, prompt)
        print(base_ans)

        print("\n--- 🚀 微调模型回答 ---")
        ft_ans = chat(ft_model, ft_tokenizer, prompt)
        print(ft_ans)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()