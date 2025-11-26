# json2prompt_response.py
import json
import argparse
from pathlib import Path

TMPL = (
    "Instruction:\n{instruction}\n\n"
    "Input:\n{input_str}\n\n"
    "Answer:"
)

def convert_one(item: dict) -> dict:
    instruction = item.get("instruction", "")
    input_cont  = item.get("input", "")
    output      = item.get("output", "")

    # 如果 input 为空，则填固定占位符
    input_str = "< noinput >" if not input_cont else input_cont

    prompt = TMPL.format(instruction=instruction, input_str=input_str)
    return {"prompt": prompt, "response": output}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="原始 instruction/input/output 数组 JSON 文件")
    parser.add_argument("output_jsonl", help="输出 prompt/response 格式 .jsonl")
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(convert_one(item), ensure_ascii=False) + "\n")
    print("✅ 完成 ->", args.output_jsonl)

if __name__ == "__main__":
    main()