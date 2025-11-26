import pandas as pd
import json
import argparse
import sys
import os
import glob
from tqdm import tqdm

def format_dataset_to_prompt_response(row):
    """
    转换 OpenCodeInstruct-like 的 DataFrame行 (row) 为 'prompt/response' 格式。

    假设: 
    - 新数据集的列名是 'input' (编码问题) 和 'output' (大模型的回答)。
    """
    
    # -----------------------------------------------------------------
    # 👇 重要：根据你的新数据集列名进行映射
    # -----------------------------------------------------------------
    # 你的新格式中，'input' 是编码问题，'output' 是回答。
    
    instruction = row.get('input')
    output = row.get('output')
    
    # 如果列名不存在或内容为空，则跳过
    if not instruction or not output:
        return None

    # -----------------------------------------------------------------
    
    # 根据 code_alpaca 格式构建 prompt
    # 新格式的 'input' 字段本身就是完整的指令，所以 'input' 设为空
    input_str = "< noinput >"

    prompt_template = (
        "Instruction:\n{instruction}\n\nInput:\n{input_str}\n\nAnswer:"
    )
    prompt = prompt_template.format(instruction=instruction, input_str=input_str)

    return {"prompt": prompt, "response": output}

def main():
    parser = argparse.ArgumentParser(description="将一个目录中的所有 .parquet 转换为 'prompt/response' (.jsonl) 格式")
    parser.add_argument(
        '--source_dir', 
        type=str, 
        required=True, 
        help="包含 .parquet 文件的源目录路径"
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True, 
        help="合并后的输出 .jsonl 文件路径"
    )
    parser.add_argument(
        '--max_entries',
        type=int,
        default=None,
        help="（可选）处理 N 条数据后停止，用于快速测试"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # 👇 更改：搜索目录而不是读取单个文件
    # -----------------------------------------------------------------
    print(f"🚀 开始搜索目录: {args.source_dir}")
    
    # 使用 glob 递归搜索所有 .parquet 文件
    search_path = os.path.join(args.source_dir, "**", "*.parquet")
    source_files = glob.glob(search_path, recursive=True)

    if not source_files:
        print(f"错误: 在 '{args.source_dir}' 中未找到 .parquet 文件。", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 找到了 {len(source_files)} 个 .parquet 文件:")
    for f in source_files:
        print(f"  - {f}")
    # -----------------------------------------------------------------

    print(f"\n🎯 开始转换并写入到: {args.output_file}")

    processed_count = 0
    skipped_count = 0
    stop_processing = False

    try:
        # 一次性打开输出文件
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            
            # 遍历找到的每个文件
            for source_file in source_files:
                if stop_processing:
                    break
                
                print(f"\nProcessing file: {source_file}")
                
                try:
                    df = pd.read_parquet(source_file)
                    print(f"  源文件行数: {len(df)}")
                    
                    # 打印列名检查 (仅第一个文件)
                    if processed_count == 0 and len(df) > 0:
                        print("\n  --- 列名检查 (文件首行) ---")
                        preview_cols = [col for col in ['input', 'output'] if col in df.columns]
                        if not preview_cols:
                            print(f"  警告: 在源文件中未找到 'input' 或 'output' 列。")
                            print(f"  找到的列: {df.columns.tolist()}")
                        else:
                            print(df[preview_cols].head(1))
                        print("  ----------------------------\n")

                except Exception as e:
                    print(f"  错误: 无法读取 Parquet 文件 {source_file}. {e}", file=sys.stderr)
                    print("  请确保你已运行: pip install pandas pyarrow", file=sys.stderr)
                    skipped_count += 1 # 标记整个文件为“跳过”
                    continue # 继续处理下一个文件

                # 使用 tqdm 显示当前文件的处理进度
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  -> {os.path.basename(source_file)}"):
                    
                    # 检查是否达到了 max_entries 限制
                    if args.max_entries and processed_count >= args.max_entries:
                        print(f"\n⚠️ 已达到 {args.max_entries} 条数据的最大限制，停止处理。")
                        stop_processing = True
                        break

                    formatted_entry = format_dataset_to_prompt_response(row)
                    
                    if formatted_entry:
                        json.dump(formatted_entry, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        processed_count += 1
                    else:
                        skipped_count += 1

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        
    print("\n" + "="*50)
    print(f"✅ 处理完成！")
    print(f"已处理条目: {processed_count}")
    print(f"已跳过条目: {skipped_count} (可能包含无效行或无法读取的整个文件)")
    print(f"总输出文件: {args.output_file}")

if __name__ == "__main__":
    main()