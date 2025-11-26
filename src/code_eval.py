import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------- Chat 函数 -------------------------
def chat(model, tokenizer, prompt):
    """
    使用模型进行对话生成，用于代码评测。
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=400,       # 稍微短一点，减少废话
            do_sample=False,          # 评测时禁用采样，减少随机性
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只保留新生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# ------------------------- 工具函数：提取第一个代码块 -------------------------
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_first_code_block(text: str):
    """
    从模型回答中提取第一个 ```python ... ``` 代码块内部的内容。
    若找不到代码块则返回 None。
    """
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()

# ------------------------- 工具函数：执行代码并跑测试 -------------------------
def run_code_and_check(code_str: str, func_name: str, tests):
    """
    code_str: 模型生成的代码字符串
    func_name: 期望定义的函数名
    tests: 列表，每个元素是 dict: {"args": (...,), "kwargs": {...}, "expected": xxx}

    返回: (passed: bool, error_msg: str)
    """
    global_env = {}
    try:
        exec(code_str, global_env, global_env)
    except Exception as e:
        return False, f"exec error: {e}"

    if func_name not in global_env:
        return False, f"function `{func_name}` not defined"

    func = global_env[func_name]

    for idx, case in enumerate(tests):
        args = case.get("args", ())
        kwargs = case.get("kwargs", {})
        expected = case["expected"]
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            return False, f"runtime error on test {idx}: {e}"
        if result != expected:
            return False, f"wrong answer on test {idx}: expected {expected}, got {result}"

    return True, "ok"

# ------------------------- 20 道题：前 7 道保持不变，后 13 道为新增 -------------------------
TASKS = [
    # ===== 1 =====
    {
        "name": "sum_of_squares",
        "prompt": (
            "Write a Python function sum_of_squares(nums) that takes a list of integers and "
            "returns the sum of the squares of all numbers. Only output Python code."
        ),
        "func_name": "sum_of_squares",
        "tests": [
            {"args": ([1, 2, 3],), "expected": 1 + 4 + 9},
            {"args": ([0, -1, 2],), "expected": 0 + 1 + 4},
            {"args": ([],), "expected": 0},
        ],
    },
    # ===== 2 =====
    {
        "name": "second_largest",
        "prompt": (
            "Write a Python function second_largest(nums) that returns the second largest "
            "distinct integer in the list. If there is no such number, return None. "
            "Only output Python code."
        ),
        "func_name": "second_largest",
        "tests": [
            {"args": ([1, 2, 3, 4],), "expected": 3},
            {"args": ([5, 5, 5],), "expected": None},
            {"args": ([2],), "expected": None},
            {"args": ([2, 2, 3],), "expected": 2},
        ],
    },
    # ===== 3 =====
    {
        "name": "rotate_right",
        "prompt": (
            "Write a Python function rotate_right(nums, k) that takes a list of integers nums "
            "and a non-negative integer k, and returns a new list where the elements are rotated "
            "to the right by k positions. Only output Python code."
        ),
        "func_name": "rotate_right",
        "tests": [
            {"args": ([1, 2, 3, 4, 5], 2), "expected": [4, 5, 1, 2, 3]},
            {"args": ([1, 2, 3], 0), "expected": [1, 2, 3]},
            {"args": ([1, 2, 3], 3), "expected": [1, 2, 3]},
            {"args": ([1, 2, 3], 4), "expected": [3, 1, 2]},
        ],
    },
    # ===== 4 =====
    {
        "name": "flatten_once",
        "prompt": (
            "Write a Python function flatten_once(nested) that takes a list of lists of integers "
            "and returns a new flat list containing all the integers in order. Only output Python code."
        ),
        "func_name": "flatten_once",
        "tests": [
            {"args": ([[1, 2], [3, 4]],), "expected": [1, 2, 3, 4]},
            {"args": ([[], [1], [], [2, 3]],), "expected": [1, 2, 3]},
            {"args": ([],), "expected": []},
        ],
    },
    # ===== 5 =====
    {
        "name": "longest_word",
        "prompt": (
            "Write a Python function longest_word(text) that takes a string and returns the longest word. "
            "Words are separated by spaces. If there is a tie, return the first one with that length. "
            "Only output Python code."
        ),
        "func_name": "longest_word",
        "tests": [
            {"args": ("hello world",), "expected": "hello"},
            {"args": ("a bb ccc dd",), "expected": "ccc"},
            {"args": ("one two three",), "expected": "three"},
        ],
    },
    # ===== 6 =====
    {
        "name": "char_frequency",
        "prompt": (
            "Write a Python function char_frequency(s) that returns a dictionary mapping each character "
            "in the string s to its count. Only output Python code."
        ),
        "func_name": "char_frequency",
        "tests": [
            {"args": ("aba",), "expected": {"a": 2, "b": 1}},
            {"args": ("",), "expected": {}},
            {"args": ("aaabb",), "expected": {"a": 3, "b": 2}},
        ],
    },
    # ===== 7 =====
    {
        "name": "intersection_sorted_unique",
        "prompt": (
            "Write a Python function intersection(nums1, nums2) that returns a new sorted list of distinct integers "
            "that appear in both nums1 and nums2. Only output Python code."
        ),
        "func_name": "intersection",
        "tests": [
            {"args": ([1, 2, 2, 3], [2, 2, 4],), "expected": [2]},
            {"args": ([1, 3, 5], [2, 4, 6],), "expected": []},
            {"args": ([1, 2, 3], [3, 2, 1],), "expected": [1, 2, 3]},
        ],
    },

    # ===== 8 新增 =====
    {
        "name": "count_unique",
        "prompt": (
            "Write a Python function count_unique(nums) that returns the number of distinct integers in the list. "
            "Only output Python code."
        ),
        "func_name": "count_unique",
        "tests": [
            {"args": ([1, 2, 2, 3],), "expected": 3},
            {"args": ([5, 5, 5],), "expected": 1},
            {"args": ([],), "expected": 0},
        ],
    },
    # ===== 9 新增 =====
    {
        "name": "unique_elements",
        "prompt": (
            "Write a Python function unique_elements(nums) that returns a new list containing only the elements "
            "that appear exactly once in the list, preserving their original order. Only output Python code."
        ),
        "func_name": "unique_elements",
        "tests": [
            {"args": ([1, 2, 2, 3, 4, 4, 5],), "expected": [1, 3, 5]},
            {"args": ([1, 1, 1],), "expected": []},
            {"args": ([1, 2, 3],), "expected": [1, 2, 3]},
        ],
    },
    # ===== 10 新增 =====
    {
        "name": "compress_string_rle",
        "prompt": (
            "Write a Python function compress_string(s) that applies a simple run-length encoding to the string s. "
            "Consecutive repeated characters are replaced by the character followed by the count. "
            "For example, 'aaabb' becomes 'a3b2'. If the string is empty, return an empty string. "
            "Only output Python code."
        ),
        "func_name": "compress_string",
        "tests": [
            {"args": ("aaabb",), "expected": "a3b2"},
            {"args": ("abcd",), "expected": "a1b1c1d1"},
            {"args": ("",), "expected": ""},
        ],
    },
    # ===== 11 新增 =====
    {
        "name": "two_sum_indices",
        "prompt": (
            "Write a Python function two_sum(nums, target) that takes a list of integers nums and an integer target, "
            "and returns a tuple (i, j) such that i < j and nums[i] + nums[j] == target. "
            "If no such pair exists, return None. Only output Python code."
        ),
        "func_name": "two_sum",
        "tests": [
            {"args": ([2, 7, 11, 15], 9), "expected": (0, 1)},
            {"args": ([3, 2, 4], 6), "expected": (1, 2)},
            {"args": ([1, 2, 3], 100), "expected": None},
        ],
    },
    # ===== 12 新增 =====
    {
        "name": "remove_consecutive_duplicates",
        "prompt": (
            "Write a Python function remove_consecutive_duplicates(nums) that takes a list of integers nums "
            "and returns a new list where any group of consecutive duplicate numbers is replaced by a single one. "
            "Only output Python code."
        ),
        "func_name": "remove_consecutive_duplicates",
        "tests": [
            {"args": ([1, 1, 2, 2, 2, 3],), "expected": [1, 2, 3]},
            {"args": ([4, 4, 4, 4],), "expected": [4]},
            {"args": ([],), "expected": []},
            {"args": ([1, 2, 3],), "expected": [1, 2, 3]},
        ],
    },

    # ===== 13 新增 =====
    {
        "name": "lis_length",
        "prompt": (
            "Write a Python function lis_length(nums) that returns the length of the longest strictly increasing "
            "subsequence in the list of integers nums. You may use an O(n^2) dynamic programming solution. "
            "Only output Python code."
        ),
        "func_name": "lis_length",
        "tests": [
            {"args": ([10, 9, 2, 5, 3, 7, 101, 18],), "expected": 4},
            {"args": ([0, 1, 0, 3, 2, 3],), "expected": 4},
            {"args": ([7, 7, 7, 7],), "expected": 1},
        ],
    },
    # ===== 14 新增 =====
    {
        "name": "matrix_transpose",
        "prompt": (
            "Write a Python function transpose(matrix) that takes a matrix represented as a list of lists, "
            "and returns its transpose. You may assume all rows have the same length. Only output Python code."
        ),
        "func_name": "transpose",
        "tests": [
            {"args": ([[1, 2, 3], [4, 5, 6]],), "expected": [[1, 4], [2, 5], [3, 6]]},
            {"args": ([[1], [2], [3]],), "expected": [[1, 2, 3]]},
            {"args": ([[1, 2]],), "expected": [[1], [2]]},
        ],
    },
    # ===== 15 新增 =====
    {
        "name": "group_by_parity",
        "prompt": (
            "Write a Python function group_by_parity(nums) that takes a list of integers and returns a dictionary with "
            "two keys: 'even' and 'odd'. The value for 'even' is a list of all even numbers in nums (in order), and "
            "the value for 'odd' is a list of all odd numbers in nums (in order). Only output Python code."
        ),
        "func_name": "group_by_parity",
        "tests": [
            {"args": ([1, 2, 3, 4, 5],), "expected": {"even": [2, 4], "odd": [1, 3, 5]}},
            {"args": ([2, 4, 6],), "expected": {"even": [2, 4, 6], "odd": []}},
            {"args": ([1, 3, 5],), "expected": {"even": [], "odd": [1, 3, 5]}},
        ],
    },
    # ===== 16 新增 =====
    {
        "name": "count_word_frequencies",
        "prompt": (
            "Write a Python function count_word_frequencies(text) that takes a string text and returns a dictionary "
            "mapping each lowercase word to its frequency. Words are separated by whitespace or punctuation. "
            "Ignore case. Only output Python code."
        ),
        "func_name": "count_word_frequencies",
        "tests": [
            {"args": ("Hello, hello world!",), "expected": {"hello": 2, "world": 1}},
            {"args": ("One, two, two; THREE three THREE.",), "expected": {"one": 1, "two": 2, "three": 3}},
        ],
    },
    # ===== 17 新增 =====
    {
        "name": "is_palindrome_clean",
        "prompt": (
            "Write a Python function is_palindrome(s) that returns True if the string s is a palindrome, "
            "ignoring case and non-alphanumeric characters. Otherwise return False. Only output Python code."
        ),
        "func_name": "is_palindrome",
        "tests": [
            {"args": ("A man, a plan, a canal: Panama",), "expected": True},
            {"args": ("race a car",), "expected": False},
            {"args": ("",), "expected": True},
        ],
    },
    # ===== 18 新增 =====
    {
        "name": "chunk_list",
        "prompt": (
            "Write a Python function chunk_list(lst, size) that splits the list lst into smaller lists (chunks) of "
            "length size. The last chunk may be shorter if there are not enough elements. Only output Python code."
        ),
        "func_name": "chunk_list",
        "tests": [
            {"args": ([1, 2, 3, 4, 5], 2), "expected": [[1, 2], [3, 4], [5]]},
            {"args": ([1, 2, 3], 1), "expected": [[1], [2], [3]]},
            {"args": ([], 3), "expected": []},
        ],
    },
    # ===== 19 新增 =====
    {
        "name": "prefix_sums",
        "prompt": (
            "Write a Python function prefix_sums(nums) that returns a new list where the i-th element is the sum of "
            "nums[0] to nums[i]. Only output Python code."
        ),
        "func_name": "prefix_sums",
        "tests": [
            {"args": ([1, 2, 3],), "expected": [1, 3, 6]},
            {"args": ([0, 0, 0],), "expected": [0, 0, 0]},
            {"args": ([],), "expected": []},
        ],
    },
    # ===== 20 新增 =====
    {
        "name": "find_mode_smallest",
        "prompt": (
            "Write a Python function find_mode(nums) that returns the integer that appears most frequently in the list. "
            "If multiple numbers have the same highest frequency, return the smallest one. "
            "If the list is empty, return None. Only output Python code."
        ),
        "func_name": "find_mode",
        "tests": [
            {"args": ([1, 2, 2, 3, 3, 3],), "expected": 3},
            {"args": ([4, 4, 5, 5],), "expected": 4},
            {"args": ([],), "expected": None},
        ],
    },
]

# ------------------------- 主流程：跑评测并统计 AC 率 -------------------------
def main():
    base_path = "./Qwen1.5-1.8B-Chat"
    adapter_path = "./outputs_10000/checkpoint-10000"

    print(f"🔹 加载基础模型 {base_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    print(f"🔹 加载微调模型 {adapter_path}")
    try:
        ft_tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    except Exception:
        print("⚠️ Adapter 路径未找到 tokenizer，复用基座 tokenizer")
        ft_tokenizer = base_tokenizer

    print("... 正在加载微调模型的基座 ...")
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    print("... 正在应用 LoRA adapter ...")
    ft_model = PeftModel.from_pretrained(ft_base_model, adapter_path, local_files_only=True)

    print("\n✅ 模型加载完成，开始批量评测代码 AC 率\n")

    base_pass = 0
    ft_pass = 0
    total = len(TASKS)

    for idx, task in enumerate(TASKS, 1):
        print(f"🧩 Task {idx}/{total}: {task['name']}")
        prompt = task["prompt"]
        func_name = task["func_name"]
        tests = task["tests"]

        # -------- 基础模型 --------
        base_resp = chat(base_model, base_tokenizer, prompt)
        base_code = extract_first_code_block(base_resp)
        if base_code is None:
            base_ok, base_msg = False, "no code block found"
        else:
            base_ok, base_msg = run_code_and_check(base_code, func_name, tests)

        # -------- 微调模型 --------
        ft_resp = chat(ft_model, ft_tokenizer, prompt)
        ft_code = extract_first_code_block(ft_resp)
        if ft_code is None:
            ft_ok, ft_msg = False, "no code block found"
        else:
            ft_ok, ft_msg = run_code_and_check(ft_code, func_name, tests)

        if base_ok:
            base_pass += 1
        if ft_ok:
            ft_pass += 1

        print(f"   - Base model:      {'✅ PASS' if base_ok else '❌ FAIL'} ({base_msg})")
        print(f"   - Finetuned model: {'✅ PASS' if ft_ok else '❌ FAIL'} ({ft_msg})")
        print("-" * 60)

    print("\n📊 最终统计：")
    print(f"   基础模型  AC 数量: {base_pass} / {total}  (AC rate = {base_pass / total:.2%})")
    print(f"   微调模型  AC 数量: {ft_pass} / {total}  (AC rate = {ft_pass / total:.2%})")

if __name__ == "__main__":
    main()
