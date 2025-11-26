# **基于 QLoRA 的 Qwen1.5-1.8B 代码生成能力微调 **

---

**项目简介**
在  RTX 2080 Ti* 8（11 GB） 上即可完成的微调实验，让 1.8 B 小模型释放生产级代码生成能力。  
我们使用 QLoRA 4-bit 量化 + 指令微调，在 OpenCodeInstruct 上实现：

| 指标 | 基座 | 微调后 | 提升 |
|---|---|---|---|
| ROUGE-L F1 | 19.39 % | 45.35 % | +133 % |
| Pass@1（20 题） | 55 % | 75 % | +20 % |

---

**关键特性**
| 特性 | 说明 |
|---|---|
|  极致高效 | 4-bit NF4 + 双重量化|
|  分布式就绪 | 内置 DDP 同步 + 梯度检查点修复，支持多卡并行 |
|  深度优化 | LoRA 目标层覆盖 全部线性层（含 MLP），专为代码任务调参 |
|  全链路评估 | 训练监控 + ROUGE 计算 + 沙箱执行 一键完成 |

---

##  快速开始
### 1. 硬件要求
- NVIDIA GPU ≥ 40 GB（2080Ti / 3060 / 4060Ti 等）
- Linux / WSL2 + CUDA ≥ 11.8

### 2. 安装
```bash
git clone https://github.com/your-username/Qwen-Code-SFT.git
cd Qwen-Code-SFT

conda create -n qwen-sft python=3.10 -y
conda activate qwen-sft

# PyTorch 示例（CUDA 11.8）
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. 推理
```bash
python src/inference.py \
    --base_model "Qwen/Qwen1.5-1.8B-Chat" \
    --lora_adapter "Qwen/Code-SFT-1.8B" \
    --interactive
```

---

##  训练自己的模型
### 数据格式
```jsonl
{"prompt": "Write a Python function to merge two dictionaries.", "response": "def merge_dicts(d1, d2):\n    return {d1, d2}"}
```
放入 `data/YourDataset.jsonl` 即可。

### 一键训练（双卡示例）
```bash
bash scripts/run_train.sh
```
脚本已写好 accelerate 参数，默认 2 卡 DDP，2500 steps ≈ 1.5 小时 完成。

---

##  项目结构
```
Qwen-Code-SFT/
├── src/
│   ├── train.py          # 训练入口
│   ├── inference.py      # 推理交互
│   └── utils.py          # 数据 / 指标
├── scripts/
│   ├── run_train.sh      # 一键训练
│   └── run_eval.sh       # 一键评测
├── data/                 # 放置 .jsonl
├── outputs/              # 日志 + LoRA 权重
├── requirements.txt
└── README.md
```
---

##  许可证
- 本项目代码：Apache 2.0  
- 基座模型权重：遵循 [Qwen1.5 官方协议](https://github.com/QwenLM/Qwen1.5/blob/main/Tongyi_License.pdf)

---

## 📧 联系我们
有问题请直接提 [Issue](https://github.com/your-username/Qwen-Code-SFT/issues) 或 Discussion，维护者会在 24 h 内回复！
```
