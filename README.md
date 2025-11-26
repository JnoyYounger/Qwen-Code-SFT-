🚀 Qwen-Code-SFT: 基于 QLoRA 的 Qwen1.5-1.8B 代码生成能力微调

提升轻量级模型的编程极限。本项目通过 QLoRA (Quantized Low-Rank Adaptation) 技术，在单张消费级显卡（如 RTX 2080 Ti）上高效微调 Qwen1.5-1.8B 模型，使其在代码生成任务上的表现显著增强。

通过在 OpenCodeInstruct 数据集上的指令微调（SFT），本模型在代码生成准确率（ROUGE-L）上提升了 133%，并在功能性正确性测试（Pass@1）中达到了 75% 的通过率。

✨ 关键特性

⚡ 高效微调: 采用 4-bit NF4 量化 + QLoRA 技术，将训练显存需求降低至 11GB 以下。

🔄 分布式优化: 内置针对 DDP 环境的进程同步机制与梯度检查点修复，支持多卡并行训练。

🧠 增强逻辑: 针对代码任务优化的 Prompt 模板与 LoRA 模块配置（覆盖所有线性层）。

📊 完整评估: 提供从 ROUGE 指标计算到功能性代码测试（Sandbox Execution）的全套评估脚本。

🛠️ 开箱即用: 提供一键启动脚本、环境配置清单及预处理好的数据集加载逻辑。

🛠️ 快速开始

1. 系统要求

OS: Linux (推荐 Ubuntu 20.04+)

Python: 3.10+

CUDA: 11.8 或 12.1+

GPU: 至少 11GB 显存 (RTX 2080Ti / 3090 / 4090)

2. 安装指南

首先克隆仓库并进入目录：

git clone [https://github.com/your-username/Qwen-Code-SFT.git](https://github.com/your-username/Qwen-Code-SFT.git)
cd Qwen-Code-SFT


创建并激活 Conda 环境：

conda create -n qwen-sft python=3.10
conda activate qwen-sft


安装依赖（推荐使用国内镜像源）：

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt


3. 数据准备

本项目使用 JSONL 格式的数据集。请确保数据放置在 data/ 目录下：

data/Opencode.jsonl: 训练数据

data/code_alpaca.jsonl: 验证数据

数据格式示例：

{"prompt": "Write a python function to...", "response": "def func():..."}


💻 开发与训练指南

项目结构

Qwen-Code-SFT/
├── src/
│   ├── train.py           # 核心训练脚本 (QLoRA + Trainer)
│   ├── inference.py       # 模型推理与测试脚本
│   └── utils.py           # 数据处理辅助函数
├── scripts/
│   ├── run_train.sh       # 一键训练 Shell 脚本
│   └── run_eval.sh        # 评估脚本
├── data/                  # 数据集存放目录
├── outputs/               # 训练输出 (Logs, Checkpoints)
├── requirements.txt       # 依赖列表
└── README.md              # 项目文档


启动训练

单机单卡模式：

python src/train.py \
    --model_path "./Qwen1.5-1.8B-Chat" \
    --data_path "data/Opencode.jsonl" \
    --output_dir "outputs/single_gpu"


单机多卡分布式模式 (推荐)：

使用 accelerate 启动多卡训练（自动处理 DDP）：

bash scripts/run_train.sh


run_train.sh 配置示例：

accelerate launch --multi_gpu --num_processes 2 src/train.py \
    --batch_size 2 \
    --grad_accum 8 \
    --max_steps 2500 \
    --learning_rate 2e-4


代码风格

使用 black 进行代码格式化。

函数与类必须包含 Docstrings。

🚢 模型推理与部署

本地推理

训练完成后，使用以下命令加载 LoRA 权重并进行对话测试：

python src/inference.py \
    --base_model "./Qwen1.5-1.8B-Chat" \
    --lora_adapter "outputs/checkpoint-2500" \
    --prompt "写一个Python函数实现快速排序"


显存占用说明

精度模式

显存占用 (Base)

显存占用 (Training)

推荐显卡

FP16 Full

~4GB

>24GB

A100 / A800

4-bit QLoRA

~1.5GB

~9GB

RTX 2080Ti / 3060

🤝 贡献指南

我们非常欢迎社区贡献！请遵循以下流程：

Fork 本仓库。

创建一个新的分支 (git checkout -b feature/AmazingFeature)。

提交您的更改 (git commit -m 'Add some AmazingFeature')。

推送到分支 (git push origin feature/AmazingFeature)。

开启一个 Pull Request。

提交规范

feat: 新功能

fix: 修复 Bug

docs: 文档更新

style: 代码格式调整 (不影响逻辑)

📄 许可证信息

本项目遵循 Apache 2.0 License 开源许可证。

基座模型 Qwen1.5 遵循其原始许可证（Tongyi Qianwen LICENSE）。

您可以自由地使用、修改和分发本项目代码，但需保留版权声明。

📞 联系方式

维护者: Your Name

Email: your.email@example.com

GitHub Issues: 如遇 Bug 或有功能建议，请直接提 Issue。

<div align="center">
Made with ❤️ by the Qwen-Code-SFT Team
</div>
