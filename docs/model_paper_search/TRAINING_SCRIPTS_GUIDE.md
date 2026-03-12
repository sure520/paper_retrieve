# 论文总结模型训练脚本使用指南

## 📋 概述

本文档介绍如何使用为 verl 框架创建的完整训练脚本来训练 Qwen3-4B-Instruct-2507 论文总结模型。

## 🎯 脚本列表

所有训练脚本位于 `scripts/model_paper_search/` 目录：

```
scripts/model_paper_search/
├── README.md                      # 脚本使用说明
├── train_all.sh                   # 一键完整训练流程
├── prepare_data.sh                # 数据准备和验证
├── run_qwen3_4b_sft.sh           # SFT 监督微调
├── run_qwen3_4b_grpo.sh          # GRPO 强化学习
├── evaluate.py                    # 模型评估
└── validate_data.py               # 数据格式验证
```

## 🚀 快速开始

### 1. 环境检查

确保你的云服务器满足以下要求：

```bash
# GPU 配置
nvidia-smi
# 应该显示 4×RTX 4090

# 检查依赖
python -c "import torch; print(torch.__version__)"
python -c "import verl; print('verl installed')"
python -c "import vllm; print('vllm installed')"

# 检查 WandB
wandb login
```

### 2. 数据准备

确保原始数据文件存在：

```bash
ls -lh data/paper_train.json
ls -lh data/paper_test.json
```

### 3. 一键训练（推荐）

```bash
cd /path/to/paper_retrieve
bash scripts/model_paper_search/train_all.sh
```

这将自动执行：
1. 数据转换和验证
2. SFT 监督微调（3 epochs）
3. GRPO 强化学习（1 epoch）

### 4. 分阶段训练

```bash
# 阶段 1: 数据准备
bash scripts/model_paper_search/prepare_data.sh

# 阶段 2: SFT 监督微调
bash scripts/model_paper_search/run_qwen3_4b_sft.sh

# 阶段 3: GRPO 强化学习
bash scripts/model_paper_search/run_qwen3_4b_grpo.sh
```

## 📊 训练配置说明

### SFT 阶段配置

在 `run_qwen3_4b_sft.sh` 中：

```bash
# GPU 配置
NPROC_PER_NODE=4              # 使用 4 张 GPU

# 批次配置
MICRO_BATCH_SIZE=2            # 每卡 micro batch
GRADIENT_ACCUMULATION=4       # 梯度累积
# 有效 batch size = 4 × 2 × 4 = 32

# 学习率
LEARNING_RATE=1e-5            # SFT 学习率

# 训练轮数
NUM_EPOCHS=3                  # 3 个 epoch

# 序列长度
MAX_PROMPT_LENGTH=4096        # 输入最大长度
MAX_RESPONSE_LENGTH=1024      # 输出最大长度
```

### GRPO 阶段配置

在 `run_qwen3_4b_grpo.sh` 中：

```bash
# GPU 配置
N_GPUS_PER_NODE=4             # 使用 4 张 GPU

# 批次配置
TRAIN_BATCH_SIZE=128          # 总训练 batch size

# GRPO 特定参数
NUM_ANSWERS_PER_QUESTION=8    # 每个问题采样 8 个答案（组大小）

# 序列长度
MAX_PROMPT_LENGTH=4096        # 输入最大长度
MAX_RESPONSE_LENGTH=1024      # 输出最大长度
MAX_TOKEN_LEN_PER_GPU=6144    # 每 GPU 最大 token 数

# 学习率
ACTOR_LEARNING_RATE=1.0e-5    # Actor 学习率

# KL 控制
KL_LOSS_COEF=0.001            # KL 惩罚系数

# 训练控制
TOTAL_EPOCHS=1                # GRPO 通常 1 个 epoch
SAVE_FREQ=20                  # 每 20 步保存一次
TEST_FREQ=5                   # 每 5 步验证一次
```

## 🔧 自定义配置

### 修改模型路径

如果模型不在默认位置：

```bash
# 编辑 run_qwen3_4b_sft.sh 或 run_qwen3_4b_grpo.sh
MODEL_PATH="/your/path/to/Qwen3-4B-Instruct-2507"
```

### 修改数据路径

```bash
# 在脚本中修改
TRAIN_FILES="/your/path/to/paper_train.parquet"
VAL_FILES="/your/path/to/paper_val.parquet"
```

### 显存不足时的调整

```bash
# 在 run_qwen3_4b_sft.sh 中
MICRO_BATCH_SIZE=1            # 减少 micro batch
GRADIENT_ACCUMULATION=8       # 增加梯度累积

# 或启用 LoRA
USE_LORA=true
LORA_RANK=32
LORA_ALPHA=64

# 在 run_qwen3_4b_grpo.sh 中
PARAM_OFFLOAD=true            # 启用参数卸载
NUM_ANSWERS_PER_QUESTION=4    # 减少组大小
```

## 📈 训练监控

### WandB 日志

训练脚本会自动记录到 WandB：

```bash
# 登录 WandB（如果还未登录）
wandb login

# 访问日志面板
# SFT: https://wandb.ai/your-username/paper_summary_sft
# GRPO: https://wandb.ai/your-username/paper_summary_grpo
```

### 关键指标

**SFT 阶段：**
- `train_loss`: 应持续下降
- `eval_loss`: 验证集损失
- `learning_rate`: 学习率变化

**GRPO 阶段：**
- `train/reward`: 平均奖励，应逐渐上升
- `train/kl_divergence`: KL 散度，应 < 0.1
- `train/entropy`: 策略熵，应逐渐降低

### 日志文件位置

```
outputs/paper_summary_sft/qwen3_4b_sft_<timestamp>/
├── train_log.txt          # 训练日志
├── metrics.jsonl          # 指标日志
├── tensorboard_log/       # TensorBoard 日志
└── checkpoint-xxx/        # 模型检查点
```

## ⚠️ 常见问题

### 1. OOM 显存不足

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 方案 1: 减少 batch size
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8

# 方案 2: 启用参数卸载
PARAM_OFFLOAD=true

# 方案 3: 启用 LoRA
USE_LORA=true
LORA_RANK=32

# 方案 4: 减少序列长度
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
```

### 2. 训练不稳定

**症状：**
- loss 震荡剧烈
- reward 不增长

**解决方案：**
```bash
# 降低学习率
LEARNING_RATE=5e-6

# 增加 warmup
WARMUP_RATIO=0.1

# 减少 GRPO 组大小
NUM_ANSWERS_PER_QUESTION=4
```

### 3. 数据格式错误

**症状：**
```
ValueError: Missing required fields
```

**解决方案：**
```bash
# 运行数据验证
python verl/utils/model_paper_search/validate_data.py \
    --train_file data/data_verl/paper_train.parquet \
    --val_file data/data_verl/paper_val.parquet
```

### 4. vLLM 初始化失败

**症状：**
```
ImportError: vllm not installed
```

**解决方案：**
```bash
pip install vllm
# 或
pip install vllm==0.x.x  # 根据 verl 要求安装特定版本
```

## 🎯 训练完成后的步骤

### 1. 评估模型

```bash
python scripts/model_paper_search/evaluate.py \
    --model_path outputs/paper_summary_grpo/qwen3_4b_grpo_*/checkpoint-xxx \
    --test_data data/data_verl/paper_val.parquet \
    --output_file eval_results.json
```

**评估指标：**
- `format_accuracy`: JSON 格式准确率
- `completeness`: 字段完整率
- `quality_avg`: 平均质量分数
- 各字段得分：`summary_score`, `algorithm_score` 等

### 2. 导出模型

如果使用 LoRA，需要合并权重：

```bash
# 创建合并脚本
cat > scripts/model_paper_search/merge_lora_weights.py << 'EOF'
from peft import AutoPeftModelForCausalLM
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    "path/to/lora/checkpoint",
    torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained("path/to/merged_model")
EOF

python scripts/model_paper_search/merge_lora_weights.py
```

### 3. 部署模型

```python
from vllm import LLM

llm = LLM(
    model="path/to/final/model",
    tensor_parallel_size=1,
    max_model_len=8192
)

prompt = """你是一个专业的学术论文分析专家...

论文标题：xxx
摘要：xxx
...

请总结这篇论文的核心内容。"""

outputs = llm.generate([prompt])
print(outputs[0].outputs[0].text)
```

## 📚 相关文档

- [训练指南](TRAINING_GUIDE.md) - 完整训练流程和理论
- [奖励函数实现](work_summary/paper_summary_reward_implementation.md) - 奖励函数详解
- [脚本 README](../../scripts/model_paper_search/README.md) - 脚本参数说明

## 📞 技术支持

如有问题，请检查：
1. WandB 日志：https://wandb.ai/your-username/
2. 本地日志：`outputs/*/train_log.txt`
3. 数据验证结果
4. GPU 状态：`nvidia-smi`

---

**最后更新**: 2026-03-11  
**维护者**: 开发团队
