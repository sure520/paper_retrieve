# 论文总结模型训练项目

基于 verl 框架和 Qwen3-4B-Instruct-2507 模型的学术论文智能总结系统。

## 📋 项目概述

本项目使用强化学习（GRPO 算法）训练一个能够自动提取论文核心思想、算法细节和关键词的 AI 模型，支持结构化的 JSON 格式输出。

### 核心能力

- 📄 **论文理解**：深度理解学术论文的研究背景、问题和方法
- 📝 **摘要生成**：生成结构化的论文总结，包含 5 个核心字段
- 🔍 **关键词提取**：自动提取研究问题和算法的中英文关键词
- ⚖️ **对比分析**：识别并对比相关算法的优劣
- 🎯 **格式规范**：输出符合 JSON Schema 的标准化结果

## 🚀 快速开始

### 环境要求

- **GPU**: 4×RTX 4090 (24GB) 或同等配置
- **CPU**: 16 核+
- **内存**: 64GB+
- **存储**: 200GB SSD
- **软件**: PyTorch 2.0+, CUDA 12.1+, verl, vllm

### 快速启动

```bash
# 1. 克隆项目
cd d:\python\code\paper_retrieve

# 2. 准备数据
bash scripts/model_paper_search/prepare_data.sh

# 3. 一键训练（SFT + GRPO）
bash scripts/model_paper_search/train_all.sh
```

### 分阶段训练

```bash
# 阶段 1: SFT 监督微调（2-3 小时）
bash scripts/model_paper_search/run_qwen3_4b_sft.sh

# 阶段 2: GRPO 强化学习（4-6 小时）
bash scripts/model_paper_search/run_qwen3_4b_grpo.sh

# 阶段 3: 模型评估
python scripts/model_paper_search/evaluate.py \
    --model_path outputs/paper_summary_grpo/qwen3_4b_grpo_*/checkpoint-xxx \
    --test_data data/data_verl/paper_val.parquet \
    --output_file eval_results.json
```

## 📁 项目结构

```
paper_retrieve/
├── scripts/model_paper_search/      # 训练脚本
│   ├── train_all.sh                 # 一键完整训练
│   ├── prepare_data.sh              # 数据准备
│   ├── run_qwen3_4b_sft.sh         # SFT 训练
│   ├── run_qwen3_4b_grpo.sh        # GRPO 训练
│   ├── evaluate.py                  # 模型评估
│   └── README.md                    # 脚本说明
│
├── verl/utils/
│   ├── model_paper_search/
│   │   ├── data_convert_to_verl_rl.py  # 数据转换
│   │   └── validate_data.py            # 数据验证
│   └── reward_score/
│       ├── paper_summary.py       # 奖励函数实现
│       └── __init__.py
│
├── data/
│   ├── paper_train.json           # 原始训练数据
│   ├── paper_test.json            # 原始测试数据
│   └── data_verl/                 # 转换后的 parquet 数据
│
├── outputs/                       # 训练输出
│   ├── paper_summary_sft/         # SFT 检查点
│   └── paper_summary_grpo/        # GRPO 检查点
│
└── docs/model_paper_search/       # 文档
    ├── TRAINING_GUIDE.md          # 训练指南
    ├── TRAINING_SCRIPTS_GUIDE.md  # 脚本使用指南
    └── work_summary/
        ├── paper_summary_reward_implementation.md  # 奖励函数文档
        └── session_summary_20260311.md             # 会话总结
```

## 📊 数据格式

### 输出格式（JSON Schema）

```json
{
  "summary": "论文概览（场景、问题、方法、效果）",
  "algorithm": "算法详细介绍（核心思想、创新点、实现步骤）",
  "compare_result": "核心对比算法及对比结果",
  "keyword_problem": "研究场景和业务问题的关键词（中英文 + 缩写）",
  "keyword_algorithm": "算法关键词（中英文 + 缩写）"
}
```

### 示例输出

```json
{
  "summary": "本文针对强化学习中的样本效率问题，提出了一种新的基于模型的 RL 算法...",
  "algorithm": "核心思想是利用学习到的环境模型生成合成轨迹，通过策略蒸馏...",
  "compare_result": "相比 SAC、TD3 等主流算法，在 MuJoCo 基准上平均提升 35%...",
  "keyword_problem": "强化学习，Reinforcement Learning, RL; 样本效率，Sample Efficiency",
  "keyword_algorithm": "基于模型的 RL, Model-Based RL; 策略蒸馏，Policy Distillation"
}
```

## 🎯 训练配置

### SFT 阶段

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU | 4 卡 | RTX 4090 |
| Batch Size | 32 | 有效 batch size |
| 学习率 | 1e-5 | SFT 学习率 |
| Epochs | 3 | 训练轮数 |
| 序列长度 | 4096+1024 | 输入 + 输出 |

### GRPO 阶段

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU | 4 卡 | RTX 4090 |
| Batch Size | 128 | 总训练 batch |
| 组大小 | 8 | 每个问题采样数 |
| 学习率 | 1.0e-5 | Actor 学习率 |
| Epochs | 1 | 训练轮数 |

## 📈 训练监控

### WandB 日志

```bash
# 登录 WandB
wandb login

# 访问日志面板
# SFT: https://wandb.ai/your-username/paper_summary_sft
# GRPO: https://wandb.ai/your-username/paper_summary_grpo
```

### 关键指标

- **SFT**: `train_loss`, `eval_loss`, `learning_rate`
- **GRPO**: `train/reward`, `train/kl_divergence`, `train/entropy`
- **评估**: `format_accuracy`, `completeness`, `quality_avg`

## 🔧 自定义配置

### 修改模型路径

```bash
# 编辑脚本中的 MODEL_PATH
MODEL_PATH="/your/path/to/Qwen3-4B-Instruct-2507"
```

### 显存不足时的调整

```bash
# 减少 batch size
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8

# 启用 LoRA
USE_LORA=true
LORA_RANK=32
```

## 📚 文档导航

- **[快速开始](scripts/model_paper_search/QUICKSTART.md)** - 5 分钟上手
- **[脚本说明](scripts/model_paper_search/README.md)** - 详细参数说明
- **[训练指南](docs/model_paper_search/TRAINING_SCRIPTS_GUIDE.md)** - 完整使用指南
- **[奖励函数](docs/model_paper_search/work_summary/paper_summary_reward_implementation.md)** - 奖励函数详解
- **[会话总结](docs/model_paper_search/work_summary/session_summary_20260311.md)** - 开发记录

## ⚠️ 常见问题

### OOM 显存不足

```bash
# 方案 1: 减少 batch size
MICRO_BATCH_SIZE=1

# 方案 2: 启用参数卸载
PARAM_OFFLOAD=true

# 方案 3: 启用 LoRA
USE_LORA=true
```

### 训练不稳定

```bash
# 降低学习率
LEARNING_RATE=5e-6

# 增加 warmup
WARMUP_RATIO=0.1
```

### 数据格式错误

```bash
# 运行数据验证
python verl/utils/model_paper_search/validate_data.py \
    --train_file data/data_verl/paper_train.parquet
```

## 📊 预期训练时间

基于 4×RTX 4090 配置：

| 阶段 | 数据量 | 预计时间 |
|------|--------|----------|
| SFT | 3600 条 | 2-3 小时 |
| GRPO | 3600 条 | 4-6 小时 |
| **总计** | - | **6-9 小时** |

## 🎓 技术栈

- **框架**: verl, PyTorch
- **模型**: Qwen3-4B-Instruct-2507
- **算法**: SFT, GRPO (Group Relative Policy Optimization)
- **推理**: vLLM
- **监控**: WandB, TensorBoard

## 📝 相关项目

- [verl 框架](verl/README.md) - 强化学习训练框架
- [Qwen3 模型](https://www.modelscope.cn/Qwen/Qwen3-4B-Instruct-2507) - 基础语言模型

## 📞 技术支持

如有问题，请检查：
1. [WandB 日志](https://wandb.ai/your-username/)
2. [本地日志](outputs/*/train_log.txt)
3. [数据验证结果](verl/utils/model_paper_search/validate_data.py)
4. [常见问题](#常见问题)

## 📄 许可证

本项目采用 Apache 2.0 许可证。

---

**最后更新**: 2026-03-11  
**维护者**: 开发团队
