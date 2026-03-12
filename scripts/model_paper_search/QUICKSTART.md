# 快速开始 - 论文总结模型训练

## 📦 已创建的脚本

以下脚本已创建并可直接使用：

```
scripts/model_paper_search/
├── train_all.sh                   ✅ 一键完整训练
├── prepare_data.sh                ✅ 数据准备和验证
├── run_qwen3_4b_sft.sh           ✅ SFT 监督微调
├── run_qwen3_4b_grpo.sh          ✅ GRPO 强化学习
├── evaluate.py                    ✅ 模型评估
└── README.md                      ✅ 详细文档
```

## 🚀 三步开始训练

### 步骤 1: 检查环境

```bash
# 确认 GPU 配置
nvidia-smi

# 确认依赖已安装
python -c "import verl, vllm, wandb; print('环境就绪')"
```

### 步骤 2: 准备数据

```bash
# 确保原始数据存在
ls data/paper_train.json
ls data/paper_test.json

# 转换数据为 verl 格式
bash scripts/model_paper_search/prepare_data.sh
```

### 步骤 3: 开始训练

**方式 1: 一键训练（推荐）**
```bash
bash scripts/model_paper_search/train_all.sh
```

**方式 2: 分阶段训练**
```bash
# SFT 阶段
bash scripts/model_paper_search/run_qwen3_4b_sft.sh

# GRPO 阶段
bash scripts/model_paper_search/run_qwen3_4b_grpo.sh
```

## 📊 预期训练时间

基于 4×RTX 4090 配置：

| 阶段 | 数据量 | 预计时间 | 输出目录 |
|------|--------|----------|----------|
| SFT | 3600 条 | 2-3 小时 | `outputs/paper_summary_sft/` |
| GRPO | 3600 条 | 4-6 小时 | `outputs/paper_summary_grpo/` |
| **总计** | - | **6-9 小时** | - |

## 🎯 训练完成后

### 评估模型
```bash
python scripts/model_paper_search/evaluate.py \
    --model_path outputs/paper_summary_grpo/qwen3_4b_grpo_*/checkpoint-xxx \
    --test_data data/data_verl/paper_val.parquet \
    --output_file eval_results.json
```

### 查看结果
```bash
cat eval_results.json
```

## ⚙️ 关键配置参数

### SFT 配置
- **GPU**: 4 卡
- **Batch Size**: 32 (4×2×4)
- **学习率**: 1e-5
- **Epochs**: 3
- **序列长度**: 4096+1024

### GRPO 配置
- **GPU**: 4 卡
- **Batch Size**: 128
- **组大小**: 8 (每个问题采样 8 个答案)
- **学习率**: 1.0e-5
- **Epochs**: 1

## 🔧 常见问题快速修复

### OOM 显存不足
```bash
# 编辑 run_qwen3_4b_sft.sh
MICRO_BATCH_SIZE=1  # 从 2 改为 1
GRADIENT_ACCUMULATION=8  # 从 4 改为 8
```

### 训练不稳定
```bash
# 编辑对应脚本
LEARNING_RATE=5e-6  # 降低学习率
```

### 数据格式错误
```bash
python verl/utils/model_paper_search/validate_data.py \
    --train_file data/data_verl/paper_train.parquet
```

## 📚 详细文档

- [脚本使用说明](README.md) - 完整参数说明
- [训练指南](../../docs/model_paper_search/TRAINING_SCRIPTS_GUIDE.md) - 详细使用指南
- [奖励函数](../../docs/model_paper_search/work_summary/paper_summary_reward_implementation.md) - 奖励函数详解

## 📞 需要帮助？

1. 检查 WandB 日志
2. 查看 `outputs/*/train_log.txt`
3. 运行数据验证脚本
4. 检查 GPU 状态：`nvidia-smi`

---

**准备就绪！现在可以开始训练了 🎉**
