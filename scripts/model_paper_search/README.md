# 论文总结模型训练脚本

本目录包含使用 verl 框架训练 Qwen3-4B-Instruct-2507 模型进行论文总结任务的完整脚本。

## 📋 目录结构

```
scripts/model_paper_search/
├── README.md                      # 本说明文档
├── train_all.sh                   # 完整训练流程（一键启动）
├── prepare_data.sh                # 数据准备和验证
├── run_qwen3_4b_sft.sh           # SFT 监督微调训练
├── run_qwen3_4b_grpo.sh          # GRPO 强化学习训练
└── evaluate.py                    # 模型评估脚本（待创建）
```

## 🚀 快速开始

### 前置条件

1. **硬件配置**
   - GPU: 4×RTX 4090 (24GB) 或同等配置
   - CPU: 16 核+
   - 内存：64GB+
   - 存储：200GB SSD

2. **软件环境**
   ```bash
   # 已安装 verl 框架及相关依赖
   # PyTorch 2.0+ / CUDA 12.1+
   ```

3. **数据准备**
   ```bash
   # 原始数据文件
   data/paper_train.json
   data/paper_test.json
   ```

### 一键训练（推荐）

```bash
# 执行完整训练流程（数据准备 + SFT + GRPO）
cd /path/to/paper_retrieve
bash scripts/model_paper_search/train_all.sh
```

### 分阶段训练

```bash
# 阶段 1: 数据准备
bash scripts/model_paper_search/prepare_data.sh

# 阶段 2: SFT 监督微调
bash scripts/model_paper_search/run_qwen3_4b_sft.sh

# 阶段 3: GRPO 强化学习
bash scripts/model_paper_search/run_qwen3_4b_grpo.sh
```

## 📊 训练参数配置

### SFT 阶段参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NPROC_PER_NODE` | 4 | GPU 数量 |
| `MICRO_BATCH_SIZE` | 2 | 每卡 micro batch size |
| `GRADIENT_ACCUMULATION` | 4 | 梯度累积步数 |
| `LEARNING_RATE` | 1e-5 | 学习率 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `MAX_PROMPT_LENGTH` | 4096 | 输入最大长度 |
| `MAX_RESPONSE_LENGTH` | 1024 | 输出最大长度 |

### GRPO 阶段参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_GPUS_PER_NODE` | 4 | GPU 数量 |
| `TRAIN_BATCH_SIZE` | 128 | 总训练 batch size |
| `NUM_ANSWERS_PER_QUESTION` | 8 | GRPO 组大小 |
| `MAX_PROMPT_LENGTH` | 4096 | 输入最大长度 |
| `MAX_RESPONSE_LENGTH` | 1024 | 输出最大长度 |
| `ACTOR_LEARNING_RATE` | 1.0e-5 | Actor 学习率 |
| `TOTAL_EPOCHS` | 1 | 训练轮数 |
| `SAVE_FREQ` | 20 | 保存 checkpoint 频率 |

## 🔧 自定义配置

### 修改模型路径

如果模型不在默认位置，修改脚本中的 `MODEL_PATH`：

```bash
# 在 run_qwen3_4b_sft.sh 或 run_qwen3_4b_grpo.sh 中
MODEL_PATH="/your/path/to/Qwen3-4B-Instruct-2507"
```

### 修改数据路径

```bash
# 在脚本中修改
TRAIN_FILES="/your/path/to/paper_train.parquet"
VAL_FILES="/your/path/to/paper_val.parquet"
```

### 启用 LoRA（显存不足时）

```bash
# 在 run_qwen3_4b_sft.sh 中
USE_LORA=true
LORA_RANK=32
LORA_ALPHA=64
```

### 调整批次大小

如果显存不足，可以减少 batch size：

```bash
# 在 run_qwen3_4b_sft.sh 中
MICRO_BATCH_SIZE=1  # 从 2 减少到 1
GRADIENT_ACCUMULATION=8  # 从 4 增加到 8，保持有效 batch size 不变
```

## 📈 训练监控

### WandB 日志

训练脚本会自动记录到 WandB：

```bash
# 查看训练日志
wandb login  # 如果还未登录
# 访问 https://wandb.ai/your-username/paper_summary_sft 或 paper_summary_grpo
```

### 关键指标

**SFT 阶段：**
- `train_loss`: 应持续下降
- `eval_loss`: 验证集损失
- `learning_rate`: 学习率变化

**GRPO 阶段：**
- `train/reward`: 平均奖励，应逐渐上升
- `train/kl_divergence`: KL 散度，应保持在 0.1 以下
- `train/entropy`: 策略熵，应逐渐降低

## ⚠️ 常见问题

### 1. OOM 显存不足

**解决方案：**
```bash
# 减少 micro batch size
MICRO_BATCH_SIZE=1

# 启用参数卸载
PARAM_OFFLOAD=true

# 启用 LoRA
USE_LORA=true
```

### 2. 训练不稳定

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

**解决方案：**
```bash
# 运行数据验证
python verl/utils/model_paper_search/validate_data.py \
    --train_file data/data_verl/paper_train.parquet \
    --val_file data/data_verl/paper_val.parquet
```

## 📝 训练日志

训练日志保存在：
```
outputs/paper_summary_sft/qwen3_4b_sft_<timestamp>/
├── train_log.txt          # 训练日志
├── metrics.jsonl          # 指标日志
├── tensorboard_log/       # TensorBoard 日志
└── checkpoint-xxx/        # 模型检查点

outputs/paper_summary_grpo/qwen3_4b_grpo_<timestamp>/
└── ... (类似结构)
```

## 🎯 训练完成后的步骤

1. **评估模型性能**
   ```bash
   python scripts/model_paper_search/evaluate.py \
       --model_path outputs/paper_summary_grpo/qwen3_4b_grpo_*/checkpoint-xxx \
       --test_data data/data_verl/paper_val.parquet \
       --output_file eval_results.json
   ```

2. **导出模型**
   ```bash
   # 如果使用 LoRA，需要合并权重
   python scripts/model_paper_search/merge_lora_weights.py \
       --model_path Qwen/Qwen3-4B-Instruct-2507 \
       --lora_path outputs/paper_summary_grpo/.../checkpoint-xxx \
       --output_path models/paper_summary_qwen3_4b
   ```

3. **部署模型**
   ```python
   from vllm import LLM
   
   llm = LLM(
       model="models/paper_summary_qwen3_4b",
       tensor_parallel_size=1,
       max_model_len=8192
   )
   ```

## 📚 相关文档

- [训练指南](../../docs/model_paper_search/TRAINING_GUIDE.md)
- [奖励函数实现](../../docs/model_paper_search/work_summary/paper_summary_reward_implementation.md)
- [verl 框架文档](../../verl/README.md)

## 📞 技术支持

如有问题，请检查：
1. WandB 日志
2. 本地日志文件
3. 数据验证结果

---

**最后更新**: 2026-03-11  
**维护者**: 开发团队
