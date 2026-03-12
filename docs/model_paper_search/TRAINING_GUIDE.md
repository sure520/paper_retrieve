# Qwen3-4B-Instruct-2507 论文总结模型训练指南

## 📋 环境准备

### 1. 云服务器配置（推荐）

```bash
# 推荐配置
- GPU: 4 × RTX 4090 (24GB)
- CPU: 16核+
- 内存: 64GB+
- 存储: 200GB SSD
- 镜像: PyTorch 2.0+ / CUDA 12.1+
```

### 2. 依赖安装

```bash
# 克隆项目
git clone <your-repo>
cd paper_retrieve

# 安装依赖
pip install -r requirements.txt
pip install wandb

# 登录WandB
wandb login
```

### 3. 模型下载

```bash
# 使用git下载模型 (请确保 lfs 已经被正确安装)
git clone https://www.modelscope.cn/Qwen/Qwen3-4B-Instruct-2507.git
```

---

## 🚀 训练流程

### 阶段一：SFT监督微调（约2-3小时）

```bash

```

**SFT阶段目标**：
- 让模型学习JSON格式输出
- 掌握论文总结的5个字段结构
- 建立基础的学术语言理解能力

**监控指标**：
- `eval_loss`: 应持续下降，最终<1.0
- `train_loss`: 平滑下降，无剧烈震荡

---

### 阶段二：GRPO强化学习（约4-6小时）

#### 2.1 启动采样进程

```bash

```

**GRPO阶段目标**：
- 优化奖励函数各维度
- 提高JSON格式准确率到95%+
- 提升摘要质量和关键词准确性

---

## 📊 WandB监控配置

### 关键指标说明

| 指标 | 说明 | 目标值 |
|------|------|--------|
| `train/loss` | 训练损失 | < 0.5 |
| `train/reward` | 平均奖励 | > 0.7 |
| `train/kl_divergence` | KL散度 | < 0.1 |
| `train/entropy` | 策略熵 | 逐渐降低 |
| `eval/format_accuracy` | JSON格式准确率 | > 95% |
| `eval/field_completeness` | 字段完整率 | > 90% |

### WandB面板设置

```python
# 在wandb界面创建自定义面板
# 1. 训练损失曲线: train/loss
# 2. 奖励曲线: train/reward
# 3. KL散度监控: train/kl_divergence
# 4. 奖励分解: train/reward_*
# 5. 评估指标: eval/*
```

---

## ⚙️ 参数调优策略

### 学习率调优

```yaml
# 如果训练不稳定，降低学习率
learning_rate: 5e-6  # 默认
learning_rate: 2e-6  # 如果震荡
learning_rate: 1e-6  # 如果仍不稳定

# LoRA模式下需要更大学习率
lora_learning_rate: 1e-4  # LoRA专用
```

### GRPO组大小调优

```yaml
# 组大小影响训练稳定性
num_answers_per_question: 4   # 默认，平衡稳定与效率
num_answers_per_question: 8   # 更稳定但显存占用高
num_answers_per_question: 2   # 显存友好但可能不稳定
```

### 奖励权重调优

```yaml
# 根据验证结果调整权重
reward:
  weights:
    summary: 0.35          # 如果摘要质量差，提高权重
    algorithm: 0.25
    comparison: 0.20       # 如果对比结果不重要，降低权重
    keyword_problem: 0.10
    keyword_algorithm: 0.10
```

---

## 🔧 常见问题解决

### 1. OOM显存不足

```bash
# 解决方案1: 启用LoRA
lora:
  enabled: true
  rank: 32

# 解决方案2: 减小批次大小
train_batch_size: 4  # 减小到4
gradient_accumulation_steps: 4  # 增加梯度累积

# 解决方案3: 启用梯度检查点
# 已在代码中默认启用
```

### 2. 训练不稳定

```yaml
# 降低学习率
learning_rate: 2e-6

# 增加warmup
warmup_steps: 200

# 使用GSPO替代GRPO
training:
  use_gspo: true  # GSPO更稳定
```

### 3. JSON格式错误率高

```bash
# 增加SFT阶段训练轮数
num_train_epochs: 5  # 默认3，增加到5

# 或在GRPO中增加格式奖励权重
# 修改reward.py中的格式检查逻辑
```

### 4. 奖励不增长

```bash
# 检查奖励函数
python tests/test_reward.py

# 检查数据格式
python utils/data_convert_to_verl_rl.py --validate

# 降低KL惩罚系数
kl_beta: 0.01  # 默认0.04，降低让策略更自由
```

---

## 📈 评估模型

```bash
# 运行评估
python evaluate.py \
    --model_path ./output/run_xxx/sft_final \
    --test_data train/data/paper_test.json \
    --output_file eval_results.json

# 评估指标包括:
# - ROUGE-1/2/L
# - JSON格式准确率
# - 字段完整率
# - 人工评分（可选）
```

---

## 💾 模型导出与部署

```python
# 合并LoRA权重（如果使用LoRA）
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./checkpoints/qwen3_4b_grpo",
    torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained("./models/paper_summary_qwen3_4b")

# 使用vLLM部署
from vllm import LLM

llm = LLM(
    model="./models/paper_summary_qwen3_4b",
    tensor_parallel_size=1,
    max_model_len=8192
)
```

---

## 📅 训练时间表参考

| 阶段 | 数据量 | 步数 | 预计时间 | 监控频率 |
|------|--------|------|----------|----------|
| SFT | 3600条 | 3 epoch | 2-3小时 | 每100步 |
| GRPO | 3600条 | 2000步 | 4-6小时 | 每10步 |
| 评估 | 400条 | - | 10分钟 | - |
| **总计** | - | - | **6-9小时** | - |

---

## 🔗 相关文件

- [config_qwen3_4b.yaml](train/config_qwen3_4b.yaml) - 主配置文件
- [ds_config_zero3.json](train/ds_config_zero3.json) - DeepSpeed配置
- [wandb_logger.py](train/wandb_logger.py) - WandB日志模块
- [training_pipeline.py](train/training_pipeline.py) - 完整训练流程
- [reward.py][def] - 奖励函数实现

---

## 📞 技术支持

如有问题，请检查：
1. WandB日志: https://wandb.ai/<your-username>/paper-summary-grpo
2. 本地日志: `logs/train_*.log`
3. 可视化: `visualizations/` 目录


[def]: train/reward.py