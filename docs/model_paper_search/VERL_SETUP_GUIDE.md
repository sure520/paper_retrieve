# VERL + GRPO 训练方案指南

## ⚠️ 重要说明

本方案使用 **VERL (Volcano Engine Reinforcement Learning)** 框架进行GRPO训练，而非直接使用Transformers。

VERL是字节跳动Seed团队开源的高性能RL训练框架，专为LLM设计。

---

## 📦 环境安装

### 1. 安装VERL框架

```bash
# 克隆VERL仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 安装（推荐从源码安装）
pip install -e . --no-build-isolation

# 或使用pip安装特定版本
pip install verl==0.5.0
```

### 2. 安装依赖

```bash
# 基础依赖
pip install torch>=2.1.0 transformers>=4.40.0 accelerate

# 数据处理
pip install datasets pandas pyarrow

# 推理加速（可选但推荐）
pip install vllm>=0.4.0

# WandB
pip install wandb
wandb login
```

### 3. 验证安装

```bash
python -c "import verl; print(verl.__version__)"
python -c "from verl.trainer.main_ppo import main"
```

---

## 🚀 快速开始

### 步骤1: 准备数据

```bash
# 将您的4000条数据转换为VERL格式
python prepare_verl_data.py \
    --input train/data/paper_train.json \
    --output_dir train/data \
    --tokenizer Qwen/Qwen3-4B-Instruct \
    --verify
```

**输出文件**:
- `train/data/paper_train.parquet` (3600条)
- `train/data/paper_val.parquet` (400条)

### 步骤2: 下载模型

```bash
# 使用modelscope下载（国内推荐）
pip install modelscope
modelscope download Qwen/Qwen3-4B-Instruct --local-dir ./models/Qwen3-4B-Instruct

# 或使用huggingface
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir ./models/Qwen3-4B-Instruct
```

### 步骤3: 启动训练

```bash
# 方式1: 使用脚本启动
chmod +x run_verl_grpo.sh
./run_verl_grpo.sh

# 方式2: 直接启动（更多控制）
python3 -m verl.trainer.main_ppo \
    algorithm=grpo \
    data.train_files=train/data/paper_train.parquet \
    data.val_files=train/data/paper_val.parquet \
    data.prompt_key=prompt \
    data.reward_key=ground_truth \
    model.path=Qwen/Qwen3-4B-Instruct \
    training.n_gpus=4
```

---

## ⚙️ 配置文件说明

### 主配置文件: `verl_grpo_config.yaml`

```yaml
# 关键参数说明
algorithm:
  name: "grpo"              # 使用GRPO算法
  num_generations: 4        # 每组采样4个回答
  kl_coeff: 0.04           # KL惩罚系数
  clip_ratio: 0.2          # PPO裁剪参数
  learning_rate: 1e-6      # 学习率

training:
  n_gpus: 4                # 4卡4090
  data_parallel_size: 4     # 数据并行度
```

### 奖励函数: `verl_paper_reward.py`

VERL框架会自动加载此奖励函数，评估5个维度:
- `summary`: 论文摘要质量 (30%)
- `algorithm`: 算法描述质量 (25%)
- `comparison`: 对比结果质量 (25%)
- `keyword_problem`: 问题关键词 (10%)
- `keyword_algorithm`: 算法关键词 (10%)

---

## 📊 WandB监控

### 自动记录指标

VERL框架会自动将以下指标记录到WandB:

| 指标 | 说明 |
|------|------|
| `train/loss` | 训练损失 |
| `train/reward` | 平均奖励 |
| `train/kl_divergence` | KL散度 |
| `train/entropy` | 策略熵 |
| `eval/mean_reward` | 验证集平均奖励 |
| `eval/format_accuracy` | JSON格式准确率 |

### 查看结果

训练开始后，访问:
```
https://wandb.ai/<your-username>/paper-summary-verl-grpo
```

---

## 🔧 高级配置

### 使用vLLM加速生成

```bash
# 在配置中启用vLLM
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
```

### 调整GRPO参数

```bash
# 增加组大小（更稳定但显存占用高）
algorithm.num_generations=8

# 降低KL惩罚（策略更自由）
algorithm.kl_coeff=0.01

# 调整学习率
algorithm.learning_rate=5e-7
```

### 使用LoRA训练

```bash
# 如果显存不足，启用LoRA
model.lora_enabled=true \
model.lora_rank=64 \
model.lora_alpha=128
```

---

## 🐛 常见问题

### Q1: VERL安装失败

```bash
# 确保CUDA版本匹配
nvcc --version

# 从源码安装
pip install -e . --no-build-isolation
```

### Q2: 奖励函数未生效

```bash
# 检查reward_fn配置路径
+reward_fn@reward_model=paper_summary_reward

# 确保verl_paper_reward.py在PYTHONPATH中
export PYTHONPATH="${PYTHONPATH}:."
```

### Q3: 显存OOM

```bash
# 减小批次大小
data.train_batch_size=64

# 启用梯度检查点
model.enable_gradient_checkpointing=true

# 使用LoRA
model.lora_enabled=true
```

### Q4: WandB未记录

```bash
# 检查环境变量
export WANDB_PROJECT="paper-summary-verl-grpo"
export WANDB_API_KEY="your-api-key"

# 在配置中启用
wandb.enabled=true
```

---

## 📈 预期结果

### 训练时间（4卡4090）

| 配置 | 时间 |
|------|------|
| 2000步, batch=128 | 4-6小时 |
| 4000步, batch=128 | 8-12小时 |

### 预期指标

| 指标 | 目标值 |
|------|--------|
| JSON格式准确率 | > 95% |
| 平均奖励 | > 0.75 |
| KL散度 | < 0.1 |

---

## 🔗 相关文件

- `verl_grpo_config.yaml` - 主配置文件
- `verl_paper_reward.py` - 奖励函数实现
- `run_verl_grpo.sh` - 启动脚本
- `prepare_verl_data.py` - 数据预处理

---

## 📚 参考资料

- [VERL GitHub](https://github.com/volcengine/verl)
- [GRPO论文](https://arxiv.org/pdf/2402.03300)
- [Qwen3文档](https://huggingface.co/Qwen/Qwen3-4B-Instruct)
