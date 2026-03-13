# AI Agent 开发指南

本文档面向 AI 助手和开发者，提供基于 verl 框架训练论文总结模型的完整技术细节和最佳实践。

## 📋 目录

- [项目概述](#项目概述)
- [架构设计](#架构设计)
- [数据流程](#数据流程)
- [训练流程](#训练流程)
- [奖励函数](#奖励函数)
- [配置参数](#配置参数)
- [故障排查](#故障排查)
- [性能优化](#性能优化)

## 项目概述

### 任务定义

训练一个 AI 模型，输入学术论文全文，输出结构化的 JSON 格式总结，包含 5 个核心字段：

1. **summary**: 论文概览（场景、问题、方法、效果）
2. **algorithm**: 算法详细介绍（核心思想、创新点、实现步骤）
3. **compare_result**: 核心对比算法及对比结果
4. **keyword_problem**: 研究场景和业务问题的关键词（中英文 + 缩写）
5. **keyword_algorithm**: 算法关键词（中英文 + 缩写）

### 技术选型

| 组件 | 技术 | 理由 |
|------|------|------|
| 基础模型 | Qwen3-4B-Instruct-2507 | 强大的中文理解和生成能力 |
| 训练框架 | verl | 支持 GRPO 等先进 RL 算法 |
| RL 算法 | GRPO | 组相对策略优化，稳定性好 |
| 推理引擎 | vLLM | 高性能采样和推理 |
| 监控 | WandB | 实时训练指标可视化 |

## 架构设计

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练系统架构                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   数据准备   │ -> │  SFT 训练    │ -> │ GRPO 训练 │  │
│  │  (Parquet)   │    │  (3 epochs)  │    │ (1 epoch)│  │
│  └──────────────┘    └──────────────┘    └──────────┘  │
│         |                   |                   |        │
│         v                   v                   v        │
│  ┌──────────────┐    ┌──────────────┐    └──────────┘  │
│  │  数据验证    │    │  Checkpoint  │         |        │
│  │  (validate)  │    │  (保存)      │         v        │
│  └──────────────┘    └──────────────┘    ┌──────────┐  │
│                                          │  评估    │  │
│                                          │ (evaluate)│  │
│                                          └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
原始 JSON (paper_train.json)
    ↓
data_convert_to_verl_rl.py (转换)
    ↓
Parquet 格式 (paper_train.parquet)
    ↓
validate_data.py (验证)
    ↓
verl 训练系统
    ↓
    ├─> SFT 训练 → checkpoint-sft-xxx
    └─> GRPO 训练 → checkpoint-grpo-xxx
```

### verl 数据格式

```python
{
    "data_source": "paper_summary",  # 奖励函数路由标识
    "prompt": [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "用户输入（包含论文全文）"}
    ],
    "ability": "academic_summarization",
    "reward_model": {
        "style": "rule",
        "ground_truth": "<JSON 字符串>",  # 标准答案
        "ground_truth_dict": {...}        # 解析后的字典
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "original_instruction": "...",
        "paper_title": "..."
    }
}
```

## 数据流程

### 1. 数据准备

```bash
bash scripts/model_paper_search/prepare_data.sh
```

**步骤：**
1. 读取原始 JSON 文件
2. 转换为 verl 格式
3. 保存为 Parquet 文件
4. 验证数据格式

**验证项：**
- 必需字段完整性
- prompt 格式正确性
- reward_model 结构
- extra_info 字段

### 2. 数据增强策略

```python
# 在 data_convert_to_verl_rl.py 中
def build_verl_item(item, idx, task_type):
    # 构建 prompt
    prompt_content = instruction
    
    # 构建 ground_truth
    ground_truth = {
        "summary": output_json.get("summary", ""),
        "algorithm": output_json.get("algorithm", ""),
        "compare_result": output_json.get("compare_result", ""),
        "keyword_problem": output_json.get("keyword_problem", ""),
        "keyword_algorithm": output_json.get("keyword_algorithm", "")
    }
    
    # 构建 verl 格式
    verl_item = {
        "data_source": task_type,
        "prompt": [...],
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth),
            "ground_truth_dict": ground_truth
        },
        "extra_info": {...}
    }
    
    return verl_item
```

## 训练流程

### SFT 阶段（监督微调）

**目标：** 让模型学习 JSON 格式输出和基础学术语言理解

**配置：**
```bash
NPROC_PER_NODE=4              # 4 卡训练
MICRO_BATCH_SIZE=2            # 每卡 micro batch
GRADIENT_ACCUMULATION=4       # 梯度累积
# 有效 batch size = 4 × 2 × 4 = 32

LEARNING_RATE=1e-5            # 学习率
NUM_EPOCHS=3                  # 3 个 epoch
MAX_PROMPT_LENGTH=4096        # 输入长度
MAX_RESPONSE_LENGTH=1024      # 输出长度
```

**监控指标：**
- `train_loss`: 应持续下降
- `eval_loss`: 验证集损失
- `learning_rate`: 学习率调度

**预期结果：**
- JSON 格式准确率 > 90%
- 字段完整率 > 85%
- train_loss < 1.0

### GRPO 阶段（强化学习）

**目标：** 优化奖励函数各维度，提升生成质量

**GRPO 算法原理：**
```
对于每个问题，采样 N 个答案（组大小）
计算每个答案的奖励
组内归一化奖励：(r_i - mean) / std
使用归一化奖励作为优势估计
```

**配置：**
```bash
TRAIN_BATCH_SIZE=128          # 总 batch size
NUM_ANSWERS_PER_QUESTION=8    # 组大小
MAX_TOKEN_LEN_PER_GPU=6144    # 每 GPU token 数

ACTOR_LEARNING_RATE=1.0e-5    # Actor 学习率
KL_LOSS_COEF=0.001            # KL 惩罚系数
TOTAL_EPOCHS=1                # 1 个 epoch
```

**监控指标：**
- `train/reward`: 平均奖励，应逐渐上升
- `train/kl_divergence`: KL 散度，应 < 0.1
- `train/entropy`: 策略熵，应逐渐降低

**预期结果：**
- 平均奖励 > 0.7
- JSON 格式准确率 > 95%
- 字段完整率 > 90%

## 奖励函数

### 奖励函数架构

```python
def compute_score(
    data_source: str,      # "paper_summary"
    solution_str: str,     # 模型生成的 JSON 字符串
    ground_truth: str,     # 标准答案 JSON 字符串
    extra_info: dict       # 额外信息
) -> float:
    """
    计算综合奖励
    返回范围：[0.0, 1.0]
    """
    # 1. 格式合规性检查（硬约束）
    format_reward = check_format_compliance(generated_json)
    
    # 2. 各维度质量奖励
    r_summary = reward_summary(paper, generated, reference)
    r_algorithm = reward_algorithm(paper, generated, reference)
    r_compare = reward_comparison(paper, generated, reference)
    r_kw_problem = reward_keywords_problem(paper, generated, reference)
    r_kw_algorithm = reward_keywords_algorithm(paper, generated, reference)
    
    # 3. 加权融合
    weights = [0.25, 0.30, 0.20, 0.125, 0.125]
    total_reward = (
        weights[0] * r_summary +
        weights[1] * r_algorithm +
        weights[2] * r_compare +
        weights[3] * r_kw_problem +
        weights[4] * r_kw_algorithm
    )
    
    # 4. 应用格式约束
    final_reward = total_reward * format_reward
    
    return final_reward
```

### 各维度奖励详解

#### 1. Summary 奖励（权重 0.25）

```python
def reward_summary(paper, generated, reference):
    # 语义相似度（LLM 评判）
    semantic_sim = compute_semantic_similarity(generated, reference)
    
    # 关键信息覆盖率
    coverage_score = check_summary_coverage(generated)
    
    # 原文一致性检查
    consistency_score = check_consistency_with_paper(paper, generated)
    
    return 0.5 * semantic_sim + 0.3 * coverage_score + 0.2 * consistency_score
```

#### 2. Algorithm 奖励（权重 0.30）

```python
def reward_algorithm(paper, generated, reference):
    semantic_sim = compute_semantic_similarity(generated, reference)
    structure_score = check_algorithm_structure(generated)
    term_accuracy = check_technical_terms(paper, generated)
    
    return 0.5 * semantic_sim + 0.3 * structure_score + 0.2 * term_accuracy
```

#### 3. Comparison 奖励（权重 0.20）

```python
def reward_comparison(paper, generated, reference):
    semantic_sim = compute_semantic_similarity(generated, reference)
    mention_rate = check_comparison_algorithms(generated, reference)
    dimension_score = check_comparison_dimensions(generated)
    
    return 0.6 * semantic_sim + 0.25 * mention_rate + 0.15 * dimension_score
```

#### 4. Problem Keywords 奖励（权重 0.125）

```python
def reward_keywords_problem(paper, generated, reference):
    overlap_score = compute_keyword_overlap(generated, reference)
    relevance_score = compute_keyword_relevance(paper, generated)
    format_score = check_keyword_format(generated)
    
    return 0.5 * overlap_score + 0.3 * relevance_score + 0.2 * format_score
```

#### 5. Algorithm Keywords 奖励（权重 0.125）

```python
def reward_keywords_algorithm(paper, generated, reference):
    overlap_score = compute_keyword_overlap(generated, reference)
    relevance_score = compute_keyword_relevance(paper, generated)
    format_score = check_keyword_format(generated)
    
    return 0.5 * overlap_score + 0.3 * relevance_score + 0.2 * format_score
```

### 核心工具函数

#### 语义相似度计算

```python
def compute_semantic_similarity(text1, text2):
    """
    使用 LLM 评判语义相似度
    失败时使用词重叠兜底
    """
    try:
        # 调用 qwen-plus API
        score = compute_similarity_with_llm(text1, text2)
        return score
    except:
        # 兜底：Jaccard 相似度
        return compute_semantic_similarity_fallback(text1, text2)
```

#### 格式合规性检查

```python
def check_format_compliance(generated_json):
    """
    硬约束：格式错误返回 0.0
    """
    required_fields = ["summary", "algorithm", "compare_result", 
                       "keyword_problem", "keyword_algorithm"]
    
    # 检查字段完整性
    for field in required_fields:
        if field not in generated_json:
            return 0.0
        if not isinstance(generated_json[field], str):
            return 0.0
        if not generated_json[field].strip():
            return 0.0
    
    # 检查关键词格式
    for kw_field in ["keyword_problem", "keyword_algorithm"]:
        if not check_keyword_format(generated_json[kw_field]):
            return 0.0
    
    return 1.0
```

## 配置参数

### 完整参数列表

```yaml
# SFT 配置
sft:
  nproc_per_node: 4
  micro_batch_size: 2
  gradient_accumulation: 4
  learning_rate: 1e-5
  num_epochs: 3
  max_prompt_length: 4096
  max_response_length: 1024
  warmup_ratio: 0.05
  lr_scheduler: cosine

# GRPO 配置
grpo:
  n_gpus_per_node: 4
  train_batch_size: 128
  num_answers_per_question: 8
  actor_learning_rate: 1.0e-5
  critic_learning_rate: 1.0e-5
  kl_loss_coef: 0.001
  kl_loss_type: low_var_kl
  total_epochs: 1
  save_freq: 20
  test_freq: 5

# 硬件配置
hardware:
  gpu_type: RTX 4090
  gpu_count: 4
  gpu_memory: 24GB
  cpu_cores: 16+
  memory: 64GB+
  storage: 200GB SSD
```

### 参数调优指南

#### 学习率调优

```yaml
# 如果训练不稳定
learning_rate: 5e-6  # 降低学习率
warmup_ratio: 0.1    # 增加 warmup

# 如果使用 LoRA
lora_learning_rate: 1e-4  # LoRA 专用更大学习率
```

#### 批次大小调优

```yaml
# 显存不足时
micro_batch_size: 1           # 减少 micro batch
gradient_accumulation: 8      # 增加梯度累积

# 显存充足时
micro_batch_size: 4           # 提高吞吐量
gradient_accumulation: 2
```

#### GRPO 组大小调优

```yaml
# 训练不稳定时
num_answers_per_question: 4   # 减少组大小

# 追求更好效果
num_answers_per_question: 16  # 增加组大小，更稳定
```

## 故障排查

### 常见问题诊断树

```
问题：训练失败
├─ OOM 显存不足
│  ├─ 减少 batch size
│  ├─ 启用参数卸载
│  └─ 启用 LoRA
├─ 数据格式错误
│  ├─ 运行 validate_data.py
│  └─ 检查 parquet 字段
└─ 依赖问题
   ├─ 检查 verl 版本
   └─ 检查 vllm 版本

问题：训练不稳定
├─ loss 震荡
│  ├─ 降低学习率
│  ├─ 增加 warmup
│  └─ 减少梯度裁剪
└─ reward 不增长
   ├─ 检查奖励函数
   ├─ 降低 KL 惩罚
   └─ 增加组大小

问题：格式准确率低
├─ 增加 SFT epochs
├─ 增加格式奖励权重
└─ 检查数据质量
```

### 诊断命令

```bash
# 检查 GPU 状态
nvidia-smi

# 检查依赖版本
python -c "import verl; print(verl.__version__)"
python -c "import vllm; print(vllm.__version__)"

# 验证数据格式
python verl/utils/model_paper_search/validate_data.py \
    --train_file data/data_verl/paper_train.parquet

# 测试奖励函数
python -c "
from verl.utils.reward_score import default_compute_score
score = default_compute_score(
    data_source='paper_summary',
    solution_str='{\"summary\": \"test\"}',
    ground_truth='{\"summary\": \"test\"}'
)
print(f'Score: {score}')
"

# 查看训练日志
tail -f outputs/*/train_log.txt

# 监控 WandB
# 访问 https://wandb.ai/your-username/
```

## 性能优化

### 训练加速

1. **梯度累积**
   - 减少通信开销
   - 保持有效 batch size

2. **混合精度训练**
   - 使用 bfloat16
   - 减少显存占用

3. **FSDP2 策略**
   - 分布式训练
   - 参数分片

4. **vLLM 采样**
   - 异步采样
   - 张量并行

### 显存优化

```bash
# 启用参数卸载
PARAM_OFFLOAD=true

# 启用优化器卸载
OPTIMIZER_OFFLOAD=true

# 启用梯度检查点
model.enable_gradient_checkpointing=True

# 减少 vLLM 显存占用
VLLM_GPU_MEMORY_UTILIZATION=0.5
```

### 奖励函数优化

```python
# 缓存 LLM 评判结果
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_similarity_score(text1_hash, text2_hash):
    # 缓存实现
    pass

# 批量处理
def batch_compute_scores(pairs):
    # 批量调用 LLM
    pass
```

## 最佳实践

### 数据准备

1. **数据清洗**
   - 去除低质量样本
   - 统一格式规范

2. **数据平衡**
   - 各字段长度平衡
   - 领域分布平衡

3. **数据验证**
   - 运行 validate_data.py
   - 人工抽查样本

### 训练监控

1. **实时指标**
   - train_loss 平滑下降
   - reward 逐渐上升
   - KL 散度 < 0.1

2. **异常检测**
   - loss 突增：检查梯度
   - reward 下降：检查奖励函数
   - OOM：调整 batch size

3. **定期评估**
   - 每 50 步验证
   - 保存 checkpoint
   - 生成样本检查

### 模型导出

```python
# 合并 LoRA 权重（如果使用）
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "path/to/lora/checkpoint",
    torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.save_pretrained("path/to/merged_model")

# 使用 vLLM 部署
from vllm import LLM

llm = LLM(
    model="path/to/merged_model",
    tensor_parallel_size=1,
    max_model_len=8192
)
```

## 相关资源

- [verl 框架文档](verl/README.md)
- [GRPO 论文](https://arxiv.org/abs/...)
- [Qwen3 模型文档](https://www.modelscope.cn/...)
- [vLLM 文档](https://vllm.readthedocs.io/)

---

**最后更新**: 2026-03-11  
**维护者**: 开发团队
