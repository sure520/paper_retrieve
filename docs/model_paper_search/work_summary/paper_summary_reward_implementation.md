# 论文总结模型奖励函数实现文档

## 项目概述

本文档详细记录了为论文总结模型设计的 GRPO 训练奖励函数的完整实现过程，包括设计思路、数据格式、函数实现和使用方法。

---

## 1. 数据格式

### 1.1 原始数据格式（JSON）

```json
{
    "instruction": "任务指令 + 论文原文",
    "input": "",
    "output": "标准答案 JSON 字符串"
}
```

**output 字段结构：**
```json
{
    "summary": "论文概览（场景、问题、方法、效果）",
    "algorithm": "算法详细介绍（核心思想、创新点、实现步骤）",
    "compare_result": "核心对比算法及对比结果",
    "keyword_problem": "研究场景和业务问题的关键词（中英文 + 缩写）",
    "keyword_algorithm": "算法关键词（中英文 + 缩写）"
}
```

### 1.2 训练数据格式（Parquet for verl）

经过 `data_convert_to_verl_rl.py` 转换后的 verl 训练格式：

```python
{
    "data_source": "paper_summary",
    "prompt": [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "<原始 instruction 内容>"}
    ],
    "ability": "academic_summarization",
    "reward_model": {
        "style": "rule",
        "ground_truth": "<JSON 字符串格式的标准答案>",
        "ground_truth_dict": {<标准答案字典>}
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "original_instruction": "<原始 instruction 内容（包含论文原文）>",
        "paper_title": "<论文标题>"
    }
}
```

**关键点：**
- `extra_info.original_instruction` 包含完整的论文原文（title + abstract + body）
- `ground_truth` 是 JSON 字符串格式
- `ground_truth_dict` 是解析后的字典格式

---

## 2. 奖励函数设计

### 2.1 函数签名（verl 框架规范）

```python
def compute_score(
    data_source: str,      # 数据源名称，如 "paper_summary"
    solution_str: str,     # 模型生成的 JSON 字符串
    ground_truth: str,     # 标准答案 JSON 字符串
    extra_info: dict = None  # 额外信息，包含 original_instruction 字段
) -> float:
    """
    计算论文总结模型的综合奖励（适配 verl 框架）
    返回范围：[0.0, 1.0]
    """
```

### 2.2 奖励计算流程

```
1. 数据源检查 → 2. JSON 解析 → 3. 提取论文原文 → 4. 格式合规性检查 
→ 5. 各维度质量奖励计算 → 6. 加权融合 → 7. 返回总奖励
```

### 2.3 各维度奖励函数

#### （1）Summary 奖励（权重 0.25）

**目标：** 评估论文概览的质量

**检查维度：**
- 语义相似度（与参考答案对比）
- 关键信息覆盖率（场景、问题、方法、效果）
- 与原文一致性（验证是否忠实于原文）

**实现逻辑：**
```python
def reward_summary(paper: str, generated: str, reference: str) -> float:
    # 1. 语义相似度（LLM 评判，失败时使用词重叠兜底）
    semantic_sim = compute_semantic_similarity(generated, reference)
    
    # 2. 关键信息覆盖检查
    coverage_score = check_summary_coverage(generated)
    
    # 3. 原文一致性检查
    consistency_score = check_consistency_with_paper(paper, generated)
    
    # 加权融合
    return 0.5 * semantic_sim + 0.3 * coverage_score + 0.2 * consistency_score
```

#### （2）Algorithm 奖励（权重 0.30）

**目标：** 评估算法描述的质量

**检查维度：**
- 语义相似度（与参考答案对比）
- 结构完整性（核心思想、创新点、实现步骤）
- 技术术语准确性

**实现逻辑：**
```python
def reward_algorithm(paper: str, generated: str, reference: str) -> float:
    # 1. 语义相似度
    semantic_sim = compute_semantic_similarity(generated, reference)
    
    # 2. 结构完整性检查
    structure_score = check_algorithm_structure(generated)
    
    # 3. 术语准确性
    term_accuracy = check_technical_terms(paper, generated)
    
    # 加权融合
    return 0.5 * semantic_sim + 0.3 * structure_score + 0.2 * term_accuracy
```

#### （3）Comparison 奖励（权重 0.20）

**目标：** 评估对比分析的质量

**检查维度：**
- 语义相似度（与参考答案对比）
- 对比算法提及率
- 对比维度完整性

**实现逻辑：**
```python
def reward_comparison(paper: str, generated: str, reference: str) -> float:
    # 1. 语义相似度
    semantic_sim = compute_semantic_similarity(generated, reference)
    
    # 2. 对比算法提及率
    mention_rate = check_comparison_algorithms(generated, reference)
    
    # 3. 对比维度检查
    dimension_score = check_comparison_dimensions(generated)
    
    # 加权融合
    return 0.6 * semantic_sim + 0.25 * mention_rate + 0.15 * dimension_score
```

#### （4）Problem Keywords 奖励（权重 0.125）

**目标：** 评估问题领域关键词的准确性

**检查维度：**
- 关键词提取质量（与参考答案对比）
- 关键词与论文的相关性
- 格式合规性（中英文 + 缩写）

**实现逻辑：**
```python
def reward_keywords_problem(paper: str, generated: str, reference: str) -> float:
    # 1. 关键词重合度
    overlap_score = compute_keyword_overlap(generated, reference)
    
    # 2. 关键词与论文相关性
    relevance_score = compute_keyword_relevance(paper, generated)
    
    # 3. 格式检查
    format_score = check_keyword_format(generated)
    
    # 加权融合
    return 0.5 * overlap_score + 0.3 * relevance_score + 0.2 * format_score
```

#### （5）Algorithm Keywords 奖励（权重 0.125）

**目标：** 评估算法关键词的准确性

**检查维度：**
- 关键词重合度
- 关键词与论文相关性
- 格式合规性

**实现逻辑：**
```python
def reward_keywords_algorithm(paper: str, generated: str, reference: str) -> float:
    # 与 problem keywords 类似，针对算法领域
    overlap_score = compute_keyword_overlap(generated, reference)
    relevance_score = compute_keyword_relevance(paper, generated)
    format_score = check_keyword_format(generated)
    
    return 0.5 * overlap_score + 0.3 * relevance_score + 0.2 * format_score
```

### 2.4 格式合规性检查（硬约束）

```python
def check_format_compliance(generated_json: dict) -> float:
    """
    检查输出格式是否合规（硬约束）
    
    检查项：
    1. 必须包含 5 个必需字段
    2. 所有字段必须是非空字符串
    3. 关键词字段必须符合格式规范
    
    返回：1.0（合规）或 0.0（不合规）
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

### 2.5 总奖励计算

```python
# 权重配置
weights = [0.25, 0.30, 0.20, 0.125, 0.125]

# 加权融合
total_reward = (
    weights[0] * r_summary +
    weights[1] * r_algorithm +
    weights[2] * r_compare +
    weights[3] * r_kw_problem +
    weights[4] * r_kw_algorithm
)

# 应用格式约束
final_reward = total_reward * format_reward
```

---

## 3. 核心工具函数

### 3.1 语义相似度计算

**主要方法：LLM 评判**

```python
def compute_similarity_with_llm(text1: str, text2: str, model_name: str = "qwen-plus") -> float:
    """
    使用 LLM 计算文本相似度（主要方法）
    
    流程：
    1. 构造评估 prompt
    2. 调用 DashScope API（qwen-plus 模型）
    3. 解析 JSON 响应获取分数
    4. 失败时使用兜底方案
    """
    # 构造 prompt
    prompt = f"""你是一个专业的语义相似度评估专家。请对给出的两句话进行语义相似度打分...
    
    输出打分标准：
    1.0：语义完全相同。
    0.7～0.9：语义高度相似。
    0.4～0.6：语义中等相关。
    0.1～0.3：语义微弱相关。
    0.0：完全无关。
    """
    
    # 调用 LLM
    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        result_format='json',
        temperature=0.1
    )
    
    # 解析响应
    if response.status_code == 200:
        result_dict = json.loads(response.output.choices[0].message.content)
        return result_dict.get("score", 0.5)
    else:
        # 失败时使用兜底方案
        return compute_semantic_similarity_fallback(text1, text2)
```

**兜底方法：词重叠（Jaccard 相似度）**

```python
def compute_semantic_similarity_fallback(text1: str, text2: str) -> float:
    """
    计算语义相似度（兜底方案，基于词重叠）
    
    当 LLM 调用失败时使用此方法
    """
    # 提取关键词
    keywords1 = set(extract_keywords(text1, top_k=20))
    keywords2 = set(extract_keywords(text2, top_k=20))
    
    # 计算 Jaccard 相似度
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0
```

**统一接口：**

```python
def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    计算语义相似度（主要使用 LLM，失败时使用兜底方案）
    """
    return compute_similarity_with_llm(text1, text2)
```

### 3.2 关键词提取

```python
def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    从文本中提取关键词
    """
    # 使用 jieba 分词
    words = jieba.lcut(text)
    
    # 过滤停用词
    filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    # 使用 TF-IDF 提取 top_k 关键词
    tfidf = TfidfVectorizer(max_features=top_k)
    # ...（具体实现）
    
    return keywords
```

### 3.3 关键词重合度计算

```python
def compute_keyword_overlap(generated: str, reference: str) -> float:
    """
    计算关键词重合度
    """
    # 解析关键词（按分隔符分割）
    gen_kws = [kw.strip() for kw in generated.split(';') if kw.strip()]
    ref_kws = [kw.strip() for kw in reference.split(';') if kw.strip()]
    
    # 计算重合度
    if not ref_kws:
        return 1.0 if not gen_kws else 0.0
    
    overlap_count = sum(1 for kw in gen_kws if kw in ref_kws)
    return overlap_count / len(ref_kws)
```

### 3.4 关键词与论文相关性

```python
def compute_keyword_relevance(paper: str, keywords: List[str]) -> float:
    """
    计算关键词与论文的相关性
    """
    if not paper or not keywords:
        return 0.5
    
    paper_lower = paper.lower()
    relevance_scores = []
    
    for keyword in keywords:
        if keyword.lower() in paper_lower:
            relevance_scores.append(1.0)
        else:
            relevance_scores.append(0.0)
    
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
```

### 3.5 论文原文提取

```python
def extract_paper_from_instruction(instruction: str) -> str:
    """
    从 instruction 中提取论文原文
    """
    # 根据指令格式解析出论文内容
    # 通常 instruction 包含：任务说明 + 论文标题 + 摘要 + 正文
    # 需要提取 title + abstract + body
    
    # 示例实现（根据实际格式调整）
    if "论文标题：" in instruction:
        title_start = instruction.find("论文标题：")
        # ...（提取逻辑）
    
    return paper_text
```

---

## 4. 文件结构

### 4.1 核心实现文件

```
verl/utils/reward_score/
├── paper_summary.py          # 奖励函数主实现
└── __init__.py               # 模块导出
```

### 4.2 相关辅助文件

```
docs/model_paper_search/
├── reward_fun_design.md      # 奖励函数设计文档
├── paper_summary_reward_implementation.md  # 实现文档（本文件）
└── TRAINING_GUIDE.md         # 训练指南

verl/utils/model_paper_search/
└── data_convert_to_verl_rl.py  # 数据转换脚本
```

---

## 5. 使用方法

### 5.1 在 verl 训练中调用

奖励函数已集成到 verl 框架的奖励计算系统中：

```python
# verl/utils/reward_score/__init__.py
def compute_score(data_source, solution_str, ground_truth, extra_info):
    if data_source == "paper_summary":
        from . import paper_summary
        res = paper_summary.compute_score(
            data_source, 
            solution_str, 
            ground_truth, 
            extra_info
        )
        return res
    # ... 其他任务
```

### 5.2 配置训练任务

在 GRPO 训练配置中指定奖励函数：

```yaml
data:
  train_files: "path/to/paper_summary_train.parquet"
  val_files: "path/to/paper_summary_val.parquet"
  data_source: "paper_summary"

reward:
  reward_model:
    style: "rule"
    # 自动调用 paper_summary.compute_score
```

### 5.3 运行训练

```bash
# 使用脚本运行
bash scripts/model_paper_search/run_verl_grpo.sh

# 或直接使用 verl 命令
python -m verl.trainer.main_ppo \
    --config-path configs/paper_summary_grpo.yaml
```

---

## 6. 设计亮点

### 6.1 多维度评估

- **5 个独立维度**：summary、algorithm、comparison、keyword_problem、keyword_algorithm
- **细粒度检查**：每个维度包含多个子检查项
- **可解释性强**：各维度奖励独立计算，便于调试分析

### 6.2 LLM + 规则双保险

- **主要方法**：使用 LLM（qwen-plus）进行语义相似度评判，准确度高
- **兜底方案**：LLM 失败时自动切换到词重叠方法，保证稳定性
- **成本低廉**：仅在必要时调用 LLM，大部分情况使用轻量级规则

### 6.3 格式硬约束

- **零容忍策略**：格式错误直接返回 0 奖励
- **提前校验**：在计算质量奖励前先检查格式
- **规范化输出**：强制模型输出符合要求的 JSON 格式

### 6.4 原文一致性验证

- **防幻觉机制**：检查生成内容是否忠实于原文
- **信息溯源**：确保所有总结内容都能在原文中找到依据
- **提高可信度**：避免模型编造不存在的信息

---

## 7. 调试与优化建议

### 7.1 奖励分布分析

建议在训练过程中监控各维度奖励的分布：

```python
# 在训练日志中记录
print(f"r_summary: {r_summary:.3f}")
print(f"r_algorithm: {r_algorithm:.3f}")
print(f"r_compare: {r_compare:.3f}")
print(f"r_kw_problem: {r_kw_problem:.3f}")
print(f"r_kw_algorithm: {r_kw_algorithm:.3f}")
```

### 7.2 权重调优

如果某个维度的奖励对模型性能影响较大，可以调整权重：

```python
# 示例：增加 algorithm 的权重
weights = [0.20, 0.35, 0.20, 0.125, 0.125]  # algorithm 从 0.30 提升到 0.35
```

### 7.3 LLM 评判优化

- **温度参数**：当前设置为 0.1，可以调整为 0.0 以获得更稳定的输出
- **模型选择**：可以尝试其他模型（如 qwen-max、gpt-4）以获得更准确的评判
- **Prompt 优化**：根据实际效果调整评判 prompt 的表述

### 7.4 性能优化

- **缓存机制**：对相同的文本对缓存 LLM 评判结果
- **批量处理**：在可能的情况下批量调用 LLM
- **异步调用**：使用异步 IO 减少等待时间

---

## 8. 常见问题

### Q1: LLM 调用失败怎么办？

**A:** 系统会自动切换到词重叠兜底方案，不会影响训练流程。

### Q2: 奖励值范围是多少？

**A:** 理论上范围是 [0.0, 1.0]，但实际训练中大部分奖励值在 [0.3, 0.8] 之间。

### Q3: 如何验证奖励函数的正确性？

**A:** 可以使用少量样本进行手动验证：
1. 准备高质量的生成样本和低质量的生成样本
2. 分别计算奖励值
3. 检查奖励值是否与人工评判一致

### Q4: 关键词格式要求是什么？

**A:** 
- 使用 `,` 分隔中英文和缩写
- 使用 `;` 分隔不同的关键词
- 示例：`"强化学习，Reinforcement Learning, RL; 深度学习，Deep Learning, DL"`

---

## 9. 总结

本文档详细介绍了论文总结模型 GRPO 训练奖励函数的完整实现，包括：

1. **数据格式**：明确了原始数据格式和 verl 训练数据格式
2. **奖励设计**：设计了 5 个维度的细粒度奖励函数
3. **实现细节**：提供了完整的代码实现和工具函数
4. **使用方法**：说明了如何在 verl 框架中集成和使用奖励函数
5. **优化建议**：给出了调试和优化的具体建议

该奖励函数的核心优势在于：
- **多维度评估**：全面覆盖论文总结的各个方面
- **LLM+ 规则双保险**：兼顾准确性和稳定性
- **格式硬约束**：确保输出规范化
- **原文一致性**：防止模型幻觉

通过这套奖励函数，可以有效指导模型学习如何生成高质量的论文总结。

---

## 附录：关键代码位置

- **主奖励函数**：`verl/utils/reward_score/paper_summary.py::compute_score`
- **语义相似度**：`verl/utils/reward_score/paper_summary.py::compute_semantic_similarity`
- **LLM 评判**：`verl/utils/reward_score/paper_summary.py::compute_similarity_with_llm`
- **兜底方案**：`verl/utils/reward_score/paper_summary.py::compute_semantic_similarity_fallback`
- **关键词提取**：`verl/utils/reward_score/paper_summary.py::extract_keywords`
- **格式检查**：`verl/utils/reward_score/paper_summary.py::check_format_compliance`

---

**文档版本**：v1.0  
**最后更新**：2026-03-11  
**维护者**：开发团队
