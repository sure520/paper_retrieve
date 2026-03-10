以下是一个**完整的 reward 函数伪代码**，专为论文总结模型的 JSON 输出格式（包含 `summary`、`algorithm`、`compare_result`、`keyword_problem`、`keyword_algorithm` 五个字段）量身定制。

该函数假设已能提取原始英文论文全文（或摘要+方法+实验部分），并使用若干自动化工具（如 NLI 模型、LLM-as-a-Judge、关键词匹配等）进行评估。

---

### ✅ 输入
- `paper_text: str` —— 论文原文（建议包含 Abstract + Introduction + Method + Experiment）
- `generated_output: dict` —— 模型生成的结构化中文摘要，格式如下：
  ```python
  {
    "summary": "本文提出了一种新方法...",
    "algorithm": "核心思想是... 具体步骤包括...",
    "compare_result": "相比ResNet，在ImageNet上提升2.1%...",
    "keyword_problem": ["图像分割", "小样本学习"],
    "keyword_algorithm": ["DiffSeg", "跨模态对齐"]
  }
  ```

---

### 🧮 Reward 函数伪代码（Python 风格）

```python
def compute_reward(paper_text: str, generated_output: dict, use_llm_judge: bool = True) -> float:
    """
    Compute a composite reward for a structured Chinese paper summary.
    Returns a scalar in [0.0, 1.0] (higher is better).
    """
    # --- 安全检查：确保输出是合法 JSON 且字段完整 ---
    required_keys = {"summary", "algorithm", "compare_result", "keyword_problem", "keyword_algorithm"}
    if not all(k in generated_output for k in required_keys):
        return 0.0  # 格式错误，直接惩罚

    try:
        summary = str(generated_output["summary"]).strip()
        algorithm = str(generated_output["algorithm"]).strip()
        compare_result = str(generated_output["compare_result"]).strip()
        kw_problem = generated_output["keyword_problem"]
        kw_algorithm = generated_output["keyword_algorithm"]

        if not summary or not algorithm:
            return 0.0
    except Exception:
        return 0.0

    # --- 1. Factuality & Coverage Reward (基于 NLI + 规则) ---
    def reward_summary_factuality(summary: str, paper: str) -> float:
        # 使用中英 NLI 模型判断摘要是否被原文蕴含（entailment）
        # 若无现成模型，可用 Qwen-Max 判断："以下中文摘要是否忠实于英文原文？"
        entail_score = nli_entailment_score(
            premise=paper, 
            hypothesis=summary,
            lang_pair=("en", "zh")
        )  # 返回 [0,1]，1 表示完全忠实
        # 加分项：是否包含“问题-方法-效果”三要素
        has_problem = any(w in summary for w in ["解决", "针对", "挑战", "问题"])
        has_method = any(w in summary for w in ["提出", "设计", "引入", "方法"])
        has_result = any(w in summary for w in ["提升", "优于", "达到", "准确率", "效果"])
        structure_bonus = (has_problem + has_method + has_result) / 3.0
        return 0.7 * entail_score + 0.3 * structure_bonus

    # --- 2. Algorithm Clarity & Completeness ---
    def reward_algorithm(algo_desc: str) -> float:
        if use_llm_judge:
            # 调用强 LLM（如 Qwen-Max）打分
            prompt = f"""你是一个AI审稿人。请对以下算法描述的完整性打分（0~1）：
            - 是否说明了核心思想？
            - 是否包含关键步骤或公式？
            - 是否指出创新点？
            算法描述：{algo_desc}
            只返回一个数字，保留两位小数。"""
            score = call_qwen_max(prompt)  # 返回 float in [0,1]
            return min(max(score, 0.0), 1.0)
        else:
            # 回退到规则：检查关键词密度
            innovation_words = ["创新", "不同于", "首次", "改进", "新", "提出"]
            step_words = ["步骤", "首先", "然后", "最后", "流程", "过程"]
            score = (
                int(any(w in algo_desc for w in innovation_words)) * 0.5 +
                int(any(w in algo_desc for w in step_words)) * 0.5
            )
            return score

    # --- 3. Comparison Result Quality ---
    def reward_comparison(compare_text: str, paper: str) -> float:
        if not compare_text or "无" in compare_text or "未" in compare_text:
            return 0.0
        # 检查是否包含具体方法名 + 指标 + 数值
        has_baseline = any(model in compare_text for model in ["ResNet", "BERT", "ViT", "SOTA", "现有方法"])
        has_metric = any(metric in compare_text for metric in ["准确率", "F1", "mAP", "BLEU", "指标", "性能"])
        has_number = bool(re.search(r'\d+(\.\d+)?%', compare_text))  # 匹配 "2.1%" 这类
        completeness = (has_baseline + has_metric + has_number) / 3.0
        # 额外：若原文有实验表格，检查是否引用关键结果
        if has_number and extract_numbers_from_paper_experiments(paper):
            completeness = min(1.0, completeness + 0.2)
        return completeness

    # --- 4. Problem Keywords Relevance ---
    def reward_kw_problem(keywords: list, paper: str) -> float:
        if not keywords:
            return 0.0
        paper_keywords = extract_top_tfidf_keywords(paper, top_k=10, lang="en")
        # 中文关键词需翻译为英文再比对（或用 multilingual embedding）
        translated = [translate_zh_to_en(kw) for kw in keywords]
        overlap = len(set(translated) & set(paper_keywords))
        return min(1.0, overlap / max(1, len(keywords)))

    # --- 5. Algorithm Keywords Accuracy ---
    def reward_kw_algorithm(keywords: list, paper: str) -> float:
        if not keywords:
            return 0.0
        # 检查是否包含论文中提出的新方法名称（通常出现在标题/摘要）
        proposed_method = extract_proposed_method_name(paper)  # e.g., "LightGCN"
        if not proposed_method:
            return 0.5  # 无法提取时给中等分
        match = any(proposed_method.lower() in kw.lower() for kw in keywords)
        return 1.0 if match else 0.2

    # --- 计算各维度 reward ---
    r1 = reward_summary_factuality(summary, paper_text)
    r2 = reward_algorithm(algorithm)
    r3 = reward_comparison(compare_result, paper_text)
    r4 = reward_kw_problem(kw_problem, paper_text)
    r5 = reward_kw_algorithm(kw_algorithm, paper_text)

    # --- 加权融合（可根据验证集调整）---
    weights = [0.30, 0.25, 0.25, 0.10, 0.10]  # summary 和 algorithm 权重更高
    total_reward = (
        weights[0] * r1 +
        weights[1] * r2 +
        weights[2] * r3 +
        weights[3] * r4 +
        weights[4] * r5
    )

    # 最终裁剪到 [0, 1]
    return max(0.0, min(1.0, total_reward))
```

---

### 🔧 辅助函数说明（需你实现或调用）

| 函数 | 说明 | 建议实现方式 |
|------|------|-------------|
| `nli_entailment_score(premise, hypothesis, lang_pair)` | 判断假设是否被前提蕴含 | 使用 [Chinese-BERT-NLI](https://huggingface.co/moritzlaurer/DeBERTa-v3-base-mnli-fever-anli) + 机器翻译，或直接调用 Qwen-Max |
| `call_qwen_max(prompt)` | 调用 Qwen-Max API 打分 | 阿里云 DashScope API |
| `extract_numbers_from_paper_experiments(paper)` | 判断原文是否有量化实验结果 | 正则匹配 `\d+\.\d+%` 或表格识别 |
| `extract_top_tfidf_keywords(text, lang)` | 提取关键词 | sklearn TfidfVectorizer + 英文分词 |
| `translate_zh_to_en(text)` | 中译英 | 使用 `googletrans` 或 Qwen-Max |
| `extract_proposed_method_name(paper)` | 提取论文提出的新方法名 | 规则：标题中非通用术语，或摘要中“we propose XXX” |

---

### 💡 使用建议

- **初期可关闭 LLM Judge**（设 `use_llm_judge=False`），用规则快速迭代。
- **人工校准 reward**：随机抽 50 个样本，对比自动 reward 与人工评分的相关性（Spearman）。
- **加入 length penalty**（可选）：
  ```python
  length_penalty = 1.0 - abs(len(summary) - 300) / 500  # 理想长度 300 字
  total_reward *= max(0.5, length_penalty)
  ```

---