import json
import re
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional, List, Set
import dashscope


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> float:
    """
    论文总结任务的奖励函数（verl框架规范）
    
    Args:
        data_source: 数据源名称，如 "paper_summary"
        solution_str: 模型生成的JSON字符串
        ground_truth: 标准答案JSON字符串
        extra_info: 额外信息，需包含 "original_instruction" 字段（包含论文原文）
    
    Returns:
        float: 奖励值，范围 [0.0, 1.0]
    """
    # 仅处理论文总结任务
    if data_source != "paper_summary":
        return 0.0
    
    # 1. 解析模型生成的JSON
    try:
        generated_json = json.loads(solution_str)
    except json.JSONDecodeError:
        return 0.0  # JSON解析失败，直接返回0
    
    # 2. 解析标准答案JSON
    try:
        ground_truth_dict = json.loads(ground_truth)
    except json.JSONDecodeError:
        return 0.0
    
    # 3. 获取论文原文（从 extra_info 中的 original_instruction 提取）
    paper_text = ""
    if extra_info and "original_instruction" in extra_info:
        paper_text = extract_paper_from_instruction(extra_info["original_instruction"])
    
    # 4. 格式合规性检查（硬约束）
    format_reward = check_format_compliance(generated_json)
    if format_reward == 0:
        return 0.0  # 格式错误直接返回0
    
    # 5. 各维度质量奖励
    r_summary = reward_summary(
        paper_text,
        generated_json.get("summary", ""),
        ground_truth_dict.get("summary", "")
    )
    r_algorithm = reward_algorithm(
        paper_text,
        generated_json.get("algorithm", ""),
        ground_truth_dict.get("algorithm", "")
    )
    r_compare = reward_comparison(
        paper_text,
        generated_json.get("compare_result", ""),
        ground_truth_dict.get("compare_result", "")
    )
    r_kw_problem = reward_keywords_problem(
        paper_text,
        generated_json.get("keyword_problem", ""),
        ground_truth_dict.get("keyword_problem", "")
    )
    r_kw_algorithm = reward_keywords_algorithm(
        paper_text,
        generated_json.get("keyword_algorithm", ""),
        ground_truth_dict.get("keyword_algorithm", "")
    )
    
    # 6. 加权融合
    weights = [0.25, 0.30, 0.20, 0.125, 0.125]
    total_reward = (
        weights[0] * r_summary +
        weights[1] * r_algorithm +
        weights[2] * r_compare +
        weights[3] * r_kw_problem +
        weights[4] * r_kw_algorithm
    )
    
    return total_reward * format_reward


def extract_paper_from_instruction(instruction: str) -> str:
    """
    从 instruction 中提取论文原文
    instruction 格式: 任务指令 + #论文 + 论文内容
    """
    marker = "#论文"
    if marker in instruction:
        return instruction.split(marker, 1)[1].strip()
    return instruction


def check_format_compliance(output: dict) -> float:
    """
    检查输出格式是否合规
    """
    required_keys = ["summary", "algorithm", "compare_result", "keyword_problem", "keyword_algorithm"]

    # 检查字段存在性
    if not all(k in output for k in required_keys):
        return 0.0

    # 检查非空性
    for key in required_keys:
        if not isinstance(output[key], str) or not output[key].strip():
            return 0.0

    # 检查关键词格式
    for kw_key in ["keyword_problem", "keyword_algorithm"]:
        if not validate_keyword_format(output[kw_key]):
            return 0.0

    return 1.0


def validate_keyword_format(keyword_str: str) -> bool:
    """
    验证关键词格式: 中文, English, Abbr; 中文2, English2
    """
    if not keyword_str:
        return False
    keywords = keyword_str.split(";")
    for kw in keywords:
        parts = kw.split(",")
        if len(parts) < 2:  # 至少要有中英文
            return False
    return True


def reward_summary(paper: str, gen_summary: str, ref_summary: str) -> float:
    """
    计算 summary 部分的奖励
    """
    # 1. 与参考答案的相似度（核心）
    semantic_sim = compute_semantic_similarity(gen_summary, ref_summary)

    # 1.2 关键词重叠度
    gen_keywords = set(extract_keywords(gen_summary, top_k=10))
    ref_keywords = set(extract_keywords(ref_summary, top_k=10))
    keyword_overlap = len(gen_keywords & ref_keywords) / max(len(ref_keywords), 1)

    reference_sim = 0.6 * semantic_sim + 0.4 * keyword_overlap

    # 2. 内容完整性检查
    has_problem = any(w in gen_summary for w in ["解决", "针对", "挑战", "问题", "场景", "应用"])
    has_method = any(w in gen_summary for w in ["提出", "设计", "引入", "方法", "算法", "模型"])
    has_result = any(w in gen_summary for w in ["提升", "优于", "达到", "准确率", "效果", "性能"])
    completeness = (has_problem + has_method + has_result) / 3.0

    # 3. 事实一致性（与原文的匹配度）
    paper_keywords = extract_keywords(paper, top_k=20)
    gen_summary_keywords = extract_keywords(gen_summary, top_k=10)
    fact_overlap = len(set(gen_summary_keywords) & set(paper_keywords)) / max(len(gen_summary_keywords), 1)

    return 0.5 * reference_sim + 0.3 * completeness + 0.2 * fact_overlap


def reward_algorithm(paper: str, gen_algo: str, ref_algo: str) -> float:
    """
    计算 algorithm 部分的奖励
    """
    # 1. 与参考答案的相似度（核心）
    semantic_sim = compute_semantic_similarity(gen_algo, ref_algo)

    # 提取技术术语（大写字母开头的专业名词）
    def extract_terms(text):
        return set(re.findall(r'[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*', text))

    gen_terms = extract_terms(gen_algo)
    ref_terms = extract_terms(ref_algo)
    term_overlap = len(gen_terms & ref_terms) / max(len(ref_terms), 1)

    reference_sim = 0.6 * semantic_sim + 0.4 * term_overlap

    # 2. 结构完整性
    has_core_idea = any(phrase in gen_algo for phrase in ["核心思想", "核心观点", "主要思想"])
    has_innovation = any(phrase in gen_algo for phrase in ["创新点", "创新", "首次", "改进", "不同于"])
    has_steps = any(phrase in gen_algo for phrase in ["步骤", "流程", "首先", "然后", "最后", "Step", "1.", "2."])
    structure_score = (has_core_idea + has_innovation + has_steps) / 3.0

    # 3. 步骤清晰度（检查步骤编号）
    step_patterns = [r"\d+\.", r"步骤\s*\d+", r"Step\s*\d+", r"首先|第一步", r"然后|第二步", r"最后|最终"]
    step_count = sum(1 for p in step_patterns if re.search(p, gen_algo))
    clarity_score = min(1.0, step_count / 3)

    # 4. 技术术语匹配（与参考答案对比）
    technical_score = term_overlap

    return 0.4 * reference_sim + 0.25 * structure_score + 0.2 * clarity_score + 0.15 * technical_score


def reward_comparison(paper: str, gen_compare: str, ref_compare: str) -> float:
    """
    计算 compare_result 部分的奖励
    """
    if not gen_compare or len(gen_compare) < 20:
        return 0.0

    # 1. 与参考答案的相似度（核心）
    semantic_sim = compute_semantic_similarity(gen_compare, ref_compare)

    # 提取对比方法名（大写字母缩写或特定格式）
    def extract_methods(text):
        # 匹配大写缩写 (如 ResNet, BERT, GPT)
        abbreviations = set(re.findall(r'\b[A-Z]{2,}\b', text))
        # 匹配引用格式 [1], [2]
        citations = set(re.findall(r'\[\d+\]', text))
        return abbreviations | citations

    gen_methods = extract_methods(gen_compare)
    ref_methods = extract_methods(ref_compare)
    method_overlap = len(gen_methods & ref_methods) / max(len(ref_methods), 1)

    reference_sim = 0.6 * semantic_sim + 0.4 * method_overlap

    # 2. 对比完整性检查
    has_baseline = any(indicator in gen_compare for indicator in [
        "相比", "对比", "优于", "vs", "versus", "baseline", "SOTA", "现有方法"
    ])
    has_metric = any(metric in gen_compare for metric in [
        "准确率", "精度", "F1", "mAP", "BLEU", "ROUGE", "准确率", "性能", "效果", "%"
    ])
    has_number = bool(re.search(r"\d+\.?\d*\s*%?", gen_compare))
    completeness = (has_baseline + has_metric + has_number) / 3.0

    # 3. 数值格式正确性
    number_patterns = [
        r"\d+\.\d+%",  # 百分比
        r"提升?\s*\d+\.?\d*",  # 提升数值
        r"\d+\.\d+\s*[±\+\-]\s*\d+\.\d+"  # 带误差范围
    ]
    number_quality = min(1.0, sum(1 for p in number_patterns if re.search(p, gen_compare)) / 2)

    # 4. 方法识别（与参考答案对比）
    method_score = method_overlap

    return 0.4 * reference_sim + 0.3 * completeness + 0.2 * number_quality + 0.1 * method_score


def reward_keywords_problem(paper: str, gen_kw: str, ref_kw: str) -> float:
    """
    计算 keyword_problem 部分的奖励
    """
    if not gen_kw:
        return 0.0

    gen_keywords = [k.strip() for k in gen_kw.split(";") if k.strip()]
    ref_keywords = [k.strip() for k in ref_kw.split(";") if k.strip()]

    if not gen_keywords:
        return 0.0

    # 1. 与参考答案的关键词匹配度（核心）
    def normalize_keyword(kw):
        """归一化关键词用于比较（取中文部分）"""
        parts = kw.split(",")
        for p in parts:
            if any('\u4e00' <= c <= '\u9fff' for c in p):
                return p.strip().lower()
        return kw.strip().lower()

    gen_kw_set = set(normalize_keyword(k) for k in gen_keywords)
    ref_kw_set = set(normalize_keyword(k) for k in ref_keywords)

    # 计算重叠度和覆盖率
    overlap = len(gen_kw_set & ref_kw_set)
    precision = overlap / len(gen_kw_set) if gen_kw_set else 0
    recall = overlap / len(ref_kw_set) if ref_kw_set else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 2. 格式正确性
    format_scores = []
    for kw in gen_keywords:
        parts = [p.strip() for p in kw.split(",")]
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in kw)
        has_english = any(c.isascii() and c.isalpha() for c in kw)
        format_scores.append(1.0 if (has_chinese and has_english) else 0.5)
    format_score = sum(format_scores) / len(format_scores)

    # 3. 语义相关性（与原文的匹配度）
    chinese_parts = [normalize_keyword(k) for k in gen_keywords]
    relevance = compute_keyword_relevance(paper, chinese_parts)

    return 0.5 * f1_score + 0.25 * format_score + 0.25 * relevance


def reward_keywords_algorithm(paper: str, gen_kw: str, ref_kw: str) -> float:
    """
    计算 keyword_algorithm 部分的奖励
    """
    if not gen_kw:
        return 0.0

    gen_keywords = [k.strip() for k in gen_kw.split(";") if k.strip()]
    ref_keywords = [k.strip() for k in ref_kw.split(";") if k.strip()]

    if not gen_keywords:
        return 0.0

    # 1. 与参考答案的关键词匹配度（核心）
    def normalize_keyword(kw):
        """归一化关键词用于比较"""
        parts = [p.strip() for p in kw.split(",")]
        # 优先使用英文部分（算法名通常是大写）
        for p in parts:
            p = p.strip()
            if re.match(r'^[A-Z][a-zA-Z0-9\-]*$', p):  # 匹配算法名格式
                return p.lower()
        # 否则使用中文部分
        for p in parts:
            if any('\u4e00' <= c <= '\u9fff' for c in p):
                return p.strip().lower()
        return kw.strip().lower()

    gen_kw_set = set(normalize_keyword(k) for k in gen_keywords)
    ref_kw_set = set(normalize_keyword(k) for k in ref_keywords)

    overlap = len(gen_kw_set & ref_kw_set)
    precision = overlap / len(gen_kw_set) if gen_kw_set else 0
    recall = overlap / len(ref_kw_set) if ref_kw_set else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 2. 格式正确性
    format_scores = []
    for kw in gen_keywords:
        parts = [p.strip() for p in kw.split(",")]
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in kw)
        has_english = any(c.isascii() and c.isalpha() for c in kw)
        format_scores.append(1.0 if (has_chinese and has_english) else 0.5)
    format_score = sum(format_scores) / len(format_scores)

    # 3. 算法识别（从原文提取算法名进行匹配）
    proposed_methods = extract_proposed_methods(paper)
    keyword_text = gen_kw.lower()
    matched = sum(1 for m in proposed_methods if m.lower() in keyword_text)
    method_score = min(1.0, matched / max(1, len(proposed_methods))) if proposed_methods else 0.5

    return 0.5 * f1_score + 0.25 * format_score + 0.25 * method_score


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    简单的关键词提取（基于词频）
    """
    if not text:
        return []
    
    # 简单的中文分词（基于正则）
    import re
    # 提取中文词语
    chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
    # 提取英文单词
    english_words = re.findall(r'[a-zA-Z]+', text)
    # 提取数字+单位
    numbers = re.findall(r'\d+\.?\d*[%a-zA-Z]*', text)
    
    all_words = chinese_words + english_words + numbers
    
    # 简单词频统计
    word_count = {}
    for word in all_words:
        if len(word) > 1:  # 过滤单字
            word_count[word] = word_count.get(word, 0) + 1
    
    # 按词频排序，取前 top_k
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_k]]


def compute_semantic_similarity_fallback(text1: str, text2: str) -> float:
    """
    计算语义相似度（兜底方案，基于词重叠）
    """
    if not text1 or not text2:
        return 0.0
    
    # 提取关键词
    keywords1 = set(extract_keywords(text1, top_k=20))
    keywords2 = set(extract_keywords(text2, top_k=20))
    
    # 计算 Jaccard 相似度
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    计算语义相似度（主要使用 LLM，失败时使用兜底方案）
    """
    # 主要使用 LLM 计算相似度
    return compute_similarity_with_llm(text1, text2)

def compute_similarity_with_llm(text1: str, text2: str, model_name: str = "qwen-plus") -> float:
    """
    使用 LLM 计算文本相似度
    """
    prompt = f"""你是一个专业的语义相似度评估专家。请对给出的两句话进行语义相似度打分，
句子1: {text1}
句子2: {text2}

输出打分标准：
1.0：语义完全相同。
0.7～0.9：语义高度相似。
0.4～0.6：语义中等相关。
0.1～0.3：语义微弱相关。
0.0：完全无关。

输出要求：
1. 严格按照以下JSON格式输出，不要添加任何多余文字、解释或换行；
2. 字段说明：
    - score：0-1的小数，语义相似度分数（1.0完全相同，0.0完全无关）；
    - reason：简短说明打分理由（不超过50字）；
3. JSON输出示例：
    {{
        "score": 0.9,
        "reason": "两句话语义高度相似，仅表述略有差异"
    }}"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    if not text1 or not text2:
        return 0.0
    
    # 调用 LLM 模型计算相似度
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model=model_name, # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='json',
        temperature=0.1
    )
    # 4. 解析响应（结构化数据）
    if response.status_code == 200:
        # 提取 JSON 结果
        structured_result = response.output.choices[0].message.content
        # 转换为 Python 字典（方便后续处理）
        import json
        try:
            result_dict = json.loads(structured_result)
            return result_dict.get("score", 0.5)
        except json.JSONDecodeError:
            return compute_semantic_similarity_fallback(text1, text2)
    else:
        return compute_semantic_similarity_fallback(text1, text2)



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


def extract_proposed_methods(paper: str) -> List[str]:
    """
    从论文中提取提出的方法名
    """
    if not paper:
        return []
    
    patterns = [
        r"we propose[d]?\s+([A-Z][a-zA-Z0-9\-]+)",
        r"called\s+([A-Z][a-zA-Z0-9\-]+)",
        r"named\s+([A-Z][a-zA-Z0-9\-]+)",
        r"\b([A-Z]{2,})\b"  # 大写缩写
    ]
    
    methods = []
    for pattern in patterns:
        matches = re.findall(pattern, paper, re.IGNORECASE)
        methods.extend(matches)
    
    return list(set(methods))


if __name__ == "__main__":
    # 批量测试示例（覆盖0-1分）
    test_cases = [
        ("我周末想去上海迪士尼乐园游玩", "周末我打算去上海迪士尼乐园玩"),
        ("今天下午我要去超市买牛奶和面包", "今日午后我计划到超市采购牛奶与面包"),
        ("小明每天早上7点跑步30分钟", "小明每天早晨7点左右跑步半小时"),
        ("公司下周要组织员工去团建旅游", "公司下个月计划安排员工外出活动"),
        ("夏天适合吃西瓜解暑", "夏天适合喝绿豆汤降温"),
        ("猫咪喜欢在阳光下睡觉", "手机充满电需要大约2小时")
    ]
    for i, test in enumerate(test_cases):
        print(f"测试案例 {i+1}: {test}")
        score = compute_similarity_with_llm(test[0], test[1])
        print(type(score))
        print(score)
