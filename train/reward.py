import re
import json
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from googletrans import Translator
import numpy as np

# 全局变量
nli_model = None
translator = None

# 初始化NLI模型
def init_nli_model():
    global nli_model
    if nli_model is None:
        nli_model = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# 初始化翻译器
def init_translator():
    global translator
    if translator is None:
        translator = Translator()

# 调用Qwen-Max API打分, 使用DashScope格式
def call_qwen_max(prompt: str) -> float:
    import requests
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("QWEN_API_KEY")
    
    if not api_key:
        raise ValueError("QWEN_API_KEY not found in environment variables")
    
    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise ValueError("BASE_URL not found in environment variables")
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen3-max",
        "input": {
            "messages": [
                {
                "role": "system",
                "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        "parameters": {
            "temperature": 0.01,
            "top_p": 0.9,
            "max_tokens": 10
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    
    if "output" in response_data and "text" in response_data["output"]:
        text = response_data["output"]["text"].strip()
        try:
            return float(text)
        except ValueError:
            # 如果返回的不是数字，尝试提取数字
            match = re.search(r"\d+\.\d+", text)
            if match:
                return float(match.group())
            return 0.5
    return 0.5

# 判断假设是否被前提蕴含
def nli_entailment_score(premise: str, hypothesis: str, lang_pair: tuple = ("en", "zh")) -> float:
    try:
        init_nli_model()
        
        # 如果是中英对比，需要将中文假设翻译成英文
        if lang_pair == ("en", "zh"):
            init_translator()
            translated_hypothesis = translator.translate(hypothesis, dest="en").text
            premise_text = premise
        elif lang_pair == ("zh", "en"):
            init_translator()
            premise_text = translator.translate(premise, dest="en").text
            translated_hypothesis = hypothesis
        else:
            premise_text = premise
            translated_hypothesis = hypothesis
        
        result = nli_model(premise_text, [translated_hypothesis], hypothesis_template="This text: {}")
        entail_score = result["scores"][0]
        return entail_score
    except Exception as e:
        # 如果无法连接到模型或其他错误，返回一个默认值
        print(f"警告：NLI模型调用失败，使用默认值 0.7。错误信息：{e}")
        return 0.7

# 从论文中提取量化实验结果
def extract_numbers_from_paper_experiments(paper: str) -> bool:
    # 正则匹配百分比或数字
    pattern = r"\d+\.\d+%|\d+\.\d+\s+\w+|\d+\s+\w+"
    return bool(re.search(pattern, paper))

# 提取关键词
def extract_top_tfidf_keywords(text: str, lang: str = "en", top_k: int = 10) -> List[str]:
    if lang == "zh":
        # 中文按字符分词
        def char_tokenizer(text):
            return list(text)
        
        vectorizer = TfidfVectorizer(tokenizer=char_tokenizer, max_features=top_k)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_k)
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        return list(feature_names)
    except ValueError:
        # 如果词汇表为空，返回空列表
        return []

# 中译英
def translate_zh_to_en(text: str) -> str:
    try:
        init_translator()
        return translator.translate(text, dest="en").text
    except Exception as e:
        # 如果翻译服务不可用，返回原文本
        print(f"警告：翻译服务调用失败，使用原文本。错误信息：{e}")
        return text

# 从论文中提取提出的新方法名称
def extract_proposed_method_name(paper: str) -> str:
    # 从标题中提取
    title_match = re.search(r"^(.*?)$", paper, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        # 查找大写首字母的连续单词，可能是方法名
        method_matches = re.findall(r"[A-Z][a-zA-Z0-9]+(?:-[A-Z][a-zA-Z0-9]+)*", title)
        if method_matches:
            return method_matches[0]
    
    # 从摘要中提取"we propose XXX"
    abstract_match = re.search(r"abstract.*?(?:we propose|introduce|present)\s+([a-zA-Z0-9_-]+)", paper, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        return abstract_match.group(1)
    
    return ""

# 计算奖励函数
def compute_reward(paper_text: str, generated_output: Dict[str, Any], use_llm_judge: bool = True) -> float:
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
            try:
                score = call_qwen_max(prompt)
                return min(max(score, 0.0), 1.0)
            except Exception:
                # 如果LLM调用失败，回退到规则
                pass
        
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
    def reward_kw_problem(keywords: str, paper: str) -> float:
        if not keywords:
            return 0.0
        
        # 解析关键词字符串，格式为 "关键词1,Keyword1,K1;关键词2,Keyword2,K2"
        kw_list = [kw.strip() for kw in keywords.split(";")]
        if not kw_list:
            return 0.0
        
        # 提取中文关键词
        zh_keywords = []
        for kw in kw_list:
            parts = kw.split(",")
            if parts:
                zh_keywords.append(parts[0])
        
        # 提取论文关键词
        paper_keywords = extract_top_tfidf_keywords(paper, top_k=10, lang="en")
        
        # 中文关键词需翻译为英文再比对
        translated = [translate_zh_to_en(kw) for kw in zh_keywords]
        overlap = len(set(translated) & set(paper_keywords))
        
        return min(1.0, overlap / max(1, len(zh_keywords)))

    # --- 5. Algorithm Keywords Accuracy ---
    def reward_kw_algorithm(keywords: str, paper: str) -> float:
        if not keywords:
            return 0.0
        
        # 解析关键词字符串
        kw_list = [kw.strip() for kw in keywords.split(";")]
        if not kw_list:
            return 0.0
        
        # 检查是否包含论文中提出的新方法名称（通常出现在标题/摘要）
        proposed_method = extract_proposed_method_name(paper)  # e.g., "LightGCN"
        if not proposed_method:
            return 0.5  # 无法提取时给中等分
        
        # 提取所有关键词部分
        all_kw_parts = []
        for kw in kw_list:
            parts = kw.split(",")
            all_kw_parts.extend(parts)
        
        # 检查是否包含提出的方法名
        match = any(proposed_method.lower() in kw.lower() for kw in all_kw_parts)
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

# 解析生成的输出字符串为JSON
def parse_generated_output(generated_text: str) -> Optional[Dict[str, Any]]:
    try:
        # 尝试直接解析
        return json.loads(generated_text)
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None

# 论文总结奖励函数
def paper_summary_reward_function(
    response: str,
    answer: str = None,
    paper_text: str = None
) -> Dict[str, Any]:
    """Reward function for paper summary tasks.
    
    Args:
        response: Generated output from the model
        answer: Ground truth answer (not used in this case)
        paper_text: Original paper text
        
    Returns:
        Dict containing reward and reward info
    """
    if not paper_text:
        return {
            "reward": 0.0,
            "reward_info": {
                "error": "Missing paper_text"
            }
        }
    
    # 解析生成的输出
    generated_output = parse_generated_output(response)
    if not generated_output:
        return {
            "reward": 0.0,
            "reward_info": {
                "error": "Invalid JSON format"
            }
        }
    
    # 计算奖励
    reward = compute_reward(paper_text, generated_output, use_llm_judge=False)  # 初期使用规则模式
    
    return {
        "reward": reward,
        "reward_info": {
            "reward": reward
        }
    }
