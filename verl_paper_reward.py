"""
VERL框架的奖励函数实现 - 论文总结任务
与VERL的reward_model接口兼容
"""

import re
import json
import torch
from typing import Dict, Any, List, Optional
from verl.utils.reward_score import compute_score


class PaperSummaryRewardFunction:
    """
    论文总结任务的奖励函数
    适配VERL框架的reward函数接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get("weights", {
            "summary": 0.30,
            "algorithm": 0.25,
            "comparison": 0.25,
            "keyword_problem": 0.10,
            "keyword_algorithm": 0.10
        })
        self.use_llm_judge = config.get("use_llm_judge", False)
    
    def __call__(self, prompts: List[str], responses: List[str], 
                 ground_truths: List[str]) -> torch.Tensor:
        """
        VERL框架调用的奖励计算接口
        
        Args:
            prompts: 输入prompt列表
            responses: 模型生成的响应列表
            ground_truths: 标准答案列表（JSON格式字符串）
            
        Returns:
            rewards: 奖励值张量 [batch_size]
        """
        rewards = []
        
        for prompt, response, ground_truth in zip(prompts, responses, ground_truths):
            reward = self.compute_reward(prompt, response, ground_truth)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def compute_reward(self, prompt: str, response: str, 
                       ground_truth: str) -> float:
        """计算单个样本的奖励"""
        
        # 1. 检查JSON格式
        parsed_response = self._parse_json(response)
        if parsed_response is None:
            return 0.0  # JSON格式错误，直接惩罚
        
        # 2. 检查必需字段
        required_keys = ["summary", "algorithm", "compare_result", 
                        "keyword_problem", "keyword_algorithm"]
        if not all(k in parsed_response for k in required_keys):
            return 0.1  # 字段不完整，轻度惩罚
        
        # 3. 解析ground truth
        try:
            parsed_gt = json.loads(ground_truth)
        except:
            parsed_gt = {}
        
        # 4. 计算各维度奖励
        r_summary = self._reward_summary(
            parsed_response.get("summary", ""),
            parsed_gt.get("summary", "")
        )
        
        r_algorithm = self._reward_algorithm(
            parsed_response.get("algorithm", ""),
            parsed_gt.get("algorithm", "")
        )
        
        r_comparison = self._reward_comparison(
            parsed_response.get("compare_result", ""),
            parsed_gt.get("compare_result", "")
        )
        
        r_kw_problem = self._reward_keywords(
            parsed_response.get("keyword_problem", ""),
            parsed_gt.get("keyword_problem", "")
        )
        
        r_kw_algorithm = self._reward_keywords(
            parsed_response.get("keyword_algorithm", ""),
            parsed_gt.get("keyword_algorithm", "")
        )
        
        # 5. 加权求和
        total_reward = (
            self.weights["summary"] * r_summary +
            self.weights["algorithm"] * r_algorithm +
            self.weights["comparison"] * r_comparison +
            self.weights["keyword_problem"] * r_kw_problem +
            self.weights["keyword_algorithm"] * r_kw_algorithm
        )
        
        return max(0.0, min(1.0, total_reward))
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            # 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    return None
            return None
    
    def _reward_summary(self, summary: str, ground_truth: str) -> float:
        """评估摘要质量"""
        if not summary:
            return 0.0
        
        score = 0.5
        
        # 检查结构要素
        has_problem = any(w in summary for w in ["解决", "针对", "挑战", "问题"])
        has_method = any(w in summary for w in ["提出", "设计", "引入", "方法"])
        has_result = any(w in summary for w in ["提升", "优于", "达到", "准确率"])
        
        structure_score = (has_problem + has_method + has_result) / 3.0
        score += 0.3 * structure_score
        
        # 长度检查
        if 50 <= len(summary) <= 500:
            score += 0.2
        
        return min(1.0, score)
    
    def _reward_algorithm(self, algorithm: str, ground_truth: str) -> float:
        """评估算法描述质量"""
        if not algorithm:
            return 0.0
        
        score = 0.4
        
        # 检查创新点描述
        innovation_words = ["创新", "不同于", "首次", "改进", "新", "提出"]
        has_innovation = any(w in algorithm for w in innovation_words)
        if has_innovation:
            score += 0.3
        
        # 检查步骤描述
        step_words = ["步骤", "首先", "然后", "最后", "流程", "过程"]
        has_steps = any(w in algorithm for w in step_words)
        if has_steps:
            score += 0.3
        
        return min(1.0, score)
    
    def _reward_comparison(self, comparison: str, ground_truth: str) -> float:
        """评估对比结果质量"""
        if not comparison or "无" in comparison:
            return 0.0
        
        score = 0.3
        
        # 检查是否包含具体指标
        has_metric = bool(re.search(r'\d+(\.\d+)?%', comparison))
        if has_metric:
            score += 0.4
        
        # 检查是否提到对比方法
        baseline_words = ["对比", "相比", "优于", "高于", "低于"]
        has_baseline = any(w in comparison for w in baseline_words)
        if has_baseline:
            score += 0.3
        
        return min(1.0, score)
    
    def _reward_keywords(self, keywords: str, ground_truth: str) -> float:
        """评估关键词质量"""
        if not keywords:
            return 0.0
        
        # 解析关键词
        kw_list = [kw.strip() for kw in keywords.split(";") if kw.strip()]
        if not kw_list:
            return 0.0
        
        score = 0.5
        
        # 检查格式（中英文对照）
        well_formatted = sum(1 for kw in kw_list if "," in kw)
        format_score = well_formatted / max(1, len(kw_list))
        score += 0.3 * format_score
        
        # 数量检查
        if 3 <= len(kw_list) <= 10:
            score += 0.2
        
        return min(1.0, score)


# VERL框架要求的函数接口
def make_reward_function(config: Dict[str, Any]):
    """创建奖励函数实例"""
    return PaperSummaryRewardFunction(config)


def compute_verl_reward(prompts: List[str], responses: List[str], 
                        ground_truths: List[str], config: Dict[str, Any]) -> torch.Tensor:
    """
    VERL框架调用的全局奖励计算函数
    """
    reward_fn = PaperSummaryRewardFunction(config)
    return reward_fn(prompts, responses, ground_truths)
