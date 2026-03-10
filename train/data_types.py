from torch.utils.data import Dataset
from transformers import AutoTokenizer
#from tokenizers import Tokenizer
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import threading
from collections import deque
import torch
import json

# 论文总结系统提示
PAPER_SUMMARY_SYSTEM_PROMPT = """你是一个学术论文总结专家，你的任务是给定你一篇论文，你需要总结出该论文的核心思想、核心算法方案细节以及该论文的关键词。
#任务
1. 分析该论文的应用场景，解决了什么问题，提出了什么方法，达到了什么样的效果等核心内容。以上几项如果没有则不用总结，比如应用场景没有则可以不总结。
2. 分析该论文中提出的算法，详细总结该算法的核心思想和创新点，与其他算法的不同，并最后总结其具体的实现步骤。
3. 罗列和分析核心对比的算法，以及本论文提出的算法的各自效果。能让人清晰的看到对比的效果。对比算法如果有缩写就用其缩写，如果没有就给出该算法的描述。
4. 分析能代表该论文的关键词，关键词包括应用的场景、研究的业务问题、提出的算法、核心使用的算法(不包括评估指标)、研究的领域，包括其中文名称、其英文全称、及其英文缩写（如果没有英文缩写则不写）。中文、英文全称和缩写用逗号隔开。
请按照以上步骤一步步分析。

#任务补充说明
1. 任务3中分析论文核心对比算法时如果该算法在论文中有相应的简称则使用其简称，如果没有则使用其引用格式。
2. 论文的关键词一定要是跟本论文主题明确相关，能代表该论文方向、领域、提出的算法的关键词。通过该关键词就能了解该论文大致的研究内容和方法。

#输出要求
必须以json格式输出，格式为{{"summary":"论文概览，分析论文的场景、解决的问题、提出的方法等","algorithm":"论文提出的算法，及其明确的介绍","compare_result":"核心对比的算法及其对比结果","keyword_problem":"论文研究的场景和业务问题的关键词，同一个keyword的中英文和简称用逗号','分隔，多个keyword之间用分号';'分隔","keyword_algorithm":"论文提出的和使用的算法的关键词，能够代表该论文核心方法的关键词，如果该论文提出的算法有简称或者全称，也必须输出，同一个keyword的中英文和简称用逗号','分隔，多个keyword之间用分号';'分隔。"}}

#论文
{paper_text}"""

SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, data):
        with self.lock:
            self.buffer.append(data)
    
    def sample(self, batch_size=32):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    prefix_tokens: List[str]
    prefix_token_ids: List[int]
    generated_token_ids: List[int]
    whole_token_ids: list[int]
    is_finished: bool
    text: str
    reward: float
    reward_info: Dict[str, float]

    old_policy_log_probs: np.ndarray
    ref_policy_log_probs: np.ndarray

@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    question: list[str]
    answer: list[str]

class Gsm8kTasksDataset(Dataset):
    """Prepare GSM8K Tasks for training"""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "main")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["question"]))
        return item

    def encode_prefix(self, question: str):
        """Prefix is the *actual* input to the model."""
        prefix = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=False
        )
        tokens = self.tokenizer.tokenize(prefix)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens,
            "prefix_token_ids": tokens_ids
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        question = [item["question"] for item in batch]
        answer = [item["answer"].split('####')[-1].strip() for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            question=question,
            answer=answer,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )

class PaperSummaryDataset(Dataset):
    """Prepare Paper Summary Tasks for training"""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        """
        初始化PaperSummaryDataset数据集

        参数:
        tokenizer (AutoTokenizer): 用于对文本进行tokenize的tokenizer
        data_path (str): 数据集文件路径，支持JSON和JSONL格式
        split (str, 可选): 数据集类型划分，"train"或"test"，默认"train"
        test_size (int, 可选): 用于测试集的样本数，默认100
        """
        self.file_path = Path(data_path)
        if self.file_path.suffix == '.json':
            self.json_to_jsonl(str(self.file_path), str(self.file_path.with_suffix('.jsonl')))
            self.file_path = self.file_path.with_suffix('.jsonl')
        # 读取JSONL文件, 转换为DataFrame
        self.data = pd.read_json(self.file_path, lines=True, encoding='utf-8')

        # 检验是否包含充分的测试样本
        if len(self.data) < test_size:
            raise ValueError(f"测试集样本数不足{test_size}个，当前样本数{len(self.data)}")
        
        # 使用最后`test_size`个样本用于测试集
        if use_test_set and split == "train":
            self.data = self.data.iloc[:-test_size]
        else:
            self.data = self.data.iloc[-test_size:]
        
        self.tokenizer = tokenizer

    def json_to_jsonl(self, json_file: str, jsonl_file: str):
        """将JSON文件转换为JSONL文件"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        # 对于论文总结任务，instruction字段包含论文文本，output字段包含Ground Truth JSON
        paper_text_with_instruction = item.get("instruction", "")
        item["question"] = paper_text_with_instruction  # 将指令+论文文本作为question字段
        item["answer"] = item.get("output", "")
        item.update(self.encode_prefix(paper_text_with_instruction))
        return item

    def encode_prefix(self, paper_text_with_instruction: str):
        """Prefix is the *actual* input to the model."""
        prefix = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是一个有帮助的助手"},
                {"role": "user", "content": paper_text_with_instruction}
            ], tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.tokenize(prefix)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens,
            "prefix_token_ids": tokens_ids
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        question = [item["question"] for item in batch]
        answer = [item["answer"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            question=question,
            answer=answer,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )