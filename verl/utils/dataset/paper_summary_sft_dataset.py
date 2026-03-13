# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Paper Summary SFT Dataset

专门为论文总结任务设计的SFT数据集类。
数据格式基于verl RL格式，包含prompt和reward_model.ground_truth字段。
"""

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PaperSummarySFTDataset(Dataset):
    """
    论文总结任务的SFT数据集

    期望的数据格式（Parquet）：
    {
        "data_source": "paper_summary",
        "prompt": [
            {"role": "system", "content": "系统提示词"},
            {"role": "user", "content": "用户输入（包含论文全文）"}
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": "<JSON字符串>",  # 标准答案
            "ground_truth_dict": {...}        # 解析后的字典
        },
        "extra_info": {
            "split": "train",
            "index": 0,
            "original_instruction": "...",
            "paper_title": "..."
        }
    }

    Args:
        parquet_files: Parquet文件路径或路径列表
        tokenizer: 用于文本token化的tokenizer
        config: 配置选项，包含max_length, truncation, pad_mode等
        processor: 多模态预处理器（可选）
        max_samples: 最大样本数，-1表示使用所有样本
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # 设置默认配置
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.max_prompt_length = config.get("max_prompt_length", 4096)
        self.max_response_length = config.get("max_response_length", 1024)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples

        # 数据字段配置
        self.prompt_key = config.get("prompt_key", "prompt")
        self.ground_truth_key = config.get("ground_truth_key", "ground_truth")
        self.ground_truth_dict_key = config.get("ground_truth_dict_key", "ground_truth_dict")
        self.data_source_key = config.get("data_source_key", "data_source")
        self.extra_info_key = config.get("extra_info_key", "extra_info")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._download()
        self._read_files_and_process()

    def _download(self):
        """下载远程文件到本地"""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        """读取并处理Parquet文件"""
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"PaperSummarySFTDataset loaded: {total} samples")

        # 限制样本数
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"Selected {self.max_samples} samples out of {total}")

        # 提取prompt列表
        if self.prompt_key in self.dataframe.columns:
            self.prompts = self.dataframe[self.prompt_key].apply(
                convert_nested_value_to_list_recursive
            ).tolist()
        else:
            raise ValueError(f"Required column '{self.prompt_key}' not found in dataset")

        # 提取ground_truth
        self.ground_truths = self._extract_ground_truths()

        # 提取extra_info（可选）
        if self.extra_info_key in self.dataframe.columns:
            self.extra_infos = self.dataframe[self.extra_info_key].apply(
                lambda x: convert_nested_value_to_list_recursive(x) if isinstance(x, (dict, list)) else x
            ).tolist()
        else:
            self.extra_infos = [None] * len(self.dataframe)

        # 提取data_source（可选）
        if self.data_source_key in self.dataframe.columns:
            self.data_sources = self.dataframe[self.data_source_key].tolist()
        else:
            self.data_sources = ["paper_summary"] * len(self.dataframe)

    def _extract_ground_truths(self) -> list[str]:
        """
        从reward_model字段中提取ground_truth
        支持两种格式：
        1. reward_model.ground_truth: JSON字符串
        2. reward_model.ground_truth_dict: 字典对象
        """
        ground_truths = []

        if "reward_model" not in self.dataframe.columns:
            raise ValueError("Required column 'reward_model' not found in dataset")

        for idx, row in self.dataframe.iterrows():
            reward_model = row["reward_model"]

            if isinstance(reward_model, dict):
                # 优先使用ground_truth_dict
                if self.ground_truth_dict_key in reward_model:
                    gt_dict = reward_model[self.ground_truth_dict_key]
                    if isinstance(gt_dict, dict):
                        ground_truths.append(json.dumps(gt_dict, ensure_ascii=False))
                    else:
                        ground_truths.append(str(gt_dict))
                elif self.ground_truth_key in reward_model:
                    gt = reward_model[self.ground_truth_key]
                    if isinstance(gt, str):
                        ground_truths.append(gt)
                    else:
                        ground_truths.append(json.dumps(gt, ensure_ascii=False))
                else:
                    raise ValueError(
                        f"Row {idx}: 'reward_model' must contain '{self.ground_truth_key}' "
                        f"or '{self.ground_truth_dict_key}'"
                    )
            elif isinstance(reward_model, str):
                # 尝试解析为JSON
                try:
                    rm_dict = json.loads(reward_model)
                    if self.ground_truth_key in rm_dict:
                        ground_truths.append(rm_dict[self.ground_truth_key])
                    elif self.ground_truth_dict_key in rm_dict:
                        gtd = rm_dict[self.ground_truth_dict_key]
                        ground_truths.append(json.dumps(gtd, ensure_ascii=False))
                    else:
                        ground_truths.append(reward_model)
                except json.JSONDecodeError:
                    ground_truths.append(reward_model)
            else:
                raise ValueError(f"Row {idx}: 'reward_model' must be dict or string")

        return ground_truths

    def __len__(self):
        return len(self.prompts)

    def _build_messages(self, prompt: list[dict], ground_truth: str) -> list[dict]:
        """
        构建多轮对话消息列表

        将prompt（系统提示+用户输入）和ground_truth（助手回复）组合成
        标准的多轮对话格式。

        Args:
            prompt: 输入prompt列表，包含system和user消息
            ground_truth: 标准答案JSON字符串

        Returns:
            messages: 多轮对话消息列表
        """
        messages = []

        # 添加prompt中的消息
        for msg in prompt:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            else:
                logger.warning(f"Invalid message format: {msg}")

        # 添加assistant的回复（ground_truth）
        messages.append({
            "role": "assistant",
            "content": ground_truth
        })

        return messages

    def _tokenize_messages(self, messages: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对消息列表进行tokenize

        Args:
            messages: 多轮对话消息列表

        Returns:
            input_ids: token ID列表
            attention_mask: 注意力掩码
            loss_mask: 损失掩码（只计算assistant回复部分的损失）
        """
        processor = self.processor if self.processor is not None else self.tokenizer

        # 使用apply_chat_template处理多轮对话
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,  # 不添加生成提示，因为已经有assistant回复
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # 构建loss_mask：只计算assistant回复部分的损失
        # 方法：单独tokenize每一轮消息，确定assistant回复的起始位置
        loss_mask = torch.zeros_like(attention_mask)

        # 找到assistant消息的位置
        # 策略：遍历所有消息，找到最后一个assistant消息（ground_truth）
        # 从该位置开始到结尾的token都计算损失
        current_pos = 0
        for i, msg in enumerate(messages):
            # 单独tokenize到当前消息（不包含当前消息）
            prefix_messages = messages[:i]
            if prefix_messages:
                prefix_inputs = processor.apply_chat_template(
                    prefix_messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prefix_length = prefix_inputs["input_ids"].shape[1]
            else:
                prefix_length = 0

            # 如果当前是assistant消息，标记为计算损失
            if msg.get("role") == "assistant":
                # tokenize到当前消息（包含当前消息）
                current_inputs = processor.apply_chat_template(
                    messages[:i+1],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                current_length = current_inputs["input_ids"].shape[1]
                # 标记assistant回复部分
                loss_mask[prefix_length:current_length] = 1

        return input_ids, attention_mask, loss_mask

    def _truncate_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        截断序列以适应最大长度限制

        策略：
        1. 优先保证prompt完整
        2. 如果总长度超过限制，从response部分截断
        3. 如果prompt本身超过限制，根据truncation策略处理
        """
        total_length = input_ids.shape[0]

        if total_length <= self.max_length:
            return input_ids, attention_mask, loss_mask

        # 如果prompt部分超过最大长度
        if prompt_length >= self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"Prompt length ({prompt_length}) exceeds max_length ({self.max_length}). "
                    f"Consider increasing max_length or truncating prompt."
                )
            elif self.truncation == "left":
                # 从左侧截断
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "right":
                # 从右侧截断
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
        else:
            # 保留完整prompt，从response截断
            keep_length = self.max_length
            input_ids = input_ids[:keep_length]
            attention_mask = attention_mask[:keep_length]
            loss_mask = loss_mask[:keep_length]

        return input_ids, attention_mask, loss_mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        获取单个样本

        Returns:
            包含以下字段的字典：
            - input_ids: token ID张量
            - attention_mask: 注意力掩码
            - position_ids: 位置编码ID
            - loss_mask: 损失掩码
            - data_source: 数据来源（可选）
            - extra_info: 额外信息（可选）
        """
        prompt = self.prompts[idx]
        ground_truth = self.ground_truths[idx]
        data_source = self.data_sources[idx]
        extra_info = self.extra_infos[idx]

        # 构建多轮对话消息
        messages = self._build_messages(prompt, ground_truth)

        # Tokenize
        input_ids, attention_mask, loss_mask = self._tokenize_messages(messages)

        # 计算prompt长度（用于截断策略）
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        processor = self.processor if self.processor is not None else self.tokenizer
        prompt_inputs = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,  # 添加生成提示
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]

        # 截断处理
        input_ids, attention_mask, loss_mask = self._truncate_sequences(
            input_ids, attention_mask, loss_mask, prompt_length
        )

        # 生成position_ids
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        # Padding处理
        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # 右侧padding
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full(
                    (self.max_length - sequence_length,),
                    pad_token_id,
                    dtype=input_ids.dtype
                )
                padded_attention_mask = torch.zeros(
                    (self.max_length - sequence_length,),
                    dtype=attention_mask.dtype
                )
                padded_loss_mask = torch.zeros(
                    (self.max_length - sequence_length,),
                    dtype=loss_mask.dtype
                )

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)

        # 构建返回结果
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        # 添加可选字段
        if data_source is not None:
            result["data_source"] = data_source
        if extra_info is not None:
            result["extra_info"] = extra_info

        return result

    def get_ground_truth(self, idx: int) -> str:
        """获取指定索引的ground_truth（用于验证和调试）"""
        return self.ground_truths[idx]

    def get_prompt(self, idx: int) -> list[dict]:
        """获取指定索引的prompt（用于验证和调试）"""
        return self.prompts[idx]

    def validate_sample(self, idx: int) -> dict:
        """
        验证单个样本的格式

        Returns:
            验证结果字典，包含is_valid和error信息
        """
        try:
            prompt = self.prompts[idx]
            ground_truth = self.ground_truths[idx]

            # 验证prompt格式
            if not isinstance(prompt, list):
                return {"is_valid": False, "error": f"prompt must be a list, got {type(prompt)}"}

            for msg in prompt:
                if not isinstance(msg, dict):
                    return {"is_valid": False, "error": f"Each message must be a dict, got {type(msg)}"}
                if "role" not in msg or "content" not in msg:
                    return {"is_valid": False, "error": f"Message must have 'role' and 'content' keys: {msg}"}

            # 验证ground_truth是有效的JSON
            try:
                gt_dict = json.loads(ground_truth)
                required_fields = ["summary", "algorithm", "compare_result", "keyword_problem", "keyword_algorithm"]
                missing_fields = [f for f in required_fields if f not in gt_dict]
                if missing_fields:
                    return {
                        "is_valid": False,
                        "error": f"ground_truth missing required fields: {missing_fields}"
                    }
            except json.JSONDecodeError as e:
                return {"is_valid": False, "error": f"ground_truth is not valid JSON: {e}"}

            return {"is_valid": True, "error": None}

        except Exception as e:
            return {"is_valid": False, "error": str(e)}
