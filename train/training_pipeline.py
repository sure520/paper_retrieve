"""
完整训练流程脚本 - SFT + GRPO两阶段训练
适用于Qwen3-4B模型，4卡4090配置
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import yaml


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_sft_data(data_path: str, tokenizer, max_length: int = 2048):
    """准备SFT训练数据"""
    
    def format_instruction(sample):
        """格式化指令数据"""
        instruction = sample["instruction"]
        output = sample["output"]
        
        # Qwen3的chat template
        messages = [
            {"role": "system", "content": "你是一个专业的学术论文分析专家，擅长提取论文的核心思想、算法细节和关键词。"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
        
        return {
            "prompt": prompt,
            "completion": output,
            "full_text": full_text
        }
    
    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 格式化数据
    formatted_data = [format_instruction(item) for item in raw_data]
    
    # 创建数据集
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["full_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        # 设置labels（用于计算loss）
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # 将prompt部分的labels设为-100（不计算loss）
        prompt_tokens = tokenizer(
            examples["prompt"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        prompt_length = len([t for t in prompt_tokens["input_ids"] if t != tokenizer.pad_token_id])
        
        for i in range(len(model_inputs["labels"])):
            model_inputs["labels"][i][:prompt_length] = [-100] * prompt_length
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def sft_training(config: dict, data_path: str, output_dir: str):
    """
    第一阶段：SFT监督微调
    让模型学习生成符合格式的JSON输出
    """
    print("=" * 60)
    print("🚀 开始SFT训练阶段")
    print("=" * 60)
    
    model_path = config["model"]["pretrained_model_path"]
    
    # 加载模型和tokenizer
    print(f"📥 加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据
    print("📊 准备训练数据...")
    dataset = prepare_sft_data(data_path, tokenizer)
    
    # 分割训练集和验证集
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"✅ 训练集: {len(train_dataset)}条, 验证集: {len(eval_dataset)}条")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else None,
        run_name=f"{config.get('wandb', {}).get('name', 'sft')}-sft"
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("🏃 开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"💾 保存SFT模型到: {output_dir}/sft_final")
    trainer.save_model(f"{output_dir}/sft_final")
    tokenizer.save_pretrained(f"{output_dir}/sft_final")
    
    print("✅ SFT训练完成!")
    return f"{output_dir}/sft_final"


def grpo_training(config: dict, sft_model_path: str):
    """
    第二阶段：GRPO强化学习训练
    使用SFT模型作为初始策略和参考策略
    """
    print("=" * 60)
    print("🚀 开始GRPO训练阶段")
    print("=" * 60)
    
    # 更新配置，使用SFT模型
    config["model"]["pretrained_model_path"] = sft_model_path
    config["model"]["ref_model_path"] = sft_model_path
    
    # 保存更新后的配置
    config_path = "config_grpo.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    print(f"📋 配置已更新: {config_path}")
    print("⚠️ 请手动启动GRPO训练进程:")
    print("  1. 启动采样进程: python sampling_worker.py --config config_grpo.yaml")
    print("  2. 启动训练进程: deepspeed --num_gpus=3 training_worker.py --config config_grpo.yaml")


def main():
    parser = argparse.ArgumentParser(description="论文总结模型训练流程")
    parser.add_argument("--config", default="config_qwen3_4b.yaml", help="配置文件路径")
    parser.add_argument("--data", default="train/data/paper_train.json", help="训练数据路径")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--stage", choices=["sft", "grpo", "all"], default="all", 
                       help="训练阶段: sft(仅SFT), grpo(仅GRPO), all(完整流程)")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输出目录: {run_dir}")
    
    # 执行训练阶段
    sft_model_path = None
    
    if args.stage in ["sft", "all"]:
        sft_model_path = sft_training(config, args.data, str(run_dir / "sft"))
    
    if args.stage in ["grpo", "all"]:
        if sft_model_path is None:
            # 如果跳过SFT，需要指定SFT模型路径
            sft_model_path = config["model"]["pretrained_model_path"]
            print(f"⚠️ 使用预训练模型作为初始策略: {sft_model_path}")
        
        grpo_training(config, sft_model_path)


if __name__ == "__main__":
    main()
