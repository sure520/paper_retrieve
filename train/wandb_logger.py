"""
WandB实验追踪模块 - 用于GRPO/GSPO训练过程监控
"""

import wandb
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class WandbLogger:
    """WandB日志记录器，支持GRPO/GSPO训练指标追踪"""
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled and config.get("wandb", {}).get("enabled", True)
        self.config = config
        
        if self.enabled:
            wandb_config = config.get("wandb", {})
            
            # 初始化WandB
            wandb.init(
                project=wandb_config.get("project", "paper-summary-grpo"),
                name=wandb_config.get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                tags=wandb_config.get("tags", ["grpo", "paper-summary"]),
                config=config,
                reinit=True
            )
            
            # 定义指标汇总方式
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")
            
            self.log_interval = wandb_config.get("log_interval", 10)
            self.log_artifacts = wandb_config.get("log_artifacts", True)
            
            print(f"✅ WandB已初始化: {wandb.run.url}")
    
    def log_train_step(self, step: int, metrics: Dict[str, float]):
        """记录训练步骤指标"""
        if not self.enabled or step % self.log_interval != 0:
            return
        
        log_dict = {"train/step": step}
        
        # 核心训练指标
        if "loss" in metrics:
            log_dict["train/loss"] = metrics["loss"]
        if "reward" in metrics:
            log_dict["train/reward"] = metrics["reward"]
        if "reward_std" in metrics:
            log_dict["train/reward_std"] = metrics["reward_std"]
        
        # GRPO/GSPO特定指标
        if "kl_divergence" in metrics:
            log_dict["train/kl_divergence"] = metrics["kl_divergence"]
        if "entropy" in metrics:
            log_dict["train/entropy"] = metrics["entropy"]
        if "importance_ratio_mean" in metrics:
            log_dict["train/importance_ratio_mean"] = metrics["importance_ratio_mean"]
        if "importance_ratio_std" in metrics:
            log_dict["train/importance_ratio_std"] = metrics["importance_ratio_std"]
        
        # 学习率
        if "learning_rate" in metrics:
            log_dict["train/learning_rate"] = metrics["learning_rate"]
        
        # 奖励分解（如果可用）
        for key in ["reward_summary", "reward_algorithm", "reward_comparison", 
                    "reward_keyword_problem", "reward_keyword_algorithm"]:
            if key in metrics:
                log_dict[f"train/{key}"] = metrics[key]
        
        wandb.log(log_dict)
    
    def log_eval_step(self, step: int, metrics: Dict[str, Any]):
        """记录评估指标"""
        if not self.enabled:
            return
        
        log_dict = {"train/step": step}
        
        # 准确率指标
        if "accuracy" in metrics:
            log_dict["eval/accuracy"] = metrics["accuracy"]
        if "format_accuracy" in metrics:
            log_dict["eval/format_accuracy"] = metrics["format_accuracy"]
        if "answer_accuracy" in metrics:
            log_dict["eval/answer_accuracy"] = metrics["answer_accuracy"]
        
        # ROUGE分数（用于文本生成质量）
        if "rouge1" in metrics:
            log_dict["eval/rouge1"] = metrics["rouge1"]
        if "rouge2" in metrics:
            log_dict["eval/rouge2"] = metrics["rouge2"]
        if "rougeL" in metrics:
            log_dict["eval/rougeL"] = metrics["rougeL"]
        
        # JSON格式正确率
        if "json_valid_rate" in metrics:
            log_dict["eval/json_valid_rate"] = metrics["json_valid_rate"]
        
        # 字段完整率
        if "field_completeness" in metrics:
            log_dict["eval/field_completeness"] = metrics["field_completeness"]
        
        wandb.log(log_dict)
    
    def log_sample_outputs(self, step: int, samples: list):
        """记录模型生成样本（用于人工检查）"""
        if not self.enabled:
            return
        
        # 创建表格展示生成样本
        columns = ["step", "input_prompt", "generated_output", "ground_truth", "reward"]
        data = []
        
        for sample in samples[:5]:  # 只记录前5个样本
            data.append([
                step,
                sample.get("prompt", "")[:200] + "...",
                sample.get("output", "")[:500] + "...",
                sample.get("ground_truth", "")[:500] + "...",
                sample.get("reward", 0)
            ])
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"generated_samples": table}, step=step)
    
    def log_checkpoint(self, ckpt_path: str, step: int, metadata: Optional[Dict] = None):
        """保存模型checkpoint到WandB"""
        if not self.enabled or not self.log_artifacts:
            return
        
        artifact = wandb.Artifact(
            name=f"model-checkpoint-step-{step}",
            type="model",
            metadata=metadata or {"step": step}
        )
        
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_file():
            artifact.add_file(str(ckpt_path))
        elif ckpt_path.is_dir():
            artifact.add_dir(str(ckpt_path))
        
        wandb.log_artifact(artifact)
        print(f"📦 Checkpoint已保存到WandB: step {step}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录完整配置"""
        if not self.enabled:
            return
        
        # 保存配置为JSON文件并上传
        config_path = Path("config_logged.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        wandb.save(str(config_path))
    
    def finish(self):
        """结束WandB会话"""
        if self.enabled:
            wandb.finish()
            print("✅ WandB会话已结束")


# 奖励函数指标追踪器
class RewardMetricsTracker:
    """专门用于追踪奖励函数各维度的指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            "summary": [],
            "algorithm": [],
            "comparison": [],
            "keyword_problem": [],
            "keyword_algorithm": [],
            "total": []
        }
    
    def update(self, rewards: Dict[str, float]):
        """更新奖励指标"""
        for key in self.metrics.keys():
            if key in rewards:
                self.metrics[key].append(rewards[key])
    
    def get_averages(self) -> Dict[str, float]:
        """获取平均奖励值"""
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.metrics.items()
        }
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取详细统计信息"""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                import numpy as np
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        return stats
