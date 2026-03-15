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
SFT (Supervised Fine-Tuning) 训练器模块

本模块实现了监督微调训练流程，用于训练大语言模型。
主要功能包括：
- 构建训练数据集和数据加载器
- 初始化训练引擎和优化器
- 执行训练循环和验证
- 保存检查点和恢复训练

使用方式：
    python sft_trainer.py

配置文件位于 config/sft_trainer_engine.yaml
"""

import os
from functools import partial

from tensordict.tensorclass import NonTensorData

# 设置 NCCL 调试级别为警告，避免过多日志输出
os.environ["NCCL_DEBUG"] = "WARN"
# 启用分词器并行处理，加速数据预处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

# 初始化日志记录器
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    """
    SFT 训练器类
    
    负责管理整个监督微调训练流程，包括：
    - 配置解析和验证
    - 数据集和数据加载器构建
    - 训练引擎初始化
    - 训练循环执行
    - 验证和检查点保存
    
    Attributes:
        config: Hydra 配置对象，包含所有训练参数
        rank: 当前进程的全局排名
        model_config: 模型配置
        engine_config: 引擎配置
        optimizer_config: 优化器配置
        train_dataset: 训练数据集
        val_dataset: 验证数据集（可选）
        train_dataloader: 训练数据加载器
        engine: 训练引擎
        ckpt_handler: 检查点处理器
    """

    def __init__(
        self,
        config,
    ):
        """
        初始化 SFT 训练器
        
        Args:
            config: Hydra 配置对象，包含模型、数据、训练等所有配置参数
        """
        self.config = config

        # 记录初始化前的 GPU 内存使用情况
        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        # 获取当前进程的全局排名
        self.rank = torch.distributed.get_rank()

        # 按顺序构建各个组件
        self._build_config()      # 解析配置
        self._build_dataset()     # 构建数据集
        self._build_engine()      # 构建训练引擎
        self._build_dataloader()  # 构建数据加载器
        self._init_engine()       # 初始化引擎

        self._build_ckpt_handler()  # 构建检查点处理器

        # 从检查点恢复训练，获取恢复的全局步数
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        # 获取设备名称（cuda/npu/cpu）
        self.device_name = self.config.trainer.device

        # 仅在主进程打印配置信息
        if self.rank == 0:
            print(self.config)

        # 记录初始化后的 GPU 内存使用情况
        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        """
        构建检查点处理器
        
        用于保存和加载训练检查点，支持：
        - 自动恢复训练
        - 保留最近 N 个检查点
        - 支持 HDFS 存储
        - 支持 LoRA 训练元数据
        """
        # 获取检查点相关配置
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)
        lora_train_meta = self._get_lora_train_meta()

        # 创建检查点处理器实例
        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            lora_train_meta=lora_train_meta,
        )

    def _get_lora_train_meta(self):
        """
        获取 LoRA 训练元数据
        
        如果启用了 LoRA 微调，返回 LoRA 相关配置信息，
        包括秩（rank）、alpha 缩放因子和任务类型。
        
        Returns:
            dict | None: LoRA 元数据字典，如果未启用 LoRA 则返回 None
        """
        # 检查是否启用 LoRA
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        lora_rank = int(getattr(self.config.model, "lora_rank", 0) or 0)

        # 如果没有配置 LoRA 路径且秩为 0，则未启用 LoRA
        if lora_adapter_path is None and lora_rank <= 0:
            return None

        # 获取 LoRA alpha 参数
        raw_lora_alpha = self.config.model.get("lora_alpha", None)
        if raw_lora_alpha is None:
            log_with_rank(
                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            lora_alpha = 0
        else:
            lora_alpha = int(raw_lora_alpha)
            if lora_alpha == 0:
                log_with_rank(
                    "LoRA is enabled but `model.lora_alpha` is 0; this may lead to ineffective LoRA scaling.",
                    logger=logger,
                    rank=self.rank,
                    level=logging.WARNING,
                    log_only_rank_0=True,
                )

        # 获取任务类型，默认为因果语言模型
        task_type = self.config.model.get("task_type", None)
        if task_type is None:
            task_type = "CAUSAL_LM"

        return {
            "r": lora_rank,
            "lora_alpha": int(lora_alpha or 0),
            "task_type": str(task_type),
        }

    def _build_config(self):
        """
        构建和验证配置对象
        
        将 Hydra 的 OmegaConf 配置转换为数据类对象，
        便于类型检查和属性访问。
        """
        from verl.utils.config import omega_conf_to_dataclass

        # 转换各模块配置
        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # 获取性能分析区间配置
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        """
        验证性能分析区间配置
        
        确保 profiler_interval 是一个有效的二元组，
        且结束步数大于等于开始步数。
        """
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        # 如果开始步数为负数，结束步数也必须为负数（表示不启用）
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        """
        构建训练引擎
        
        初始化训练工作器（TrainingWorker），设置损失函数，
        并获取引擎实例用于后续训练操作。
        """
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        # 设置 SFT 损失函数（交叉熵损失）
        self.loss_fn = partial(sft_loss, config=None)

        # 创建训练工作器配置
        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        # 初始化训练工作器并设置损失函数
        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # 获取引擎实例，用于分布式训练操作
        # 注意：在 SPMD 模式下，这个抽象需要打破
        self.engine = self.training_client.engine

    def _init_engine(self):
        """
        初始化训练引擎
        
        设置总训练步数、每个 epoch 的步数，
        以及保存和验证的频率。
        """
        # 计算总训练步数
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            # 如果未指定，则根据 epoch 数和数据加载器长度计算
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        # 每个 epoch 的步数
        self.steps_per_epoch = len(self.train_dataloader)

        # 设置保存频率
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        # 设置验证频率
        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        # 重置训练客户端状态
        self.training_client.reset()

    def _build_dataset(self):
        """
        构建训练和验证数据集
        
        根据配置创建数据集实例，支持自定义数据集类。
        """
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor

        # 创建训练数据集
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )

        # 创建验证数据集（如果配置了验证文件）
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        """
        构建数据加载器
        
        创建分布式数据采样器和数据加载器，
        支持数据并行训练和状态保存（用于恢复训练）。
        """
        config = self.config

        # 获取设备名称，用于 pin_memory_device 设置
        device_name = get_device_name()

        # 获取数据并行的排名和大小
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        # 创建分布式采样器，确保每个 GPU 获取不同的数据分片
        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        # 计算每个数据并行进程的批次大小
        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size

        # 创建数据整理函数，用于填充和批处理
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        # 创建有状态的数据加载器，支持从检查点恢复
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        # 创建验证数据加载器（如果存在验证集）
        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        """
        获取批次中每个样本的序列长度
        
        计算当前批次中每个样本的实际序列长度，
        并在数据并行组内收集所有样本的长度信息。
        
        Args:
            data: 包含 input_ids 和 attention_mask 的数据字典
            
        Returns:
            list: 所有数据并行进程上的序列长度列表
        """
        # 检查是否为嵌套张量（用于变长序列优化）
        is_nested = data["input_ids"].is_nested
        if is_nested:
            # 嵌套张量通过 offsets 计算长度
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            # 普通张量通过 attention_mask 求和计算长度
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)

        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)

        # 获取数据并行组信息
        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        # 如果只有一个数据并行进程，直接返回
        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        # 在数据并行组内收集所有样本的序列长度
        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        """
        执行训练循环
        
        主要流程：
        1. 初始化日志追踪器
        2. 遍历所有 epoch
        3. 对每个批次执行前向传播、反向传播和参数更新
        4. 定期执行验证和保存检查点
        5. 记录训练指标和性能数据
        """
        # 判断当前进程是否需要记录日志（主进程且有输出）
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # 初始化日志追踪器（WandB/TensorBoard 等）
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        # 从恢复的步数开始训练
        global_step = self.resume_global_step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # 使用 StatefulDataLoader 时，数据加载器会自动从上次中断的位置恢复
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # 计算起始 epoch，用于设置采样器的 epoch
        start_epoch = global_step // self.steps_per_epoch

        # 准备元数据，传递给训练批次
        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0

        # 主训练循环：遍历所有 epoch
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # 设置采样器的 epoch，确保每个 epoch 的数据顺序不同
            self.train_sampler.set_epoch(epoch=epoch)

            # 清理 GPU 缓存
            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            # 遍历当前 epoch 的所有批次
            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # 构建张量字典，包含数据张量和元数据
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                # 将序列长度包装为 NonTensorData，避免被解释为 NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                # 分配非张量数据，用于学习率调度器
                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                # 在指定步数开始性能分析（SPMD 模式）
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                # 执行一个批次的训练
                output = self.training_client.train_batch(data=data)

                # 在指定步数停止性能分析
                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                # 记录训练指标（仅在有输出的进程上）
                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    # 重命名指标，添加 train/ 前缀
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    # 计算当前批次的总 token 数
                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    # 仅在数据并行主进程上记录日志
                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                # 判断是否为最后一步、验证步或保存步
                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # 执行验证（最后一步或达到验证频率时）
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if self.engine.is_mp_src_rank_with_outputs():
                            metrics = tu.get(output, "metrics")
                            val_losses.append(metrics["loss"])

                    # 计算平均验证损失
                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        # 在数据并行组内平均
                        dp_group = self.engine.get_data_parallel_group()
                        if dp_group is not None:
                            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

                    # 记录验证指标
                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric

                    # 同步所有进程
                    torch.distributed.barrier()

                # 保存检查点（最后一步或达到保存频率时）
                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                # 训练完成，退出循环
                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    """
    运行 SFT 训练的主函数
    
    初始化分布式进程组，创建训练器实例并执行训练。
    
    Args:
        config: Hydra 配置对象
    """
    from verl.utils.distributed import initialize_global_process_group

    # 初始化分布式进程组
    initialize_global_process_group()
    # 创建训练器并执行训练
    trainer = SFTTrainer(config=config)
    trainer.fit()
    # 销毁进程组
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    """
    程序入口点
    
    使用 Hydra 进行配置管理，自动从 config/sft_trainer_engine.yaml 加载配置。
    支持命令行参数覆盖配置。
    
    Args:
        config: Hydra 自动注入的配置对象
    """
    # 在 Ascend NPU 上运行时，自动设置设备为 npu
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """
    创建 SFT 数据集
    
    根据配置创建数据集实例，支持使用自定义数据集类。
    
    Args:
        data_paths: 数据文件路径列表
        data_config: 数据配置对象
        tokenizer: 分词器实例
        processor: 数据处理器实例
        max_samples: 最大样本数，-1 表示不限制
        
    Returns:
        数据集实例
    """
    # 检查是否指定了自定义数据集类
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        # 动态加载自定义数据集类
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # 默认使用多轮对话数据集
        dataset_cls = MultiTurnSFTDataset

    # 创建数据集实例
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()