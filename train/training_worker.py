#!/usr/bin/env python3
"""
训练进程脚本：负责接收采样数据并进行模型训练
通过ZeroMQ接收采样进程的数据，支持DeepSpeed分布式训练
"""

import time
import torch
import zmq
import yaml
import json
import pickle
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.distributed as dist
from data_types import Gsm8kTasksDataset, PaperSummaryDataset, Episode
from utils import group_advantages, grpo_loss, gspo_loss, get_batch_log_probs, sample_trajectory
from reward import paper_summary_reward_function
import deepspeed
import numpy as np
from peft import LoraConfig, get_peft_model

class TrainingWorker:
    def __init__(self, config: dict):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        self.gpu_id = config["gpu"]["training_gpu"]
        self.pretrained_model_path = config["model"]["pretrained_model_path"]
        self.ref_model_path = config["model"]["ref_model_path"]
        self.dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
        self.data_path = config["data"]["data_path"]
        self.max_gen_len = config["data"]["max_gen_len"]
        self.train_batch_size = config["data"]["train_batch_size"]
        self.sample_batch_size = config["data"]["sample_batch_size"]
        self.test_size = config["data"]["test_size"]
        self.test_batch_size = config["data"]["test_batch_size"]
        self.data_path = config["data"]["data_path"]
        self.num_answers_per_question = config["data"]["num_answers_per_question"]
        self.num_questions_per_batch = self.train_batch_size // self.num_answers_per_question
        self.use_gspo = config["training"]["use_gspo"]
        self.eval_interval = config["training"]["eval_interval"]
        self.sync_interval = config["training"]["sync_interval"]
        self.max_train_steps = config["training"].get("max_train_steps", 0)
        self.vis_interval = config["training"].get("vis_interval", self.eval_interval)  # 默认与评估间隔相同
        self.zmq_data_port = config["communication"]["data_port"]
        self.ds_config_path = config["deepspeed"]["config_path"]
        self.ckpt_dir = Path(config["checkpoint"]["ckpt_dir"])
        self.ckpt_file = config["checkpoint"]["ckpt_file"]
        self.use_lora = config["lora"]["enabled"]
        self.lora_rank = config["lora"]["rank"]
        self.lora_alpha = config["lora"]["alpha"]
        self.lora_lr = float(config["lora"]["learning_rate"])
        self.lora_dropoutp = config["lora"]["dropout"]
        self.lora_adapter_dir = Path(config["lora"]["adapter_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.lora_adapter_dir.mkdir(parents=True, exist_ok=True)

        # DeepSpeed多进程相关属性(后续初始化模型时会重新设置)
        self.rank = 0
        self.world_size = 1
        self.is_main_process = True
        
        # 日志配置
        self.setup_logging()

        self.setup_model()
        self.setup_zmq()
        
        # 可视化数据存储
        self.train_metrics = {
            'step': [],
            'loss': [],
            'reward': [],
            'accuracy': [],
            'format_accuracy': [],
            'answer_accuracy': [],
            'entropy': []
        }

    def setup_logging(self):
        """配置日志记录"""
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        # 配置日志记录器
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 定义日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if self.is_main_process:
            self.logger.info(f"日志文件已创建: {log_file}")
            self.logger.info(f"训练配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
            
            # 创建可视化目录
            self.vis_dir = Path("visualizations")
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"可视化目录已创建: {self.vis_dir}")

    def setup_model(self):
        """初始化模型和优化器"""
        # 初始化新策略模型
        self.new_policy_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path, 
            dtype=self.dtype, 
            _attn_implementation="sdpa"
        )
        self.new_policy_model.train()
        self.new_policy_model.requires_grad_(True)
        # 启用梯度检查点
        self.new_policy_model.gradient_checkpointing_enable()
        self.logger.info("梯度检查点已启用")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, padding_side='left')
        # 启用LoRA配置和初始化
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.new_policy_model = get_peft_model(self.new_policy_model, lora_config)
            if self.is_main_process:
                self.logger.info("LoRA配置已启用")

        # DeepSpeed配置和初始化
        if self.is_main_process:
            self.logger.info("正在使用DeepSpeed进行优化训练...")

        dist.init_process_group(backend='gloo')  # autodl的vgpu没法用nccl通信, 需要设置为gloo

        # 获取当前进程的rank和world_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = (self.rank == 0)

        if self.is_main_process:
            self.logger.info(f"DeepSpeed进程初始化 - Rank: {self.rank}, World Size: {self.world_size}, Is Main Process: {self.is_main_process}")

        # 加载DeepSpeed配置
        try:
            with open(self.ds_config_path, 'r') as f:
                ds_config = json.load(f)
            if self.is_main_process:
                self.logger.info(f"成功加载DeepSpeed配置: {self.ds_config_path}")
        except FileNotFoundError:
            if self.is_main_process:
                self.logger.error(f"警告: DeepSpeed配置文件 {self.ds_config_path}")
            raise
        except json.JSONDecodeError as e:
            if self.is_main_process:
                self.logger.error(f"错误: DeepSpeed配置文件格式错误: {e}")
            raise

        # 加载模型参数
        model_parameters = list(self.new_policy_model.parameters())

        if self.use_lora:
            ds_config['optimizer']['params']['lr'] = self.lora_lr

        # 初始化DeepSpeed引擎
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.new_policy_model,
            model_parameters=model_parameters,
            config=ds_config
        )

        self.device=self.model_engine.device
        if self.is_main_process:
            self.logger.info(f"DeepSpeed初始化完成，世界大小: {self.model_engine.world_size}")

        # 梯度裁剪参数
        self.max_grad_norm = 1.0
        if self.is_main_process:
            self.logger.info("模型和优化器初始化完成")

    def setup_zmq(self):
        """初始化ZeroMQ通信"""
        self.context = zmq.Context()

        # 只有主进程(rank=0)连接ZMQ，避免多进程竞争
        if self.is_main_process:
            # 数据接收socket（PULL模式）
            self.data_receiver = self.context.socket(zmq.PULL)
            self.data_receiver.connect(f"tcp://localhost:{self.zmq_data_port}")
            self.logger.info(f"ZeroMQ数据接收端口连接: {self.zmq_data_port}")
        else:
            # 非主进程不需要ZMQ连接
            self.data_receiver = None
            self.logger.info(f"Rank {self.rank} 进程跳过ZMQ连接（由rank 0处理）")

    def deserialize_episodes(self, serialized_data):
        """反序列化episodes数据"""
        episodes = []
        for data in serialized_data:
            episode = Episode(
                prefix=data['prefix'],
                prefix_tokens=data['prefix_tokens'],
                prefix_token_ids=data['prefix_token_ids'],
                generated_token_ids=data['generated_token_ids'],
                whole_token_ids=data['whole_token_ids'],
                is_finished=data['is_finished'],
                text=data['text'],
                reward=data['reward'],
                reward_info=data['reward_info'],
                old_policy_log_probs=data['old_policy_log_probs'],
                ref_policy_log_probs=data['ref_policy_log_probs']
            )
            episodes.append(episode)
        return episodes

    def broadcast_episodes(self, episodes):
        """将episodes数据从rank=0广播到所有其他rank"""
        if self.world_size > 1:
            if self.is_main_process:
                serialized_data = pickle.dumps(episodes)
                np_data = np.frombuffer(serialized_data, dtype=np.uint8)
                size_tensor = torch.from_numpy(np.array([len(np_data)], dtype=np.int64)).to(self.device)
                episodes_tensor = torch.from_numpy(np_data.copy()).to(self.device)
            else:
                size_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
                episodes_tensor = None

            # 广播数据长度
            dist.broadcast(size_tensor, src=0)
            data_size = size_tensor.item()

            if not self.is_main_process:
                episodes_tensor = torch.zeros(data_size, dtype=torch.uint8, device=self.device)

            # 广播实际数据（每个rank接收自己的部分）
            dist.broadcast(episodes_tensor, src=0)

            # 反序列化数据
            serialized_data = episodes_tensor.cpu().numpy().tobytes()
            episodes = pickle.loads(serialized_data)

            # 不同的rank分割episodes
            sample_batch_size = len(episodes)
            sample_questions_per_batch = sample_batch_size // self.num_answers_per_question
            num_questions_per_rank = sample_questions_per_batch // self.world_size
            num_data_per_rank = num_questions_per_rank * self.num_answers_per_question

            episodes_per_rank = episodes[self.rank*num_data_per_rank : self.rank*num_data_per_rank+num_data_per_rank]

            return episodes_per_rank
        else:
            # 单进程模式，直接返回
            return episodes

    def train_step(self, episodes):
        """执行一个训练步骤"""
        prefix_len = len(episodes[0].whole_token_ids) - len(episodes[0].generated_token_ids)

        # 计算新策略的概率分布
        batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
        attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()

        new_policy_log_probs = get_batch_log_probs(
            model=self.model_engine,
            batch_token_ids=batch_token_ids,
            attention_mask=attention_mask,
            enable_grad=True  # 新策略训练，需要梯度
        )

        # 计算优势函数
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=self.dtype, device=self.device)
        advantages = group_advantages(rewards=rewards, num_answers_per_question=self.num_answers_per_question).to(self.device)

        # 计算grpo算法的loss
        ref_policy_log_probs = torch.tensor(np.array([episode.ref_policy_log_probs for episode in episodes]), dtype=self.dtype, device=self.device)
        old_policy_log_probs = torch.tensor(np.array([episode.old_policy_log_probs for episode in episodes]), dtype=self.dtype, device=self.device)

        if self.use_gspo:
            loss = gspo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )
        else:
            loss = grpo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )

        # 反向传播和优化步骤
        self.model_engine.backward(loss)
        self.model_engine.step()

        return loss

    def evaluate(self):
        with torch.no_grad():
            self.model_engine.module.eval() # 模型调整为评估模式
            
            # 根据数据路径判断任务类型
            is_paper_summary = True
            
            if is_paper_summary:
                # 加载论文总结评估数据集
                test_dataset = PaperSummaryDataset(
                    data_path=self.data_path,
                    tokenizer=self.tokenizer,
                    split="test",
                    test_size=self.test_size
                )
                collate_fn = PaperSummaryDataset.collate_fn
                current_reward_function = paper_summary_reward_function
            else:
                # 加载GSM8K评估数据集
                test_dataset = Gsm8kTasksDataset(
                    data_path=self.data_path,
                    tokenizer=self.tokenizer,
                    split="test",
                    test_size=self.test_size
                )
                collate_fn = Gsm8kTasksDataset.collate_fn
                from utils import reward_function as current_reward_function
            
            generator = torch.Generator(device="cpu")
            test_dataloader = DataLoader(
                test_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                generator=generator,
                batch_size=self.test_batch_size,
            )

            # 最新模型参数进行采样评估
            reward_sum = 0.0
            entropy_sum = 0.0
            json_success_num = 0
            
            # 收集样本预测结果
            sample_results = []
            max_samples = 5  # 最多保存5个样本
            
            for batch in test_dataloader:
                episodes = sample_trajectory(
                    model=self.model_engine.module,
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_gen_len=self.max_gen_len,
                    num_answer_per_question=1,
                    reward_function=current_reward_function,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # 评估奖励和指标
                for i, episode in enumerate(episodes):
                    reward_sum += episode.reward
                    
                    # 检查JSON格式是否正确
                    if "error" not in episode.reward_info or episode.reward_info["error"] != "Invalid JSON format":
                        json_success_num += 1
                    
                    # 收集样本预测结果
                    if len(sample_results) < max_samples:
                        sample_results.append({
                            'question': batch.question[i] if hasattr(batch, 'question') else "",
                            'prediction': episode.text,
                            'ground_truth': batch.answer[i] if hasattr(batch, 'answer') else "",
                            'reward': episode.reward,
                            'reward_info': episode.reward_info
                        })

                # 评估entropy
                batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
                attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()
                with torch.no_grad():
                    batch_logits = self.model_engine(input_ids=batch_token_ids, attention_mask=attention_mask).logits
                batch_logits = batch_logits[:, :-1, :]       # 去掉最后一个logits, 因为最后一个用于预测下一个token
                batch_probs = torch.softmax(batch_logits, dim=-1)
                batch_log_probs = torch.log(batch_probs + 1e-12)
                batch_token_entropy = -torch.sum(batch_probs * batch_log_probs, dim=-1) # batch_size * seq_len
                batch_entropy = batch_token_entropy.mean(dim=-1)
                entropy_sum += batch_entropy.sum().item()
                
                if len(sample_results) >= max_samples:
                    break

            average_reward = reward_sum / self.test_size
            json_success_rate = json_success_num / self.test_size
            entropy = entropy_sum / self.test_size
            self.model_engine.module.train() # 模型调整回训练模式
            
            # 保存样本预测结果
            self.sample_results = sample_results
            
            # 返回统一格式，适配不同任务
            return average_reward, json_success_rate, json_success_rate, entropy

    def run(self):
        """主运行循环"""
        self.logger.info(f"Rank {self.rank} 开始训练循环...")
        if self.max_train_steps > 0:
            self.logger.info(f"最大训练步数: {self.max_train_steps}")
        else:
            self.logger.info("训练步数不限制（将一直运行直到手动停止）")
        train_step = 0

        # 评估原始模型的准确率
        if self.is_main_process and train_step % self.eval_interval == 0:
            eval_start_time = time.time()
            accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
            self.logger.info(f"第 {train_step} 步训练后评估模型性能, 格式准确率: {format_accuracy}, 回答准确率: {answer_accuracy}, 平均熵: {entropy}, 评估时间: {time.time() - eval_start_time:.2f}")
        
        # 初始化tqdm进度条（仅主进程显示）
        pbar = None
        if self.is_main_process and self.max_train_steps > 0:
            pbar = tqdm(total=self.max_train_steps, desc="训练进度", unit="步", dynamic_ncols=True)
            pbar.set_postfix(loss=0.0, reward=0.0, accuracy=0.0)
        
        # 用于接收广播停止信号的张量（所有 rank 都要有）
        should_stop_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
        
        try:
            while True:
                try:
                    episodes = None

                    # 数据接收逻辑：只有rank=0从ZMQ接收，其他rank等待广播
                    if self.is_main_process:
                        # 主进程从ZMQ接收数据
                        if self.data_receiver and self.data_receiver.poll(100):  # 100ms超时
                            data = self.data_receiver.recv()
                            serialized_episodes = pickle.loads(data)
                            episodes = self.deserialize_episodes(serialized_episodes)
                            sample_batch_size = len(episodes)
                            assert sample_batch_size % self.num_answers_per_question == 0 # 检查sample_batch_size是否能被num_answers_per_question整除
                            sample_questions_per_batch = sample_batch_size // self.num_answers_per_question
                            assert sample_questions_per_batch % self.world_size == 0 # 检查sample_questions_per_batch是否能被world_size整除
                        else:
                            # 没有数据，继续循环
                            time.sleep(0.01)
                            continue

                    # 广播数据到所有rank
                    if self.world_size > 1:
                        episodes = self.broadcast_episodes(episodes)

                    # 如果所有rank都没有数据，继续循环
                    if episodes is None or len(episodes) == 0:
                        time.sleep(0.01)
                        continue

                    rewards = [episode.reward for episode in episodes]
                    avg_reward = sum(rewards) / len(rewards)
                    self.logger.info(f"Rank {self.rank} 开始训练步骤{train_step}，数据批次大小: {len(episodes)}, 平均奖励: {avg_reward:.4f}")

                    loss = self.train_step(episodes)
                    train_step += 1

                    # 记录训练指标
                    if self.is_main_process:
                        self.train_metrics['step'].append(train_step)
                        self.train_metrics['loss'].append(loss.item())
                        self.train_metrics['reward'].append(avg_reward)
                        
                        # 更新tqdm进度条
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix(
                                loss=f"{loss.item():.4f}",
                                reward=f"{avg_reward:.4f}",
                                accuracy=f"{self.train_metrics['accuracy'][-1] if self.train_metrics['accuracy'] else 0.0:.4f}"
                            )

                    # 检查是否达到最大训练步数（仅主进程判断）
                    if self.is_main_process:
                        if self.max_train_steps > 0 and train_step >= self.max_train_steps:
                            self.logger.info(f"Rank {self.rank} 达到最大训练步数 {self.max_train_steps}")
                            should_stop_tensor[0] = 1  # 设置停止标志

                    # 广播停止信号给所有进程
                    dist.broadcast(should_stop_tensor, src=0)
                    
                    # 所有进程检查是否应该停止
                    if should_stop_tensor.item() == 1:
                        self.logger.info(f"Rank {self.rank} 收到停止信号，退出训练循环")
                        break

                    # 定期同步模型参数（只有主进程需要同步到采样进程）
                    if self.is_main_process and train_step % self.sync_interval == 0:
                        if self.use_lora:
                            self.model_engine.save_pretrained(self.lora_adapter_dir)
                            self.logger.info(f"第 {train_step} 步训练后保存LoRA模型参数至 {self.lora_adapter_dir}")
                        else:
                            output_file = self.ckpt_dir / self.ckpt_file
                            torch.save(self.model_engine.module.state_dict(), output_file)
                            self.logger.info(f"第 {train_step} 步训练后保存全量模型参数至 {output_file}")

                    # 定期评估模型性能(只在主进程进行评估)
                    if self.is_main_process and train_step % self.eval_interval == 0:
                        eval_start_time = time.time()
                        accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
                        eval_time = time.time() - eval_start_time
                        self.logger.info(f"第 {train_step} 步训练后评估模型性能, 格式准确率: {format_accuracy:.4f}, 回答准确率: {answer_accuracy:.4f}, 平均熵: {entropy:.4f}, 评估时间: {eval_time:.2f}s")
                        
                        # 记录评估指标
                        self.train_metrics['accuracy'].append(accuracy)
                        self.train_metrics['format_accuracy'].append(format_accuracy)
                        self.train_metrics['answer_accuracy'].append(answer_accuracy)
                        self.train_metrics['entropy'].append(entropy)
                        
                        # 定期更新可视化图像
                        if train_step % self.vis_interval == 0:
                            self.update_visualizations()

                except zmq.Again:
                    # 没有数据，继续循环
                    continue
                except Exception as e:
                    print(f"Rank {self.rank} 训练步骤错误: {e}")
                    continue

                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.info(f"Rank {self.rank} 训练进程收到中断信号")
        except Exception as e:
            self.logger.error(f"Rank {self.rank} 训练进程错误: {e}")
        finally:
            # 关闭进度条
            if pbar:
                pbar.close()
            self.cleanup()

    def plot_loss_reward(self):
        """绘制损失和奖励曲线"""
        if not self.train_metrics['step']:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_metrics['step'], self.train_metrics['loss'], label='损失', color='red')
        plt.xlabel('训练步骤')
        plt.ylabel('损失值')
        plt.title('训练损失曲线')
        plt.grid(True)
        plt.legend()
        
        # 绘制奖励曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_metrics['step'], self.train_metrics['reward'], label='奖励', color='green')
        plt.xlabel('训练步骤')
        plt.ylabel('奖励值')
        plt.title('训练奖励曲线')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.vis_dir / f"loss_reward_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"损失/奖励曲线已保存至: {save_path}")
    
    def plot_accuracy(self):
        """绘制准确率曲线"""
        if not self.train_metrics['accuracy']:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 绘制准确率曲线
        eval_steps = self.train_metrics['step'][::self.eval_interval] if self.train_metrics['step'] else []
        if len(eval_steps) > len(self.train_metrics['accuracy']):
            eval_steps = eval_steps[:len(self.train_metrics['accuracy'])]
        
        plt.plot(eval_steps, self.train_metrics['accuracy'], label='总体准确率', color='blue')
        plt.plot(eval_steps, self.train_metrics['format_accuracy'], label='格式准确率', color='orange')
        plt.plot(eval_steps, self.train_metrics['answer_accuracy'], label='回答准确率', color='purple')
        
        plt.xlabel('训练步骤')
        plt.ylabel('准确率')
        plt.title('训练准确率曲线')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.vis_dir / f"accuracy_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"准确率曲线已保存至: {save_path}")
    
    def plot_sample_predictions(self):
        """绘制样本预测结果对比"""
        if not hasattr(self, 'sample_results') or not self.sample_results:
            return
            
        # 保存样本预测结果到JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.vis_dir / f"sample_predictions_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_results, f, ensure_ascii=False, indent=2)
        
        # 创建可视化图像
        plt.figure(figsize=(15, 10))
        
        for i, result in enumerate(self.sample_results):
            plt.subplot(len(self.sample_results), 1, i+1)
            plt.axis('off')
            
            # 显示问题和预测结果
            text = f"问题: {result['question'][:100]}...\n\n"
            text += f"预测: {result['prediction'][:200]}...\n\n"
            text += f"真实: {result['ground_truth'][:200]}...\n\n"
            text += f"奖励: {result['reward']:.4f}"
            
            plt.text(0.1, 0.9, text, fontsize=10, ha='left', va='top', transform=plt.gca().transAxes, wrap=True)
            plt.title(f"样本 {i+1}", fontsize=12, pad=10)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.vis_dir / f"sample_comparison_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"样本预测结果对比已保存至: {save_path}")
    
    def update_visualizations(self):
        """更新所有可视化图像"""
        if self.is_main_process:
            self.plot_loss_reward()
            self.plot_accuracy()
            self.plot_sample_predictions()
    
    def cleanup(self):
        """清理资源"""
        self.logger.info(f"Rank {self.rank} 清理训练进程资源...")
        
        # 保存最终的可视化图像
        if self.is_main_process:
            self.update_visualizations()

        # 只有主进程有ZMQ连接需要清理
        if self.is_main_process:
            if hasattr(self, 'data_receiver') and self.data_receiver:
                self.data_receiver.close()
            if hasattr(self, 'sync_sender') and self.sync_sender:
                self.sync_sender.close()

        if hasattr(self, 'context'):
            self.context.term()

        self.logger.info(f"Rank {self.rank} 训练进程已清理完成")


def main():
    """主函数"""
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 创建训练进程实例
    worker = TrainingWorker(config=config)

    print("=== GRPO训练进程 ===")
    print(f"数据端口: {config['communication']['data_port']}")
    print(f"DeepSpeed: 启用")
    print(f"当前进程: Rank {worker.rank}/{worker.world_size-1}, 主进程: {worker.is_main_process}")

    print("初始化成功")
    worker.run()


if __name__ == "__main__":
    main()