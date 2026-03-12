#!/bin/bash
# Qwen3-4B-Instruct-2507 论文总结任务 GRPO 强化学习训练脚本
# 适配 4×RTX 4090 (24GB) 配置
# 基于 verl 框架的 GRPO 算法实现

set -x

# ==================== 配置参数 ====================
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
PROJECT_NAME="paper_summary_grpo"
EXPERIMENT_NAME="qwen3_4b_grpo_lora_${TIMESTAMP}"
TRAIN_DIR="outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

# 创建输出目录
mkdir -p $TRAIN_DIR
export TENSORBOARD_DIR=$TRAIN_DIR/tensorboard_log/
export VERL_FILE_LOGGER_PATH=$TRAIN_DIR/metrics.jsonl

# 模型路径（使用 SFT 训练后的模型作为起点）
# 如果使用 SFT 检查点，修改为实际路径：
# MODEL_PATH="outputs/paper_summary_sft/qwen3_4b_sft_*/checkpoint-xxx"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

# 数据路径
TRAIN_FILES="data/data_verl/paper_train.parquet"
VAL_FILES="data/data_verl/paper_val.parquet"

# ==================== GRPO 训练参数 ====================
# 4 卡 4090 优化配置
N_GPUS_PER_NODE=4
NNODES=1

# 批次大小配置
TRAIN_BATCH_SIZE=128              # 总训练 batch size
GRADIENT_ACCUMULATION_STEPS=1     # 梯度累积
# 有效 batch size = 128

# 序列长度配置
MAX_PROMPT_LENGTH=4096            # 论文输入最大长度
MAX_RESPONSE_LENGTH=1024          # 生成最大长度
MAX_TOKEN_LEN_PER_GPU=6144        # 每 GPU 最大 token 数 (4096+1024)

# GRPO 特定参数
NUM_ANSWERS_PER_QUESTION=8        # 每个问题的采样数（GRPO 组大小）
# 更大的组提供更稳定的训练，但占用更多显存

# 学习率配置
ACTOR_LEARNING_RATE=1.0e-5        # Actor 学习率
CRITIC_LEARNING_RATE=1.0e-5       # Critic 学习率（如果使用）

# PPO 配置
PPO_MINI_BATCH_SIZE=64            # PPO mini batch size
PPO_EPOCHS=2                      # PPO 更新轮数

# KL 散度控制
KL_LOSS_COEF=0.001                # KL 惩罚系数
KL_LOSS_TYPE="low_var_kl"         # 低方差 KL 估计
USE_KL_IN_REWARD=false            # 不在奖励中使用 KL

# ==================== 模型配置 ====================
# FSDP2 配置（推荐用于 4090）
STRATEGY="fsdp2"
MODEL_DTYPE="bf16"                # 使用 bfloat16 训练
PARAM_OFFLOAD=false               # 参数卸载（显存不足时启用）
OPTIMIZER_OFFLOAD=false           # 优化器卸载

# vLLM 采样配置
VLLM_GPU_MEMORY_UTILIZATION=0.6   # vLLM 显存占用率
VLLM_TENSOR_PARALLEL_SIZE=2       # 张量并行大小（2 卡用于采样）

# LoRA 配置（可选，显存不足时启用）
USE_LORA=false
LORA_RANK=32
LORA_ALPHA=64
MERGE_LORA=false                  # 训练完成后合并 LoRA 权重

# ==================== 训练控制 ====================
TOTAL_EPOCHS=1                    # GRPO 通常训练 1 个 epoch
SAVE_FREQ=20                      # 保存 checkpoint 频率
TEST_FREQ=5                       # 验证频率
VAL_BEFORE_TRAIN=true             # 训练前验证
CRITIC_WARMUP=0                   # Critic warmup 步数

# 日志配置
LOGGER='["console", "wandb"]'
LOG_VAL_GENERATIONS=0             # 验证生成日志数量

# ==================== 启动 GRPO 训练 ====================
echo "=========================================="
echo "开始 GRPO 强化学习训练"
echo "项目：${PROJECT_NAME}"
echo "实验：${EXPERIMENT_NAME}"
echo "输出目录：${TRAIN_DIR}"
echo "模型：${MODEL_PATH}"
echo "训练数据：${TRAIN_FILES}"
echo "验证数据：${VAL_FILES}"
echo "GPU 数量：${N_GPUS_PER_NODE}"
echo "组大小：${NUM_ANSWERS_PER_QUESTION}"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4090 计算能力
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand（单机训练）
export NCCL_NET_GDR_LEVEL=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 构建训练命令
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
    algorithm.kl_ctrl.kl_coef=${KL_LOSS_COEF} \
    algorithm.kl_ctrl.type=fixed \
    \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger=${LOGGER} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${TRAIN_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LEARNING_RATE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=${STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${NUM_ANSWERS_PER_QUESTION} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=${STRATEGY} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE} \
    \
    reward.num_workers=8 \
    reward.reward_manager.name=naive \
    2>&1 | tee ${TRAIN_DIR}/train_log.txt

# ==================== 训练完成 ====================
echo "=========================================="
echo "GRPO 训练完成！"
echo "检查点保存在：${TRAIN_DIR}"
echo "日志文件：${TRAIN_DIR}/train_log.txt"
echo "WandB 项目：${PROJECT_NAME}"
echo "=========================================="

# 可选：如果使用了 LoRA，合并权重
if [ "$USE_LORA" = true ] && [ "$MERGE_LORA" = true ]; then
    echo "正在合并 LoRA 权重..."
    python3 scripts/model_paper_search/merge_lora_weights.py \
        --model_path ${MODEL_PATH} \
        --lora_path ${TRAIN_DIR}/checkpoint-xxx \
        --output_path ${TRAIN_DIR}/merged_model
fi
