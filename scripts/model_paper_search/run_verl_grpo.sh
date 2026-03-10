#!/bin/bash
# VERL + GRPO 训练启动脚本
# 适用于4卡4090训练Qwen3-4B模型

set -e

# 配置
MODEL_PATH="Qwen/Qwen3-4B-Instruct"
TRAIN_DATA="train/data/paper_train.parquet"
VAL_DATA="train/data/paper_val.parquet"
OUTPUT_DIR="./verl_output"
N_GPUS=4

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "🚀 启动 VERL + GRPO 训练"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $TRAIN_DATA"
echo "输出: $OUTPUT_DIR"
echo "GPU数: $N_GPUS"
echo "=========================================="

# 设置环境变量
export WANDB_PROJECT="paper-summary-verl-grpo"
export WANDB_NAME="qwen3-4b-grpo-$(date +%Y%m%d-%H%M%S)"
export PYTHONPATH="${PYTHONPATH}:."

# 启动训练
python3 -m verl.trainer.main_ppo \
    algorithm=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=prompt \
    data.reward_key=ground_truth \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    model.path=$MODEL_PATH \
    model.trust_remote_code=true \
    algorithm.num_generations=4 \
    algorithm.kl_coeff=0.04 \
    algorithm.clip_ratio=0.2 \
    algorithm.learning_rate=1e-6 \
    algorithm.lr_scheduler=cosine \
    algorithm.warmup_steps=100 \
    algorithm.total_training_steps=2000 \
    training.n_gpus=$N_GPUS \
    training.nnodes=1 \
    training.save_interval=200 \
    training.output_dir=$OUTPUT_DIR \
    training.log_interval=10 \
    training.eval_interval=50 \
    +reward_fn@reward_model=paper_summary_reward \
    reward_model.weights.summary=0.30 \
    reward_model.weights.algorithm=0.25 \
    reward_model.weights.comparison=0.25 \
    reward_model.weights.keyword_problem=0.10 \
    reward_model.weights.keyword_algorithm=0.10

echo "=========================================="
echo "✅ 训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
