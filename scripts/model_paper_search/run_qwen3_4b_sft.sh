#!/bin/bash
# Qwen3-4B-Instruct-2507 论文总结任务 SFT 监督微调训练脚本
# 适配 4×RTX 4090 (24GB) 配置

set -x

# ==================== 配置参数 ====================
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
PROJECT_NAME="paper_summary_sft"
EXPERIMENT_NAME="qwen3_4b_sft_${TIMESTAMP}"
TRAIN_DIR="outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

# 创建输出目录
mkdir -p $TRAIN_DIR

# 模型和数据路径
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
# 如果使用本地模型，修改为：
# MODEL_PATH="/path/to/Qwen3-4B-Instruct-2507"

TRAIN_FILES="data/data_verl/paper_train.parquet"
VAL_FILES="data/data_verl/paper_val.parquet"

# 如果使用 HDFS 或远程存储，可以修改为实际路径
# TRAIN_FILES="/mnt/hdfs/data/paper_summary/train.parquet"
# VAL_FILES="/mnt/hdfs/data/paper_summary/val.parquet"

# ==================== 训练参数 ====================
# 4 卡 4090 优化配置
NPROC_PER_NODE=4
MICRO_BATCH_SIZE=2          # 每卡 micro batch size
GRADIENT_ACCUMULATION=4     # 梯度累积步数
# 有效 batch size = 4 GPUs × 2 × 4 = 32

LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

NUM_EPOCHS=3
WARMUP_RATIO=0.05
LR_SCHEDULER_TYPE="cosine"

MAX_PROMPT_LENGTH=4096      # 论文输入最大长度
MAX_RESPONSE_LENGTH=1024    # 输出最大长度

# LoRA 配置（如显存不足可启用）
USE_LORA=false
LORA_RANK=32
LORA_ALPHA=64

# ==================== 启动 SFT 训练 ====================
echo "=========================================="
echo "开始 SFT 监督微调训练"
echo "项目：${PROJECT_NAME}"
echo "实验：${EXPERIMENT_NAME}"
echo "输出目录：${TRAIN_DIR}"
echo "模型：${MODEL_PATH}"
echo "训练数据：${TRAIN_FILES}"
echo "验证数据：${VAL_FILES}"
echo "GPU 数量：${NPROC_PER_NODE}"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4090 计算能力
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand（单机训练）
export NCCL_NET_GDR_LEVEL=2

# 构建训练命令
CMD="torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \\
    -m verl.trainer.sft_trainer \\
    data.train_files=${TRAIN_FILES} \\
    data.val_files=${VAL_FILES} \\
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \\
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \\
    data.max_response_length=${MAX_RESPONSE_LENGTH} \\
    model.path=${MODEL_PATH} \\
    model.use_remove_padding=True \\
    model.enable_gradient_checkpointing=True \\
    optim.lr=${LEARNING_RATE} \\
    optim.weight_decay=${WEIGHT_DECAY} \\
    optim.max_grad_norm=${MAX_GRAD_NORM} \\
    optim.lr_scheduler_type=${LR_SCHEDULER_TYPE} \\
    optim.warmup_ratio=${WARMUP_RATIO} \\
    trainer.total_epochs=${NUM_EPOCHS} \\
    trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION} \\
    trainer.default_local_dir=${TRAIN_DIR} \\
    trainer.project_name=${PROJECT_NAME} \\
    trainer.experiment_name=${EXPERIMENT_NAME} \\
    trainer.logger='[\"console\", \"wandb\"]' \\
    trainer.save_freq=100 \\
    trainer.test_freq=50 \\
    trainer.val_before_train=True \\
    engine=fsdp"

# 如果启用 LoRA，添加以下参数
if [ "$USE_LORA" = true ]; then
    CMD="${CMD} \\
    model.lora_rank=${LORA_RANK} \\
    model.lora_alpha=${LORA_ALPHA} \\
    model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'"
fi

# 执行训练命令
eval $CMD 2>&1 | tee ${TRAIN_DIR}/train_log.txt

# ==================== 训练完成 ====================
echo "=========================================="
echo "SFT 训练完成！"
echo "检查点保存在：${TRAIN_DIR}"
echo "日志文件：${TRAIN_DIR}/train_log.txt"
echo "=========================================="

# 可选：上传到 WandB
# wandb artifact upload ${TRAIN_DIR} --type model
