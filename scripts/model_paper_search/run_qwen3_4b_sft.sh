#!/bin/bash
# Qwen3-4B-Instruct-2507 论文总结任务 SFT 监督微调训练脚本
# 适配 2×RTX 3090 (24GB) 配置，启用 LoRA 防止 OOM

set -x

# 设置时区为中国标准时间
export TZ=Asia/Shanghai

# ==================== 配置参数 ====================
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
PROJECT_NAME="paper_summary_sft"
EXPERIMENT_NAME="qwen3_4b_sft_${TIMESTAMP}"
TRAIN_DIR="outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

# 创建输出目录
mkdir -p $TRAIN_DIR

# 模型和数据路径
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

TRAIN_FILES="data/data_verl/paper_train.parquet"
VAL_FILES="data/data_verl/paper_test.parquet"

# ==================== 训练参数 ====================
# 2 卡 RTX 3090 优化配置
NPROC_PER_NODE=2
MICRO_BATCH_SIZE=1          # 每卡 micro batch size (3090 显存较小，设为 1)
TRAIN_BATCH_SIZE=8       # 全局 batch size (进一步降低)

LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
CLIP_GRAD=1.0               # 梯度裁剪值
SP_SIZE=2                  # 序列并行大小（3090 显存较小，设为 2）

NUM_EPOCHS=3
WARMUP_RATIO=0.05
LR_SCHEDULER_TYPE="cosine"

MAX_LENGTH=28106           # 最大序列长度（论文输入 + JSON 输出）

# LoRA 配置（启用 LoRA 以减少显存占用）
USE_LORA=true
LORA_RANK=8                 # LoRA 秩（降低以减少显存）
LORA_ALPHA=16               # LoRA alpha (通常是 rank 的 2 倍)

# 显存优化配置（关键：解决 OOM 问题）
MODEL_DTYPE="bfloat16"      # 使用 bf16 混合精度训练
OFFLOAD_POLICY=false        # 禁用 offload_policy（与 LoRA 不兼容）
PARAM_OFFLOAD=false         # 禁用参数卸载（与 LoRA 不兼容）
OPTIMIZER_OFFLOAD=false     # 禁用优化器卸载（与 LoRA 不兼容）

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
echo "Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "全局 Batch Size: ${TRAIN_BATCH_SIZE}"
echo "最大序列长度：${MAX_LENGTH}"
echo "LoRA 秩：${LORA_RANK}"
echo "序列并行大小：${SP_SIZE}"
echo "模型精度：${MODEL_DTYPE}"
echo "显存碎片优化：enabled"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3090 计算能力
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand（单机训练）
export NCCL_NET_GDR_LEVEL=2
export NCCL_ASYNC_TIMEOUT=1800     # 增加 NCCL 超时时间（秒）

# 显存碎片优化（关键！防止显存碎片化）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRAG=1

# 执行训练命令（只使用配置文件中存在的字段）
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    -m verl.trainer.sft_trainer \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.custom_cls.path=pkg://verl/utils/dataset/paper_summary_sft_dataset \
    data.custom_cls.name=PaperSummarySFTDataset \
    data.truncation=left \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=${LEARNING_RATE} \
    optim.weight_decay=${WEIGHT_DECAY} \
    optim.clip_grad=${CLIP_GRAD} \
    optim.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    optim.lr_warmup_steps_ratio=${WARMUP_RATIO} \
    trainer.total_epochs=${NUM_EPOCHS} \
    trainer.default_local_dir=${TRAIN_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    $(if [ "$USE_LORA" = true ]; then echo "model.target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"] model.lora_rank=${LORA_RANK} model.lora_alpha=${LORA_ALPHA}"; fi) \
    engine=fsdp \
    engine.strategy=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.model_dtype=${MODEL_DTYPE} \
    engine.offload_policy=${OFFLOAD_POLICY} \
    engine.param_offload=${PARAM_OFFLOAD} \
    engine.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    2>&1 | tee ${TRAIN_DIR}/train_log.txt

# ==================== 训练完成 ====================
echo "=========================================="
echo "SFT 训练完成！"
echo "检查点保存在：${TRAIN_DIR}"
echo "日志文件：${TRAIN_DIR}/train_log.txt"
echo "WandB 项目：${PROJECT_NAME}"
echo "WandB 实验：${EXPERIMENT_NAME}"
echo "=========================================="
echo ""
echo "训练指标已同步到 WandB，请访问 https://wandb.ai/ 查看"
echo "WandB 项目：${PROJECT_NAME}"
echo "WandB 实验：${EXPERIMENT_NAME}"
echo ""
