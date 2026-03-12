#!/bin/bash
# 论文总结数据准备和验证脚本
# 将原始 JSON 数据转换为 verl 训练格式并验证

set -x

# ==================== 配置参数 ====================
INPUT_JSON="data/paper_train.json"
OUTPUT_PARQUET="data/data_verl/paper_train.parquet"

# 验证集配置
VAL_INPUT_JSON="data/paper_test.json"
VAL_OUTPUT_PARQUET="data/data_verl/paper_val.parquet"

BATCH_SIZE=500  # 批处理大小

# ==================== 创建输出目录 ====================
mkdir -p data/data_verl

# ==================== 转换训练集 ====================
echo "=========================================="
echo "开始转换训练集数据"
echo "输入：${INPUT_JSON}"
echo "输出：${OUTPUT_PARQUET}"
echo "=========================================="

python verl/utils/model_paper_search/data_convert_to_verl_rl.py \
    --input ${INPUT_JSON} \
    --output ${OUTPUT_PARQUET} \
    --batch-size ${BATCH_SIZE}

# ==================== 转换验证集 ====================
echo "=========================================="
echo "开始转换验证集数据"
echo "输入：${VAL_INPUT_JSON}"
echo "输出：${VAL_OUTPUT_PARQUET}"
echo "=========================================="

python verl/utils/model_paper_search/data_convert_to_verl_rl.py \
    --input ${VAL_INPUT_JSON} \
    --output ${VAL_OUTPUT_PARQUET} \
    --batch-size ${BATCH_SIZE}

# ==================== 验证数据 ====================
echo "=========================================="
echo "验证数据格式和统计信息"
echo "=========================================="

python verl/utils/model_paper_search/validate_data.py \
    --train_file ${OUTPUT_PARQUET} \
    --val_file ${VAL_OUTPUT_PARQUET}

echo "=========================================="
echo "数据准备完成！"
echo "训练集：${OUTPUT_PARQUET}"
echo "验证集：${VAL_OUTPUT_PARQUET}"
echo "=========================================="
