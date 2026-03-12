#!/bin/bash
# 论文总结模型完整训练流程脚本
# 包含：数据准备 -> SFT 监督微调 -> GRPO 强化学习
# 适配 4×RTX 4090 (24GB) 配置

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 切换到项目根目录
cd ${PROJECT_ROOT}

echo "=========================================="
echo "论文总结模型完整训练流程"
echo "项目根目录：${PROJECT_ROOT}"
echo "=========================================="

# 阶段选择
STAGE=${1:-"all"}  # 可选：all, data, sft, grpo

case ${STAGE} in
    "all")
        # ========== 完整流程 ==========
        echo "=========================================="
        echo "阶段 1/3: 数据准备"
        echo "=========================================="
        bash ${SCRIPT_DIR}/prepare_data.sh
        
        echo ""
        echo "=========================================="
        echo "阶段 2/3: SFT 监督微调"
        echo "=========================================="
        bash ${SCRIPT_DIR}/run_qwen3_4b_sft.sh
        
        # 获取 SFT 检查点路径（假设最后一个检查点）
        SFT_CKPT_DIR=$(ls -td outputs/paper_summary_sft/qwen3_4b_sft_*/checkpoint-* | tail -1)
        if [ -z "${SFT_CKPT_DIR}" ]; then
            echo "❌ 未找到 SFT 检查点，请检查 SFT 训练是否成功"
            exit 1
        fi
        echo "使用 SFT 检查点：${SFT_CKPT_DIR}"
        
        # 修改 GRPO 脚本中的模型路径
        sed -i "s|MODEL_PATH=.*|MODEL_PATH=\"${SFT_CKPT_DIR}\"|g" ${SCRIPT_DIR}/run_qwen3_4b_grpo.sh
        
        echo ""
        echo "=========================================="
        echo "阶段 3/3: GRPO 强化学习"
        echo "=========================================="
        bash ${SCRIPT_DIR}/run_qwen3_4b_grpo.sh
        
        # 恢复 GRPO 脚本中的模型路径
        sed -i "s|MODEL_PATH=.*|MODEL_PATH=\"Qwen/Qwen3-4B-Instruct-2507\"|g" ${SCRIPT_DIR}/run_qwen3_4b_grpo.sh
        
        echo ""
        echo "=========================================="
        echo "✅ 完整训练流程完成！"
        echo "=========================================="
        ;;
        
    "data")
        # ========== 仅数据准备 ==========
        echo "=========================================="
        echo "阶段：数据准备"
        echo "=========================================="
        bash ${SCRIPT_DIR}/prepare_data.sh
        ;;
        
    "sft")
        # ========== 仅 SFT 训练 ==========
        echo "=========================================="
        echo "阶段：SFT 监督微调"
        echo "=========================================="
        bash ${SCRIPT_DIR}/run_qwen3_4b_sft.sh
        ;;
        
    "grpo")
        # ========== 仅 GRPO 训练 ==========
        echo "=========================================="
        echo "阶段：GRPO 强化学习"
        echo "=========================================="
        bash ${SCRIPT_DIR}/run_qwen3_4b_grpo.sh
        ;;
        
    *)
        echo "用法：$0 {all|data|sft|grpo}"
        echo ""
        echo "  all   - 完整流程（数据准备 + SFT + GRPO）"
        echo "  data  - 仅数据准备"
        echo "  sft   - 仅 SFT 监督微调"
        echo "  grpo  - 仅 GRPO 强化学习"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 检查 WandB 日志监控训练指标"
echo "2. 运行评估脚本测试模型性能"
echo "3. 导出最终模型用于部署"
echo ""
echo "评估命令示例："
echo "  python scripts/model_paper_search/evaluate.py \\"
echo "      --model_path outputs/paper_summary_grpo/qwen3_4b_grpo_*/checkpoint-xxx \\"
echo "      --test_data data/data_verl/paper_val.parquet \\"
echo "      --output_file eval_results.json"
echo ""
