#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文总结模型评估脚本
评估生成质量、JSON 格式准确率、字段完整率等指标
"""

import argparse
import json
import os
from typing import Dict, List, Any
from collections import defaultdict
import pyarrow.parquet as pq
from tqdm import tqdm


def load_parquet(file_path: str) -> List[Dict]:
    """加载 parquet 文件"""
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df.to_dict('records')


def extract_json_from_response(text: str) -> Dict:
    """从模型响应中提取 JSON"""
    try:
        return json.loads(text)
    except:
        import re
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    
    return None


def compute_metrics(generated: Dict, ground_truth: Dict) -> Dict[str, float]:
    """计算单个样本的评估指标"""
    metrics = {}
    required_fields = ["summary", "algorithm", "compare_result", 
                       "keyword_problem", "keyword_algorithm"]
    
    if generated is None:
        return {field: 0.0 for field in required_fields + ["format_accuracy", "completeness"]}
    
    present_fields = [f for f in required_fields if f in generated and generated[f]]
    metrics["completeness"] = len(present_fields) / len(required_fields)
    
    from difflib import SequenceMatcher
    for field in required_fields:
        if field in generated and field in ground_truth:
            gen_text = str(generated[field])
            gt_text = str(ground_truth[field])
            similarity = SequenceMatcher(None, gen_text, gt_text).ratio()
            metrics[f"{field}_score"] = similarity
        else:
            metrics[f"{field}_score"] = 0.0
    
    quality_scores = [metrics[f"{f}_score"] for f in required_fields]
    metrics["quality_avg"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return metrics


def evaluate_model(predictions: List[str], ground_truths: List[Dict]) -> Dict[str, Any]:
    """评估模型整体性能"""
    all_metrics = defaultdict(list)
    
    for pred, gt in zip(predictions, ground_truths):
        generated = extract_json_from_response(pred) if isinstance(pred, str) else pred
        gt_dict = json.loads(gt) if isinstance(gt, str) else gt
        metrics = compute_metrics(generated, gt_dict)
        
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    format_accurate = sum(1 for pred in predictions if extract_json_from_response(pred) is not None)
    avg_metrics["format_accuracy"] = format_accurate / len(predictions)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="评估论文总结模型性能")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（检查点或最终模型）")
    parser.add_argument("--test_data", type=str, required=True,
                        help="测试数据 parquet 文件路径")
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                        help="评估结果输出文件")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="推理批次大小")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大评估样本数，-1 表示全部")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("论文总结模型评估")
    print("=" * 60)
    print(f"模型路径：{args.model_path}")
    print(f"测试数据：{args.test_data}")
    print(f"输出文件：{args.output_file}")
    print()
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = load_parquet(args.test_data)
    
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    
    print(f"测试样本数：{len(test_data)}")
    
    # 准备推理
    print("\n准备模型推理...")
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.8
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
        
        # 构建 prompts
        prompts = []
        for item in test_data:
            prompt_messages = item["prompt"]
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt_messages])
            prompts.append(prompt_text)
        
        # 生成预测
        print("开始生成预测...")
        predictions = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for output in outputs:
                predictions.append(output.outputs[0].text)
        
        # 提取 ground truths
        ground_truths = [item["reward_model"]["ground_truth"] for item in test_data]
        
        # 评估
        print("\n评估结果...")
        results = evaluate_model(predictions, ground_truths)
        
        # 打印结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        for key, value in sorted(results.items()):
            print(f"{key}: {value:.4f}")
        
        # 保存结果
        output_data = {
            "model_path": args.model_path,
            "test_data": args.test_data,
            "num_samples": len(test_data),
            "metrics": results,
            "predictions": predictions,
            "ground_truths": ground_truths
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到：{args.output_file}")
        
    except ImportError:
        print("错误：vllm 未安装，请使用 pip install vllm 安装")
        exit(1)
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()