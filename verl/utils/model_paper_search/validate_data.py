#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证脚本
验证 parquet 文件是否符合 verl 训练格式要求
"""

import argparse
import json
import pyarrow.parquet as pq
import pandas as pd
from typing import Dict, Any


def validate_parquet_file(file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    验证 parquet 文件格式
    
    Args:
        file_path: parquet 文件路径
        verbose: 是否打印详细信息
        
    Returns:
        验证结果字典
    """
    print(f"\n验证文件：{file_path}")
    print("-" * 60)
    
    # 读取 parquet 文件
    table = pq.read_table(file_path)
    df = table.to_pandas()
    
    # 基本统计信息
    print(f"样本数量：{len(df)}")
    print(f"列名：{list(df.columns)}")
    print()
    
    # 验证必需字段
    required_fields = [
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info"
    ]
    
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        print(f"❌ 缺少必需字段：{missing_fields}")
        return {"valid": False, "error": f"Missing fields: {missing_fields}"}
    
    print(f"✓ 所有必需字段存在")
    
    # 验证字段内容
    errors = []
    warnings = []
    
    # 1. 验证 data_source
    data_sources = df["data_source"].unique()
    print(f"\ndata_source 取值：{data_sources}")
    
    # 2. 验证 prompt 格式
    print(f"\n验证 prompt 字段...")
    for idx, row in df.head(5).iterrows():
        prompt = row["prompt"]
        if not isinstance(prompt, list):
            errors.append(f"样本 {idx}: prompt 不是列表格式")
            continue
        
        # 检查是否包含 system 和 user 角色
        roles = [msg.get("role") for msg in prompt if isinstance(msg, dict)]
        if "system" not in roles:
            warnings.append(f"样本 {idx}: prompt 缺少 system 角色")
        if "user" not in roles:
            errors.append(f"样本 {idx}: prompt 缺少 user 角色")
    
    if not errors:
        print(f"✓ prompt 格式正确")
    
    # 3. 验证 reward_model 字段
    print(f"\n验证 reward_model 字段...")
    for idx, row in df.head(5).iterrows():
        reward_model = row["reward_model"]
        if not isinstance(reward_model, dict):
            errors.append(f"样本 {idx}: reward_model 不是字典格式")
            continue
        
        if "style" not in reward_model:
            errors.append(f"样本 {idx}: reward_model 缺少 style 字段")
        
        if "ground_truth" not in reward_model:
            errors.append(f"样本 {idx}: reward_model 缺少 ground_truth 字段")
        
        # 验证 ground_truth 是否为有效的 JSON 字符串
        if "ground_truth" in reward_model:
            try:
                gt = json.loads(reward_model["ground_truth"])
                if not isinstance(gt, dict):
                    warnings.append(f"样本 {idx}: ground_truth 不是字典")
            except:
                errors.append(f"样本 {idx}: ground_truth 不是有效的 JSON 字符串")
    
    if not errors:
        print(f"✓ reward_model 格式正确")
    
    # 4. 验证 extra_info 字段
    print(f"\n验证 extra_info 字段...")
    for idx, row in df.head(5).iterrows():
        extra_info = row["extra_info"]
        if not isinstance(extra_info, dict):
            errors.append(f"样本 {idx}: extra_info 不是字典格式")
            continue
        
        required_extra = ["split", "index", "original_instruction", "paper_title"]
        missing_extra = [k for k in required_extra if k not in extra_info]
        if missing_extra:
            warnings.append(f"样本 {idx}: extra_info 缺少字段 {missing_extra}")
    
    if not errors:
        print(f"✓ extra_info 格式正确")
    
    # 5. 检查数据分布
    print(f"\n数据分布统计:")
    print(f"  - 唯一 data_source 数量：{len(data_sources)}")
    print(f"  - ability 取值：{df['ability'].unique()}")
    
    # 6. 检查序列长度分布
    print(f"\n序列长度统计:")
    prompt_lengths = df["prompt"].apply(lambda x: sum(len(msg.get("content", "")) for msg in x) if isinstance(x, list) else 0)
    print(f"  - 平均 prompt 长度：{prompt_lengths.mean():.1f}")
    print(f"  - 最大 prompt 长度：{prompt_lengths.max()}")
    print(f"  - 最小 prompt 长度：{prompt_lengths.min()}")
    
    # 打印警告和错误
    if warnings:
        print(f"\n⚠️ 警告 ({len(warnings)} 条):")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... 还有 {len(warnings) - 5} 条警告")
    
    if errors:
        print(f"\n❌ 错误 ({len(errors)} 条):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 条错误")
        return {"valid": False, "errors": errors}
    
    print("\n✅ 数据验证通过！")
    return {"valid": True, "num_samples": len(df)}


def main():
    parser = argparse.ArgumentParser(description="验证 verl 格式的 parquet 数据文件")
    parser.add_argument("--train_file", type=str, help="训练集 parquet 文件路径")
    parser.add_argument("--val_file", type=str, help="验证集 parquet 文件路径")
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.train_file:
        result = validate_parquet_file(args.train_file, args.verbose)
        results["train"] = result
    
    if args.val_file:
        result = validate_parquet_file(args.val_file, args.verbose)
        results["val"] = result
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    for name, result in results.items():
        if result.get("valid"):
            print(f"✓ {name}: 通过 ({result.get('num_samples', 0)} 样本)")
        else:
            print(f"❌ {name}: 失败 - {result.get('error', '未知错误')}")
    
    # 如果所有验证都通过，返回 0
    all_valid = all(r.get("valid", False) for r in results.values())
    exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
