# convert_to_verl_rl.py
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import ijson
import re


def convert_to_verl_format(input_file, output_file, task_type="paper_summary", batch_size=500):
    """
    将论文总结数据转换为 verl RL 训练格式
    
    策略：将结构化输出作为 ground_truth，训练模型生成符合格式的总结
    
    使用流式处理避免内存溢出，支持大文件处理
    """
    writer = None
    batch_data = []
    total_count = 0
    
    with open(input_file, 'rb') as f:
        # 使用 ijson 流式解析 JSON 数组
        items = ijson.items(f, 'item')
        
        for idx, item in enumerate(items):
            try:
                verl_item = build_verl_item(item, idx, task_type)
                batch_data.append(verl_item)
                total_count += 1
                
                # 当批次达到指定大小时，写入 Parquet
                if len(batch_data) >= batch_size:
                    df_batch = pd.DataFrame(batch_data)
                    table = pa.Table.from_pandas(df_batch)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                    writer.write_table(table)
                    
                    print(f"  已处理 {total_count} 条样本...")
                    batch_data = []  # 清空批次
                    
            except Exception as e:
                print(f"  ⚠️ 处理第 {idx} 条数据时出错: {e}")
                continue
        
        # 写入剩余的数据
        if batch_data:
            df_batch = pd.DataFrame(batch_data)
            table = pa.Table.from_pandas(df_batch)
            
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            writer.write_table(table)
    
    if writer:
        writer.close()
    
    print(f"✅ 转换完成：共 {total_count} 条样本")
    print(f"📁 保存至：{output_file}")
    return total_count


def build_verl_item(item, idx, task_type="paper_summary"):
    """构建 verl 格式的数据项"""
    instruction = item["instruction"]
    output_json = json.loads(item["output"])
    
    # 构建 prompt（给模型的输入）
    prompt_content = instruction

    # 构建 ground_truth（用于奖励计算）
    # 将标准答案的各个字段拼接，用于规则奖励匹配
    ground_truth = {
        "summary": output_json.get("summary", ""),
        "algorithm": output_json.get("algorithm", ""),
        "compare_result": output_json.get("compare_result", ""),
        "keyword_problem": output_json.get("keyword_problem", ""),
        "keyword_algorithm": output_json.get("keyword_algorithm", "")
    }
    
    verl_item = {
        "data_source": task_type,
        "prompt": [
            {"role": "system", "content": "你是一个专业的学术论文分析专家，擅长提取论文的核心思想、算法细节和关键词。"},
            {"role": "user", "content": prompt_content}
        ],
        "ability": "academic_summarization",
        "reward_model": {
            "style": "rule",  # 使用规则奖励
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            "ground_truth_dict": ground_truth  # 保留字典形式用于解析
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "original_instruction": instruction,  # 保留原始信息
            "paper_title": extract_title(instruction)
        }
    }
    
    return verl_item


def extract_paper_content(instruction):
    """从 instruction 中提取论文内容（去掉前面的提示词）"""
    marker = "#论文\n"
    if marker in instruction:
        return instruction.split(marker, 1)[1].strip()
    return instruction


def extract_title(instruction):
    """提取论文标题"""
    match = re.search(r'<title>(.*?)</title>', instruction)
    return match.group(1) if match else "Unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将论文总结数据转换为 verl RL 训练格式（流式处理版）")
    parser.add_argument("--input", default=r"D:\python\code\paper_retrieve\data\paper_test.json", 
                        help="输入 JSON 文件路径")
    parser.add_argument("--output", default=r"D:\python\code\paper_retrieve\data\data_verl\paper_test.parquet", 
                        help="输出 Parquet 文件路径")
    parser.add_argument("--batch-size", type=int, default=500, 
                        help="批处理大小，默认 500")
    args = parser.parse_args()
    
    convert_to_verl_format(args.input, args.output, batch_size=args.batch_size)
