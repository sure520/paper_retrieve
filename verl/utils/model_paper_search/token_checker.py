#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字符串 token 数量和大小检查工具
"""
import argparse
import sys
from pathlib import Path
import re


def simple_token_count(text):
    """
    简单的 token 计算方法（默认使用）
    对于中文：每 1-2 个字符约为 1 个 token
    对于英文：每 4 个字符约为 1 个 token
    
    Args:
        text: 要检查的文本
    
    Returns:
        token_count: token 数量
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    chinese_tokens = chinese_chars // 2
    english_tokens = english_chars // 4
    other_tokens = other_chars // 4
    
    total_tokens = chinese_tokens + english_tokens + other_tokens
    return max(total_tokens, 1)


def count_tokens(text, tokenizer_name="gpt2", use_transformers=False):
    """
    计算 token 数量
    
    Args:
        text: 要检查的文本
        tokenizer_name: tokenizer 名称，默认为 gpt2
        use_transformers: 是否使用 transformers 库
    
    Returns:
        token_count: token 数量
    """
    if use_transformers:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except ImportError:
            print("⚠️  transformers 库未安装，使用简单计算方法")
            return simple_token_count(text)
        except Exception as e:
            print(f"⚠️  使用 tokenizer 时出错: {e}，使用简单计算方法")
            return simple_token_count(text)
    else:
        return simple_token_count(text)


def approx_token_count(text):
    """
    近似计算 token 数量（备用方法）
    
    Args:
        text: 要检查的文本
    
    Returns:
        近似 token 数量
    """
    words = text.split()
    chars = len(text)
    return max(int(len(words) * 1.3), chars // 4)


def get_text_info(text, tokenizer_name="gpt2", use_transformers=False):
    """
    获取文本的完整信息
    
    Args:
        text: 要检查的文本
        tokenizer_name: tokenizer 名称
        use_transformers: 是否使用 transformers 库
    
    Returns:
        包含各种信息的字典
    """
    char_count = len(text)
    word_count = len(text.split())
    token_count = count_tokens(text, tokenizer_name, use_transformers)
    size_bytes = len(text.encode('utf-8'))
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'token_count': int(token_count) if isinstance(token_count, float) else token_count,
        'size_bytes': size_bytes,
        'size_kb': round(size_bytes / 1024, 2),
        'size_mb': round(size_bytes / (1024 * 1024), 4)
    }


def print_text_info(info, text_preview=None):
    """
    打印文本信息
    
    Args:
        info: 文本信息字典
        text_preview: 可选的文本预览
    """
    print("=" * 60)
    print("📊 文本信息统计")
    print("=" * 60)
    print(f"字符数 (Char):  {info['char_count']:,}")
    print(f"词数 (Word):    {info['word_count']:,}")
    print(f"Token 数:       {info['token_count']:,}")
    print("-" * 60)
    print(f"大小 (Bytes):   {info['size_bytes']:,} B")
    print(f"大小 (KB):      {info['size_kb']} KB")
    print(f"大小 (MB):      {info['size_mb']} MB")
    print("=" * 60)
    
    if text_preview:
        print("\n📝 文本预览:")
        print("-" * 60)
        if len(text_preview) > 500:
            print(text_preview[:500] + "...")
        else:
            print(text_preview)
        print("-" * 60)


def check_file(file_path, tokenizer_name="gpt2", use_transformers=False):
    """
    检查文件内容
    
    Args:
        file_path: 文件路径
        tokenizer_name: tokenizer 名称
        use_transformers: 是否使用 transformers 库
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n📂 检查文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        info = get_text_info(text, tokenizer_name, use_transformers)
        print_text_info(info, text)
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="检查字符串或文件的 token 数量和大小"
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="要检查的文本或文件路径（如果未提供，则从标准输入读取）"
    )
    
    parser.add_argument(
        "-f", "--file",
        action="store_true",
        help="将输入视为文件路径"
    )
    
    parser.add_argument(
        "-t", "--tokenizer",
        default="gpt2",
        help="使用的 tokenizer 名称（默认: gpt2），例如: bert-base-chinese, Qwen/Qwen-7B-Chat"
    )
    
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="使用 transformers 库进行精确 token 计算（需要网络连接下载 tokenizer）"
    )
    
    args = parser.parse_args()
    
    if args.file:
        if not args.input:
            print("❌ 请提供文件路径")
            return
        check_file(args.input, args.tokenizer, args.use_transformers)
        return
    
    if args.input:
        text = args.input
        print(f"\n📝 检查文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        info = get_text_info(text, args.tokenizer, args.use_transformers)
        print_text_info(info, text)
        return
    
    print("📝 请输入要检查的文本（按 Ctrl+D 或 Ctrl+Z 结束）:")
    try:
        text = sys.stdin.read()
        if text.strip():
            info = get_text_info(text, args.tokenizer, args.use_transformers)
            print_text_info(info, text)
        else:
            print("⚠️  未输入文本")
    except KeyboardInterrupt:
        print("\n\n已取消")


if __name__ == "__main__":
    main()
