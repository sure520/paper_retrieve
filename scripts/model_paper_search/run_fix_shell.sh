#!/bin/bash
SCRIPT="$1"
ins_dos2unix=True

if [ -z "$SCRIPT" ]; then
  echo "用法: $0 <要执行的脚本路径>"
  exit 1
fi

if [ins_dos2unix]; then
  apt update -y
  apt install -y dos2unix
fi
dos2unix "$SCRIPT"
chmod +x "$SCRIPT"
exec sudo "$SCRIPT"
