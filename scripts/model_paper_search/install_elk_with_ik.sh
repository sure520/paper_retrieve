#!/bin/bash

# ==============================================================================
# ELK Stack + IK 分词器一键部署脚本
# ==============================================================================
# 
# 功能概述：
#   本脚本用于在 Debian/Ubuntu 系统上自动化部署 ELK Stack（Elasticsearch 8.15.2 + Kibana 8.15.2）
#   以及中文 IK 分词器，配置为单节点模式，支持远程访问。
# 
# 部署组件：
#   1. Elasticsearch 8.15.2 - 分布式搜索引擎和分析引擎
#   2. Kibana 8.15.2 - 数据可视化和管理界面
#   3. IK 分词器 - 中文分词插件，支持 ik_smart（智能分词）和 ik_max_word（最细粒度分词）
# 
# 主要步骤：
#   1. 系统更新和依赖准备
#   2. 配置阿里云 Elastic 软件源（加速下载）
#   3. 导入 Elastic GPG 密钥
#   4. 安装 OpenJDK 17（Java 运行环境）
#   5. 安装 Elasticsearch 和 Kibana 8.15.2
#   6. 系统参数调优（vm.max_map_count、fs.file-max）
#   7. 配置 Elasticsearch：
#      - 单节点模式
#      - 允许远程访问（0.0.0.0）
#      - 禁用 HTTP SSL（测试环境）
#   8. 配置 Kibana：
#      - 中文界面
#      - 允许远程访问
#      - 连接到本地 Elasticsearch
#   9. 修复文件和目录权限
#   10. 启用并启动 Elasticsearch 和 Kibana 服务
#   11. 等待 Elasticsearch 启动完成
#   12. 提取 elastic 用户初始密码
#   13. 下载并安装 IK 分词器插件
#   14. 重启 Elasticsearch 加载插件
#   15. 输出访问信息和使用提示
# 
# 访问地址：
#   - Elasticsearch: http://<服务器IP>:9200
#   - Kibana: http://<服务器IP>:5601
# 
# 默认账号：
#   - 用户名: elastic
#   - 密码: 自动生成并输出，或查看日志
# 
# 注意事项：
#   1. 本脚本适用于 Debian/Ubuntu 系统
#   2. 需要 root 权限运行
#   3. 测试环境配置，生产环境请启用 HTTPS 并调整安全设置
#   4. 确保防火墙/安全组开放 9200 和 5601 端口
#   5. 脚本使用阿里云镜像源加速下载
# 
# 作者: Model Paper Search Project
# 版本: 1.0 (最终修复版)
# ==============================================================================

set -e

echo "🚀 开始部署 ELK Stack + IK 分词器（最终修复版）..."

# 1. 更新系统
apt update -y

# 2. 配置阿里云 Elastic 源
cat > /etc/apt/sources.list.d/elasticsearch.list << 'EOF'
deb https://mirrors.aliyun.com/elasticstack/8.x/apt/ stable main
EOF

# 3. 导入 GPG 密钥
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | gpg --dearmor -o /etc/apt/trusted.gpg.d/elasticsearch.gpg

# 4. 安装 Java
apt install -y openjdk-17-jre-headless

# 5. 安装 Elasticsearch 和 Kibana
apt update
apt install -y elasticsearch=8.15.2 kibana=8.15.2

# 6. 系统参数调优
sysctl -w vm.max_map_count=262144
sysctl -w fs.file-max=65536
echo "vm.max_map_count=262144" >> /etc/sysctl.conf
echo "fs.file-max=65536" >> /etc/sysctl.conf

# 7. 【关键修复】追加 Elasticsearch 配置（支持远程访问 + 允许 HTTP）
echo "📝 配置 Elasticsearch（启用远程访问）..."
cat >> /etc/elasticsearch/elasticsearch.yml << 'EOF'

# --- 用户自定义配置 ---
cluster.name: my-es-cluster
node.name: node-1
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
# 允许在安全模式下使用 HTTP（仅限测试环境！）
xpack.security.http.ssl:
  enabled: false
EOF

# 8. 修复 elasticsearch.yml 权限（必须！）
chown root:elasticsearch /etc/elasticsearch/elasticsearch.yml
chmod 640 /etc/elasticsearch/elasticsearch.yml

# 9. 配置 Kibana（中文 + 远程访问）
cat > /etc/kibana/kibana.yml << 'EOF'
server.port: 5601
server.host: "0.0.0.0"
elasticsearch.hosts: ["http://localhost:9200"]
i18n.locale: "zh-CN"
elasticsearch.ssl.verificationMode: none
EOF

# 10. 修复数据目录权限
chown -R elasticsearch:elasticsearch /var/lib/elasticsearch
chown -R elasticsearch:elasticsearch /var/log/elasticsearch
chown -R kibana:kibana /var/lib/kibana

# 11. 启用并启动服务
systemctl daemon-reload
systemctl enable --now elasticsearch
systemctl enable --now kibana

# 12. 等待 Elasticsearch 启动
echo "⏳ 等待 Elasticsearch 启动（最多 60 秒）..."
sleep 10
for i in {1..6}; do
    if curl -s http://localhost:9200 >/dev/null; then
        echo "✅ Elasticsearch 已成功启动！"
        break
    fi
    sleep 10
done

# 13. 获取 elastic 用户密码
PASSWORD_FILE="/tmp/es_password.txt"
if [ -f /var/log/elasticsearch/my-es-cluster.log ]; then
    grep -A1 "The generated password for the elastic built-in superuser is" /var/log/elasticsearch/my-es-cluster.log 2>/dev/null | tail -n1 | awk '{print $NF}' > "$PASSWORD_FILE"
fi

if [ -s "$PASSWORD_FILE" ]; then
    ELASTIC_PASSWORD=$(cat "$PASSWORD_FILE")
else
    ELASTIC_PASSWORD="<请手动查看日志: sudo grep 'password' /var/log/elasticsearch/my-es-cluster.log>"
fi

# 14. 安装 IK 分词器
echo "🔤 正在安装 IK 分词器..."
IK_URL="https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.15.2/elasticsearch-analysis-ik-8.15.2.zip"
cd /tmp
rm -f ik.zip
wget -O ik.zip "$IK_URL"
/usr/share/elasticsearch/bin/elasticsearch-plugin install file:///tmp/ik.zip --batch

# 15. 重启 Elasticsearch 加载插件
systemctl restart elasticsearch
echo "🔄 等待 IK 插件加载..."
sleep 20

# 16. 输出结果
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "🎉 ELK + IK 部署完成！"
echo "----------------------------------------"
echo "Elasticsearch 地址: http://$IP:9200"
echo "Kibana 地址:        http://$IP:5601"
echo "用户名:             elastic"
echo "密码:               $ELASTIC_PASSWORD"
echo "IK 分词器:          已安装（支持 ik_smart / ik_max_word）"
echo "----------------------------------------"
echo "💡 提示："
echo "  1. 若无法访问，请检查云服务器安全组是否开放 9200 和 5601 端口"
echo "  2. 生产环境请禁用 xpack.security.http.ssl.enabled=false 并配置 HTTPS"