import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

import dashscope
from elasticsearch import Elasticsearch


load_dotenv()


es = Elasticsearch(
    hosts=["http://localhost:9200"],  # ES服务器地址
    basic_auth=("xxx", "xxx")  # 用户名和密码
)


def insert(es, file_path, index_name):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Indexing"):
        data = json.loads(line)
        es.index(
            index=index_name,
            #id="my_document_id",
            document=data,
        )



# 插入论文总结数据
insert(es, "/home/ecs-user/data/merged_summary_org_infos_emb_data_tmp.jsonl", 'papers-index')