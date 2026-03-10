import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

import dashscope
from elasticsearch import Elasticsearch


load_dotenv()


query_rewrite_prompt='''现在是一个论文搜索引擎助手，现在用户给定了一个论文搜索的query，你需要帮我理解和改写该query，我们搜索引擎支持基于关键字的论文搜索、论文发布机构/学校/公司搜索、发布日期搜索。因此你首先帮我分析用户的搜索意图，>然后再帮我改写query。
#你的任务
1、分析用户意图，首先分析用户想要搜索的论文需求，包括论文的具体内容要求、发布的机构/学校/公司要求、发布的日期要求，如果没有机构/学校/公司和日期要求，可以没有。
2、改写query，需要改写成两个query，一个query是具有完整语义的论文内容要求的query，用于embedding检索，一个query是论文内容的关键字，用于文本检索。
3、提取出query里论文发布机构/学校/公司，提取出时间要求，只提取有近期、近一周、近一个月、近一年类似于这样的近期的时间要求，其他的时间要求可以忽略。
**注意：以上改写的语义query里不要包含论文发布机构/学校/公司搜索、发布日期。语义query只包含论文内容**
#输出要求
Json格式输出，只输出json，不输出其他内容，输出格式如下：{{“分析”:”分析用户意图、内容要去、发布机构和时间要求等”,”语义query”:”改写的具有语义的query”,”关键词”:”改写的关键词query”,”机构”:”发布的机构，如果没有则写无”,时间要
求”:”为整数，具体的天数”}}
#你的知识库
时间知识：近期表示近一个月，1周有7天，一个月有30天，半年有180天，一年有365天。
#示例
##示例1
原始query:openai近期发表的论文。
输出:{{“分析”:”用户没有具体的论文内容要求，但是机构为openai，时间为近期，表示30天”,”语义query”:”无”,关键词”:”无”,”机构”:”openai”,时间要求”:”30”}}
##示例2
原始query:大模型用于销售相关的论文。
输出:{{“分析”:”用户论文内容要求为大模型用于销售相关的论文，机构和时间没有要求”,”语义query”:”大模型用于销售相关的论文”,关键词”:”大模型 销售”,”机构”:”无”,时间要求”:”无”}}
##示例3
原始query:2021年以前视觉领域里的最火的论文是什么
输出:{{“分析”:”用户论文内容要求为视觉领域，时间为2021年以前可以忽略，机构没有要求”,”语义query”:”视觉领域相关论文”,关键词”:”视觉相关论文”,”机构”:”无”,时间要求”:”无”}}
{example}
#分析以下query
原始query:{query}
'''

example='''##示例{number}
原始query:{ori_query}
输出:{rewrite_query}
'''


# 相似query查询, 返回组装后的query改写prompt
def query_similar(es, user_query):

    query = {
        "query": {
            "match": {
                "ori_query": user_query
            }
        },
        "size":3,
        "_source":["ori_query","rewrite_query"]
    }

    # 执行查询
    response = es.search(index="querys", body=query)

    # 打印结果
    ii = 1
    exas = []
    for hit in response['hits']['hits']:
        exas.append(example.format(number=ii, ori_query=hit['_source']['ori_query'], rewrite_query=hit['_source']['rewrite_query']))
        ii += 1
    example_str = ''.join(exas)
    prompt_final = query_rewrite_prompt.format(example=example_str, query=user_query)
    return prompt_final


def rewrite_query(prompt):
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{'role': 'user', 'content': prompt}],
        result_format='message'
    )
    return response['messages'][-1]['content']