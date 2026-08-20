# test_siliconflow_rerank.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/rerank"

if not API_KEY:
    print("❌ 请设置 SILICONFLOW_API_KEY")
    exit(1)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "Qwen/Qwen3-Reranker-8B",
    "query": "怎么测这块主板的短路问题？",
    "documents": [
        "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。",
        "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
    ],
    "top_n": 2,
}

try:
    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.text}")
except Exception as e:
    print(f"请求异常: {e}")