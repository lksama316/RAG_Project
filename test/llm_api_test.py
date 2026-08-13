import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # ← 加载 .env 文件

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
