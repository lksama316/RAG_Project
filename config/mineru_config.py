# config/mineru_config.py
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# 获取当前文件所在目录的父目录（项目根目录）
BASE_DIR = Path(__file__).parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)  # 指定绝对路径

@dataclass
class MineruConfig:
    base_url: str
    api_token: str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_token=os.getenv("MINERU_API_TOKEN")
)


