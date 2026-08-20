# config/reranker_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    api_key: str
    model: str
    base_url: str


reranker_config = RerankerConfig(
    api_key=os.getenv("SILICONFLOW_API_KEY", ""),
    model=os.getenv("SILICONFLOW_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/rerank"),
)