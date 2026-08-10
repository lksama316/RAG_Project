# config/lm_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    vl_model: str
    llm_model: str
    llm_temperature: float

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    vl_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)

# config/lm_config.py 末尾添加

vl_model = lm_config.vl_model
llm_model = lm_config.llm_model
base_url = lm_config.base_url
api_key = lm_config.api_key
llm_temperature = lm_config.llm_temperature