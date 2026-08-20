# utils/reranker_http_utils.py

import requests
from config.reranker_config import reranker_config
from tool.logger import logger


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """
    使用 SiliconFlow Rerank API 对文档进行重排序打分。
    """
    # 过滤空文档，避免 API 报错
    valid_docs = []
    valid_indices = []
    for idx, doc in enumerate(documents):
        if doc and doc.strip():
            valid_docs.append(doc.strip())
            valid_indices.append(idx)

    if not valid_docs:
        return [0.5] * len(documents)

    headers = {
        "Authorization": f"Bearer {reranker_config.api_key}",
        "Content-Type": "application/json",
    }

    # 严格参考测试脚本的 payload 格式
    payload = {
        "model": reranker_config.model,
        "query": query,
        "documents": valid_docs,
        "top_n": len(valid_docs),
    }

    try:
        response = requests.post(
            reranker_config.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # 解析分数，按原始顺序排列
        scores = [0.0] * len(documents)
        for item in result.get("results", []):
            api_index = item.get("index")      # 在 valid_docs 中的索引
            score = item.get("relevance_score", 0.0)
            if api_index is not None and 0 <= api_index < len(valid_indices):
                original_index = valid_indices[api_index]
                scores[original_index] = float(score)

        logger.info(f"Rerank 完成，共 {len(documents)} 篇文档")
        return scores

    except requests.exceptions.RequestException as e:
        logger.error(f"Rerank API 请求失败: {e}")
        # 降级返回均匀分数
        return [0.5] * len(documents)
    except Exception as e:
        logger.error(f"Rerank 处理异常: {e}")
        return [0.5] * len(documents)