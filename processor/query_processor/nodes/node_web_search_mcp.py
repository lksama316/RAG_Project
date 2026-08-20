# processor/query_processor/nodes/node_web_search_mcp.py
import json
import httpx

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json


class NodeWebSearchMcp(NodeBase):
    """
    节点功能：调用百炼 MCP 联网搜索补充信息
    使用原生 httpx + JSON-RPC 2.0 协议，不依赖 openai-agents SDK
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行 MCP 联网搜索
        :param state: 会话状态，包含 rewritten_query
        :return: 包含 web_search_docs 的状态字典
        """
        query = state.get("rewritten_query", "")
        docs = []

        # 如果没有查询内容，直接返回
        if query:
            try:
                result = self._mcp_call(query)
                if result:
                    docs = result
                    logger.info(f"MCP 搜索成功，返回 {len(docs)} 条结果")
            except Exception as e:
                logger.error(f"MCP 搜索失败: {e}")

        if docs:
            return {"web_search_docs": docs}
        return {}

    def _mcp_call(self, query: str) -> list:
        """
        使用 httpx 直接调用百炼 MCP 服务（JSON-RPC 2.0 协议）
        流程：initialize 握手 → tools/call 调用搜索工具
        :param query: 搜索关键词（改写后的问题）
        :return: 格式化后的文档列表 [{title, url, snippet}, ...]
        """
        url = mcp_config.mcp_base_url
        headers = {
            "Authorization": f"Bearer {mcp_config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=30) as client:
            # ===== 步骤1: Initialize 握手 =====
            init_payload = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {}
                },
                "id": 1
            }

            logger.info("MCP: 正在初始化连接...")
            resp = client.post(url, headers=headers, json=init_payload)
            resp.raise_for_status()
            init_result = resp.json()

            if "error" in init_result:
                raise Exception(f"Initialize 失败: {init_result['error']}")

            logger.info("MCP: Initialize 成功")

            # ===== 步骤2: 调用搜索工具 =====
            call_payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "bailian_web_search",
                    "arguments": {
                        "query": query,
                        "count": 5
                    }
                },
                "id": 2
            }

            logger.info(f"MCP: 正在搜索: {query}")
            resp = client.post(url, headers=headers, json=call_payload)
            resp.raise_for_status()
            call_result = resp.json()

            if "error" in call_result:
                raise Exception(f"搜索调用失败: {call_result['error']}")

            # ===== 步骤3: 解析返回结果 =====
            # 百炼 MCP 返回格式: result.content[0].text 包含 JSON 字符串
            content = call_result.get("result", {}).get("content", [])
            if not content:
                logger.warning("MCP: 返回内容为空")
                return []

            # 提取 text 字段并解析
            text_content = content[0].get("text", "{}")
            data = json.loads(text_content)
            pages = data.get("pages", [])

            # ===== 步骤4: 格式化为统一结构 =====
            # 遵循文档规范：仅保留 title/url/snippet，过滤空 snippet
            docs = []
            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                if not snippet:
                    continue
                docs.append({
                    "title": (item.get("title") or "").strip(),
                    "url": (item.get("url") or "").strip(),
                    "snippet": snippet,
                })

            logger.info(f"MCP: 解析完成，有效结果 {len(docs)} 条")
            return docs


# ===================== 单元测试 =====================
if __name__ == "__main__":

    init_state = {
        "rewritten_query": "HAK180烫金机 如何调节转印温度"
    }

    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))