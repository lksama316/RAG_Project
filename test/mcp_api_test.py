# test/mcp_diagnose.py
"""
百炼 MCP 服务诊断脚本
模拟 MCP 协议的 initialize → tools/list → tools/call 流程
"""

import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("MCP_DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp")

if not API_KEY:
    print("❌ 错误: DASHSCOPE_API_KEY 未在 .env 中配置")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

# ===== 工具函数 =====
def log_response(title: str, resp: httpx.Response):
    print(f"\n{'='*60}")
    print(f"📡 {title}")
    print(f"状态码: {resp.status_code}")
    print(f"响应头: {dict(resp.headers)}")
    try:
        body = resp.json()
        print(f"响应体 (JSON): {json.dumps(body, ensure_ascii=False, indent=2)}")
    except:
        print(f"响应体 (原始): {resp.text[:500]}...")
    print(f"{'='*60}\n")


# ===== 测试1: 基础连通性 (GET) =====
def test_connectivity():
    print("\n🔍 测试1: 基础连通性 (GET /)")
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(BASE_URL, headers=HEADERS)
            log_response("GET 请求结果", resp)
            return resp.status_code == 200 or resp.status_code in [400, 405]
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False


# ===== 测试2: MCP Initialize =====
def test_initialize():
    print("\n🔍 测试2: MCP Initialize (标准握手)")
    payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        },
        "id": 1
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(BASE_URL, headers=HEADERS, json=payload)
            log_response("Initialize 请求结果", resp)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("result"):
                    print("✅ Initialize 成功！服务端返回:", data.get("result"))
                    return True
                else:
                    print(f"⚠️ Initialize 响应异常: {data}")
                    return False
            else:
                print(f"❌ Initialize 失败，状态码: {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ Initialize 异常: {e}")
        return False


# ===== 测试3: 获取工具列表 =====
def test_tools_list():
    print("\n🔍 测试3: 获取工具列表 (tools/list)")
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(BASE_URL, headers=HEADERS, json=payload)
            log_response("tools/list 请求结果", resp)
            if resp.status_code == 200:
                data = resp.json()
                tools = data.get("result", {}).get("tools", [])
                print(f"✅ 获取到 {len(tools)} 个工具:")
                for t in tools:
                    print(f"  - {t.get('name')}: {t.get('description', '无描述')}")
                return True
            else:
                print(f"❌ tools/list 失败，状态码: {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ tools/list 异常: {e}")
        return False


# ===== 测试4: 实际调用搜索工具 =====
def test_web_search(query: str = "HAK180烫金机"):
    print(f"\n🔍 测试4: 调用搜索工具 (bailian_web_search), query='{query}'")
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "bailian_web_search",
            "arguments": {
                "query": query,
                "count": 3
            }
        },
        "id": 3
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(BASE_URL, headers=HEADERS, json=payload)
            log_response("tools/call 请求结果", resp)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                content = result.get("content", [])
                if content:
                    print(f"✅ 搜索成功！返回 {len(content)} 条结果")
                    for item in content[:3]:
                        print(f"  - {item.get('text', '')[:100]}...")
                else:
                    print("⚠️ 搜索成功但无内容返回")
                return True
            else:
                print(f"❌ 搜索失败，状态码: {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ 搜索异常: {e}")
        return False


# ===== 主入口 =====
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 百炼 MCP 服务诊断工具")
    print(f"📍 服务地址: {BASE_URL}")
    print(f"🔑 API Key 前缀: {API_KEY[:8]}...")
    print("=" * 60)

    # 1. 基础连通性
    if not test_connectivity():
        print("\n❌ 基础连通性测试失败，请检查网络和 URL")
        exit(1)

    # 2. Initialize
    init_ok = test_initialize()

    # 3. tools/list
    tools_ok = test_tools_list()

    # 4. 实际搜索
    search_ok = test_web_search()

    # ===== 诊断结论 =====
    print("\n" + "=" * 60)
    print("📊 诊断结论")
    print(f"  ✅ 基础连通性 (GET): 通过")
    print(f"  {'✅' if init_ok else '❌'} Initialize: {'通过' if init_ok else '失败'}")
    print(f"  {'✅' if tools_ok else '❌'} tools/list: {'通过' if tools_ok else '失败'}")
    print(f"  {'✅' if search_ok else '❌'} 实际搜索: {'成功' if search_ok else '失败'}")
    print("=" * 60)

    if not init_ok:
        print("\n💡 建议: Initialize 失败，说明百炼 MCP 可能不支持标准 MCP 协议.")
        print("   请查阅百炼官方文档，确认正确的调用方式。")
    elif not tools_ok:
        print("\n💡 建议: tools/list 失败，但 Initialize 成功.")
        print("   可能是工具列表需要特定参数，或者百炼 MCP 实现有差异。")
    elif not search_ok:
        print("\n💡 建议: 连接和初始化正常，但搜索工具调用失败.")
        print("   请检查 query 参数格式，或联系百炼技术支持。")
    else:
        print("\n✅ 所有测试通过！请检查你的代码是否与标准 MCP 协议一致。")