import json
from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.state import get_default_state

if __name__ == "__main__":
    setup_logging()

    # 1. 加载你已有的 chunks.json（带 item_name 的那个）
    with open(r"E:\AI_Study\pdf\output\HUAWEI MateBook B3-420 用户指南-(NDZ,Windows11_01,zh-cn)\chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. 构造 state（入库节点需要这些字段）
    state = get_default_state()
    state["task_id"] = "test-001"
    state["file_title"] = "HUAWEI MateBook B3-420 用户指南"
    state["item_name"] = "HUAWEIMateBookB3-420笔记本电脑"
    state["chunks"] = chunks

    # 3. 执行入库节点
    node = NodeImportMilvus()
    result = node(state)
    print("✅ 入库完成！")