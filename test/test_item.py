# test_item_name.py
import json
import logging
from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition

if __name__ == "__main__":
    setup_logging(logging.INFO)

    # 1. 加载你现有的 chunks.json
    with open(r"E:\AI_Study\pdf\output\HUAWEI MateBook B3-420 用户指南-(NDZ,Windows11_01,zh-cn)\chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. 模拟 state（file_title 就用你改好的那个）
    mock_state = {
        "file_title": "HUAWEI MateBook B3-420 用户指南-(NDZ,Windows11_01,zh-cn)",
        "chunks": chunks
    }

    # 3. 执行主体识别节点
    node = NodeItemNameRecognition()
    result_state = node(mock_state)

    # 4. 打印识别出的商品名
    print(f"\n✅ 识别出的商品名: {result_state.get('item_name')}")

    # 5. 检查第一个切片是否带上了 item_name
    if result_state.get("chunks"):
        print(f"✅ 第一个切片的 item_name: {result_state['chunks'][0].get('item_name')}")