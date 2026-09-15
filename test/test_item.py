import logging
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.state import create_default_state
from utils.task_utils import clear_task, get_done_task_list, get_running_task_list


def test_node_item_name_recognition():
    """验证主体识别节点的编排、状态回填和任务追踪。"""
    setup_logging(logging.INFO)

    task_id = "test-node-item-name-recognition"
    file_title = "HUAWEI MateBook B3-420 用户指南-(NDZ,Windows11_01,zh-cn)"
    expected_item_name = "HUAWEI MateBook B3-420"
    dense_vector = [0.0] * 1024
    sparse_vector = {0: 1.0}

    state = create_default_state(
        task_id=task_id,
        file_title=file_title,
        chunks=[
            {
                "title": "产品概述",
                "content": "HUAWEI MateBook B3-420 用户指南。",
            },
            {
                "title": "安全信息",
                "content": "使用本产品前，请先阅读安全说明。",
            },
        ],
    )

    node = NodeItemNameRecognition()

    try:
        with (
            patch.object(node, "_step_3_call_llm", return_value=expected_item_name) as mock_llm,
            patch.object(
                node,
                "_step_5_generate_vectors",
                return_value=(dense_vector, sparse_vector),
            ) as mock_generate_vectors,
            patch.object(node, "_step_6_save_to_milvus") as mock_save_to_milvus,
        ):
            result_state = node(state)

        assert result_state["item_name"] == expected_item_name
        assert all(
            chunk["item_name"] == expected_item_name for chunk in result_state["chunks"]
        )
        assert get_done_task_list(task_id) == ["主体名称识别"]
        assert get_running_task_list(task_id) == []

        mock_llm.assert_called_once()
        mock_generate_vectors.assert_called_once_with(expected_item_name)
        mock_save_to_milvus.assert_called_once_with(
            state,
            file_title,
            expected_item_name,
            dense_vector,
            sparse_vector,
        )

        print("测试通过：主体名称识别节点状态回填和任务追踪正常。")
    finally:
        clear_task(task_id)


if __name__ == "__main__":
    test_node_item_name_recognition()
