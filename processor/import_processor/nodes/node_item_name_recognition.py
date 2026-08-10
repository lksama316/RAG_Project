# processor/import_processor/nodes/node_item_name_recognition.py
import json
import logging
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType

#from config.lm_config import lm_config
#from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
#from utils.embedding_utils import generate_embeddings
#from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name: str = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 核心节点：商品主体名称识别
        流程总览：
            1. 提取输入
            2. 构建大模型上下文
            3. 调用大模型识别商品名称
            4. 回填商品名称到状态和切片
            5. 生成商品名称的稠密/稀疏向量
            6. 将数据存入Milvus向量数据库

        必要参数：task_id、file_title, chunks
        更新参数：item_name

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤5：为商品名称生成稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)

        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)


        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} ---")

        return state