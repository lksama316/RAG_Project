# processor/import_processor/nodes/node_bge_embedding.py

import json
import logging
from typing import List, Dict

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
#from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        流程总览：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        chunks = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        return state
