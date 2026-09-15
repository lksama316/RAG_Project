"""
RAGAS 评估程序。

功能：
1. 读取脚本当前目录下的 qa.csv；
2. 调用 KBQueryWorkflow 获取 answer 和 reranked_docs；
3. 使用项目提供的 LLM 和 BGE-M3 执行 Ragas 的 5 项指标评估；
4. 将结果写入脚本当前目录下的 qa_result.csv，编码为 UTF-8 BOM。

运行方式：
    python eval/ragas_eval.py

依赖：
    ragas
    datasets
    pandas
    langchain-core
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings


def _patch_ragas_compat_imports() -> None:
    """
    修复 Ragas 0.4.x 与 langchain-community 0.4.x 的 Vertex AI 导入兼容问题。

    Ragas 0.4.3 仍直接导入 langchain_community.chat_models.vertexai，
    但该模块在新版 langchain-community 中已移除。本项目不使用 Vertex AI，
    因此仅在模块缺失时注入占位类，避免影响 OpenAI 兼容模型调用。

    :return: 无返回值，必要时直接修改 sys.modules。
    """
    import importlib
    import types

    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = types.ModuleType(module_name)

    class ChatVertexAI:
        """Ragas 导入检查使用的占位类；当前评估流程不会实例化该类。"""

        pass

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module

    parent_module = importlib.import_module("langchain_community.chat_models")
    setattr(parent_module, "vertexai", compatibility_module)


_patch_ragas_compat_imports()

try:
    import ragas  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "ragas":
        raise
    raise SystemExit(
        "缺少 ragas 依赖，请先执行：python -m pip install -U ragas datasets"
    ) from exc

# 保证从仓库任意目录运行脚本时，都能正确导入项目模块和根目录 .env。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.embedding_utils import get_bge_m3_ef
from utils.llm_utils import get_llm_client

QA_FILE = SCRIPT_DIR / "qa.csv"
RESULT_FILE = SCRIPT_DIR / "qa_result.csv"
REQUIRED_INPUT_COLUMNS = ("question", "ground_truth")
OUTPUT_COLUMNS = (
    "question",
    "answer",
    "context",
    "ground_truth",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
METRIC_COLUMNS = OUTPUT_COLUMNS[4:]
METRIC_COLUMN_ALIASES = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "context_precision": (
        "context_precision",
        "llm_context_precision_with_reference",
    ),
    "context_recall": ("context_recall", "llm_context_recall"),
    "answer_correctness": ("answer_correctness",),
}


class BGEM3LangChainEmbeddings(Embeddings):
    """将项目现有 BGE-M3 对象适配为 LangChain Embeddings 接口。"""

    def __init__(self) -> None:
        """初始化适配器，并复用项目提供的 BGE-M3 单例对象。"""
        self._embedding_function = get_bge_m3_ef()

    @staticmethod
    def _extract_dense_vectors(result: dict[str, Any]) -> list[list[float]]:
        """
        从 BGE-M3 的编码结果中提取 dense 向量。

        :param result: BGEM3EmbeddingFunction 返回的编码结果字典。
        :return: 可供 LangChain/Ragas 使用的二维浮点向量列表。
        """
        dense_vectors = result.get("dense")
        if dense_vectors is None:
            raise ValueError("BGE-M3 编码结果中缺少 dense 字段")

        if hasattr(dense_vectors, "tolist"):
            dense_vectors = dense_vectors.tolist()

        if not dense_vectors:
            return []

        # 兼容单条文本返回一维向量的情况。
        if isinstance(dense_vectors[0], (int, float)):
            dense_vectors = [dense_vectors]

        return [
            [float(value) for value in vector]
            for vector in dense_vectors
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        将文档文本批量转换为 dense 向量。

        :param texts: 待编码的文档文本列表。
        :return: 与输入顺序一致的向量列表。
        """
        if not texts:
            return []
        result = self._embedding_function.encode_documents(texts)
        return self._extract_dense_vectors(result)

    def embed_query(self, text: str) -> list[float]:
        """
        将查询文本转换为 dense 向量。

        :param text: 待编码的查询文本。
        :return: 查询文本对应的 dense 向量。
        """
        result = self._embedding_function.encode_queries([text])
        vectors = self._extract_dense_vectors(result)
        if not vectors:
            raise ValueError("BGE-M3 未返回查询向量")
        return vectors[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        异步批量编码文档，避免阻塞 Ragas 的异步评估流程。

        :param texts: 待编码的文档文本列表。
        :return: 与输入顺序一致的向量列表。
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """
        异步编码查询文本，避免阻塞 Ragas 的异步评估流程。

        :param text: 待编码的查询文本。
        :return: 查询文本对应的 dense 向量。
        """
        return await asyncio.to_thread(self.embed_query, text)


def _extract_contexts(reranked_docs: Sequence[Any]) -> list[str]:
    """
    从工作流返回的 reranked_docs 中提取 Ragas 所需上下文列表。

    :param reranked_docs: KBQueryWorkflow 返回的重排文档列表。
    :return: 去除空内容后的上下文字符串列表。
    """
    contexts: list[str] = []
    for document in reranked_docs:
        if isinstance(document, dict):
            content = document.get("content")
        else:
            content = document

        if content is None:
            continue

        content_text = str(content).strip()
        if content_text:
            contexts.append(content_text)

    return contexts


def _run_rag_with_retry(
    workflow: KBQueryWorkflow,
    initial_state: dict[str, Any],
    max_retries: int,
) -> tuple[str, list[str]]:
    """
    执行单次 RAG 查询，并在失败时进行有限次重试。

    :param workflow: 已初始化的 KBQueryWorkflow 实例。
    :param initial_state: 包含 original_query、session_id、is_stream 的初始状态。
    :param max_retries: 首次执行失败后的最大重试次数。
    :return: answer 字符串以及 reranked_docs 中提取的上下文列表。
    """
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)

    for attempt_index in range(attempts):
        try:
            state = workflow.run(initial_state=initial_state, stream=False)
            if not isinstance(state, dict):
                raise TypeError("KBQueryWorkflow.run 未返回字典类型的 state")

            answer = str(state.get("answer") or "").strip()
            if not answer:
                raise ValueError("KBQueryWorkflow 返回的 answer 为空")

            contexts = _extract_contexts(state.get("reranked_docs") or [])
            return answer, contexts
        except Exception as exc:
            last_error = exc
            if attempt_index >= attempts - 1:
                break
            wait_seconds = min(2 ** attempt_index, 5)
            print(
                f"RAG 查询失败，{wait_seconds} 秒后重试 "
                f"({attempt_index + 1}/{max_retries})：{exc}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"RAG 查询连续失败：{last_error}") from last_error


def step_1_load_qa_data(qa_path: Path) -> pd.DataFrame:
    """
    读取并校验评估问题集。

    :param qa_path: qa.csv 文件路径，必须包含 question、ground_truth 两列。
    :return: 去除空白问题和答案后的 DataFrame。
    """
    if not qa_path.exists():
        raise FileNotFoundError(f"未找到评估输入文件：{qa_path}")

    qa_data = pd.read_csv(
        qa_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    qa_data.columns = [str(column).strip() for column in qa_data.columns]

    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in qa_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"qa.csv 缺少必要列：{missing_columns}；"
            f"当前列为：{list(qa_data.columns)}"
        )

    qa_data = qa_data[list(REQUIRED_INPUT_COLUMNS)].copy()
    for column in REQUIRED_INPUT_COLUMNS:
        qa_data[column] = qa_data[column].astype(str).str.strip()

    qa_data = qa_data[
        (qa_data["question"] != "") & (qa_data["ground_truth"] != "")
    ].reset_index(drop=True)

    if qa_data.empty:
        raise ValueError("qa.csv 中没有有效的 question 和 ground_truth 数据")

    return qa_data


def step_2_collect_rag_results(
    qa_data: pd.DataFrame,
    workflow: KBQueryWorkflow,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    逐条调用 RAG 工作流并收集回答与检索上下文。

    :param qa_data: 包含 question、ground_truth 的评估数据。
    :param workflow: 已初始化的 KBQueryWorkflow 实例。
    :param max_retries: 每条问题首次失败后的最大重试次数。
    :return: 包含 question、answer、retrieved_contexts、ground_truth 的记录列表。
    """
    rag_results: list[dict[str, Any]] = []
    total = len(qa_data)

    for row_index, row in qa_data.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip()
        session_id = f"ragas_eval_{row_index}_{uuid.uuid4().hex[:8]}"

        initial_state = {
            "original_query": question,
            "session_id": session_id,
            "is_stream": False,
        }

        print(f"[{row_index + 1}/{total}] 正在执行 RAG：{question}")
        answer, contexts = _run_rag_with_retry(
            workflow=workflow,
            initial_state=initial_state,
            max_retries=max_retries,
        )

        rag_results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return rag_results


def step_3_create_ragas_llm() -> Any:
    """
    使用项目 get_llm_client() 创建 Ragas 评估 LLM。

    :return: 适配 Ragas 后的 LLM；若当前 Ragas 无需显式包装，则返回 LangChain LLM。
    """
    llm_client = get_llm_client()
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        return llm_client

    return LangchainLLMWrapper(llm_client)


def step_4_create_ragas_embeddings() -> Any:
    """
    使用项目 get_bge_m3_ef() 创建 Ragas 评估 Embedding。

    :return: 适配 Ragas 后的 Embedding；若当前 Ragas 无需显式包装，则返回适配器对象。
    """
    embeddings = BGEM3LangChainEmbeddings()
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        return embeddings

    return LangchainEmbeddingsWrapper(embeddings)


def step_5_create_ragas_dataset(records: Sequence[dict[str, Any]]) -> Any:
    """
    将 RAG 结果转换为 Ragas 数据集。

    :param records: 包含 question、answer、retrieved_contexts、ground_truth 的记录列表。
    :return: Ragas EvaluationDataset；旧版 Ragas 则返回 Hugging Face Dataset。
    """
    try:
        from ragas import EvaluationDataset

        try:
            from ragas.dataset_schema import SingleTurnSample
        except ImportError:
            from ragas import SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=record["question"],
                response=record["answer"],
                retrieved_contexts=record["retrieved_contexts"],
                reference=record["ground_truth"],
            )
            for record in records
        ]
        return EvaluationDataset(samples=samples)
    except ImportError:
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "question": [record["question"] for record in records],
                "answer": [record["answer"] for record in records],
                "contexts": [record["retrieved_contexts"] for record in records],
                "ground_truth": [record["ground_truth"] for record in records],
            }
        )


def _resolve_metric(*metric_names: str) -> Any:
    """
    按候选名称从 ragas.metrics 中查找指标，以兼容不同 Ragas 版本。

    :param metric_names: 指标的候选类名或对象名，按优先级排列。
    :return: 找到的 Ragas 指标类或指标对象。
    """
    import ragas.metrics as ragas_metrics

    for metric_name in metric_names:
        if hasattr(ragas_metrics, metric_name):
            return getattr(ragas_metrics, metric_name)

    raise ImportError(f"当前 Ragas 版本未提供指标：{metric_names}")


def _instantiate_metric(
    metric_component: Any,
    llm: Any,
    embeddings: Any | None = None,
) -> Any:
    """
    根据 Ragas 指标类型创建实例，并注入项目 LLM/Embedding。

    :param metric_component: ragas.metrics 中的指标类或指标对象。
    :param llm: Ragas 评估使用的 LLM。
    :param embeddings: 可选的 Ragas 评估 Embedding，仅传给需要 Embedding 的指标。
    :return: 可直接传给 evaluate() 的 Ragas 指标实例。
    """
    if not inspect.isclass(metric_component):
        return metric_component

    parameters = inspect.signature(metric_component).parameters
    metric_kwargs: dict[str, Any] = {}

    if "llm" in parameters:
        metric_kwargs["llm"] = llm
    if embeddings is not None and "embeddings" in parameters:
        metric_kwargs["embeddings"] = embeddings

    return metric_component(**metric_kwargs)


def step_6_create_ragas_metrics(llm: Any, embeddings: Any) -> list[Any]:
    """
    创建要求的 5 个 Ragas 指标。

    :param llm: Ragas 评估 LLM，用于 Faithfulness、Context Precision、Context Recall、
        Answer Relevancy 和 Answer Correctness 的判定。
    :param embeddings: Ragas 评估 Embedding，用于 Answer Relevancy 和 Answer Correctness
        的语义相似度计算。
    :return: 按 Faithfulness、Answer Relevancy、Context Precision、Context Recall、
        Answer Correctness 顺序排列的指标列表。
    """
    faithfulness_metric = _resolve_metric("Faithfulness", "faithfulness")
    answer_relevancy_metric = _resolve_metric(
        "AnswerRelevancy",
        "ResponseRelevancy",
        "answer_relevancy",
    )
    context_precision_metric = _resolve_metric(
        "LLMContextPrecisionWithReference",
        "ContextPrecisionWithReference",
        "ContextPrecision",
        "context_precision",
    )
    context_recall_metric = _resolve_metric(
        "LLMContextRecall",
        "ContextRecall",
        "context_recall",
    )
    answer_correctness_metric = _resolve_metric(
        "AnswerCorrectness",
        "answer_correctness",
    )

    return [
        _instantiate_metric(faithfulness_metric, llm=llm),
        _instantiate_metric(
            answer_relevancy_metric,
            llm=llm,
            embeddings=embeddings,
        ),
        _instantiate_metric(context_precision_metric, llm=llm),
        _instantiate_metric(context_recall_metric, llm=llm),
        _instantiate_metric(
            answer_correctness_metric,
            llm=llm,
            embeddings=embeddings,
        ),
    ]


def step_7_evaluate_ragas(
    dataset: Any,
    metrics: Sequence[Any],
    llm: Any,
    embeddings: Any,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    执行 Ragas 评估并返回指标 DataFrame。

    :param dataset: step_5_create_ragas_dataset() 创建的评估数据集。
    :param metrics: step_6_create_ragas_metrics() 创建的 5 个 Ragas 指标。
    :param llm: Ragas 评估使用的 LLM。
    :param embeddings: Ragas 评估使用的 Embedding。
    :param show_progress: 是否显示 Ragas 评估进度条。
    :return: 包含样本及 5 个指标得分的 DataFrame。
    """
    from ragas import evaluate

    evaluate_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "metrics": list(metrics),
        "llm": llm,
        "embeddings": embeddings,
        "show_progress": show_progress,
        "raise_exceptions": False,
    }

    evaluate_parameters = inspect.signature(evaluate).parameters
    supports_var_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in evaluate_parameters.values()
    )
    if not supports_var_keywords:
        evaluate_kwargs = {
            key: value
            for key, value in evaluate_kwargs.items()
            if key in evaluate_parameters
        }

    result = evaluate(**evaluate_kwargs)
    return result.to_pandas()


def step_8_export_results(
    records: Sequence[dict[str, Any]],
    evaluation_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    将评估结果按指定 9 列格式写入 UTF-8 BOM CSV。

    :param records: 原始 RAG 结果记录，用于输出 question、answer、context、ground_truth。
    :param evaluation_data: Ragas 返回的指标 DataFrame。
    :param output_path: qa_result.csv 的保存路径。
    :return: 无返回值，文件会直接写入输出路径。
    """
    output_rows: list[dict[str, Any]] = []

    for row_index, record in enumerate(records):
        metric_row = (
            evaluation_data.iloc[row_index]
            if row_index < len(evaluation_data)
            else {}
        )

        output_row: dict[str, Any] = {
            "question": record["question"],
            "answer": record["answer"],
            "context": "\n\n".join(record["retrieved_contexts"]),
            "ground_truth": record["ground_truth"],
        }

        for metric_name in METRIC_COLUMNS:
            metric_value = pd.NA
            if hasattr(metric_row, "get"):
                for source_column in METRIC_COLUMN_ALIASES[metric_name]:
                    if source_column in metric_row:
                        metric_value = metric_row.get(source_column, pd.NA)
                        break
            output_row[metric_name] = metric_value

        output_rows.append(output_row)

    output_data = pd.DataFrame(output_rows, columns=list(OUTPUT_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"评估完成，结果已保存至：{output_path}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    :return: 包含输入文件、输出文件、重试次数和进度条开关的参数对象。
    """
    parser = argparse.ArgumentParser(description="KBQueryWorkflow + Ragas 评估程序")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=QA_FILE,
        help=f"评估问题 CSV 路径，默认：{QA_FILE}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=RESULT_FILE,
        help=f"评估结果 CSV 路径，默认：{RESULT_FILE}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="单条问题 RAG 失败后的最大重试次数，默认：2",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 Ragas 进度条",
    )
    return parser.parse_args()


def main() -> None:
    """执行完整的 RAGAS 评估流程。"""
    args = parse_args()

    qa_data = step_1_load_qa_data(args.qa_file)
    workflow = KBQueryWorkflow()

    rag_records = step_2_collect_rag_results(
        qa_data=qa_data,
        workflow=workflow,
        max_retries=args.max_retries,
    )

    ragas_llm = step_3_create_ragas_llm()
    ragas_embeddings = step_4_create_ragas_embeddings()
    ragas_dataset = step_5_create_ragas_dataset(rag_records)
    ragas_metrics = step_6_create_ragas_metrics(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    evaluation_data = step_7_evaluate_ragas(
        dataset=ragas_dataset,
        metrics=ragas_metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=not args.no_progress,
    )
    step_8_export_results(
        records=rag_records,
        evaluation_data=evaluation_data,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()


