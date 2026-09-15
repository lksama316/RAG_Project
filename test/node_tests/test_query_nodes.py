import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import processor.query_processor.nodes.node_answer_output as answer_module
import processor.query_processor.nodes.node_item_name_confirm as confirm_module
import processor.query_processor.nodes.node_rerank as rerank_module
import processor.query_processor.nodes.node_search_embedding as embedding_search_module
import processor.query_processor.nodes.node_search_embedding_hyde as hyde_module
import processor.query_processor.nodes.node_web_search_mcp as web_module
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp


# ---------------------------------------------------------------------------
# node_rrf
# ---------------------------------------------------------------------------

def test_node_rrf_process_merges_and_deduplicates():
    state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "a", "content": "A1"}},
            {"entity": {"chunk_id": "b", "content": "B1"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "a", "content": "A2"}},
            {"entity": {"chunk_id": "c", "content": "C1"}},
        ],
    }

    result = NodeRrf().process(state)

    assert [doc["chunk_id"] for doc in result["rrf_chunks"]] == ["a", "b", "c"]
    assert result["rrf_chunks"][0]["content"] == "A1"


def test_node_rrf_merge_respects_max_results():
    node = NodeRrf()
    results = node._rrf_merge(
        [
            ([{"chunk_id": "a"}, {"chunk_id": "b"}], 1.0),
            ([{"chunk_id": "c"}], 1.0),
        ],
        max_results=2,
    )

    assert len(results) == 2
    assert {item[0]["chunk_id"] for item in results} == {"a", "c"}


# ---------------------------------------------------------------------------
# node_search_embedding
# ---------------------------------------------------------------------------

def test_node_search_embedding_process_success(monkeypatch):
    client = object()
    requests = ["dense_req", "sparse_req"]
    embeddings = {
        "dense": [[0.1, 0.2]],
        "sparse": [{1: 0.5}],
    }
    expected_chunks = [{"id": 1, "entity": {"chunk_id": "c1", "content": "content"}}]

    generate_mock = Mock(return_value=embeddings)
    request_mock = Mock(return_value=requests)
    client_mock = Mock(return_value=client)
    search_mock = Mock(return_value=[expected_chunks])
    monkeypatch.setattr(embedding_search_module, "generate_embeddings", generate_mock)
    monkeypatch.setattr(embedding_search_module, "create_hybrid_search_requests", request_mock)
    monkeypatch.setattr(embedding_search_module, "get_milvus_client", client_mock)
    monkeypatch.setattr(embedding_search_module, "hybrid_search", search_mock)

    result = NodeSearchEmbedding().process(
        {"rewritten_query": "HAK180 如何调温", "item_names": ["HAK180 烫金机"]}
    )

    assert result == {"embedding_chunks": expected_chunks}
    request_mock.assert_called_once_with(
        dense_vector=[0.1, 0.2],
        sparse_vector={1: 0.5},
        expr="item_name in ['HAK180 烫金机']",
        limit=10,
    )
    search_mock.assert_called_once()


def test_node_search_embedding_process_returns_empty_on_error(monkeypatch):
    monkeypatch.setattr(
        embedding_search_module,
        "generate_embeddings",
        Mock(side_effect=RuntimeError("embedding failed")),
    )

    result = NodeSearchEmbedding().process({"rewritten_query": "query", "item_names": []})

    assert result == {}


# ---------------------------------------------------------------------------
# node_search_embedding_hyde
# ---------------------------------------------------------------------------

def test_node_search_embedding_hyde_process_success(monkeypatch):
    node = NodeSearchEmbeddingHyde()
    expected_chunks = [{"entity": {"chunk_id": "hyde-1"}}]
    monkeypatch.setattr(node, "_step_1_create_hyde_doc", Mock(return_value="假设答案"))
    monkeypatch.setattr(node, "_step_2_search_embedding_hyde", Mock(return_value=expected_chunks))

    result = node.process({"rewritten_query": "问题", "item_names": ["HAK180"]})

    assert result == {
        "hyde_embedding_chunks": expected_chunks,
        "hyde_doc": "假设答案",
    }


def test_node_search_embedding_hyde_search_combines_query_and_doc(monkeypatch):
    embeddings = {"dense": [[0.3]], "sparse": [{3: 0.7}]}
    expected_chunks = [{"entity": {"chunk_id": "c1"}}]
    generate_mock = Mock(return_value=embeddings)
    request_mock = Mock(return_value=["dense_req", "sparse_req"])
    search_mock = Mock(return_value=[expected_chunks])

    monkeypatch.setattr(hyde_module, "generate_embeddings", generate_mock)
    monkeypatch.setattr(hyde_module, "create_hybrid_search_requests", request_mock)
    monkeypatch.setattr(hyde_module, "get_milvus_client", Mock(return_value=object()))
    monkeypatch.setattr(hyde_module, "hybrid_search", search_mock)

    result = NodeSearchEmbeddingHyde()._step_2_search_embedding_hyde(
        "rewritten query",
        "hypothetical answer",
        ["HAK180"],
    )

    assert result == expected_chunks
    generate_mock.assert_called_once_with(["rewritten query hypothetical answer"])
    request_mock.assert_called_once_with(
        dense_vector=[0.3],
        sparse_vector={3: 0.7},
        expr="item_name in ['HAK180']",
        limit=10,
    )


def test_node_search_embedding_hyde_returns_empty_on_error(monkeypatch):
    node = NodeSearchEmbeddingHyde()
    monkeypatch.setattr(node, "_step_1_create_hyde_doc", Mock(side_effect=RuntimeError("llm failed")))

    assert node.process({"rewritten_query": "query", "item_names": []}) == {}


# ---------------------------------------------------------------------------
# node_web_search_mcp
# ---------------------------------------------------------------------------

class FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, timeout):
        self.timeout = timeout
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, headers, **kwargs):
        payload = kwargs["json"]
        self.requests.append(payload)
        if payload["method"] == "initialize":
            return FakeHttpResponse({"jsonrpc": "2.0", "result": {"protocolVersion": "2024-11-05"}})

        pages = {
            "pages": [
                {"title": " 标题 ", "url": " https://example.com ", "snippet": " 摘要 "},
                {"title": "空摘要", "url": "https://empty.example.com", "snippet": "  "},
            ]
        }
        return FakeHttpResponse(
            {"jsonrpc": "2.0", "result": {"content": [{"text": json.dumps(pages)}]}}
        )


def test_node_web_search_mcp_parses_pages(monkeypatch):
    monkeypatch.setattr(web_module.httpx, "Client", FakeHttpClient)

    result = NodeWebSearchMcp()._mcp_call("HAK180")

    assert result == [
        {
            "title": "标题",
            "url": "https://example.com",
            "snippet": "摘要",
        }
    ]


def test_node_web_search_mcp_process_success_and_failure(monkeypatch):
    node = NodeWebSearchMcp()
    monkeypatch.setattr(node, "_mcp_call", Mock(return_value=[{"title": "T", "url": "U", "snippet": "S"}]))
    assert node.process({"rewritten_query": "query"})["web_search_docs"][0]["title"] == "T"

    monkeypatch.setattr(node, "_mcp_call", Mock(side_effect=RuntimeError("network failed")))
    assert node.process({"rewritten_query": "query"}) == {}
    assert node.process({"rewritten_query": ""}) == {}


# ---------------------------------------------------------------------------
# node_rerank
# ---------------------------------------------------------------------------

def test_node_rerank_process_sorts_and_applies_cliff_cutoff(monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "rerank_documents",
        Mock(return_value=[0.95, 0.80, 0.70, 0.10, 0.05]),
    )
    state = {
        "rewritten_query": "问题",
        "rrf_chunks": [
            {"chunk_id": "l1", "title": "L1", "content": "local 1"},
            {"chunk_id": "l2", "title": "L2", "content": "local 2"},
            {"chunk_id": "l3", "title": "L3", "content": "local 3"},
        ],
        "web_search_docs": [
            {"url": "https://web/1", "title": "W1", "snippet": "web 1"},
            {"url": "https://web/2", "title": "W2", "snippet": "web 2"},
        ],
    }

    result = NodeRerank().process(state)

    assert len(result["reranked_docs"]) == 3
    assert [doc["score"] for doc in result["reranked_docs"]] == [0.95, 0.8, 0.7]
    assert result["reranked_docs"][0]["source"] == "local"


def test_node_rerank_falls_back_when_api_fails(monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "rerank_documents",
        Mock(side_effect=RuntimeError("rerank failed")),
    )
    state = {
        "rewritten_query": "问题",
        "rrf_chunks": [{"chunk_id": "l1", "title": "L1", "content": "local"}],
        "web_search_docs": [{"url": "https://web", "title": "W", "snippet": "web"}],
    }

    result = NodeRerank().process(state)

    assert len(result["reranked_docs"]) == 2
    assert all("score" not in doc for doc in result["reranked_docs"])


# ---------------------------------------------------------------------------
# node_item_name_confirm
# ---------------------------------------------------------------------------

def test_node_item_name_confirm_extracts_and_cleans_llm_json(monkeypatch):
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='```json\n{"item_names": [" HAK 180 "], "rewritten_query": "改写问题"}\n```'
    )
    monkeypatch.setattr(confirm_module, "ChatOpenAI", Mock(return_value=llm))

    result = NodeItemNameConfirm()._step_4_extract_info("原问题", [])

    assert result == {
        "item_names": ["HAK180"],
        "rewritten_query": "改写问题",
    }


def test_node_item_name_confirm_extract_failure_uses_fallback(monkeypatch):
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("llm failed")
    monkeypatch.setattr(confirm_module, "ChatOpenAI", Mock(return_value=llm))

    result = NodeItemNameConfirm()._step_4_extract_info("原问题", [])

    assert result == {"item_names": [], "rewritten_query": "原问题"}


def test_node_item_name_confirm_process_confirmed_branch(monkeypatch):
    history = [{"_id": "history-1", "item_names": []}]
    save_mock = Mock(side_effect=["message-1", "message-1", "message-1"])
    update_mock = Mock()
    monkeypatch.setattr(confirm_module, "get_recent_messages", Mock(return_value=history))
    monkeypatch.setattr(confirm_module, "save_chat_message", save_mock)
    monkeypatch.setattr(confirm_module, "update_message_item_names", update_mock)

    node = NodeItemNameConfirm()
    monkeypatch.setattr(
        node,
        "_step_4_extract_info",
        Mock(return_value={"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么用"}),
    )
    monkeypatch.setattr(
        node,
        "_step_5_vectorize_and_query",
        Mock(
            return_value=[
                {
                    "extracted_name": "HAK180",
                    "matches": [{"item_name": "HAK180 烫金机", "score": 0.92}],
                }
            ]
        ),
    )

    state = {"session_id": "session-1", "original_query": "怎么用"}
    result = node.process(state)

    assert result["item_names"] == ["HAK180 烫金机"]
    assert result["answer"] == ""
    update_mock.assert_called_once_with(["history-1"], ["HAK180 烫金机"])
    assert save_mock.call_count == 2


def test_node_item_name_confirm_process_candidate_branch(monkeypatch):
    monkeypatch.setattr(confirm_module, "get_recent_messages", Mock(return_value=[]))
    save_mock = Mock(side_effect=["message-1", "message-1", "message-1"])
    monkeypatch.setattr(confirm_module, "save_chat_message", save_mock)
    monkeypatch.setattr(confirm_module, "update_message_item_names", Mock())

    node = NodeItemNameConfirm()
    monkeypatch.setattr(
        node,
        "_step_4_extract_info",
        Mock(return_value={"item_names": ["HAK180"], "rewritten_query": "问题"}),
    )
    monkeypatch.setattr(
        node,
        "_step_5_vectorize_and_query",
        Mock(
            return_value=[
                {
                    "extracted_name": "HAK180",
                    "matches": [
                        {"item_name": "HAK180 烫金机", "score": 0.70},
                        {"item_name": "HAK180 安全手册", "score": 0.65},
                    ],
                }
            ]
        ),
    )

    result = node.process({"session_id": "session-1", "original_query": "怎么用"})

    assert result["item_names"] == []
    assert "HAK180 烫金机" in result["answer"]
    assert "请明确一下型号" in result["answer"]
    assert save_mock.call_count == 3


def test_node_item_name_confirm_process_no_item_names_branch(monkeypatch):
    monkeypatch.setattr(confirm_module, "get_recent_messages", Mock(return_value=[]))
    save_mock = Mock(return_value="message-1")
    monkeypatch.setattr(confirm_module, "save_chat_message", save_mock)
    monkeypatch.setattr(confirm_module, "update_message_item_names", Mock())

    node = NodeItemNameConfirm()
    monkeypatch.setattr(
        node,
        "_step_4_extract_info",
        Mock(return_value={"item_names": [], "rewritten_query": "通用问题"}),
    )
    query_mock = Mock()
    monkeypatch.setattr(node, "_step_5_vectorize_and_query", query_mock)

    result = node.process({"session_id": "session-1", "original_query": "通用问题"})

    assert result["item_names"] == []
    assert result["answer"] == ""
    query_mock.assert_not_called()
    assert save_mock.call_count == 2


def test_node_item_name_confirm_vectorize_and_query(monkeypatch):
    embeddings = {"dense": [[0.1]], "sparse": [{1: 0.2}]}
    monkeypatch.setattr(confirm_module, "get_milvus_client", Mock(return_value=object()))
    monkeypatch.setattr(confirm_module, "generate_embeddings", Mock(return_value=embeddings))
    monkeypatch.setattr(confirm_module, "create_hybrid_search_requests", Mock(return_value=["reqs"]))
    monkeypatch.setattr(
        confirm_module,
        "hybrid_search",
        Mock(return_value=[[{"entity": {"item_name": "HAK180 烫金机"}, "distance": 0.88}]]),
    )

    result = NodeItemNameConfirm()._step_5_vectorize_and_query(["HAK180"])

    assert result == [
        {
            "extracted_name": "HAK180",
            "matches": [{"item_name": "HAK180 烫金机", "score": 0.88}],
        }
    ]


# ---------------------------------------------------------------------------
# node_answer_output
# ---------------------------------------------------------------------------

def test_node_answer_output_process_non_stream(monkeypatch, tmp_path):
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="最终回答")
    get_llm_mock = Mock(return_value=llm)
    save_history_mock = Mock()
    push_mock = Mock()
    set_result_mock = Mock()
    monkeypatch.setattr(answer_module, "get_llm_client", get_llm_mock)
    monkeypatch.setattr(answer_module, "save_chat_message", save_history_mock)
    monkeypatch.setattr(answer_module, "push_to_session", push_mock)
    monkeypatch.setattr(answer_module, "set_task_result", set_result_mock)

    state = {
        "session_id": "session-answer",
        "original_query": "原问题",
        "rewritten_query": "改写问题",
        "item_names": ["HAK180 烫金机"],
        "history": [{"role": "user", "text": "历史"}],
        "reranked_docs": [
            {
                "content": "正文 ![图](http://img/local.png)",
                "title": "标题",
                "source": "local",
                "chunk_id": "c1",
                "url": None,
                "score": 0.9,
            },
            {
                "content": "网页内容",
                "title": "网页",
                "source": "web",
                "chunk_id": None,
                "url": "http://img/web.jpg",
                "score": 0.8,
            },
        ],
        "is_stream": False,
        "answer": "",
    }

    result = NodeAnswerOutput().process(state)

    assert result["answer"] == "最终回答"
    assert "改写问题" in result["prompt"]
    assert set(answer_module.NodeAnswerOutput()._extract_images_from_docs(result["reranked_docs"])) == {
        "http://img/local.png",
        "http://img/web.jpg",
    }
    save_history_mock.assert_called_once()
    set_result_mock.assert_called_once_with("session-answer", "answer", "最终回答")
    push_mock.assert_not_called()


def test_node_answer_output_existing_stream_answer(monkeypatch):
    push_mock = Mock()
    save_history_mock = Mock()
    llm_mock = Mock()
    monkeypatch.setattr(answer_module, "push_to_session", push_mock)
    monkeypatch.setattr(answer_module, "save_chat_message", save_history_mock)
    monkeypatch.setattr(answer_module, "get_llm_client", llm_mock)

    state = {
        "session_id": "session-stream",
        "answer": "已有回答",
        "is_stream": True,
        "item_names": [],
        "reranked_docs": [],
    }

    result = NodeAnswerOutput().process(state)

    assert result["answer"] == "已有回答"
    assert push_mock.call_count == 2
    assert push_mock.call_args_list[0].args[1] == "delta"
    assert push_mock.call_args_list[1].args[1] == "final"
    llm_mock.assert_not_called()
    save_history_mock.assert_called_once()


def test_node_answer_output_stream_generation(monkeypatch):
    chunks = [
        SimpleNamespace(content="第一段"),
        SimpleNamespace(content="第二段"),
    ]
    llm = Mock()
    llm.stream.return_value = chunks
    push_mock = Mock()
    monkeypatch.setattr(answer_module, "get_llm_client", Mock(return_value=llm))
    monkeypatch.setattr(answer_module, "push_to_session", push_mock)

    state = {"session_id": "session-stream", "is_stream": True}
    result = NodeAnswerOutput()._step_3_generate_response(state, "prompt")

    assert result["answer"] == "第一段第二段"
    assert [call.args[2]["delta"] for call in push_mock.call_args_list] == ["第一段", "第二段"]


def test_node_answer_output_extract_images_deduplicates():
    docs = [
        {"content": "![a](http://img/a.png) ![a2](http://img/a.png)"},
        {"url": "http://img/b.jpg", "content": "![c](http://img/c.webp)"},
        {"url": "https://not-image.example.com", "content": "plain"},
    ]

    result = NodeAnswerOutput()._extract_images_from_docs(docs)

    assert result == ["http://img/a.png", "http://img/b.jpg", "http://img/c.webp"]