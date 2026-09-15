import json
import zipfile
from unittest.mock import Mock

import pytest

import processor.import_processor.nodes.node_import_milvus as milvus_node_module
import processor.import_processor.nodes.node_md_img as md_img_module
import processor.import_processor.nodes.node_pdf_to_md as pdf_node_module
import utils.embedding_utils as embedding_utils
from processor.import_processor.exceptions import (
    FileProcessingError,
    MilvusError,
    PdfConversionError,
    StateFieldError,
    ValidationError,
)
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# node_entry
# ---------------------------------------------------------------------------

def test_node_entry_md_success(tmp_path):
    md_file = tmp_path / "manual.md"
    md_file.write_text("# 标题\n正文", encoding="utf-8")

    state = {"import_file_path": str(md_file)}
    result = NodeEntry().process(state)

    assert result["is_md_read_enabled"] is True
    assert result["md_path"] == str(md_file)
    assert result["md_content"] == "# 标题\n正文"
    assert result["file_title"] == "manual"


def test_node_entry_pdf_success(tmp_path):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-test")

    result = NodeEntry().process({"import_file_path": str(pdf_file)})

    assert result["is_pdf_read_enabled"] is True
    assert result["pdf_path"] == str(pdf_file)
    assert result["file_title"] == "manual"


def test_node_entry_rejects_missing_and_unsupported_files(tmp_path):
    with pytest.raises(StateFieldError):
        NodeEntry().process({})

    missing = tmp_path / "missing.md"
    with pytest.raises(FileProcessingError):
        NodeEntry().process({"import_file_path": str(missing)})

    unsupported = tmp_path / "manual.txt"
    unsupported.write_text("text", encoding="utf-8")
    with pytest.raises(ValidationError):
        NodeEntry().process({"import_file_path": str(unsupported)})


# ---------------------------------------------------------------------------
# node_pdf_to_md
# ---------------------------------------------------------------------------

def test_node_pdf_to_md_process_success(tmp_path, monkeypatch):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-test")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    md_file = output_dir / "manual.md"
    md_file.write_text("# PDF 内容", encoding="utf-8")

    node = NodePDFToMD()
    monkeypatch.setattr(node, "_step_2_upload_and_poll", Mock(return_value="https://example.com/result.zip"))
    monkeypatch.setattr(
        node,
        "_step_3_download_and_extract",
        Mock(return_value=str(md_file)),
    )

    result = node.process({"pdf_path": str(pdf_file), "file_dir": str(output_dir)})

    assert result["md_path"] == str(md_file)
    assert result["md_content"] == "# PDF 内容"


def test_node_pdf_to_md_upload_and_poll_success(tmp_path, monkeypatch):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-test")

    post_response = FakeResponse(
        payload={
            "code": 0,
            "data": {
                "file_urls": ["https://upload.example.com/file"],
                "batch_id": "batch-1",
            },
        }
    )
    upload_response = FakeResponse()
    poll_response = FakeResponse(
        payload={
            "code": 0,
            "data": {
                "extract_result": [
                    {"state": "done", "full_zip_url": "https://example.com/result.zip"}
                ]
            },
        }
    )

    monkeypatch.setattr(pdf_node_module.requests, "post", Mock(return_value=post_response))
    monkeypatch.setattr(pdf_node_module.requests, "put", Mock(return_value=upload_response))
    monkeypatch.setattr(pdf_node_module.requests, "get", Mock(return_value=poll_response))

    result = NodePDFToMD()._step_2_upload_and_poll(pdf_file)

    assert result == "https://example.com/result.zip"


def test_node_pdf_to_md_upload_business_error(tmp_path, monkeypatch):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        pdf_node_module.requests,
        "post",
        Mock(return_value=FakeResponse(payload={"code": 500, "message": "failed"})),
    )

    with pytest.raises(PdfConversionError):
        NodePDFToMD()._step_2_upload_and_poll(pdf_file)


def test_node_pdf_to_md_download_and_extract(tmp_path, monkeypatch):
    zip_file = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_file, "w") as archive:
        archive.writestr("full.md", "# extracted")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(
        pdf_node_module.requests,
        "get",
        Mock(return_value=FakeResponse(content=zip_file.read_bytes())),
    )

    md_path = NodePDFToMD()._step_3_download_and_extract(
        "https://example.com/source.zip",
        output_dir,
        "manual",
    )

    assert md_path.endswith("manual.md")
    assert (output_dir / "manual" / "manual.md").read_text(encoding="utf-8") == "# extracted"


# ---------------------------------------------------------------------------
# node_md_img
# ---------------------------------------------------------------------------

def test_node_md_img_process_success(tmp_path, monkeypatch):
    md_file = tmp_path / "manual.md"
    md_file.write_text("![old](images/a.png)", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_file = images_dir / "a.png"
    image_file.write_bytes(b"image")
    new_md_file = tmp_path / "manual_new.md"

    node = NodeMDImg()
    monkeypatch.setattr(
        node,
        "_step_1_get_content",
        Mock(return_value=("![old](images/a.png)", md_file, images_dir)),
    )
    monkeypatch.setattr(
        node,
        "_step_2_scan_images",
        Mock(return_value=[("a.png", str(image_file), ("", ""))]),
    )
    monkeypatch.setattr(
        node,
        "_step_3_generate_summaries",
        Mock(return_value={"a.png": "图片摘要"}),
    )
    monkeypatch.setattr(node, "_step_4_upload_and_replace", Mock(return_value="![图片摘要](http://img/a.png)"))
    monkeypatch.setattr(node, "_step_5_backup_new_md_file", Mock(return_value=str(new_md_file)))

    result = node.process({"md_path": str(md_file), "md_content": "original"})

    assert result["md_content"] == "![图片摘要](http://img/a.png)"
    assert result["md_path"] == str(new_md_file)


def test_node_md_img_skips_when_images_directory_missing(tmp_path):
    md_file = tmp_path / "manual.md"
    md_file.write_text("plain text", encoding="utf-8")
    state = {"md_path": str(md_file)}

    result = NodeMDImg().process(state)

    assert result is state


def test_node_md_img_find_and_replace_image_reference():
    node = NodeMDImg()
    md_content = "说明文字 ![旧图](images/a.png) 后续文字"
    context = node._find_image_in_md(md_content, "a.png")

    assert context == ("说明文字 ", " 后续文字")
    replaced = node._process_md_file(
        md_content,
        {"a.png": ("新图片描述", "http://minio/a.png")},
    )
    assert replaced == "说明文字 ![新图片描述](http://minio/a.png) 后续文字"


# ---------------------------------------------------------------------------
# node_document_split
# ---------------------------------------------------------------------------

def test_node_document_split_process_success(tmp_path):
    md_file = tmp_path / "manual.md"
    md_file.write_text("# 一级标题\n\n正文\n\n## 二级标题\n\n更多正文", encoding="utf-8")

    state = {
        "file_title": "manual",
        "md_path": str(md_file),
        "md_content": md_file.read_text(encoding="utf-8"),
    }
    result = NodeDocumentSplit().process(state)

    assert result["chunks"]
    assert all(chunk["file_title"] == "manual" for chunk in result["chunks"])
    assert all("parent_title" in chunk for chunk in result["chunks"])
    backup = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert backup == result["chunks"]


def test_node_document_split_no_title_fallback():
    result = NodeDocumentSplit()._step_3_handle_no_title(
        "没有标题的正文",
        [],
        0,
        "manual",
    )

    assert result == [{"title": "无标题", "content": "没有标题的正文", "file_title": "manual"}]


def test_node_document_split_ignores_code_block_headings():
    content = "# 真标题\n\n```python\n# 代码注释\n```\n\n## 第二标题\n\n正文"
    sections, title_count, _ = NodeDocumentSplit()._step_2_split_by_titles(content, "manual")

    assert title_count == 2
    assert len(sections) == 2
    assert "代码注释" in sections[0]["content"]


def test_node_document_split_splits_long_section():
    section = {
        "title": "长章节",
        "content": "长章节\n\n" + ("这是一个测试句子。" * 600),
        "file_title": "manual",
    }

    chunks = NodeDocumentSplit()._split_long_section(section)

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 2000 for chunk in chunks)
    assert chunks[0]["parent_title"] == "长章节"


# ---------------------------------------------------------------------------
# node_item_name_recognition
# ---------------------------------------------------------------------------

def test_node_item_name_recognition_process_success(monkeypatch):
    node = NodeItemNameRecognition()
    dense_vector = [0.0] * 8
    sparse_vector = {1: 0.5}
    state = {
        "file_title": "HAK180 产品安全手册",
        "chunks": [
            {"title": "产品信息", "content": "HAK180 烫金机安全说明"},
            {"title": "电源", "content": "请使用规定电源"},
        ],
    }

    monkeypatch.setattr(node, "_step_3_call_llm", Mock(return_value="HAK180 烫金机"))
    monkeypatch.setattr(
        node,
        "_step_5_generate_vectors",
        Mock(return_value=(dense_vector, sparse_vector)),
    )
    save_mock = Mock()
    monkeypatch.setattr(node, "_step_6_save_to_milvus", save_mock)

    result = node.process(state)

    assert result["item_name"] == "HAK180 烫金机"
    assert all(chunk["item_name"] == "HAK180 烫金机" for chunk in result["chunks"])
    save_mock.assert_called_once_with(
        state,
        "HAK180 产品安全手册",
        "HAK180 烫金机",
        dense_vector,
        sparse_vector,
    )


def test_node_item_name_recognition_context_uses_first_three_chunks():
    node = NodeItemNameRecognition()
    chunks = [
        {"title": f"标题{i}", "content": f"内容{i}"}
        for i in range(1, 5)
    ]

    context = node._step_2_build_context(chunks)

    assert "【切片3】" in context
    assert "【切片4】" not in context


def test_node_item_name_recognition_validates_inputs():
    with pytest.raises(StateFieldError):
        NodeItemNameRecognition().process({"file_title": "manual", "chunks": []})

    with pytest.raises(StateFieldError):
        NodeItemNameRecognition().process({"file_title": "", "chunks": [{"title": "x", "content": "y"}]})


# ---------------------------------------------------------------------------
# node_bge_embedding
# ---------------------------------------------------------------------------

def test_node_bge_embedding_process_batches_chunks(monkeypatch):
    calls = []

    def fake_generate_embeddings(texts):
        calls.append(texts)
        return {
            "dense": [[float(index)] * 4 for index, _ in enumerate(texts)],
            "sparse": [{index: 1.0} for index, _ in enumerate(texts)],
        }

    monkeypatch.setattr(embedding_utils, "generate_embeddings", fake_generate_embeddings)
    chunks = [
        {"item_name": "HAK180", "content": f"内容{i}", "file_title": "manual"}
        for i in range(6)
    ]

    result = NodeBGEEmbedding().process({"chunks": chunks})

    assert len(result["chunks"]) == 6
    assert all("dense_vector" in chunk and "sparse_vector" in chunk for chunk in result["chunks"])
    assert [len(call) for call in calls] == [5, 1]
    assert calls[0][0] == "HAK180\n内容0"


def test_node_bge_embedding_validates_chunks():
    with pytest.raises(StateFieldError):
        NodeBGEEmbedding().process({"chunks": []})


# ---------------------------------------------------------------------------
# node_import_milvus
# ---------------------------------------------------------------------------

class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self):
        self.collections = set()
        self.created_collection = None
        self.deleted_filters = []
        self.inserted_data = None
        self.schema = None

    def has_collection(self, name):
        return name in self.collections

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, collection_name, schema, index_params):
        self.collections.add(collection_name)
        self.created_collection = {
            "name": collection_name,
            "schema": schema,
            "index_params": index_params,
        }

    def delete(self, collection_name, filter):
        self.deleted_filters.append((collection_name, filter))

    def insert(self, collection_name, data):
        self.inserted_data = data
        return {"insert_count": len(data), "ids": [101, 102]}


def test_node_import_milvus_process_success(monkeypatch):
    client = FakeMilvusClient()
    monkeypatch.setattr(milvus_node_module, "get_milvus_client", Mock(return_value=client))
    chunks = [
        {
            "content": "内容1",
            "title": "标题1",
            "parent_title": "父标题",
            "file_title": "manual",
            "item_name": "HAK180",
            "dense_vector": [0.1, 0.2],
            "sparse_vector": {1: 0.5},
        },
        {
            "content": "内容2",
            "title": "标题2",
            "parent_title": "父标题",
            "file_title": "manual",
            "item_name": "HAK180",
            "dense_vector": [0.3, 0.4],
            "sparse_vector": {2: 0.6},
        },
    ]

    result = NodeImportMilvus().process({"chunks": chunks})

    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["101", "102"]
    assert client.created_collection is not None
    assert client.deleted_filters == [(milvus_node_module.milvus_config.chunks_collection, "file_title=='manual'")]
    assert all(item["part"] == 0 for item in client.inserted_data)


def test_node_import_milvus_rejects_missing_vectors():
    with pytest.raises(StateFieldError):
        NodeImportMilvus().process({"chunks": [{"content": "no vectors"}]})


def test_node_import_milvus_rejects_unavailable_client(monkeypatch):
    monkeypatch.setattr(milvus_node_module, "get_milvus_client", Mock(return_value=None))
    chunks = [{"dense_vector": [0.1], "sparse_vector": {0: 1.0}, "file_title": "manual"}]

    with pytest.raises(MilvusError):
        NodeImportMilvus().process({"chunks": chunks})