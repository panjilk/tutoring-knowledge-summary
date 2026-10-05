"""上传文件流式落盘测试。"""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.services.knowledge_base_service import (
    KnowledgeBaseService,
    save_upload_stream,
)


def test_save_upload_stream_without_extra_full_copy(tmp_path: Path) -> None:
    """验证大字节流会被正确写入，同时得到真实大小和 SHA-256。"""

    # 构造约 2 MB 内容，覆盖分块读取而非只测试极小文件的情况。
    content = (b"knowledge-base-content-" * 100_000) + b"end"
    source = io.BytesIO(content)
    destination = tmp_path / "upload.bin"

    file_hash, size_bytes = save_upload_stream(source, destination)

    assert size_bytes == len(content)
    assert file_hash == hashlib.sha256(content).hexdigest()
    assert destination.read_bytes() == content


def test_save_empty_upload(tmp_path: Path) -> None:
    """验证空文件仍有合法 SHA-256，但大小为零。"""

    destination = tmp_path / "empty.bin"

    file_hash, size_bytes = save_upload_stream(
        io.BytesIO(b""),
        destination,
    )

    assert size_bytes == 0
    assert file_hash == hashlib.sha256(b"").hexdigest()


def test_ingest_file_accepts_file_like_object(tmp_path: Path) -> None:
    """验证服务可直接处理 Streamlit 上传对象类似的二进制流。"""

    # 用 Mock 隔离数据库和向量库，只检查服务层的编排与落盘行为。
    settings = SimpleNamespace(
        upload_directory=tmp_path / "uploads",
        chunk_size=100,
        chunk_overlap=10,
        max_upload_batch_mb=1024,
    )
    repository = MagicMock()
    repository.find_document_by_hash.return_value = None
    repository.create_ingestion_job.return_value = "job-1"
    vector_store = MagicMock()
    service = KnowledgeBaseService(settings, repository, vector_store)

    result = service.ingest_file(
        "kb-1",
        "notes.txt",
        io.BytesIO("这是知识库内容。".encode()),
    )

    assert result.success is True
    assert result.chunk_count == 1
    repository.create_document.assert_called_once()
    vector_store.add_documents.assert_called_once()

    stored_path = Path(
        repository.create_document.call_args.kwargs["stored_path"]
    )
    assert stored_path.exists()
    assert stored_path.read_text(encoding="utf-8") == "这是知识库内容。"


def test_rebuild_index_reuses_original_file(tmp_path: Path) -> None:
    """验证重建索引读取已有原文件，而不是要求重新上传。"""

    source_file = tmp_path / "source.txt"
    source_file.write_text("用于重建索引的知识库内容。", encoding="utf-8")
    settings = SimpleNamespace(
        upload_directory=tmp_path / "uploads",
        chunk_size=100,
        chunk_overlap=10,
        max_upload_batch_mb=1024,
    )
    document = SimpleNamespace(
        id="document-1",
        stored_path=str(source_file),
        original_name="source.txt",
    )
    repository = MagicMock()
    repository.list_documents.return_value = [document]
    repository.create_ingestion_job.return_value = "job-1"
    vector_store = MagicMock()
    service = KnowledgeBaseService(settings, repository, vector_store)

    results = service.rebuild_index("kb-1")

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].chunk_count == 1
    vector_store.delete_collection.assert_called_once_with("kb-1")
    vector_store.add_documents.assert_called_once()
    assert repository.replace_chunk_records.call_count == 2
