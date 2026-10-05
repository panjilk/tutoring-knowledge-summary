"""知识库和文档导入业务逻辑。"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from src.config import Settings
from src.database import Repository
from src.database.models import DocumentModel, KnowledgeBaseModel
from src.document_loaders import (
    SUPPORTED_EXTENSIONS,
    build_document_overview,
    load_document,
    split_documents,
)
from src.embeddings import VectorStoreManager
from src.exceptions import DuplicateDocumentError, UnsupportedFileError

# 使用模块名作为日志来源，日志中可据此定位到本文件。
logger = logging.getLogger(__name__)
# 每次读取 1 MB，避免大文件导入时同时把整个文件放进内存。
UPLOAD_BUFFER_BYTES = 1024 * 1024


def save_upload_stream(
    content: bytes | BinaryIO,
    destination: Path,
) -> tuple[str, int]:
    """流式保存上传内容，并返回 SHA-256 和实际字节数。"""

    # 页面上传对象与直接传入的 bytes 统一转为“可按块读取的二进制流”。
    source: BinaryIO
    if isinstance(content, bytes):
        source = io.BytesIO(content)
    else:
        source = content

    try:
        source.seek(0)
    except (AttributeError, OSError):
        # 网络流等对象可能不支持 seek，此时直接从当前位置读取。
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0

    # 一边写入临时文件，一边累计哈希和大小；不用再额外读取一遍完整文件。
    with destination.open("wb") as destination_file:
        while True:
            data = source.read(UPLOAD_BUFFER_BYTES)
            if not data:
                break

            destination_file.write(data)
            digest.update(data)
            size_bytes += len(data)

    return digest.hexdigest(), size_bytes


@dataclass(frozen=True)
class IngestionResult:
    """单个文件导入结果。"""

    file_name: str
    success: bool
    message: str
    chunk_count: int = 0


class KnowledgeBaseService:
    """协调 MySQL、原始文件、文档解析和 Chroma。

    导入成功意味着同一份资料已经同步到三个位置：本地原文件、MySQL 元数据、
    Chroma 向量库。此类方法会在任一环节出错时尽力清理不完整的数据。
    """

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        vector_store: VectorStoreManager,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vector_store = vector_store

    def create_knowledge_base(self, name: str) -> KnowledgeBaseModel:
        """创建新知识库。"""

        return self._repository.create_knowledge_base(name)

    def list_knowledge_bases(self) -> list[KnowledgeBaseModel]:
        """列出所有知识库。"""

        return self._repository.list_knowledge_bases()

    def list_documents(self, knowledge_base_id: str) -> list[DocumentModel]:
        """列出知识库中的文档。"""

        return self._repository.list_documents(knowledge_base_id)

    @property
    def max_upload_batch_bytes(self) -> int:
        """单次批量上传允许的总字节数。"""

        return self._settings.max_upload_batch_mb * 1024 * 1024

    def ingest_file(
        self,
        knowledge_base_id: str,
        original_name: str,
        content: bytes | BinaryIO,
    ) -> IngestionResult:
        """保存、解析、分块并向量化一个上传文件。"""

        # 只保留文件名本身，丢弃用户可能传入的目录部分，避免写到上传目录之外。
        safe_name = Path(original_name).name.replace("\x00", "").strip()
        if not safe_name:
            return IngestionResult(original_name, False, "文件名无效。")

        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileError(
                f"文件 {safe_name} 格式不受支持，可用格式：{supported}。"
            )

        # 原始文件名可能重复，因此用 UUID 作为磁盘文件名和数据库主键。
        document_id = str(uuid4())
        knowledge_base_directory = (
            self._settings.upload_directory / knowledge_base_id
        )
        knowledge_base_directory.mkdir(parents=True, exist_ok=True)
        stored_path = knowledge_base_directory / f"{document_id}{extension}"
        temporary_path = knowledge_base_directory / (
            f"{document_id}{extension}.uploading"
        )

        try:
            # 先写到 .uploading 临时文件，只有校验通过后才原子替换为正式文件。
            file_hash, size_bytes = save_upload_stream(
                content,
                temporary_path,
            )
            if size_bytes == 0:
                temporary_path.unlink(missing_ok=True)
                return IngestionResult(safe_name, False, "文件内容为空。")

            # 内容哈希相同表示文件内容重复，即使用户改了文件名也能识别出来。
            duplicate = self._repository.find_document_by_hash(
                knowledge_base_id,
                file_hash,
            )
            if duplicate is not None:
                temporary_path.unlink(missing_ok=True)
                raise DuplicateDocumentError(
                    f"文件 {safe_name} 与已导入的 "
                    f"{duplicate.original_name} 内容相同。"
                )

            temporary_path.replace(stored_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        try:
            # 磁盘文件确认保存后，再创建 MySQL 元数据；失败时删除刚写入的文件。
            self._repository.create_document(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_name=safe_name,
                stored_path=str(stored_path),
                file_hash=file_hash,
                file_type=extension.removeprefix("."),
                size_bytes=size_bytes,
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        # 任务表记录导入状态，便于排查“处理中”或“失败”的文件。
        job_id = self._repository.create_ingestion_job(document_id)
        chunk_ids: list[str] = []

        try:
            self._repository.update_document_status(document_id, "processing")
            # 解析后得到带位置元数据的 LangChain Document 列表，例如 PDF 页码或 Markdown 标题。
            parsed_documents = load_document(
                stored_path,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_name=safe_name,
            )
            # 同时提取目录/标题概览，供“这本书讲什么”类整库问题使用。
            overview_text = build_document_overview(
                parsed_documents,
                safe_name,
            )
            self._repository.save_document_overview(
                document_id,
                overview_text,
            )
            # 长文需要切成重叠的小块；chunk_ids 既是 Chroma 向量 ID，也是 MySQL 中的关联记录。
            chunks, chunk_ids = split_documents(
                parsed_documents,
                document_id=document_id,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )

            if not chunks:
                raise ValueError("文档分块后没有得到可用内容。")

            # 先写向量库，成功后再记录块 ID，确保数据库不会指向不存在的向量。
            self._vector_store.add_documents(
                knowledge_base_id,
                chunks,
                chunk_ids,
            )
            self._repository.add_chunk_records(document_id, chunk_ids)
            self._repository.update_document_status(
                document_id,
                "completed",
                chunk_count=len(chunks),
            )
            self._repository.finish_ingestion_job(job_id, "completed")

            return IngestionResult(
                file_name=safe_name,
                success=True,
                message="导入成功。",
                chunk_count=len(chunks),
            )
        # 导入链路包含解析器、Embedding 和向量库，需统一记录最终失败状态。
        except Exception as exc:  # noqa: BLE001
            # 向量写入可能已完成一部分，失败时尽力清理，避免脏数据参与检索。
            if chunk_ids:
                try:
                    self._vector_store.delete_documents(
                        knowledge_base_id,
                        chunk_ids,
                    )
                except Exception:
                    logger.warning(
                        "清理导入失败后的向量数据时发生异常。",
                        exc_info=True,
                    )

            error_message = str(exc)
            self._repository.update_document_status(
                document_id,
                "failed",
                error_message=error_message,
            )
            self._repository.finish_ingestion_job(
                job_id,
                "failed",
                error_message,
            )
            return IngestionResult(safe_name, False, error_message)

    def delete_document(self, document_id: str) -> None:
        """同时删除向量、MySQL 记录和本地原始文件。"""

        document = self._repository.get_document(document_id)
        if document is None:
            return

        # 删除顺序很重要：先拿到块 ID、删除向量和数据库记录，最后才删除本地原文件。
        chunk_ids = self._repository.list_chunk_ids(document_id)
        self._vector_store.delete_documents(
            document.knowledge_base_id,
            chunk_ids,
        )
        self._repository.delete_document_record(document_id)
        Path(document.stored_path).unlink(missing_ok=True)

    def rebuild_index(
        self,
        knowledge_base_id: str,
    ) -> list[IngestionResult]:
        """保留原文件和元数据，重新生成整个知识库的向量索引。"""

        # 重建前先读取文件清单；删除 collection 后会按每个原始文件重新解析和入库。
        documents = self._repository.list_documents(knowledge_base_id)
        self._vector_store.delete_collection(knowledge_base_id)
        results: list[IngestionResult] = []

        for document in documents:
            job_id = self._repository.create_ingestion_job(document.id)
            chunk_ids: list[str] = []

            # collection 已清空，旧块 ID 不再有效，应先从 MySQL 清除。
            self._repository.replace_chunk_records(document.id, [])

            try:
                # 重建与首次导入相同，但复用已有的 document ID、磁盘文件和元数据。
                self._repository.update_document_status(
                    document.id,
                    "processing",
                    chunk_count=0,
                )
                parsed_documents = load_document(
                    Path(document.stored_path),
                    document_id=document.id,
                    knowledge_base_id=knowledge_base_id,
                    original_name=document.original_name,
                )
                overview_text = build_document_overview(
                    parsed_documents,
                    document.original_name,
                )
                self._repository.save_document_overview(
                    document.id,
                    overview_text,
                )
                chunks, chunk_ids = split_documents(
                    parsed_documents,
                    document_id=document.id,
                    chunk_size=self._settings.chunk_size,
                    chunk_overlap=self._settings.chunk_overlap,
                )
                if not chunks:
                    raise ValueError("文档分块后没有得到可用内容。")

                self._vector_store.add_documents(
                    knowledge_base_id,
                    chunks,
                    chunk_ids,
                )
                self._repository.replace_chunk_records(
                    document.id,
                    chunk_ids,
                )
                self._repository.update_document_status(
                    document.id,
                    "completed",
                    chunk_count=len(chunks),
                )
                self._repository.finish_ingestion_job(
                    job_id,
                    "completed",
                )
                results.append(
                    IngestionResult(
                        document.original_name,
                        True,
                        "索引重建成功。",
                        len(chunks),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if chunk_ids:
                    try:
                        self._vector_store.delete_documents(
                            knowledge_base_id,
                            chunk_ids,
                        )
                    except Exception:
                        logger.warning(
                            "清理索引重建失败后的向量时发生异常。",
                            exc_info=True,
                        )

                error_message = str(exc)
                self._repository.update_document_status(
                    document.id,
                    "failed",
                    chunk_count=0,
                    error_message=error_message,
                )
                self._repository.finish_ingestion_job(
                    job_id,
                    "failed",
                    error_message,
                )
                results.append(
                    IngestionResult(
                        document.original_name,
                        False,
                        error_message,
                    )
                )

        return results

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        """删除知识库的向量、本地文件和 MySQL 数据。"""

        # 删除顺序是向量库 → MySQL（级联删除相关记录）→ 本地上传目录。
        self._vector_store.delete_collection(knowledge_base_id)
        self._repository.delete_knowledge_base_record(knowledge_base_id)

        upload_directory = self._settings.upload_directory / knowledge_base_id
        if upload_directory.exists():
            shutil.rmtree(upload_directory)
