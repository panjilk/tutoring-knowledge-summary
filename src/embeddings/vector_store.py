"""本地 Embedding 模型和 Chroma 向量数据库封装。"""

from __future__ import annotations

from collections.abc import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings
from src.exceptions import IndexConfigurationMismatchError


class EmbeddingProvider:
    """延迟加载本地向量模型，避免应用启动时立即占用大量资源。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedding: Embeddings | None = None

    def get(self) -> Embeddings:
        """获取唯一的 Embedding 模型实例。"""

        if self._embedding is None:
            # 首次真正需要检索或导入时才下载/加载模型，后续重用同一实例。
            self._embedding = HuggingFaceEmbeddings(
                model_name=self._settings.embedding_model_name,
                model_kwargs={"device": self._settings.embedding_device},
                encode_kwargs={
                    # 归一化后余弦相似度只比较向量方向，和 collection 的 cosine 配置匹配。
                    "batch_size": self._settings.embedding_batch_size,
                    "normalize_embeddings": True,
                },
            )

        return self._embedding


class VectorStoreManager:
    """按知识库隔离 Chroma collection。

    一个知识库对应 Chroma 中一个 collection；这样删除或重建一个知识库不会影响其他资料。
    """

    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider

    @staticmethod
    def collection_name(knowledge_base_id: str) -> str:
        """把 UUID 转换为符合 Chroma 规则的 collection 名称。"""

        # UUID 含连字符，而 collection 名称使用下划线更稳妥。
        normalized_id = knowledge_base_id.replace("-", "_")
        return f"kb_{normalized_id}"

    def get_store(self, knowledge_base_id: str) -> Chroma:
        """获取指定知识库的向量存储。"""

        # 复用旧 collection 前检查模型和分块参数，防止不同向量空间混用。
        self._validate_collection_configuration(knowledge_base_id)
        return Chroma(
            collection_name=self.collection_name(knowledge_base_id),
            embedding_function=self._embedding_provider.get(),
            persist_directory=str(self._settings.vector_directory),
            collection_metadata=self._expected_collection_metadata(),
        )

    def add_documents(
        self,
        knowledge_base_id: str,
        documents: Sequence[Document],
        document_ids: Sequence[str],
    ) -> None:
        """分批写入向量，降低大量文档导入时的内存压力。"""

        if len(documents) != len(document_ids):
            raise ValueError("文档块数量与 ID 数量不一致。")

        # Chroma 单次写入太多会占用大量内存，固定每 128 个块写一次。
        vector_store = self.get_store(knowledge_base_id)
        batch_size = 128

        for start_index in range(0, len(documents), batch_size):
            end_index = start_index + batch_size
            vector_store.add_documents(
                documents=list(documents[start_index:end_index]),
                ids=list(document_ids[start_index:end_index]),
            )

    def search(
        self,
        knowledge_base_id: str,
        question: str,
        *,
        fetch_k: int,
        top_k: int,
        score_threshold: float,
        metadata_filter: dict[str, str | int] | None = None,
    ) -> list[tuple[Document, float]]:
        """检索并过滤相关文档块。"""

        vector_store = self.get_store(knowledge_base_id)
        search_options = {}
        if metadata_filter is not None:
            search_options["filter"] = metadata_filter

        # 先多取 fetch_k 个候选，再根据阈值过滤，最后保留 top_k 个最相关的块。
        candidates = vector_store.similarity_search_with_relevance_scores(
            question,
            k=fetch_k,
            **search_options,
        )

        relevant_results = [
            (document, score)
            for document, score in candidates
            if score >= score_threshold
        ]
        return relevant_results[:top_k]

    def get_documents_by_ids(
        self,
        knowledge_base_id: str,
        document_ids: Sequence[str],
    ) -> dict[str, Document]:
        """按向量块 ID 读取原文和元数据，供检索后补充相邻上下文。"""

        if not document_ids:
            return {}

        # dict.fromkeys 在去重的同时保留调用方给出的 ID 顺序。
        unique_ids = list(dict.fromkeys(document_ids))
        result = self.get_store(knowledge_base_id).get(
            ids=unique_ids,
            include=["documents", "metadatas"],
        )
        result_ids = result.get("ids") or []
        contents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        documents_by_id: dict[str, Document] = {}

        for result_index, chunk_id in enumerate(result_ids):
            if result_index >= len(contents):
                continue

            content = contents[result_index]
            if content is None:
                continue

            raw_metadata = (
                metadatas[result_index]
                if result_index < len(metadatas)
                else None
            )
            metadata = dict(raw_metadata or {})
            # 旧索引即使没有显式保存 chunk_id，也可以用 Chroma 返回的 ID 补齐。
            metadata.setdefault("chunk_id", chunk_id)
            documents_by_id[chunk_id] = Document(
                page_content=content,
                metadata=metadata,
            )

        return documents_by_id

    def delete_documents(
        self,
        knowledge_base_id: str,
        document_ids: Sequence[str],
    ) -> None:
        """根据向量块 ID 精确删除文档内容。"""

        # 空列表不访问 Chroma，删除不存在的 collection 也不会抛异常。
        if not document_ids:
            return

        collection = self._get_existing_collection(knowledge_base_id)
        if collection is not None:
            collection.delete(ids=list(document_ids))

    def delete_collection(self, knowledge_base_id: str) -> None:
        """删除整个知识库对应的向量 collection。"""

        client = self._get_client()
        collection_name = self.collection_name(knowledge_base_id)

        if self._get_existing_collection(knowledge_base_id) is not None:
            client.delete_collection(collection_name)

    def _get_client(self) -> chromadb.ClientAPI:
        """创建指向本地持久化目录的轻量 Chroma 客户端。"""

        return chromadb.PersistentClient(
            path=str(self._settings.vector_directory)
        )

    def _get_existing_collection(self, knowledge_base_id: str):
        """获取已存在的 collection，不存在时返回 None。"""

        client = self._get_client()
        collection_name = self.collection_name(knowledge_base_id)
        # Chroma 的 get_collection 不存在时会报错，先查询名称列表使删除/检查可幂等执行。
        collection_names = {
            collection.name
            for collection in client.list_collections()
        }
        if collection_name not in collection_names:
            return None

        return client.get_collection(collection_name)

    def _expected_collection_metadata(self) -> dict[str, str | int]:
        """生成会影响向量兼容性的索引配置。"""

        return {
            "hnsw:space": "cosine",
            "rag_embedding_model": self._settings.embedding_model_name,
            "rag_chunk_size": self._settings.chunk_size,
            "rag_chunk_overlap": self._settings.chunk_overlap,
        }

    def _validate_collection_configuration(
        self,
        knowledge_base_id: str,
    ) -> None:
        """阻止不同模型或分块规则生成的向量混在一起。"""

        collection = self._get_existing_collection(knowledge_base_id)
        if collection is None:
            return

        # 只有影响向量语义的配置才写到 metadata；它们改变后必须重建已有索引。
        expected_metadata = self._expected_collection_metadata()
        actual_metadata = collection.metadata or {}
        incompatible_keys = [
            key
            for key, expected_value in expected_metadata.items()
            if actual_metadata.get(key) != expected_value
        ]
        if not incompatible_keys:
            return

        if collection.count() == 0:
            # 空 collection 没有旧向量，直接更新其配置即可，无需要求用户重建。
            collection.modify(
                metadata={**actual_metadata, **expected_metadata}
            )
            return

        changed_names = "、".join(incompatible_keys)
        raise IndexConfigurationMismatchError(
            "当前知识库的向量索引配置与应用不一致："
            f"{changed_names}。请恢复原配置，或重建该知识库索引。"
        )
