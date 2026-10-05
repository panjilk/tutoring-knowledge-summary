"""Chroma 向量存储集成测试，不下载真实 Embedding 模型。"""

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings.vector_store import VectorStoreManager
from src.exceptions import IndexConfigurationMismatchError


class FakeEmbeddings(Embeddings):
    """根据文本关键词生成固定向量，供离线测试使用。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        if "苹果" in text:
            return [1.0, 0.0, 0.0]
        if "香蕉" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FakeEmbeddingProvider:
    """提供测试 Embedding 实例。"""

    def __init__(self) -> None:
        self._embedding = FakeEmbeddings()

    def get(self) -> Embeddings:
        return self._embedding


def test_add_search_and_delete_documents(tmp_path: Path) -> None:
    """验证向量块可写入、按语义检索，并在最后完整删除。"""

    # tmp_path 是 pytest 提供的临时目录，测试结束后会自动清理，不会污染 data 目录。
    settings = build_settings(tmp_path / "vectors")
    manager = VectorStoreManager(settings, FakeEmbeddingProvider())
    knowledge_base_id = "11111111-1111-1111-1111-111111111111"

    documents = [
        Document(page_content="苹果是一种水果。", metadata={"chunk_id": "a"}),
        Document(page_content="香蕉通常是黄色的。", metadata={"chunk_id": "b"}),
    ]
    manager.add_documents(
        knowledge_base_id,
        documents,
        ["chunk-a", "chunk-b"],
    )

    results = manager.search(
        knowledge_base_id,
        "苹果",
        fetch_k=2,
        top_k=1,
        score_threshold=-1.0,
    )

    assert len(results) == 1
    assert "苹果" in results[0][0].page_content

    manager.delete_documents(knowledge_base_id, ["chunk-a", "chunk-b"])
    manager.delete_collection(knowledge_base_id)


def test_changed_embedding_model_is_rejected(tmp_path: Path) -> None:
    """验证已有向量索引不能在更换 Embedding 模型后被误用。"""

    vector_directory = tmp_path / "vectors"
    knowledge_base_id = "22222222-2222-2222-2222-222222222222"
    original_manager = VectorStoreManager(
        build_settings(vector_directory, model_name="model-a"),
        FakeEmbeddingProvider(),
    )
    original_manager.add_documents(
        knowledge_base_id,
        [Document(page_content="苹果是一种水果。")],
        ["chunk-a"],
    )

    changed_manager = VectorStoreManager(
        build_settings(vector_directory, model_name="model-b"),
        FakeEmbeddingProvider(),
    )

    try:
        # 旧 collection 的 metadata 仍记录 model-a，因此搜索必须抛出兼容性异常。
        changed_manager.search(
            knowledge_base_id,
            "苹果",
            fetch_k=1,
            top_k=1,
            score_threshold=-1.0,
        )
    except IndexConfigurationMismatchError:
        pass
    else:
        raise AssertionError("模型改变后应拒绝使用旧向量索引。")
    finally:
        original_manager.delete_collection(knowledge_base_id)


def test_search_can_filter_by_chapter_number(tmp_path: Path) -> None:
    """验证检索时 metadata_filter 会只返回指定章节。"""

    settings = build_settings(tmp_path / "vectors")
    manager = VectorStoreManager(settings, FakeEmbeddingProvider())
    knowledge_base_id = "33333333-3333-3333-3333-333333333333"
    documents = [
        Document(
            page_content="苹果是第三章的示例。",
            metadata={"chapter_number": 3},
        ),
        Document(
            page_content="苹果是第六章的示例。",
            metadata={"chapter_number": 6},
        ),
    ]
    manager.add_documents(
        knowledge_base_id,
        documents,
        ["chapter-3", "chapter-6"],
    )

    results = manager.search(
        knowledge_base_id,
        "苹果",
        fetch_k=2,
        top_k=2,
        score_threshold=-1.0,
        metadata_filter={"chapter_number": 3},
    )

    assert len(results) == 1
    assert results[0][0].metadata["chapter_number"] == 3
    manager.delete_collection(knowledge_base_id)


def test_get_documents_by_ids_returns_text_and_metadata(tmp_path: Path) -> None:
    """验证可按块 ID 读取相邻块，并保留调用方需要的 chunk_id。"""

    settings = build_settings(tmp_path / "vectors")
    manager = VectorStoreManager(settings, FakeEmbeddingProvider())
    knowledge_base_id = "44444444-4444-4444-4444-444444444444"
    manager.add_documents(
        knowledge_base_id,
        [
            Document(
                page_content="第一页末尾内容。",
                metadata={"page_number": 1, "chunk_index": 0},
            ),
            Document(
                page_content="第二页开头内容。",
                metadata={"page_number": 2, "chunk_index": 1},
            ),
        ],
        ["chunk-0", "chunk-1"],
    )

    documents = manager.get_documents_by_ids(
        knowledge_base_id,
        ["chunk-1", "chunk-0"],
    )

    assert set(documents) == {"chunk-0", "chunk-1"}
    assert documents["chunk-0"].page_content == "第一页末尾内容。"
    assert documents["chunk-0"].metadata["chunk_id"] == "chunk-0"
    assert documents["chunk-1"].metadata["page_number"] == 2
    manager.delete_collection(knowledge_base_id)


def build_settings(
    vector_directory: Path,
    *,
    model_name: str = "fake-model",
) -> SimpleNamespace:
    """创建向量存储测试所需的最小配置。"""

    # VectorStoreManager 只访问这些配置项；SimpleNamespace 可避免测试依赖真实 .env。
    return SimpleNamespace(
        vector_directory=vector_directory,
        embedding_model_name=model_name,
        chunk_size=800,
        chunk_overlap=120,
    )
