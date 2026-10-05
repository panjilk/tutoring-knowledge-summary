"""RAG 提示词和引用映射测试。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.services.rag_service import RagService, SourceReference


def test_build_sources_uses_real_metadata() -> None:
    """验证引用卡片从检索结果的真实文件名、页码和分数构造。"""

    document = Document(
        page_content="这是知识库中的真实原文。",
        metadata={
            "source_file": "manual.pdf",
            "page_number": 8,
            "chunk_id": "chunk-1",
        },
    )

    sources = RagService._build_sources([(document, 0.88)])

    assert len(sources) == 1
    assert sources[0].number == 1
    assert sources[0].file_name == "manual.pdf"
    assert sources[0].location == "第 8 页"
    assert sources[0].score == 0.88


def test_context_contains_fixed_reference_number() -> None:
    """验证发给模型的上下文中含固定的 [1] 引用编号和来源位置。"""

    document = Document(
        page_content="系统使用本地向量模型。",
        metadata={
            "source_file": "design.md",
            "section_title": "技术选型",
        },
    )

    context = RagService._build_context([(document, 0.9)])

    assert "[1]" in context
    assert "design.md" in context
    assert "章节：技术选型" in context


def test_follow_up_question_uses_previous_user_question() -> None:
    """验证带指代词的追问会附带上一条用户问题用于检索。"""

    history = [
        {"role": "user", "content": "什么是向量数据库？"},
        {"role": "assistant", "content": "向量数据库用于相似度检索。"},
    ]

    retrieval_query = RagService._build_retrieval_query(
        "它有什么作用？",
        history,
    )

    assert "什么是向量数据库" in retrieval_query
    assert "它有什么作用" in retrieval_query


def test_invalid_citation_numbers_are_filtered() -> None:
    """验证模型编造的 [9] 不会显示为真实引用来源。"""

    sources = [
        SourceReference(1, "a.pdf", "第 1 页", "内容 A", 0.9, "a"),
        SourceReference(2, "b.pdf", "第 2 页", "内容 B", 0.8, "b"),
    ]

    result = RagService.validate_citations(
        "结论来自资料 [1]，另一个错误来源是 [9]。",
        sources,
    )

    assert [source.number for source in result.sources] == [1]
    assert result.invalid_numbers == [9]
    assert result.missing_citations is False


def test_missing_citations_keeps_retrieved_sources() -> None:
    """验证回答未标注引用时，界面仍可展示所有本次检索资料。"""

    sources = [
        SourceReference(1, "a.pdf", "第 1 页", "内容 A", 0.9, "a"),
    ]

    result = RagService.validate_citations("回答没有引用。", sources)

    assert list(result.sources) == sources
    assert result.missing_citations is True


def test_chapter_summary_uses_metadata_filter_and_more_chunks() -> None:
    """验证章节总结扩大召回量，且用元数据过滤其他章节。"""

    # Mock 向量库记录调用参数，使测试无需真实 Chroma 或 Embedding 模型。
    settings = SimpleNamespace(
        retrieval_top_k=6,
        retrieval_fetch_k=20,
        retrieval_score_threshold=0.35,
    )
    vector_store = MagicMock()
    vector_store.search.return_value = [
        (
            Document(
                page_content="第三章重点内容。",
                metadata={
                    "source_file": "数学.pdf",
                    "page_number": 81,
                    "chapter_number": 3,
                    "chapter_title": "实数",
                    "chunk_id": "chunk-3",
                },
            ),
            0.8,
        )
    ]
    service = RagService(
        settings,
        vector_store,
        MagicMock(),
        MagicMock(),
    )

    prepared = service.prepare_answer(
        "kb-1",
        "帮我总结一下第三章的内容重点",
    )

    assert prepared.mode == "chapter"
    assert prepared.sources[0].location == "第 81 页 · 第 3 章 实数"
    vector_store.search.assert_called_once_with(
        "kb-1",
        "第3章 章节总结 主要内容 知识重点 知识梳理 小结与反思",
        fetch_k=36,
        top_k=12,
        score_threshold=0.35,
        metadata_filter={"chapter_number": 3},
    )
    assert "不得混入其他章节" in prepared.messages[0].content
    assert "至少 3 个不同的引用编号" in prepared.messages[0].content


def test_retrieval_adds_previous_chunk_across_pdf_pages() -> None:
    """验证命中第二页开头时，会把第一页末尾作为相邻上下文补入。"""

    settings = SimpleNamespace(
        retrieval_top_k=6,
        retrieval_fetch_k=20,
        retrieval_score_threshold=0.35,
    )
    vector_store = MagicMock()
    vector_store.search.return_value = [
        (
            Document(
                page_content="Chroma，它主要负责保存和检索文本向量。",
                metadata={
                    "document_id": "doc-1",
                    "source_file": "设计说明.pdf",
                    "page_number": 2,
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                },
            ),
            0.91,
        )
    ]
    vector_store.get_documents_by_ids.return_value = {
        "chunk-0": Document(
            page_content="该系统使用的数据库是",
            metadata={
                "document_id": "doc-1",
                "source_file": "设计说明.pdf",
                "page_number": 1,
                "chunk_id": "chunk-0",
                "chunk_index": 0,
            },
        )
    }
    repository = MagicMock()
    repository.list_chunk_ids.return_value = ["chunk-0", "chunk-1"]
    service = RagService(
        settings,
        vector_store,
        MagicMock(),
        repository,
    )

    prepared = service.prepare_answer(
        "kb-1",
        "系统使用的数据库是什么？",
    )

    assert len(prepared.sources) == 1
    source = prepared.sources[0]
    assert source.score == 0.91
    assert source.location == "第 1 页、第 2 页（含相邻上下文）"
    assert source.excerpt.index("该系统使用的数据库是") < source.excerpt.index(
        "Chroma，它主要负责保存和检索文本向量。"
    )
    assert "【第 1 页】" in source.excerpt
    assert "【第 2 页】" in source.excerpt
    repository.list_chunk_ids.assert_called_once_with("doc-1")
    vector_store.get_documents_by_ids.assert_called_once_with(
        "kb-1",
        ["chunk-0"],
    )


def test_chinese_chapter_number_is_extracted() -> None:
    """验证章节号解析兼容中文数字、阿拉伯数字和无章节号的问题。"""

    assert RagService._extract_requested_chapter("总结第三章") == 3
    assert RagService._extract_requested_chapter("查看第 12 章") == 12
    assert RagService._extract_requested_chapter("什么是有理数") is None
