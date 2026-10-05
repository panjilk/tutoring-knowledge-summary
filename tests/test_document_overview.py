"""文档结构概览提取测试。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.document_loaders import build_document_overview
from src.services.rag_service import RagService


def test_build_document_overview_extracts_toc_and_headings() -> None:
    """验证概览同时保留目录原文、章节/小节标题和页码位置。"""

    documents = [
        Document(
            page_content=(
                "目录\n"
                "第1章 有理数\n"
                "1.1 从自然数到有理数\n"
                "第2章 有理数的运算\n"
                "2.1 有理数的加法"
            ),
            metadata={"page_number": 6},
        ),
        Document(
            page_content="第3章 实数\n3.1 平方根",
            metadata={"page_number": 81},
        ),
    ]

    overview = build_document_overview(documents, "数学教材.pdf")

    assert "数学教材.pdf" in overview
    assert "目录原文" in overview
    assert "第1章 有理数" in overview
    assert "1.1 从自然数到有理数" in overview
    assert "第3章 实数" in overview
    assert "第 6 页" in overview


def test_math_decimals_are_not_treated_as_section_headings() -> None:
    """验证数学小数和不等式不会被误识别为“1.1”小节标题。"""

    documents = [
        Document(
            page_content=(
                "计算平方根的近似值\n"
                "1.42 ＜ 2 ＜ 1.52\n"
                "1.4142 ＜ 2 ＜ 1.4152\n"
                "1.1\n"
                "真正的小节标题"
            ),
            metadata={"page_number": 10},
        )
    ]

    overview = build_document_overview(documents, "数学.pdf")

    assert "1.1 真正的小节标题" in overview
    assert "1.4142" not in overview
    assert "目录原文" not in overview


def test_overview_question_uses_saved_document_structure() -> None:
    """验证整库概览问题直接使用已保存的结构概览，不进行向量搜索。"""

    # Mock 数据库返回一份已经在导入时保存好的概览记录。
    settings = SimpleNamespace()
    vector_store = MagicMock()
    repository = MagicMock()
    repository.list_document_overviews.return_value = [
        (
            SimpleNamespace(
                id="document-1",
                original_name="数学教材.pdf",
            ),
            SimpleNamespace(
                overview_text=(
                    "文件：数学教材.pdf\n"
                    "章节结构：第1章 有理数；第2章 有理数的运算"
                )
            ),
        )
    ]
    service = RagService(
        settings,
        vector_store,
        MagicMock(),
        repository,
    )

    prepared_answer = service.prepare_answer(
        "kb-1",
        "这个知识库里有哪些知识？",
    )

    assert prepared_answer.mode == "overview"
    assert len(prepared_answer.sources) == 1
    assert prepared_answer.sources[0].file_name == "数学教材.pdf"
    assert "第1章 有理数" in prepared_answer.messages[1].content
    vector_store.search.assert_not_called()


def test_specific_question_does_not_trigger_overview_mode() -> None:
    """验证只有宽泛概览问题才进入 overview 模式。"""

    assert RagService._is_overview_question("什么是有理数？") is False
    assert RagService._is_overview_question("这本书讲了什么？") is True
