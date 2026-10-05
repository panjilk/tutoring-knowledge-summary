"""聊天历史数据库操作测试。"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, KnowledgeBaseModel
from src.database.repository import Repository


def build_repository() -> Repository:
    """创建只供当前测试使用的内存数据库。"""

    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))

    Base.metadata.create_all(engine)

    repository = Repository.__new__(Repository)
    repository._engine = engine
    repository._session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    return repository


def add_knowledge_base(repository: Repository) -> str:
    """向测试数据库写入聊天会话所属的知识库。"""

    knowledge_base_id = "kb-1"
    with repository._session_factory.begin() as session:
        session.add(
            KnowledgeBaseModel(
                id=knowledge_base_id,
                name="测试知识库",
            )
        )
    return knowledge_base_id


def test_chat_history_is_saved_in_message_order() -> None:
    """验证标题会清理空格，消息按自增序号读取，引用 JSON 会保存。"""

    repository = build_repository()
    knowledge_base_id = add_knowledge_base(repository)

    chat_session = repository.create_chat_session(
        knowledge_base_id,
        "  如何使用   向量数据库？  ",
    )
    repository.add_chat_message(
        chat_session.id,
        "user",
        "向量数据库有什么作用？",
    )
    repository.add_chat_message(
        chat_session.id,
        "assistant",
        "它可以执行相似度检索。[1]",
        [
            {
                "number": 1,
                "file_name": "说明.pdf",
                "location": "第 1 页",
                "excerpt": "向量数据库用于相似度检索。",
                "score": 0.9,
                "chunk_id": "chunk-1",
            }
        ],
    )

    sessions = repository.list_chat_sessions(knowledge_base_id)
    messages = repository.list_chat_messages(chat_session.id)

    assert sessions[0].title == "如何使用 向量数据库？"
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.sequence_number for message in messages] == [1, 2]
    assert messages[1].sources[0]["file_name"] == "说明.pdf"


def test_chat_session_can_be_cleared_and_deleted() -> None:
    """验证清空操作保留会话，而删除操作移除会话及关联消息。"""

    repository = build_repository()
    knowledge_base_id = add_knowledge_base(repository)
    chat_session = repository.create_chat_session(
        knowledge_base_id,
        "需要删除的对话",
    )
    repository.add_chat_message(
        chat_session.id,
        "user",
        "测试问题",
    )

    repository.clear_chat_messages(chat_session.id)
    assert repository.list_chat_messages(chat_session.id) == []

    repository.delete_chat_session(chat_session.id)
    assert repository.get_chat_session(chat_session.id) is None
    assert repository.list_chat_sessions(knowledge_base_id) == []


def test_document_overview_is_saved_and_listed() -> None:
    """验证已完成文档可保存并读取其结构概览。"""

    repository = build_repository()
    knowledge_base_id = add_knowledge_base(repository)
    document = repository.create_document(
        document_id="document-1",
        knowledge_base_id=knowledge_base_id,
        original_name="教材.pdf",
        stored_path="教材.pdf",
        file_hash="hash-1",
        file_type="pdf",
        size_bytes=100,
    )
    repository.update_document_status(document.id, "completed")

    repository.save_document_overview(
        document.id,
        "第1章 有理数\n第2章 有理数的运算",
    )
    overview_records = repository.list_document_overviews(
        knowledge_base_id
    )

    assert len(overview_records) == 1
    saved_document, saved_overview = overview_records[0]
    assert saved_document.original_name == "教材.pdf"
    assert "第2章" in saved_overview.overview_text
