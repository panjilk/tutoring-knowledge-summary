"""SQLAlchemy 数据表模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """返回不带时区标记的 UTC 时间，便于 MySQL 统一存储。"""

    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """所有数据表模型的基类。"""


class KnowledgeBaseModel(Base):
    """知识库基本信息。"""

    __tablename__ = "knowledge_bases"

    # UUID 以字符串保存；不用自增数字，便于本地文件和向量 collection 也使用同一标识。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ChatSessionModel(Base):
    """一次可以持续追问的聊天会话。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 删除知识库时，CASCADE 会让数据库自动删除属于它的聊天会话。
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ChatMessageModel(Base):
    """聊天会话中的一条用户或助手消息。"""

    __tablename__ = "chat_messages"
    # 同一会话中消息序号不可重复，保证历史对话可按 sequence_number 可靠还原。
    __table_args__ = (
        UniqueConstraint(
            "chat_session_id",
            "sequence_number",
            name="uq_chat_message_session_sequence",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 将 RAG 实际检索到的来源快照保存为 JSON，历史回答可继续展示原始引用。
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )


class DocumentModel(Base):
    """已上传文档和处理状态。"""

    __tablename__ = "documents"
    # 同一知识库对文件内容哈希做唯一约束，阻止同一文件被重复导入。
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "file_hash",
            name="uq_document_kb_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 数据库只保存原文件的路径；实际大文件仍放在 data/uploads 目录而非 MySQL 中。
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class DocumentOverviewModel(Base):
    """文档的目录、章节和全局结构概览。"""

    __tablename__ = "document_overviews"

    # document_id 同时是外键和主键，因此一份文档只能对应一份最新概览。
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    overview_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class DocumentChunkModel(Base):
    """记录文档块与向量库 ID 的对应关系。"""

    __tablename__ = "document_chunks"

    # ID 与 Chroma 中的向量 ID 完全相同，删除文档时可据此精确删除向量。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)


class IngestionJobModel(Base):
    """文档导入任务记录。"""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
