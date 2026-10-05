"""封装 MySQL 的常用数据操作。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings
from src.database.models import (
    Base,
    ChatMessageModel,
    ChatSessionModel,
    DocumentChunkModel,
    DocumentModel,
    DocumentOverviewModel,
    IngestionJobModel,
    KnowledgeBaseModel,
    utc_now,
)
from src.exceptions import DatabaseConnectionError, DuplicateDocumentError


class Repository:
    """知识库、文档和任务的统一数据访问入口。

    ``with self._session_factory.begin()`` 会在代码块正常结束时提交事务，
    出现异常时自动回滚；纯查询使用不带 ``begin`` 的会话即可。
    """

    def __init__(self, settings: Settings) -> None:
        # pool_pre_ping 在复用连接前探测其是否失效，适合可能重启的本地 MySQL。
        self._engine = create_engine(
            settings.mysql_url,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def initialize(self) -> None:
        """连接数据库并创建尚不存在的数据表。"""

        try:
            # 根据 models.py 中的表定义创建不存在的表，不会删除已有数据。
            Base.metadata.create_all(self._engine)
        except OperationalError as exc:
            raise DatabaseConnectionError(
                "无法连接 MySQL。请确认服务已启动、数据库已创建，"
                "并检查 .env 中的 MYSQL_* 配置。"
            ) from exc

    def check_connection(self) -> bool:
        """检查 MySQL 是否可以执行查询。"""

        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def create_knowledge_base(self, name: str) -> KnowledgeBaseModel:
        """创建知识库。"""

        # 去掉纯空白名称；唯一约束作为并发情况下的最后一道保护。
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("知识库名称不能为空。")

        knowledge_base = KnowledgeBaseModel(
            id=str(uuid4()),
            name=clean_name,
        )

        try:
            with self._session_factory.begin() as session:
                session.add(knowledge_base)
        except IntegrityError as exc:
            raise ValueError("知识库名称已存在。") from exc

        return knowledge_base

    def list_knowledge_bases(self) -> list[KnowledgeBaseModel]:
        """按更新时间倒序获取知识库。"""

        with self._session_factory() as session:
            # 按最近活动时间而非创建时间排序，方便用户继续最近使用的知识库。
            statement = select(KnowledgeBaseModel).order_by(
                KnowledgeBaseModel.updated_at.desc()
            )
            return list(session.scalars(statement).all())

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseModel | None:
        """根据 ID 获取知识库。"""

        with self._session_factory() as session:
            return session.get(KnowledgeBaseModel, knowledge_base_id)

    def delete_knowledge_base_record(self, knowledge_base_id: str) -> None:
        """删除知识库记录，关联文档由 MySQL 外键级联删除。"""

        with self._session_factory.begin() as session:
            # 外键 ondelete=CASCADE 会自动删除会话、文档、概览和块记录。
            session.execute(
                delete(KnowledgeBaseModel).where(
                    KnowledgeBaseModel.id == knowledge_base_id
                )
            )

    def create_chat_session(
        self,
        knowledge_base_id: str,
        title: str,
    ) -> ChatSessionModel:
        """创建聊天会话，标题通常取自第一条用户问题。"""

        # 连续换行/空格折叠为一个空格，避免第一条问题作为标题时显示得杂乱。
        clean_title = " ".join(title.split()).strip()
        if not clean_title:
            clean_title = "新对话"

        # 数据库字段最多保存 120 个字符，界面标题保持简短更容易浏览。
        if len(clean_title) > 50:
            clean_title = f"{clean_title[:49]}…"

        chat_session = ChatSessionModel(
            id=str(uuid4()),
            knowledge_base_id=knowledge_base_id,
            title=clean_title,
        )

        with self._session_factory.begin() as session:
            session.add(chat_session)
            self._touch_knowledge_base(session, knowledge_base_id)

        return chat_session

    def list_chat_sessions(
        self,
        knowledge_base_id: str,
    ) -> list[ChatSessionModel]:
        """按最近更新时间倒序列出知识库中的聊天会话。"""

        with self._session_factory() as session:
            statement = (
                select(ChatSessionModel)
                .where(
                    ChatSessionModel.knowledge_base_id == knowledge_base_id
                )
                .order_by(ChatSessionModel.updated_at.desc())
            )
            return list(session.scalars(statement).all())

    def get_chat_session(
        self,
        chat_session_id: str,
    ) -> ChatSessionModel | None:
        """根据 ID 获取聊天会话。"""

        with self._session_factory() as session:
            return session.get(ChatSessionModel, chat_session_id)

    def delete_chat_session(self, chat_session_id: str) -> None:
        """删除聊天会话，关联消息由 MySQL 外键级联删除。"""

        with self._session_factory.begin() as session:
            chat_session = session.get(ChatSessionModel, chat_session_id)
            if chat_session is None:
                return

            knowledge_base_id = chat_session.knowledge_base_id
            session.delete(chat_session)
            self._touch_knowledge_base(session, knowledge_base_id)

    def clear_chat_messages(self, chat_session_id: str) -> None:
        """清空会话消息，但保留会话本身。"""

        with self._session_factory.begin() as session:
            chat_session = session.get(ChatSessionModel, chat_session_id)
            if chat_session is None:
                return

            # 只删除消息表记录，不删除 chat_sessions 表中的会话标题。
            session.execute(
                delete(ChatMessageModel).where(
                    ChatMessageModel.chat_session_id == chat_session_id
                )
            )
            chat_session.updated_at = utc_now()
            self._touch_knowledge_base(
                session,
                chat_session.knowledge_base_id,
            )

    def add_chat_message(
        self,
        chat_session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> ChatMessageModel:
        """向会话末尾追加一条消息。"""

        if role not in {"user", "assistant"}:
            raise ValueError("聊天消息角色只能是 user 或 assistant。")

        with self._session_factory.begin() as session:
            chat_session = session.get(ChatSessionModel, chat_session_id)
            if chat_session is None:
                raise ValueError("聊天会话不存在或已被删除。")

            # 每个会话内部单独递增序号，以保证查询出的消息顺序稳定。
            sequence_statement = select(
                func.coalesce(
                    func.max(ChatMessageModel.sequence_number),
                    0,
                )
            ).where(ChatMessageModel.chat_session_id == chat_session_id)
            next_sequence = int(session.scalar(sequence_statement) or 0) + 1

            message = ChatMessageModel(
                id=str(uuid4()),
                chat_session_id=chat_session_id,
                role=role,
                content=content,
                sources=sources or [],
                sequence_number=next_sequence,
            )
            session.add(message)

            chat_session.updated_at = utc_now()
            self._touch_knowledge_base(
                session,
                chat_session.knowledge_base_id,
            )

        return message

    def list_chat_messages(
        self,
        chat_session_id: str,
    ) -> list[ChatMessageModel]:
        """按发送顺序获取会话中的全部消息。"""

        with self._session_factory() as session:
            statement = (
                select(ChatMessageModel)
                .where(
                    ChatMessageModel.chat_session_id == chat_session_id
                )
                .order_by(ChatMessageModel.sequence_number.asc())
            )
            return list(session.scalars(statement).all())

    def find_document_by_hash(
        self,
        knowledge_base_id: str,
        file_hash: str,
    ) -> DocumentModel | None:
        """查找知识库内内容完全相同的文件。"""

        with self._session_factory() as session:
            # 同一知识库内用 SHA-256 哈希判重，允许不同知识库存入相同文件。
            statement = select(DocumentModel).where(
                DocumentModel.knowledge_base_id == knowledge_base_id,
                DocumentModel.file_hash == file_hash,
            )
            return session.scalar(statement)

    def create_document(
        self,
        *,
        document_id: str,
        knowledge_base_id: str,
        original_name: str,
        stored_path: str,
        file_hash: str,
        file_type: str,
        size_bytes: int,
    ) -> DocumentModel:
        """保存待处理文档记录。"""

        # 新文件先标记为 pending，真正开始解析时由服务层改为 processing。
        document = DocumentModel(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            original_name=original_name,
            stored_path=stored_path,
            file_hash=file_hash,
            file_type=file_type,
            size_bytes=size_bytes,
            status="pending",
            chunk_count=0,
        )

        try:
            with self._session_factory.begin() as session:
                session.add(document)
                self._touch_knowledge_base(session, knowledge_base_id)
        except IntegrityError as exc:
            raise DuplicateDocumentError(
                f"文件 {original_name} 已存在于当前知识库。"
            ) from exc

        return document

    def list_documents(self, knowledge_base_id: str) -> list[DocumentModel]:
        """获取知识库中的全部文档。"""

        with self._session_factory() as session:
            statement = (
                select(DocumentModel)
                .where(DocumentModel.knowledge_base_id == knowledge_base_id)
                .order_by(DocumentModel.created_at.desc())
            )
            return list(session.scalars(statement).all())

    def get_document(self, document_id: str) -> DocumentModel | None:
        """根据 ID 获取文档。"""

        with self._session_factory() as session:
            return session.get(DocumentModel, document_id)

    def save_document_overview(
        self,
        document_id: str,
        overview_text: str,
    ) -> DocumentOverviewModel:
        """新增或更新文档的目录与章节概览。"""

        clean_overview = overview_text.strip()
        if not clean_overview:
            raise ValueError("文档概览不能为空。")

        with self._session_factory.begin() as session:
            # document_id 是此表主键，因此这里实现“一份文档最多一条概览”的 upsert 效果。
            overview = session.get(DocumentOverviewModel, document_id)
            if overview is None:
                overview = DocumentOverviewModel(
                    document_id=document_id,
                    overview_text=clean_overview,
                )
                session.add(overview)
            else:
                overview.overview_text = clean_overview
                overview.updated_at = utc_now()

        return overview

    def list_document_overviews(
        self,
        knowledge_base_id: str,
    ) -> list[tuple[DocumentModel, DocumentOverviewModel]]:
        """获取知识库中所有已完成文档的结构概览。"""

        with self._session_factory() as session:
            # 只返回处理成功的文档，失败或处理中资料不应用于整库概览回答。
            statement = (
                select(DocumentModel, DocumentOverviewModel)
                .join(
                    DocumentOverviewModel,
                    DocumentOverviewModel.document_id == DocumentModel.id,
                )
                .where(
                    DocumentModel.knowledge_base_id == knowledge_base_id,
                    DocumentModel.status == "completed",
                )
                .order_by(DocumentModel.created_at.asc())
            )
            return list(session.execute(statement).all())

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """更新文档处理状态和结果。"""

        with self._session_factory.begin() as session:
            document = session.get(DocumentModel, document_id)
            if document is None:
                return

            document.status = status
            document.error_message = error_message
            document.updated_at = utc_now()

            if chunk_count is not None:
                document.chunk_count = chunk_count

            self._touch_knowledge_base(
                session,
                document.knowledge_base_id,
            )

    def add_chunk_records(
        self,
        document_id: str,
        chunk_ids: Sequence[str],
    ) -> None:
        """首次保存向量块 ID，供后续精确删除。"""

        self.replace_chunk_records(document_id, chunk_ids)

    def replace_chunk_records(
        self,
        document_id: str,
        chunk_ids: Sequence[str],
    ) -> None:
        """原子替换文档的向量块 ID，支持安全重建索引。"""

        # 先构建新记录，再在同一事务中删旧写新，避免旧块 ID 残留。
        records = [
            DocumentChunkModel(
                id=chunk_id,
                document_id=document_id,
                chunk_index=index,
            )
            for index, chunk_id in enumerate(chunk_ids)
        ]

        with self._session_factory.begin() as session:
            # 与 MySQL 级联删除配合：删文档会自动删概览、块和导入任务。
            session.execute(
                delete(DocumentChunkModel).where(
                    DocumentChunkModel.document_id == document_id
                )
            )
            session.add_all(records)

    def list_chunk_ids(self, document_id: str) -> list[str]:
        """获取文档对应的全部向量块 ID。"""

        with self._session_factory() as session:
            statement = (
                select(DocumentChunkModel.id)
                .where(DocumentChunkModel.document_id == document_id)
                .order_by(DocumentChunkModel.chunk_index.asc())
            )
            return list(session.scalars(statement).all())

    def delete_document_record(self, document_id: str) -> None:
        """删除文档记录和关联块记录。"""

        with self._session_factory.begin() as session:
            document = session.get(DocumentModel, document_id)
            if document is None:
                return

            knowledge_base_id = document.knowledge_base_id
            session.delete(document)
            self._touch_knowledge_base(session, knowledge_base_id)

    def create_ingestion_job(self, document_id: str) -> str:
        """创建文档导入任务并返回任务 ID。"""

        job_id = str(uuid4())
        job = IngestionJobModel(
            id=job_id,
            document_id=document_id,
            status="running",
        )

        with self._session_factory.begin() as session:
            session.add(job)

        return job_id

    def finish_ingestion_job(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """结束导入任务。"""

        with self._session_factory.begin() as session:
            job = session.get(IngestionJobModel, job_id)
            if job is None:
                return

            job.status = status
            job.error_message = error_message
            job.finished_at = utc_now()

    @staticmethod
    def _touch_knowledge_base(
        session: Session,
        knowledge_base_id: str,
    ) -> None:
        """更新知识库时间，方便界面按最近活动排序。"""

        knowledge_base = session.get(
            KnowledgeBaseModel,
            knowledge_base_id,
        )
        if knowledge_base is not None:
            knowledge_base.updated_at = utc_now()
