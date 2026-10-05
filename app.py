"""个人知识库问答系统的 Streamlit 页面入口。

这个文件主要负责「界面层」：接收用户的操作、调用 ``src`` 中的服务、
再把结果显示出来。文档解析、向量检索、数据库读写和大模型调用等核心
业务逻辑均在 ``src`` 目录中实现。
"""

from __future__ import annotations

from dataclasses import asdict  # 将资料来源等数据类转换为字典，便于保存和展示。
from typing import Any  # 页面中部分数据来自数据库，结构在运行时才确定。

import streamlit as st

# 配置、数据库、向量检索和大模型客户端分别封装在 src 中；本文件不直接处理它们的细节。
from src.config import get_settings
from src.database import Repository
from src.embeddings import EmbeddingProvider, VectorStoreManager
from src.exceptions import (
    DatabaseConnectionError,
    DeepSeekConfigurationError,
    DuplicateDocumentError,
    NoRelevantContextError,
    RagSystemError,
)
from src.llm import DeepSeekClient
from src.logging_config import configure_logging
from src.services import KnowledgeBaseService, RagService

# 下拉框内部用这个不会与真实数据库 ID 冲突的值，表示“尚未保存的新对话”。
NEW_CHAT_OPTION = "__new_chat__"

# 必须在绝大多数 Streamlit 输出函数之前设置页面的浏览器标题、图标和布局。
st.set_page_config(
    page_title="个人知识库问答系统",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def build_services() -> tuple[
    KnowledgeBaseService,
    RagService,
    Repository,
    DeepSeekClient,
]:
    """创建可在 Streamlit 页面刷新之间复用的服务实例。

    Streamlit 每次点击控件都会从上到下重新执行 app.py。缓存此函数的返回值，
    可以避免每次重跑都重新建立数据库、向量库和大模型客户端。
    """

    # 从 .env 等来源读取数据库、模型和文件大小限制等配置。
    settings = get_settings()
    # 初始化日志和数据库表；数据库模块会负责具体的连接与建表。
    configure_logging(settings)
    repository = Repository(settings)
    repository.initialize()

    # EmbeddingProvider 把文本转换成向量；VectorStoreManager 保存并检索这些向量。
    embedding_provider = EmbeddingProvider(settings)
    vector_store = VectorStoreManager(settings, embedding_provider)
    # DeepSeekClient 封装与 DeepSeek API 的通信。
    deepseek_client = DeepSeekClient(settings)

    # Service 是界面和底层组件之间的业务层：
    # - KnowledgeBaseService 负责知识库和文件的管理；
    # - RagService 负责“检索资料 + 让模型依据资料回答”的 RAG 流程。
    knowledge_base_service = KnowledgeBaseService(
        settings,
        repository,
        vector_store,
    )
    rag_service = RagService(
        settings,
        vector_store,
        deepseek_client,
        repository,
    )
    return (
        knowledge_base_service,
        rag_service,
        repository,
        deepseek_client,
    )


def main() -> None:
    """按照从上到下的顺序绘制整个应用页面。"""

    st.title("📚 个人知识库问答系统")
    st.caption("本地保存知识库，通过 DeepSeek 根据检索资料回答问题。")

    try:
        # 取得缓存的服务对象；首次运行时会在 build_services() 中完成初始化。
        (
            knowledge_base_service,
            rag_service,
            repository,
            deepseek_client,
        ) = build_services()
    except DatabaseConnectionError as exc:
        render_database_error(str(exc))
        return
    # Streamlit 顶层必须兜底，避免未预料的第三方依赖错误导致白屏。
    except Exception as exc:  # noqa: BLE001
        st.error(f"应用初始化失败：{exc}")
        st.info("请先根据 README 完成依赖、MySQL 和 .env 配置。")
        return

    # 每次页面重跑都读取最新的知识库列表，以便创建或删除后立即反映到界面。
    knowledge_bases = knowledge_base_service.list_knowledge_bases()
    (
        selected_knowledge_base_id,
        selected_chat_session_id,
    ) = render_sidebar(
        knowledge_base_service,
        repository,
        deepseek_client,
        knowledge_bases,
    )

    # 没有知识库时无法上传文件或检索，因此只显示引导信息。
    if selected_knowledge_base_id is None:
        st.info("请先在左侧创建一个知识库。")
        return

    # 两个标签页共用同一个已选知识库：一个用于问答，一个用于维护资料。
    chat_tab, management_tab = st.tabs(
        ["💬 知识库问答", "📁 文档管理"],
        on_change="rerun",
    )

    # 仅渲染当前打开的标签页，避免切换到文档管理时仍执行模型相关的页面逻辑。
    if chat_tab.open:
        with chat_tab:
            render_chat_page(
                selected_knowledge_base_id,
                selected_chat_session_id,
                rag_service,
                deepseek_client,
                repository,
            )

    if management_tab.open:
        with management_tab:
            render_document_management(
                selected_knowledge_base_id,
                knowledge_base_service,
            )


def render_sidebar(
    knowledge_base_service: KnowledgeBaseService,
    repository: Repository,
    deepseek_client: DeepSeekClient,
    knowledge_bases: list[Any],
) -> tuple[str | None, str | None]:
    """绘制侧边栏，并返回当前选择的知识库 ID 和会话 ID。"""

    # with st.sidebar 表示：此缩进块内所有 Streamlit 控件都显示在左侧边栏。
    with st.sidebar:
        st.header("知识库")

        # form 会把输入和按钮组成一个表单；点击提交前修改输入框不会反复触发操作。
        # clear_on_submit=True 表示创建成功提交后清空名称输入框。
        with st.form("create_knowledge_base_form", clear_on_submit=True):
            new_name = st.text_input(
                "新知识库名称",
                placeholder="例如：我的学习资料",
            )
            create_submitted = st.form_submit_button(
                "创建知识库",
                width="stretch",
            )

        # form_submit_button 只有被点击时才会返回 True。
        if create_submitted:
            try:
                knowledge_base_service.create_knowledge_base(new_name)
                st.success("知识库创建成功。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        # 还没有知识库时，不能显示选择框或历史会话。
        if not knowledge_bases:
            render_system_status(repository, deepseek_client)
            return None, None

        # 下拉框展示易读的名称，后续服务调用则使用唯一的知识库 ID。
        name_to_id = {
            knowledge_base.name: knowledge_base.id
            for knowledge_base in knowledge_bases
        }
        selected_name = st.selectbox(
            "当前知识库",
            options=list(name_to_id.keys()),
        )
        selected_id = name_to_id[selected_name]

        st.divider()
        selected_chat_session_id = render_chat_history_sidebar(
            repository,
            selected_id,
        )

        st.divider()
        render_system_status(repository, deepseek_client)

        # 将破坏性操作收起，并要求先勾选确认，避免误删文件和向量数据。
        with st.expander("删除当前知识库"):
            confirmed = st.checkbox(
                "我确认删除全部文件和向量数据",
                key=f"confirm_delete_kb_{selected_id}",
            )
            if st.button(
                "永久删除",
                disabled=not confirmed,
                type="primary",
                width="stretch",
            ):
                knowledge_base_service.delete_knowledge_base(selected_id)
                st.success("知识库已删除。")
                st.rerun()

        return selected_id, selected_chat_session_id


def render_chat_history_sidebar(
    repository: Repository,
    knowledge_base_id: str,
) -> str | None:
    """在侧边栏中显示当前知识库的历史聊天会话。"""

    st.subheader("历史对话")

    # session_state 是 Streamlit 为当前浏览器会话保存的临时状态。
    # 为每个知识库生成不同 key，避免切换知识库时把对话选错。
    active_key = get_active_chat_session_key(knowledge_base_id)
    selector_key = get_chat_selector_key(knowledge_base_id)
    # 从数据库读取当前知识库的历史对话，通常按最近更新时间排序。
    chat_sessions = repository.list_chat_sessions(knowledge_base_id)
    session_ids = [chat_session.id for chat_session in chat_sessions]
    # 数据库存 ID 不便阅读，因此为下拉框准备“标题 · 时间”的显示文字。
    session_labels = {
        chat_session.id: (
            f"{chat_session.title} · "
            f"{chat_session.updated_at.strftime('%m-%d %H:%M')}"
        )
        for chat_session in chat_sessions
    }

    active_session_id = st.session_state.get(active_key)
    if active_key not in st.session_state:
        # 第一次打开知识库时，默认恢复最近使用的会话。
        active_session_id = session_ids[0] if session_ids else None
        st.session_state[active_key] = active_session_id
    elif active_session_id is not None and active_session_id not in session_ids:
        # 会话可能已在其他页面中删除，此时安全回到新对话。
        active_session_id = None
        st.session_state[active_key] = None

    # 点击后执行回调函数 start_new_chat，把活动会话设为 None。
    st.button(
        "新建对话",
        icon=":material/add:",
        width="stretch",
        key=f"new_chat_{knowledge_base_id}",
        on_click=start_new_chat,
        args=(knowledge_base_id,),
    )

    # 在控件创建前同步值，避免切换或新建会话时出现状态冲突。
    st.session_state[selector_key] = (
        st.session_state.get(active_key) or NEW_CHAT_OPTION
    )
    # key 让控件的值自动保存到 st.session_state；on_change 在切换时同步活动会话。
    st.selectbox(
        "选择历史对话",
        options=[NEW_CHAT_OPTION, *session_ids],
        format_func=lambda session_id: (
            "新对话"
            if session_id == NEW_CHAT_OPTION
            else session_labels[session_id]
        ),
        key=selector_key,
        label_visibility="collapsed",
        on_change=select_chat_session,
        args=(knowledge_base_id,),
    )

    # 回调执行后，从状态中取出最终选择的会话 ID。
    selected_session_id = st.session_state.get(active_key)
    if selected_session_id is not None:
        with st.expander("删除当前历史对话"):
            confirmed = st.checkbox(
                "确认删除该会话及全部消息",
                key=f"confirm_delete_chat_{selected_session_id}",
            )
            st.button(
                "删除历史对话",
                icon=":material/delete:",
                disabled=not confirmed,
                width="stretch",
                key=f"delete_chat_{selected_session_id}",
                on_click=delete_chat_session,
                args=(
                    repository,
                    knowledge_base_id,
                    selected_session_id,
                ),
            )

    return selected_session_id


def render_system_status(
    repository: Repository,
    deepseek_client: DeepSeekClient,
) -> None:
    """显示 MySQL 和 DeepSeek 配置状态。"""

    st.subheader("系统状态")
    # 通过简单的健康检查让用户知道后续的保存和问答是否可用。
    mysql_status = "✅ MySQL 已连接" if repository.check_connection() else "❌ MySQL 未连接"
    deepseek_status = (
        "✅ DeepSeek 已配置"
        if deepseek_client.is_configured()
        else "⚠️ DeepSeek 未配置"
    )
    st.caption(mysql_status)
    st.caption(deepseek_status)


def render_document_management(
    knowledge_base_id: str,
    service: KnowledgeBaseService,
) -> None:
    """绘制文件上传、状态列表和删除操作。"""

    st.subheader("导入文档")
    # file_uploader 返回用户选择的上传文件对象；这里只选择文件，尚未写入知识库。
    uploaded_files = st.file_uploader(
        "支持 PDF、DOCX、TXT 和 Markdown，可一次选择多个文件。",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    # 只有文件存在时才允许点击导入，避免对空列表执行后续流程。
    if st.button(
        "开始导入",
        disabled=not uploaded_files,
        type="primary",
    ):
        # 先检查整个批次大小，避免单个文件都处理了一部分后才发现总量超限。
        total_upload_bytes = sum(
            uploaded_file.size
            for uploaded_file in uploaded_files
        )
        if total_upload_bytes > service.max_upload_batch_bytes:
            limit_mb = service.max_upload_batch_bytes // (1024 * 1024)
            actual_mb = total_upload_bytes / (1024 * 1024)
            st.error(
                f"本批文件共 {actual_mb:.2f} MB，超过 {limit_mb} MB 限制。"
                "请分成多个批次导入。"
            )
            return

        # 预留进度条和消息容器；循环中会持续更新它们，而不是不断创建新控件。
        progress_bar = st.progress(0, text="准备导入……")
        result_container = st.container()
        total_files = len(uploaded_files)

        # 逐个解析、切分文本并建立向量索引。一个文件失败不会影响其他文件。
        for file_index, uploaded_file in enumerate(uploaded_files, start=1):
            progress_bar.progress(
                (file_index - 1) / total_files,
                text=f"正在处理：{uploaded_file.name}",
            )

            try:
                # ingest_file 完成“保存原文件 → 解析文本 → 分块 → 向量化 → 入库”。
                result = service.ingest_file(
                    knowledge_base_id,
                    uploaded_file.name,
                    uploaded_file,
                )
                if result.success:
                    result_container.success(
                        f"{result.file_name}：导入成功，共 {result.chunk_count} 个文档块。"
                    )
                else:
                    result_container.error(
                        f"{result.file_name}：{result.message}"
                    )
            except DuplicateDocumentError as exc:
                result_container.warning(str(exc))
            except RagSystemError as exc:
                result_container.error(str(exc))
            # 一个文件失败不能阻断同一批次中的其他文件。
            except Exception as exc:  # noqa: BLE001
                result_container.error(
                    f"{uploaded_file.name} 导入失败：{exc}"
                )

        progress_bar.progress(1.0, text="本批文件处理完成。")

    st.divider()
    st.subheader("已导入文档")
    # 从数据库获取该知识库已有文件，用于下方的表格和删除选择框。
    documents = service.list_documents(knowledge_base_id)

    if not documents:
        st.info("当前知识库还没有文档。")
        return

    # 将程序内部状态转换成给用户看的中文名称。
    status_names = {
        "pending": "等待处理",
        "processing": "处理中",
        "completed": "已完成",
        "failed": "失败",
    }
    # 数据表组件需要字典列表；此处也统一格式化文件大小和时间。
    table_rows = [
        {
            "文件名": document.original_name,
            "类型": document.file_type.upper(),
            "大小（KB）": round(document.size_bytes / 1024, 2),
            "状态": status_names.get(document.status, document.status),
            "文档块": document.chunk_count,
            "导入时间": document.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "错误": document.error_message or "",
        }
        for document in documents
    ]
    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
        key=f"document_table_{knowledge_base_id}",
    )

    # 重建索引不删除原始文件，只根据当前模型和分块配置重新生成向量。
    with st.expander("索引维护"):
        st.caption(
            "当 Embedding 模型或分块参数改变，或者导入被中断时，"
            "可以保留原文件并重新生成全部向量。"
        )
        confirm_rebuild = st.checkbox(
            "确认重建当前知识库的全部向量索引",
            key=f"confirm_rebuild_{knowledge_base_id}",
        )
        if st.button(
            "重建索引",
            disabled=not confirm_rebuild,
            width="stretch",
            key=f"rebuild_index_{knowledge_base_id}",
        ):
            # status 是可更新的进度面板；as status 用于稍后写入完成或失败状态。
            with st.status("正在重建索引……", expanded=True) as status:
                rebuild_results = service.rebuild_index(
                    knowledge_base_id
                )
                success_count = sum(
                    result.success
                    for result in rebuild_results
                )
                for result in rebuild_results:
                    if result.success:
                        st.write(
                            f"✅ {result.file_name}："
                            f"{result.chunk_count} 个文档块"
                        )
                    else:
                        st.write(
                            f"❌ {result.file_name}：{result.message}"
                        )

                if success_count == len(rebuild_results):
                    status.update(
                        label="索引重建完成。",
                        state="complete",
                    )
                else:
                    status.update(
                        label=(
                            f"重建完成：成功 {success_count} 个，"
                            f"失败 {len(rebuild_results) - success_count} 个。"
                        ),
                        state="error",
                    )

    # 与知识库选择类似：界面显示文件名，删除服务使用精确的文档 ID。
    document_name_to_id = {
        f"{document.original_name} ({document.id[:8]})": document.id
        for document in documents
    }
    selected_document_name = st.selectbox(
        "选择要删除的文档",
        options=list(document_name_to_id.keys()),
    )
    confirm_delete = st.checkbox(
        "确认删除该文档及其向量数据",
        key=f"confirm_document_delete_{knowledge_base_id}",
    )

    if st.button(
        "删除文档",
        disabled=not confirm_delete,
    ):
        document_id = document_name_to_id[selected_document_name]
        service.delete_document(document_id)
        st.success("文档已删除。")
        st.rerun()


def render_chat_page(
    knowledge_base_id: str,
    chat_session_id: str | None,
    rag_service: RagService,
    deepseek_client: DeepSeekClient,
    repository: Repository,
) -> None:
    """绘制可持久化的多轮聊天和引用来源。

    一次提问的顺序是：读取历史 → 必要时新建会话 → 保存问题 → 检索资料
    → 流式生成回答 → 校验引用 → 保存回答。
    """

    if not deepseek_client.is_configured():
        st.warning(
            "DeepSeek 尚未配置。可以先导入文档，填写 .env 并重启应用后再提问。"
        )

    # 已保存的对话从数据库读出；新对话还没有 ID，因此使用空列表开始。
    message_records = (
        repository.list_chat_messages(chat_session_id)
        if chat_session_id is not None
        else []
    )
    # 数据库记录转换为界面需要的统一结构，sources 是回答引用的资料来源。
    messages: list[dict[str, Any]] = [
        {
            "role": message.role,
            "content": message.content,
            "sources": message.sources,
        }
        for message in message_records
    ]

    # 用 4:1 的列宽在提示文字右侧放置“清空当前会话”按钮。
    top_bar_left, top_bar_right = st.columns([4, 1])
    with top_bar_left:
        st.caption("回答将显示真实检索来源；知识库证据不足时不会强行生成答案。")
    with top_bar_right:
        if st.button(
            "清空当前会话",
            disabled=chat_session_id is None or not messages,
            width="stretch",
        ):
            repository.clear_chat_messages(chat_session_id)
            st.rerun()

    # 先完整重绘已有消息。Streamlit 每次交互重跑脚本，所以历史必须主动重新展示。
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    # 使用 Streamlit 原生底部容器固定提问框。
    # 页面内容滚动时，输入框会一直停留在浏览器底部。
    # st.bottom 把输入区域固定在页面底部，适合聊天类界面。
    with st.bottom:
        question = st.chat_input(
            "请输入你想询问的内容……",
            submit_mode="disable",
            key=(
                f"chat_input_{knowledge_base_id}_"
                f"{chat_session_id or 'new'}"
            ),
        )
    # 用户还未提交问题时，页面到此结束；避免执行任何检索或模型调用。
    if not question:
        return

    # RAG 服务只需要角色和文本作为多轮对话上下文，不需要把全部来源再次传给模型。
    previous_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
    ]

    # 第一次提问才创建数据库会话，并将问题用作会话标题的初始来源。
    if chat_session_id is None:
        chat_session = repository.create_chat_session(
            knowledge_base_id,
            question,
        )
        chat_session_id = chat_session.id
        st.session_state[
            get_active_chat_session_key(knowledge_base_id)
        ] = chat_session_id

    # 先保存用户问题。即使模型调用失败，历史记录中也能看到原问题。
    repository.add_chat_message(
        chat_session_id,
        "user",
        question,
    )

    with st.chat_message("user"):
        st.markdown(question)

    # 在助手消息气泡中依次显示检索状态、流式回答和可展开的来源。
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索知识库……"):
                # prepare_answer 负责检索最相关资料、选择回答模式，并整理提示词所需信息。
                prepared_answer = rag_service.prepare_answer(
                    knowledge_base_id,
                    question,
                    previous_history,
                )

            if prepared_answer.mode == "overview":
                st.caption(
                    "已启用整库概览模式，将依据目录和章节结构回答。"
                )
            elif prepared_answer.mode == "chapter":
                st.caption(
                    "已启用章节限定检索，只会使用指定章节的资料。"
                )

            # stream_answer 是生成器：文本生成一部分就立刻显示，最后返回完整回答文本。
            answer_text = st.write_stream(
                rag_service.stream_answer(prepared_answer)
            )
            # 校验模型写出的 [1]、[2] 等编号是否确实存在于本次检索来源中。
            citation_validation = rag_service.validate_citations(
                answer_text,
                prepared_answer.sources,
            )
            # dataclass 不能直接作为 JSON/数据库字段保存，先转为普通字典。
            source_dicts = [
                asdict(source)
                for source in prepared_answer.sources
            ]

            if citation_validation.invalid_numbers:
                invalid_text = "、".join(
                    str(number)
                    for number in citation_validation.invalid_numbers
                )
                st.warning(
                    f"模型生成了不存在的引用编号：{invalid_text}。"
                    "这些编号已被忽略。"
                )
            if citation_validation.missing_citations:
                st.warning(
                    "模型回答没有标注引用编号，下面展示本次检索到的资料。"
                )

            render_sources(source_dicts)
        except NoRelevantContextError as exc:
            answer_text = str(exc)
            source_dicts = []
            st.info(answer_text)
        except DeepSeekConfigurationError as exc:
            answer_text = str(exc)
            source_dicts = []
            st.error(answer_text)
        # API SDK 可能抛出多种供应商异常，统一转换为界面提示。
        except Exception as exc:  # noqa: BLE001
            answer_text = f"回答生成失败：{exc}"
            source_dicts = []
            st.error(answer_text)

    # 无论成功或被异常处理，都保存一条 assistant 消息，保证对话记录完整。
    repository.add_chat_message(
        chat_session_id,
        "assistant",
        answer_text,
        source_dicts,
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    """在答案下方显示本次实际检索到的全部资料。"""

    if not sources:
        return

    # 来源默认折叠，避免长摘录挤占聊天记录；用户需要时可自行展开核对。
    with st.expander(f"查看本次检索来源（{len(sources)} 条）"):
        for source in sources:
            source_number = source["number"]
            file_name = source["file_name"]
            location = source["location"]
            score = source["score"]

            st.markdown(
                f"**[{source_number}] {file_name} · {location}**  "
                f"\n相关度：`{score:.3f}`"
            )
            st.write(source["excerpt"])
            st.divider()


def render_database_error(message: str) -> None:
    """显示数据库初始化步骤，而不是让页面直接崩溃。"""

    st.error(message)
    st.markdown(
        "请依次完成：\n"
        "1. 安装并启动 MySQL 8.0 或更高版本。\n"
        "2. 将 `.env.example` 复制为 `.env` 并填写 MySQL 参数。\n"
        "3. 运行 `python scripts/init_database.py`。\n"
        "4. 重新运行 `streamlit run app.py`。"
    )


def get_active_chat_session_key(knowledge_base_id: str) -> str:
    """返回当前知识库所使用的活动会话状态键。"""

    # key 中包含知识库 ID，使每个知识库的当前会话独立保存。
    return f"active_chat_session_{knowledge_base_id}"


def get_chat_selector_key(knowledge_base_id: str) -> str:
    """返回历史会话下拉框的 Streamlit 状态键。"""

    # 下拉框也使用独立 key，避免 Streamlit 控件状态互相覆盖。
    return f"chat_session_selector_{knowledge_base_id}"


def select_chat_session(knowledge_base_id: str) -> None:
    """将历史会话下拉框的值设置为当前活动会话。"""

    selector_key = get_chat_selector_key(knowledge_base_id)
    active_key = get_active_chat_session_key(knowledge_base_id)
    # 回调发生时，控件的新值已经被 Streamlit 写进 session_state。
    selected_value = st.session_state.get(selector_key)
    st.session_state[active_key] = (
        None
        if selected_value == NEW_CHAT_OPTION
        else selected_value
    )


def start_new_chat(knowledge_base_id: str) -> None:
    """切换到一个尚未保存的新对话。"""

    # None 表示尚未创建数据库会话；下一次提交问题时会自动创建。
    st.session_state[get_active_chat_session_key(knowledge_base_id)] = None


def delete_chat_session(
    repository: Repository,
    knowledge_base_id: str,
    chat_session_id: str,
) -> None:
    """删除所选历史会话并切换到新对话。"""

    # 先删数据库中的会话和消息，再清除当前页面记录的活动会话。
    repository.delete_chat_session(chat_session_id)
    st.session_state[get_active_chat_session_key(knowledge_base_id)] = None


# Streamlit 从上到下执行本文件时，以这一行开始构建页面。
main()
