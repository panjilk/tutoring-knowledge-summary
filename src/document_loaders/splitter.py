"""文档分块逻辑。"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    *,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], list[str]]:
    """将解析结果切分为可检索文档块，并生成稳定的块 ID。"""

    # 优先在段落、换行、中文句末处分割；chunk_overlap 保留相邻块的衔接上下文。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    chunk_ids: list[str] = []
    # 同一 document_id 和顺序始终生成同一个 UUID，重建索引时便于精确替换旧向量。
    for chunk_index, chunk in enumerate(chunks):
        stable_value = f"{document_id}:{chunk_index}"
        chunk_id = str(uuid5(NAMESPACE_URL, stable_value))

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = chunk_index
        chunk_ids.append(chunk_id)

    return chunks, chunk_ids
