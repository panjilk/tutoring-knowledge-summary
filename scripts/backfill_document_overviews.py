"""为已导入文档补建目录和章节概览。"""

from __future__ import annotations

import sys
from pathlib import Path

# 与 init_database.py 相同：保证从任意工作目录启动脚本都可导入项目模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.database import Repository
from src.document_loaders import build_document_overview, load_document


def main() -> int:
    """解析现有原始文件并保存概览，不重新生成向量。"""

    # 此脚本只补建概览，绝不会删除原文件或重写向量，因此可用于已有知识库的升级。
    repository = Repository(get_settings())
    repository.initialize()
    success_count = 0
    failure_count = 0

    # 双层循环遍历“知识库 → 已完成文档”；失败文件跳过，因为它们没有可靠的内容可概括。
    for knowledge_base in repository.list_knowledge_bases():
        documents = repository.list_documents(knowledge_base.id)
        for document in documents:
            if document.status != "completed":
                continue

            try:
                # 重用正常导入链路中的解析器，确保目录识别规则和新导入文档保持一致。
                parsed_documents = load_document(
                    Path(document.stored_path),
                    document_id=document.id,
                    knowledge_base_id=knowledge_base.id,
                    original_name=document.original_name,
                )
                overview_text = build_document_overview(
                    parsed_documents,
                    document.original_name,
                )
                repository.save_document_overview(
                    document.id,
                    overview_text,
                )
                success_count += 1
                print(f"概览生成成功：{document.original_name}")
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                print(f"概览生成失败：{document.original_name}：{exc}")

    print(
        f"概览补建完成：成功 {success_count} 个，"
        f"失败 {failure_count} 个。"
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    # 返回非零状态码可让命令行或自动化任务识别是否存在处理失败。
    raise SystemExit(main())
