"""从已解析文档中提取适合整库概览的目录与章节结构。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document

# 概览要进入模型上下文，因此限制总长度、目录页数量和标题数量。
MAX_OVERVIEW_CHARACTERS = 16_000
MAX_TOC_PAGES = 6
MAX_HEADING_COUNT = 120

CHAPTER_PATTERN = re.compile(
    r"第\s*([0-9一二三四五六七八九十百]+)\s*章\s*"
    r"([^\d\n…·]{1,30})"
)
SECTION_PATTERN = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){1,2})\s+"
    r"([\u4e00-\u9fffA-Za-z][^\n]{0,49})"
)
SECTION_NUMBER_PATTERN = re.compile(
    r"^\d{1,2}(?:\.\d{1,2}){1,2}$"
)


def build_document_overview(
    documents: Sequence[Document],
    original_name: str,
) -> str:
    """生成不依赖大模型的文档结构概览。"""

    if not documents:
        raise ValueError("无法为没有解析内容的文档生成概览。")

    # 分别收集章节、小节和源文件自带标题，最后按优先级组成概览。
    chapter_headings: list[str] = []
    section_headings: list[str] = []
    metadata_headings: list[str] = []
    toc_pages: list[str] = []
    fallback_lines: list[str] = []
    seen_chapters: set[str] = set()
    seen_sections: set[str] = set()
    seen_metadata_headings: set[str] = set()

    for document_index, document in enumerate(documents, start=1):
        content = document.page_content.strip()
        if not content:
            continue

        # location 让概览中的目录条目仍可追溯到 PDF 页或 Word/Markdown 标题。
        location = _get_location(document, document_index)
        lines = _clean_lines(content)
        page_chapters = _extract_chapter_headings(lines, location)
        page_sections = _extract_section_headings(lines, location)

        _extend_unique(
            chapter_headings,
            seen_chapters,
            page_chapters,
        )
        _extend_unique(
            section_headings,
            seen_sections,
            page_sections,
        )

        section_title = str(
            document.metadata.get("section_title", "")
        ).strip()
        if section_title and section_title not in seen_metadata_headings:
            seen_metadata_headings.add(section_title)
            metadata_headings.append(f"{section_title}（{location}）")

        # 目录页通常有许多章节/小节标题；评分避免把普通正文误当完整目录。
        toc_score = len(page_chapters) * 4 + len(page_sections)
        if "目录" in content:
            toc_score += 10
        if toc_score >= 8 and len(toc_pages) < MAX_TOC_PAGES:
            toc_pages.append(
                f"--- {location} ---\n{content[:5_000]}"
            )

        if lines and len(fallback_lines) < 40:
            fallback_lines.append(f"{lines[0]}（{location}）")

    overview_parts = [
        f"文件：{original_name}",
        f"已解析内容单元：{len(documents)}",
    ]

    if toc_pages:
        overview_parts.append("目录原文：\n" + "\n\n".join(toc_pages))

    # 目录原文优先，结构化提取的标题次之；两者都没有时才使用每段的首行作为线索。
    headings = [
        *chapter_headings,
        *section_headings,
        *metadata_headings,
    ][:MAX_HEADING_COUNT]
    if headings:
        overview_parts.append("章节结构：\n- " + "\n- ".join(headings))
    elif fallback_lines:
        overview_parts.append(
            "内容线索：\n- " + "\n- ".join(fallback_lines)
        )

    # 最终再次截断，保证异常长的目录不会撑爆后续提示词。
    return "\n\n".join(overview_parts)[:MAX_OVERVIEW_CHARACTERS]


def _clean_lines(content: str) -> list[str]:
    """清理空白并保留原有行边界。"""

    return [
        " ".join(line.split())
        for line in content.splitlines()
        if line.strip()
    ]


def _extract_chapter_headings(
    lines: Sequence[str],
    location: str,
) -> list[str]:
    """提取“第 N 章 标题”形式的章节名。"""

    headings: list[str] = []
    for line in lines:
        for chapter_number, title in CHAPTER_PATTERN.findall(line):
            clean_title = _clean_heading_title(title)
            if clean_title:
                headings.append(
                    f"第{chapter_number}章 {clean_title}（{location}）"
                )
    return headings


def _extract_section_headings(
    lines: Sequence[str],
    location: str,
) -> list[str]:
    """提取“1.1 标题”形式的小节名。"""

    headings: list[str] = []
    for line_index, line in enumerate(lines):
        match = SECTION_PATTERN.match(line)
        if match is not None:
            section_number = match.group(1)
            raw_title = match.group(2)
        elif (
            # 某些 PDF 会把编号“1.1”和标题分成两行，兼容这种排版。
            SECTION_NUMBER_PATTERN.fullmatch(line)
            and line_index + 1 < len(lines)
            and _looks_like_heading(lines[line_index + 1])
        ):
            section_number = line
            raw_title = lines[line_index + 1]
        else:
            continue

        title = _clean_heading_title(raw_title)
        if title:
            headings.append(f"{section_number} {title}（{location}）")
    return headings


def _clean_heading_title(title: str) -> str:
    """去掉目录点线、页码和过长的正文尾部。"""

    clean_title = re.split(r"[…·]{2,}|\s{2,}", title, maxsplit=1)[0]
    clean_title = clean_title.strip(" ：:。.;；、-")
    return clean_title[:40]


def _looks_like_heading(value: str) -> bool:
    """排除公式、页码和普通数值，保留自然语言小节标题。"""

    clean_value = value.strip()
    if not clean_value or len(clean_value) > 60:
        return False
    if not re.match(r"[\u4e00-\u9fffA-Za-z]", clean_value):
        return False

    # 数学不等式或算式经常以数字开头，不能误判成“1.1 标题”。
    math_symbols = ("=", "＜", "＞", "<", ">", "×", "÷")
    return not any(symbol in clean_value for symbol in math_symbols)


def _get_location(document: Document, document_index: int) -> str:
    """生成人类可读的位置，供概览引用。"""

    page_number = document.metadata.get("page_number")
    if page_number is not None:
        return f"第 {page_number} 页"

    section_title = document.metadata.get("section_title")
    if section_title:
        return f"章节：{section_title}"

    return f"内容单元 {document_index}"


def _extend_unique(
    target: list[str],
    seen: set[str],
    values: Sequence[str],
) -> None:
    """按出现顺序追加去重后的标题。"""

    for value in values:
        # 用不含位置的标题做去重键；相同标题在页眉重复出现时只保留首次位置。
        normalized = value.split("（", maxsplit=1)[0]
        if normalized in seen:
            continue

        seen.add(normalized)
        target.append(value)
