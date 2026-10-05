"""应用日志初始化。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import Settings


def configure_logging(settings: Settings) -> None:
    """配置控制台和按大小轮转的本地日志。"""

    # 取根日志器，使项目内各模块的 logger 都能把日志写入同一个文件。
    root_logger = logging.getLogger()
    if any(
        # Streamlit 会反复执行页面代码，此判断防止每次重跑都叠加一个文件处理器。
        getattr(handler, "name", "") == "rag_file_handler"
        for handler in root_logger.handlers
    ):
        return

    log_directory = settings.project_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_directory) / "rag-system.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # 单个日志到 5 MB 自动轮转，最多保留 5 个历史文件，避免日志无限增长。
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.name = "rag_file_handler"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
