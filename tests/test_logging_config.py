"""日志轮转配置测试。"""

import logging
from pathlib import Path
from types import SimpleNamespace

from src.logging_config import configure_logging


def test_configure_logging_does_not_add_duplicate_handlers(
    tmp_path: Path,
) -> None:
    """验证多次初始化日志仍只注册一个项目文件处理器。"""

    settings = SimpleNamespace(project_root=tmp_path)
    root_logger = logging.getLogger()

    # Streamlit 重跑时可能多次调用初始化函数，此处模拟两次调用。
    configure_logging(settings)
    configure_logging(settings)

    project_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "name", "") == "rag_file_handler"
    ]
    assert len(project_handlers) == 1
    assert (tmp_path / "logs" / "rag-system.log").exists()

    for handler in project_handlers:
        root_logger.removeHandler(handler)
        handler.close()
