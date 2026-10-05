"""从环境变量读取应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

# 从本文件向上两层得到项目根目录：src/config/settings.py → 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 只从项目自己的 .env 读取变量，避免当前终端目录不同而找不到配置。
load_dotenv(PROJECT_ROOT / ".env")


def _read_int(name: str, default: int) -> int:
    """读取整数环境变量，并在格式错误时给出明确异常。"""

    # 未设置时采用代码默认值；设置了但格式错误时不能静默继续，以免配置问题难以排查。
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数。") from exc


def _read_float(name: str, default: float) -> float:
    """读取浮点数环境变量。"""

    # 浮点数配置主要用于模型 temperature 和相似度分数阈值。
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字。") from exc


@dataclass(frozen=True)
class Settings:
    """应用运行所需的全部配置。

    frozen=True 让配置在启动后不可修改；若修改 .env，需要重启进程或清除缓存后再读取。
    """

    project_root: Path
    data_directory: Path
    upload_directory: Path
    vector_directory: Path

    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_temperature: float
    deepseek_timeout_seconds: int
    deepseek_max_retries: int

    embedding_model_name: str
    embedding_device: str
    embedding_batch_size: int

    chunk_size: int
    chunk_overlap: int
    max_upload_batch_mb: int
    retrieval_top_k: int
    retrieval_fetch_k: int
    retrieval_score_threshold: float

    @property
    def mysql_url(self) -> URL:
        """生成可正确处理特殊字符密码的 SQLAlchemy 连接地址。"""

        # 不手动拼接 mysql:// URL，避免密码里的 @、: 等字符破坏连接字符串。
        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={
                "charset": "utf8mb4",
                "connect_timeout": "5",
            },
        )

    @property
    def deepseek_is_configured(self) -> bool:
        """判断 DeepSeek 的必要参数是否已经填写。"""

        return bool(
            self.deepseek_api_key
            and self.deepseek_base_url
            and self.deepseek_model
        )

    def ensure_directories(self) -> None:
        """创建本地文件和向量索引目录。"""

        # exist_ok=True 允许每次启动都安全调用，不会因为目录已存在而报错。
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.vector_directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存配置，避免页面每次刷新都重复解析环境变量。"""

    # 所有可变数据集中在 data 目录，便于备份或迁移，不污染源代码目录。
    data_directory = Path(
        os.getenv("DATA_DIRECTORY", str(PROJECT_ROOT / "data"))
    ).resolve()

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_directory=data_directory,
        upload_directory=data_directory / "uploads",
        vector_directory=data_directory / "vector_store",
        mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        mysql_port=_read_int("MYSQL_PORT", 3306),
        mysql_user=os.getenv("MYSQL_USER", "root"),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),
        mysql_database=os.getenv("MYSQL_DATABASE", "rag_system"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", ""),
        deepseek_temperature=_read_float("DEEPSEEK_TEMPERATURE", 0.1),
        deepseek_timeout_seconds=_read_int("DEEPSEEK_TIMEOUT_SECONDS", 60),
        deepseek_max_retries=_read_int("DEEPSEEK_MAX_RETRIES", 2),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-small-zh-v1.5",
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=_read_int("EMBEDDING_BATCH_SIZE", 32),
        chunk_size=_read_int("CHUNK_SIZE", 800),
        chunk_overlap=_read_int("CHUNK_OVERLAP", 120),
        max_upload_batch_mb=_read_int("MAX_UPLOAD_BATCH_MB", 1024),
        retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 6),
        retrieval_fetch_k=_read_int("RETRIEVAL_FETCH_K", 20),
        retrieval_score_threshold=_read_float(
            "RETRIEVAL_SCORE_THRESHOLD",
            0.35,
        ),
    )

    # 重叠长度必须小于块长度，否则文本切分器无法向前推进。
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE。")
    if settings.max_upload_batch_mb <= 0:
        raise ValueError("MAX_UPLOAD_BATCH_MB 必须大于 0。")

    settings.ensure_directories()
    return settings
