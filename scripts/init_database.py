"""创建 MySQL 数据库和项目数据表。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import OperationalError

# 脚本直接用 python 执行时，先把项目根目录加入模块搜索路径，才能导入 src。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.database import Repository


def main() -> int:
    """连接 MySQL 服务，创建数据库后初始化数据表。"""

    # 这里先连 MySQL 服务本身（不指定数据库），因为目标数据库可能还不存在。
    settings = get_settings()
    database_name = settings.mysql_database

    # 数据库名会拼接进 DDL，因此先严格限制可用字符。
    if not re.fullmatch(r"[A-Za-z0-9_]+", database_name):
        print("MYSQL_DATABASE 只能包含字母、数字和下划线。")
        return 1

    # URL.create 负责转义密码中的特殊字符；此连接只用于执行 CREATE DATABASE。
    server_url = URL.create(
        drivername="mysql+pymysql",
        username=settings.mysql_user,
        password=settings.mysql_password,
        host=settings.mysql_host,
        port=settings.mysql_port,
        query={
            "charset": "utf8mb4",
            "connect_timeout": "5",
        },
    )

    try:
        server_engine = create_engine(server_url, pool_pre_ping=True)
        with server_engine.begin() as connection:
            connection.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )

        # 数据库创建完后，Repository.initialize 再依据 SQLAlchemy 模型创建数据表。
        repository = Repository(settings)
        repository.initialize()
    except OperationalError as exc:
        print("MySQL 初始化失败，请检查服务状态、账号、密码和权限。")
        print(f"详细信息：{exc}")
        return 1

    print(f"数据库 {database_name} 和项目数据表初始化成功。")
    return 0


if __name__ == "__main__":
    # 只有直接运行此脚本时才执行；被其他模块 import 时不会产生副作用。
    raise SystemExit(main())
