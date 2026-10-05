"""初始化并配置当前用户使用的本机 MySQL 实例。"""

from __future__ import annotations

import os
import re
import secrets
import socket
import subprocess
import time
from pathlib import Path

import pymysql

# 该脚本只配置当前 Windows 用户的本地实例，数据和凭据均放在用户 AppData 下。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MYSQL_BASE_DIRECTORY = Path(r"C:\Program Files\MySQL\MySQL Server 8.4")
MYSQLD_PATH = MYSQL_BASE_DIRECTORY / "bin" / "mysqld.exe"

LOCAL_APP_DATA = Path(os.environ["LOCALAPPDATA"])
INSTANCE_DIRECTORY = LOCAL_APP_DATA / "MySQL" / "RagSystem"
DATA_DIRECTORY = INSTANCE_DIRECTORY / "data"
CONFIG_PATH = INSTANCE_DIRECTORY / "my.ini"
ROOT_CREDENTIALS_PATH = INSTANCE_DIRECTORY / "root-client.ini"

# 显式使用回环地址，避免把本地开发数据库暴露到局域网。
MYSQL_HOST = "127.0.0.1"
MYSQL_DATABASE = "rag_system"
MYSQL_APP_USER = "rag_app"


def _get_mysql_port() -> int:
    """优先使用临时覆盖值，否则沿用现有实例端口。"""

    # 环境变量方便临时指定端口；未指定时优先复用已有配置，最后才默认 3306。
    override_port = os.environ.get("RAG_MYSQL_PORT")
    if override_port:
        return int(override_port)

    if CONFIG_PATH.exists():
        config_text = CONFIG_PATH.read_text(encoding="utf-8-sig")
        port_match = re.search(r"^port=(\d+)$", config_text, flags=re.MULTILINE)
        if port_match:
            return int(port_match.group(1))

    return 3306


MYSQL_PORT = _get_mysql_port()


def main() -> int:
    """完成数据目录初始化、账号创建和项目配置。"""

    if not MYSQLD_PATH.exists():
        print("未找到 mysqld.exe，请先安装 MySQL Community Server 8.4。")
        return 1

    INSTANCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # 首次运行：初始化数据目录 → 写配置 → 启动服务 → 创建账号 → 更新项目 .env。
    if not _is_initialized():
        print("正在初始化 MySQL 数据目录……")
        _initialize_data_directory()

    _write_mysql_config()

    if not _is_port_open():
        print("正在启动本机 MySQL……")
        _start_mysql()
        _wait_until_ready()

    # root 凭据仅用于本机初始化；应用日常连接使用权限更小的 rag_app 账号。
    root_password = _read_root_password()
    app_password = _read_existing_app_password() or _generate_password()

    try:
        root_password = _configure_accounts(
            root_password=root_password,
            app_password=app_password,
        )
    except pymysql.MySQLError as exc:
        print(f"MySQL 账号配置失败：{exc}")
        return 1

    _write_root_credentials(root_password)
    _update_project_env(app_password)
    _configure_startup()

    print("MySQL 本机实例配置完成。")
    print(f"监听地址：{MYSQL_HOST}:{MYSQL_PORT}")
    print(f"项目数据库：{MYSQL_DATABASE}")
    print(f"项目账号：{MYSQL_APP_USER}")
    print(f"root 凭据文件：{ROOT_CREDENTIALS_PATH}")
    print("项目 .env 已更新，密码未在终端显示。")
    return 0


def _is_initialized() -> bool:
    """通过系统库目录判断实例是否已经初始化。"""

    return (DATA_DIRECTORY / "mysql").is_dir()


def _initialize_data_directory() -> None:
    """使用无初始密码模式初始化，随后立即设置随机 root 密码。"""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # --initialize-insecure 只在首次、本机离线初始化时使用；随后会立即设定随机 root 密码。
    command = [
        str(MYSQLD_PATH),
        "--initialize-insecure",
        f"--basedir={MYSQL_BASE_DIRECTORY}",
        f"--datadir={DATA_DIRECTORY}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "MySQL 数据目录初始化失败。请查看输出：\n"
            f"{result.stderr.strip()}"
        )


def _write_mysql_config() -> None:
    """生成只监听本机地址的 MySQL 配置。"""

    # 配置文件将 MySQL 限定为本机监听，并统一采用 utf8mb4 存储中英文和表情符号。
    config_text = (
        "[mysqld]\n"
        f"basedir={_as_mysql_path(MYSQL_BASE_DIRECTORY)}\n"
        f"datadir={_as_mysql_path(DATA_DIRECTORY)}\n"
        f"port={MYSQL_PORT}\n"
        f"bind-address={MYSQL_HOST}\n"
        "character-set-server=utf8mb4\n"
        "collation-server=utf8mb4_0900_ai_ci\n"
        "skip-log-bin\n"
        f"log-error={_as_mysql_path(INSTANCE_DIRECTORY / 'mysql-error.log')}\n"
        f"pid-file={_as_mysql_path(INSTANCE_DIRECTORY / 'mysql.pid')}\n"
        "\n"
        "[client]\n"
        f"host={MYSQL_HOST}\n"
        f"port={MYSQL_PORT}\n"
        "default-character-set=utf8mb4\n"
    )
    CONFIG_PATH.write_text(config_text, encoding="utf-8")


def _start_mysql() -> None:
    """在当前用户会话中隐藏启动 mysqld。"""

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

    # 后台启动，不让 mysqld 的控制台窗口或输出干扰用户当前终端。
    subprocess.Popen(
        [str(MYSQLD_PATH), f"--defaults-file={CONFIG_PATH}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        close_fds=True,
    )


def _wait_until_ready(timeout_seconds: int = 30) -> None:
    """等待 MySQL 开始接受 TCP 连接。"""

    # monotonic 不受系统时间调整影响，适合计算重试超时。
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_port_open():
            return
        time.sleep(0.5)

    error_log = INSTANCE_DIRECTORY / "mysql-error.log"
    raise RuntimeError(
        "MySQL 启动超时。请检查错误日志："
        f"{error_log}"
    )


def _is_port_open() -> bool:
    """检查本机 3306 端口是否正在监听。"""

    try:
        with socket.create_connection(
            (MYSQL_HOST, MYSQL_PORT),
            timeout=1,
        ):
            return True
    except OSError:
        return False


def _configure_accounts(
    *,
    root_password: str,
    app_password: str,
) -> str:
    """设置 root 密码并创建权限受限的项目账号。"""

    # 使用 root 只完成一次账号/权限初始化；业务程序永远不用 root 连接。
    connection = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user="root",
        password=root_password,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=5,
    )

    try:
        with connection.cursor() as cursor:
            if not root_password:
                root_password = _generate_password()
                cursor.execute(
                    "ALTER USER 'root'@'localhost' IDENTIFIED BY %s",
                    (root_password,),
                )

            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

            # Windows/客户端可能以 localhost 或 127.0.0.1 连接，两种 host 都创建账号。
            for account_host in ("localhost", "127.0.0.1"):
                cursor.execute(
                    f"CREATE USER IF NOT EXISTS "
                    f"'{MYSQL_APP_USER}'@'{account_host}' IDENTIFIED BY %s",
                    (app_password,),
                )
                cursor.execute(
                    f"ALTER USER '{MYSQL_APP_USER}'@'{account_host}' "
                    "IDENTIFIED BY %s",
                    (app_password,),
                )
                cursor.execute(
                    f"GRANT ALL PRIVILEGES ON `{MYSQL_DATABASE}`.* TO "
                    f"'{MYSQL_APP_USER}'@'{account_host}'"
                )
    finally:
        connection.close()

    return root_password


def _read_root_password() -> str:
    """读取先前保存的 root 密码；首次配置时返回空字符串。"""

    if not ROOT_CREDENTIALS_PATH.exists():
        return ""

    content = ROOT_CREDENTIALS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^password=(.*)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _write_root_credentials(root_password: str) -> None:
    """保存仅供本机管理使用的 root 客户端配置。"""

    # 单独保存 root 客户端配置，避免把管理员密码写入项目的 .env。
    credential_text = (
        "[client]\n"
        "user=root\n"
        f"password={root_password}\n"
        f"host={MYSQL_HOST}\n"
        f"port={MYSQL_PORT}\n"
    )
    ROOT_CREDENTIALS_PATH.write_text(credential_text, encoding="utf-8")

    # 尽量移除继承权限，仅保留当前用户的读写权限。
    current_user = os.environ.get("USERNAME", "")
    if current_user:
        subprocess.run(
            [
                "icacls",
                str(ROOT_CREDENTIALS_PATH),
                "/inheritance:r",
                "/grant:r",
                f"{current_user}:(R,W)",
            ],
            capture_output=True,
            check=False,
        )


def _read_existing_app_password() -> str:
    """重复运行脚本时复用项目已有密码。"""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""

    content = env_path.read_text(encoding="utf-8-sig")
    match = re.search(r"^MYSQL_PASSWORD=(.*)$", content, flags=re.MULTILINE)
    if not match:
        return ""

    password = match.group(1).strip()
    if not password or password == "请填写MySQL密码":
        return ""
    return password


def _update_project_env(app_password: str) -> None:
    """写入项目账号，不覆盖用户后续填写的 DeepSeek 参数。"""

    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if env_path.exists():
        content = env_path.read_text(encoding="utf-8-sig")
    else:
        content = example_path.read_text(encoding="utf-8-sig")

    # 只更新 MySQL 相关键，DeepSeek 等由用户填写的其他配置会保持不变。
    values = {
        "MYSQL_HOST": MYSQL_HOST,
        "MYSQL_PORT": str(MYSQL_PORT),
        "MYSQL_USER": MYSQL_APP_USER,
        "MYSQL_PASSWORD": app_password,
        "MYSQL_DATABASE": MYSQL_DATABASE,
    }
    for key, value in values.items():
        pattern = rf"^{re.escape(key)}=.*$"
        replacement = f"{key}={value}"

        # 有该变量就原地替换；没有则追加一行，兼容缺少部分键的旧 .env。
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(
                pattern,
                replacement,
                content,
                flags=re.MULTILINE,
            )
        else:
            content = f"{content.rstrip()}\n{replacement}\n"

    env_path.write_text(content, encoding="utf-8")


def _write_startup_script() -> None:
    """在当前用户登录时隐藏启动 MySQL。"""

    startup_script = _get_startup_script_path()
    startup_script.parent.mkdir(parents=True, exist_ok=True)

    # VBS 的 Run 参数为 0 表示隐藏窗口，False 表示不等待 MySQL 进程结束。
    command = (
        f'"{MYSQLD_PATH}" '
        f'--defaults-file="{CONFIG_PATH}"'
    )
    escaped_command = command.replace('"', '""')
    script_text = (
        'Set shell = CreateObject("WScript.Shell")\n'
        f'shell.Run "{escaped_command}", 0, False\n'
    )
    startup_script.write_text(script_text, encoding="utf-8")


def _configure_startup() -> None:
    """优先使用 MySQL84 服务，避免服务和登录脚本重复启动。"""

    startup_script = _get_startup_script_path()
    service_check = subprocess.run(
        ["sc.exe", "query", "MySQL84"],
        capture_output=True,
        check=False,
    )
    if service_check.returncode == 0:
        # 已安装 Windows 服务时优先交给服务管理器，避免同一实例被启动两次。
        startup_script.unlink(missing_ok=True)
        print("检测到 MySQL84 Windows 服务，已使用服务启动方式。")
        return

    _write_startup_script()


def _get_startup_script_path() -> Path:
    """返回当前用户登录启动脚本路径。"""

    app_data = Path(os.environ["APPDATA"])
    return (
        app_data
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / "RagSystemMySQL.vbs"
    )


def _generate_password() -> str:
    """生成适用于 MySQL 的高强度随机密码。"""

    return secrets.token_urlsafe(24)


def _as_mysql_path(path: Path) -> str:
    """MySQL 配置使用正斜杠，避免 Windows 反斜杠转义问题。"""

    return path.resolve().as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
