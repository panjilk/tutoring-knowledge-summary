# 家教知识总结

这是一个本机运行的 LangChain RAG 项目。系统把 PDF、DOCX、TXT 和 Markdown 文档解析后存入本地 Chroma 向量库，使用 MySQL 保存知识库及文件元数据，再调用 DeepSeek 根据检索内容回答问题并展示引用来源。

## 当前能力

- 创建、选择和删除多个知识库。
- 批量导入 PDF、DOCX、TXT 和 Markdown。
- 使用 SHA-256 防止重复导入相同内容。
- 本地 Embedding 和 Chroma 向量索引。
- DeepSeek 流式回答。
- 展示引用文件、PDF 页码或章节、原文片段和相关度。
- 自动识别整库概览问题，并依据目录和章节结构生成完整回答。
- 自动识别“第几章”，限定章节检索并展示全部实际检索来源。
- 使用 MySQL 保存历史对话，可新建、切换、继续或删除会话。
- 删除文件时同步删除向量和 MySQL 记录。
- 支持在模型或分块参数变化后重建整个知识库索引。
- 缺少数据库或 DeepSeek 配置时提供中文提示。

## 环境要求

- Windows
- Python 3.12.10（当前开发环境版本）
- MySQL 8.0 或更高版本
- 建议至少 16 GB 内存
- 为模型、原始文件和向量索引预留足够磁盘空间

## 1. 创建虚拟环境

在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止运行激活脚本，可以不激活虚拟环境，直接使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 配置环境变量

复制示例文件：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写 MySQL 参数：

```dotenv
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的密码
MYSQL_DATABASE=rag_system
```

DeepSeek 参数等实际 API 信息确定后填写：

```dotenv
DEEPSEEK_API_KEY=你的API密钥
DEEPSEEK_BASE_URL=你的OpenAI兼容接口基础地址
DEEPSEEK_MODEL=你的模型名称
```

`DEEPSEEK_BASE_URL` 应填写 OpenAI 兼容接口的基础地址，而不是单次请求的完整路径。如果服务商只提供完整调用链接，需要根据其接口文档确认应填写的基础地址。

## 3. 初始化 MySQL

先确保 MySQL 服务已经启动，并且 `.env` 中的用户具备创建数据库和数据表的权限，然后运行：

```powershell
python scripts/init_database.py
```

脚本会创建 `rag_system` 数据库（名称以 `.env` 为准）以及以下数据表：

- `knowledge_bases`
- `chat_sessions`
- `chat_messages`
- `documents`
- `document_overviews`
- `document_chunks`
- `ingestion_jobs`

### 当前用户模式安装 MySQL

如果当前终端没有管理员权限，可以使用项目提供的脚本初始化用户级 MySQL 实例：

```powershell
.\.venv\Scripts\python.exe scripts\setup_local_mysql.py
```

该脚本会：

- 将数据保存到 `%LOCALAPPDATA%\MySQL\RagSystem`。
- 默认仅监听 `127.0.0.1:3306`；端口冲突时可使用其他本机端口。
- 创建 `rag_system` 数据库和权限受限的 `rag_app` 账号。
- 使用随机密码更新项目 `.env`，不在终端显示密码。
- 优先使用现有的 `MySQL84` Windows 服务；没有服务时才创建当前用户隐藏启动项。

这属于当前用户实例，不是系统级 Windows 服务。需要手动重新启动时，可再次运行 `setup_local_mysql.py`。检查端口状态可使用：

```powershell
Get-NetTCPConnection -LocalPort 3306 -State Listen
```

如果 3306 已被其他 MySQL 实例占用，可为项目实例指定 3307：

```powershell
$env:RAG_MYSQL_PORT = "3307"
.\.venv\Scripts\python.exe scripts\setup_local_mysql.py
Remove-Item Env:RAG_MYSQL_PORT
```

## 4. 启动应用

```powershell
python -m streamlit run app.py
```

如果没有激活虚拟环境，或同时启用了 Conda 环境，直接指定项目的 Python：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器默认访问：<http://localhost:8501>

如果使用 `streamlit run app.py` 遇到 `Fatal error in launcher`，并且报错中的
Python 路径与当前项目路径不一致，通常是移动项目后，`.venv` 中的启动器仍保存着旧的绝对路径。
可使用上面的 `.\.venv\Scripts\python.exe -m streamlit run app.py` 绕过启动器。
以后移动项目时，建议在新位置重新创建虚拟环境并安装依赖。

## 5. 使用流程

1. 在左侧输入名称并创建知识库。
2. 打开“文档管理”，上传并导入文件。
3. 第一次导入时，本地 Embedding 模型可能需要联网下载，请等待下载完成。
4. 文档状态变为“已完成”后，进入“知识库问答”。
5. 输入问题，查看 DeepSeek 回答和答案下方的引用来源。

单次批量上传总量默认限制为 1024MB，可在 `.env` 中通过
`MAX_UPLOAD_BATCH_MB` 调整。对于大量资料，建议分批导入并逐批确认状态。

## 数据存储说明

- 原始文件：`data/uploads/<知识库ID>/`
- Chroma 向量索引：`data/vector_store/`
- 知识库、文档及任务元数据：MySQL
- 聊天记录：仅保存在当前 Streamlit 会话中，关闭后不保留

系统不会把整个知识库发送给 DeepSeek，但会把与当前问题相关的少量原文片段发送给 DeepSeek 用于生成答案。

## 运行测试

```powershell
python -m pytest
```

测试不会真实调用 DeepSeek，也不会消耗 API 额度。

## 常见问题

### 页面提示 MySQL 无法连接

检查 MySQL 是否启动、端口是否正确、账号是否有权限，然后重新运行初始化脚本。

### 首次导入很慢

首次使用会下载 Embedding 模型。CPU 环境下，大量文件的向量化也需要较长时间。建议先用少量文件验证完整流程。

### PDF 导入后没有文本

第一版只支持包含可复制文本的电子 PDF。扫描版 PDF 需要 OCR，暂未实现。

### 知识库明明有内容但提示没有相关资料

可在 `.env` 中适当调低 `RETRIEVAL_SCORE_THRESHOLD`，例如从 `0.35` 调整到 `0.25`，然后重启应用。阈值过低可能引入无关资料。

### 提示向量索引配置不一致

这表示 `EMBEDDING_MODEL_NAME`、`CHUNK_SIZE` 或 `CHUNK_OVERLAP`
在建库后发生了变化。进入“文档管理”页面，展开“索引维护”，确认后点击
“重建索引”。系统会保留原文件，只重新生成向量。

### transformers 报缺少 torchvision

本项目是文本 RAG，不需要安装 `torchvision`。项目已通过
`.streamlit/config.toml` 关闭第三方模块源码扫描。停止旧进程后，必须使用
项目虚拟环境重新启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 项目结构

```text
rag-system/
├── app.py
├── scripts/
│   └── init_database.py
├── src/
│   ├── config/
│   ├── database/
│   ├── document_loaders/
│   ├── embeddings/
│   ├── llm/
│   └── services/
├── tests/
├── .env.example
├── requirements.txt
└── TECHNICAL_REQUIREMENTS.md
```
