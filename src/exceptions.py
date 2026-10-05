"""项目自定义异常。"""


class RagSystemError(Exception):
    """项目内可预期异常的基类。"""


class DatabaseConnectionError(RagSystemError):
    """MySQL 无法连接或数据库尚未初始化。"""


class DuplicateDocumentError(RagSystemError):
    """当前知识库已经存在内容相同的文件。"""


class UnsupportedFileError(RagSystemError):
    """文件格式不在允许范围内。"""


class DocumentParseError(RagSystemError):
    """文档无法正确解析。"""


class DeepSeekConfigurationError(RagSystemError):
    """DeepSeek 参数缺失。"""


class NoRelevantContextError(RagSystemError):
    """知识库没有检索到足够相关的资料。"""


class IndexConfigurationMismatchError(RagSystemError):
    """当前向量索引与应用配置不兼容。"""
