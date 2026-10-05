"""DeepSeek 对话接口封装。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.exceptions import DeepSeekConfigurationError


class DeepSeekClient:
    """通过 OpenAI 兼容协议调用 DeepSeek，并支持流式返回。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: ChatOpenAI | None = None

    def is_configured(self) -> bool:
        """返回 DeepSeek 参数是否完整。"""

        return self._settings.deepseek_is_configured

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        """逐段返回模型生成的文本。"""

        # client.stream 返回生成器，yield 能让界面在模型尚未完成时逐字显示内容。
        client = self._get_client()
        for message_chunk in client.stream(list(messages)):
            # LangChain 不同版本/不同模型的 content 可能是字符串，也可能是内容块列表。
            content = message_chunk.content

            if isinstance(content, str):
                if content:
                    yield content
                continue

            # 新版 LangChain 可能把多模态内容表示为字典列表。
            if isinstance(content, list):
                for content_part in content:
                    if not isinstance(content_part, dict):
                        continue

                    # 非文本内容（如工具调用）不在当前聊天界面展示。
                    text_value = content_part.get("text")
                    if isinstance(text_value, str) and text_value:
                        yield text_value

    def _get_client(self) -> ChatOpenAI:
        """延迟创建模型客户端。"""

        if not self._settings.deepseek_is_configured:
            raise DeepSeekConfigurationError(
                "DeepSeek 尚未配置。请在 .env 中填写 DEEPSEEK_API_KEY、"
                "DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL。"
            )

        if self._client is None:
            # 将 .env 中的 DeepSeek 参数映射到 OpenAI 兼容客户端；只创建一次以便连接复用。
            self._client = ChatOpenAI(
                api_key=self._settings.deepseek_api_key,
                base_url=self._settings.deepseek_base_url,
                model=self._settings.deepseek_model,
                temperature=self._settings.deepseek_temperature,
                timeout=self._settings.deepseek_timeout_seconds,
                max_retries=self._settings.deepseek_max_retries,
                streaming=True,
            )

        return self._client
