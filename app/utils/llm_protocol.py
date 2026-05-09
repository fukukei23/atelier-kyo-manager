"""
app/utils/llm_protocol.py - LLM クライアントの抽象化プロトコル

責務分割・結合度低減のため、LLM 呼び出しをプロトコルで抽象化する。
テスト時や別実装（スタブ・別プロバイダ）の注入を可能にする。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """LLM 生成を提供するプロトコル。AiLlmController の公開 API に合わせる。"""

    def generate(
        self,
        prompt: str,
        task_type: str = "default",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> Any:
        """プロンプトからテキストを生成し、GenerateResult 相当を返す。"""
        ...
