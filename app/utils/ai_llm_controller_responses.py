# ==============================================================================
# File: app/utils/ai_llm_controller_responses.py
# Date (JST): 2025-10-26 18:05
# Version: 1.0R (Responses API controller)
#
# 概要:
# - OpenAI Responses API 専用の LLM コントローラ
# - 既存 AILlmController と同一インターフェイスを提供（generate）
# - 既定モデルは GPT-5 を使用し、task_type=analysis/code も GPT-5 を優先する
# - Gemini等のフォールバックはこのResponses版では扱わず、必要な場合は
#   USE_RESPONSES_API を無効化して ai_llm_controller.py 側の通常ルートを使う
# - 切替は env(USE_RESPONSES_API) のみ
# ==============================================================================

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # >=1.42 系
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# 互換用の戻り値オブジェクト（既存コードが参照する属性を揃える）
@dataclass
class GenerateResult:
    text: str
    tokens: Dict[str, int]
    cost_usd: float
    model_family: str
    sentiment: str = "neutral"
    cached: bool = False

class AILlmController:
    """
    Responses API 専用版。既存の AILlmController と同名・同メソッドで差し替え可能。
    """

    def __init__(self) -> None:
        self._initialized = False
        self._init()

    def _init(self) -> None:
        if self._initialized:
            return
        # ---- env 読み込み ----
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        # Responses 用の既定モデルは gpt-5（無印）
        self.openai_model = os.environ.get("OPENAI_MODEL_RESPONSES", "gpt-5")
        # オプション：解析/コードで別モデルを振り分けたい場合
        self.openai_model_analysis = os.environ.get("OPENAI_MODEL_ANALYSIS_RESPONSES", self.openai_model)
        self.openai_model_code     = os.environ.get("OPENAI_MODEL_CODE_RESPONSES", self.openai_model)

        # ---- OpenAI クライアント ----
        self.client = None
        if OpenAI and self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                # 起動時プローブ：超短文で Responses を叩き、モデル存在と権限を確認
                _ = self.client.responses.create(model=self.openai_model, input="ping")
                logger.info("[LLM/Responses] Probed OpenAI model='%s' successfully.", self.openai_model)
            except Exception as e:
                logger.warning("[LLM/Responses] OpenAI probe failed for model='%s': %s", self.openai_model, e)

        self._initialized = True

    # ----------------------------------------------------------------------
    # 公開API: generate
    # ----------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        task_type: str = "general",
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
        json_schema: Optional[Dict[str, Any]] = None,  # 追加: 構造化出力を要求する場合
    ) -> GenerateResult:
        """
        Responses API で生成を実行。従来の AILlmController.generate と同シグネチャ。
        - tools: [{"type":"function","function":{"name": "...", "description": "...", "parameters": {JSONSchema}}}, ...]
        - json_schema: {"name": "...", "schema": {JSONSchema}, "strict": True}
        """
        if not self.client:
            raise RuntimeError("OpenAI client unavailable for Responses API. Check OPENAI_API_KEY and package versions.")

        # タスク別モデル選択（任意で上書き）
        model = self.openai_model
        lt = task_type.lower()
        if "analysis" in lt or "forensic" in lt or "repair" in lt:
            model = self.openai_model_analysis
        elif "code" in lt or "selector" in lt:
            model = self.openai_model_code

        # Responses API の呼び出しペイロードを構築
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": prompt,
        }

        # 構造化出力（スキーマ）を使う場合
        if json_schema:
            # OpenAI v1 Responses: response_format={"type":"json_schema","json_schema":{...}}
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        # ツール（function calling）を使う場合
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        usage_tokens: Dict[str, int] = {}
        text_out = ""

        if stream:
            # ストリーミング（チャンクをテキスト結合。必要あれば tool_calls も拾える）
            with self.client.responses.stream(**kwargs) as s:
                for event in s:
                    if event.type == "response.output_text.delta":
                        delta = event.delta
                        if delta:
                            if chunk_callback:
                                try:
                                    chunk_callback(delta)
                                except Exception:
                                    pass
                            text_out += delta
                    elif event.type == "response.completed":
                        try:
                            u = event.response.usage
                            usage_tokens = {"input": int(u.input_tokens), "output": int(u.output_tokens)}
                        except Exception:
                            usage_tokens = {}
                s.close()
        else:
            res = self.client.responses.create(**kwargs)
            # 出力テキストを抽出（tool呼び出し時は text と tool_calls が混在する）
            try:
                text_out = res.output_text  # SDKが用意するラクチンアクセサ
            except Exception:
                # 万一のフォールバック：content[...].text を手動で結合
                text_out = _collect_text_from_response(res)

            try:
                u = res.usage
                usage_tokens = {"input": int(u.input_tokens), "output": int(u.output_tokens)}
            except Exception:
                usage_tokens = {}

        # コスト計算は省略（レートは公開頻繁変更のため 0.0 固定）
        return GenerateResult(
            text=(text_out or "").strip(),
            tokens=usage_tokens,
            cost_usd=0.0,
            model_family="openai-responses",
            sentiment="neutral",
            cached=False,
        )

# ----------------------------------------------------------------------
# 内部ヘルパ
# ----------------------------------------------------------------------
def _collect_text_from_response(res: Any) -> str:
    """
    Responses API の生レスポンスからテキストだけ結合するフォールバック。
    res.output_text が無ければ content[].text[...] を探す。
    """
    try:
        parts = []
        for block in getattr(res, "content", []) or []:
            # block.type == "output_text" のとき
            t = getattr(block, "text", None)
            if not t:
                # text フィールドが list 構造の場合もある
                t = "".join([seg.get("text","") for seg in getattr(block, "text_annotations", []) or []])
            if t:
                parts.append(t)
        return "".join(parts)
    except Exception:
        try:
            return str(res)
        except Exception:
            return ""
