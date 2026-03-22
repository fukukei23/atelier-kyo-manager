# ==============================================================================
# File: ai_llm_controller.py
# Registry: C:\Users\USER\tools\atelier-kyo-manager\app\utils\ai_llm_controller.py
# Date & Time (JST): 2025-09-16 14:00:00
# Version: 4.1J (Standardized Model Integration)
#
# --- What's New (v4.1J) ---
#  - [Refactor] Removed the local GenerateResult dataclass definition.
#  - [Integration] Now imports the standardized `GenerateResult` from
#    `app.models.result_models` to ensure system-wide data consistency.
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Literal
from dataclasses import asdict
from datetime import timedelta

# --- 共通データモデルをインポート ---
from app.models.result_models import GenerateResult

# ---------- ロガー ----------
logger = logging.getLogger(__name__)

# ---------- OpenTelemetry ----------
from opentelemetry import trace
tracer = trace.get_tracer("ai_llm_controller")

# ---------- キャッシュ ----------
import diskcache
CACHE_DIR = Path(__file__).resolve().parents[2] / "instance" / "llm_cache"
CACHE = diskcache.Cache(CACHE_DIR, size_limit=500_000_000)  # 500 MB

# ---------- LLM ライブラリ ----------
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    from transformers import pipeline
    LOCAL_NLP = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except ImportError:
    LOCAL_NLP = None
try:
    from llama_cpp import Llama
    # 注意: モデルパスは環境に合わせて調整してください
    model_path = Path(__file__).resolve().parents[2] / "models" / "llama-7b-q4_0.gguf"
    if model_path.exists():
        LOCAL_LLAMA = Llama(model_path=str(model_path), n_ctx=2048)
    else:
        LOCAL_LLAMA = None
except Exception:
    LOCAL_LLAMA = None


# ---------- 設定ローダ ----------
def _load_config() -> Dict[str, Any]:
    try:
        from flask import current_app
        return dict(current_app.config)
    except Exception:
        try:
            from app.config.config import Config
            return {k: getattr(Config, k) for k in dir(Config) if not k.startswith("_")}
        except Exception:
            return dict(os.environ)

# ---------- ポリシー ----------
TASK_TO_MODEL_FAMILY: Dict[str, Literal["minimax", "glm", "local"]] = {
    "default": "minimax",
    "analysis": "minimax",
    "code": "minimax",
    "summarize": "minimax",
}

# ---------- コスト計算 ----------
MODEL_COST = {
    "minimax": {"input": 0.10/1_000_000, "output": 0.50/1_000_000},  # MiniMax pricing
    "glm": {"input": 0.10/1_000_000, "output": 0.50/1_000_000},  # GLM pricing
    "local": {"input": 0, "output": 0},
}

# ---------- 本体 ----------
class AILlmController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.cfg = _load_config()
        self.minimax_client = None
        self.glm_client = None
        if OpenAI:
            m_key = self.cfg.get("MINIMAX_API_KEY")
            m_base = self.cfg.get("MINIMAX_API_BASE", "https://api.minimax.chat")
            g_key = self.cfg.get("GLM_API_KEY")
            g_base = self.cfg.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
            if m_key:
                self.minimax_client = OpenAI(api_key=m_key, base_url=m_base)
            if g_key:
                self.glm_client = OpenAI(api_key=g_key, base_url=g_base)
        self._initialized = True

    # ---- 公開 generate ----
    def generate(
        self,
        prompt: str,
        task_type: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> GenerateResult:
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("task.type", task_type)
            span.set_attribute("prompt.length", len(prompt))
            family = TASK_TO_MODEL_FAMILY.get(task_type, "default")
            cache_key = self._cache_key(family, prompt, tools)
            cached = CACHE.get(cache_key)
            if cached:
                logger.info("Cache hit")
                # キャッシュから復元する際も to_dict は不要
                return GenerateResult(**cached, cached=True)

            result = self._generate_with_retry(family, prompt, tools, stream, chunk_callback)
            if result is None:
                raise ValueError(f"All LLM providers failed for task_type={family}")
            CACHE.set(cache_key, result.to_dict(), expire=300)  # 5分TTL
            return result

    # ---- キックロジック（リトライ＋フォールバック） ----
    def _generate_with_retry(
        self,
        family: Literal["minimax", "glm", "local"],
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        chunk_callback: Optional[Callable[[str], None]],
    ) -> GenerateResult:
        families = [family, "glm", "local"] if family != "local" else ["local"]
        for attempt, fam in enumerate(families, 1):
            if fam == 'local' and not LOCAL_LLAMA: continue
            try:
                logger.info(f"Attempt {attempt} with {fam}")
                if stream and fam in ("minimax", "glm"):
                    text, usage = self._stream_call(fam, prompt, chunk_callback)
                else:
                    text, usage = self._raw_call(fam, prompt, tools)
                cost = self._calc_cost(fam, usage)
                sent = self._sentiment(text)
                return GenerateResult(text=text, tokens=usage, cost_usd=cost, model_family=fam, sentiment=sent)
            except Exception as e:
                logger.warning(f"{fam} failed: {e}")
                if attempt == len(families):
                    raise
                time.sleep(1.5 ** attempt)

    # ---- 生呼び出し ----
    def _raw_call(self, fam: str, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> tuple[str, dict[str, int]]:
        if fam == "minimax" and self.minimax_client:
            cmpl = self.minimax_client.chat.completions.create(model="MiniMax-M2.7", messages=[{"role": "user", "content": prompt}], tools=tools or [])
            txt = cmpl.choices[0].message.content or ""
            usage = cmpl.usage.model_dump() if cmpl.usage else {}
            return txt, usage
        if fam == "glm" and self.glm_client:
            cmpl = self.glm_client.chat.completions.create(model="glm-5", messages=[{"role": "user", "content": prompt}], tools=tools or [])
            msg = cmpl.choices[0].message
            txt = msg.content or msg.reasoning_content or ""  # GLM uses reasoning_content
            usage = cmpl.usage.model_dump() if cmpl.usage else {}
            return txt, usage
        if fam == "local" and LOCAL_LLAMA:
            out = LOCAL_LLAMA(prompt, max_tokens=512, temperature=0.3)
            txt = out["choices"][0]["text"]
            usage = {"input": len(prompt) // 4, "output": len(txt) // 4}
            return txt, usage
        raise RuntimeError(f"No client configured for {fam}")

    # ---- ストリーミング ----
    def _stream_call(self, fam: str, prompt: str, chunk_callback: Callable[[str], None]) -> tuple[str, dict[str, int]]:
        text = ""
        if fam == "minimax" and self.minimax_client:
            for chunk in self.minimax_client.chat.completions.create(model="MiniMax-M2.7", messages=[{"role": "user", "content": prompt}], stream=True):
                delta = chunk.choices[0].delta.content or ""
                text += delta
                if chunk_callback: chunk_callback(delta)
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage
        if fam == "glm" and self.glm_client:
            for chunk in self.glm_client.chat.completions.create(model="glm-5", messages=[{"role": "user", "content": prompt}], stream=True):
                delta = chunk.choices[0].delta.content or chunk.choices[0].delta.reasoning_content or ""
                text += delta
                if chunk_callback: chunk_callback(delta)
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage
        raise RuntimeError(f"Streaming not supported for {fam}")

    # ---- ユーティリティ ----
    def _cache_key(self, fam: str, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> str:
        content = json.dumps({"fam": fam, "prompt": prompt, "tools": tools}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _calc_cost(self, fam: str, usage: dict[str, int]) -> float:
        inp = usage.get("prompt_tokens", usage.get("input", 0))
        out = usage.get("completion_tokens", usage.get("output", 0))
        c = MODEL_COST.get(fam, {"input": 0, "output": 0})
        return inp * c["input"] + out * c["output"]

    def _sentiment(self, text: str) -> Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if LOCAL_NLP:
            try:
                label = LOCAL_NLP(text[:512])[0]["label"]
                return "POSITIVE" if label == "POSITIVE" else "NEGATIVE"
            except Exception: pass
        return "NEUTRAL"

    # ---- クラスメソッド：簡易呼び出し ----
    @classmethod
    def quick(cls, prompt: str, task: str = "default") -> str:
        return cls().generate(prompt, task_type=task).text
