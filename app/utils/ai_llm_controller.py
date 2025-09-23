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
TASK_TO_MODEL_FAMILY: Dict[str, Literal["gemini", "deepseek", "openai", "local"]] = {
    "default": "openai",
    "analysis": "gemini",
    "code": "deepseek",
    "summarize": "openai",
}

# ---------- コスト計算 ----------
MODEL_COST = {
    "gemini": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
    "openai": {"input": 5.00/1_000_000, "output": 15.00/1_000_000},
    "deepseek": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
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
        self.gemini_model = None
        self.deepseek_client = None
        self.openai_client = None
        if genai:
            key = self.cfg.get("GEMINI_API_KEY")
            if key:
                genai.configure(api_key=key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        if OpenAI:
            d_key = self.cfg.get("DEEPSEEK_API_KEY")
            d_base = self.cfg.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            o_key = self.cfg.get("OPENAI_API_KEY")
            if d_key:
                self.deepseek_client = OpenAI(api_key=d_key, base_url=d_base)
            if o_key:
                self.openai_client = OpenAI(api_key=o_key)
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
            CACHE.set(cache_key, result.to_dict(), expire=300)  # 5分TTL
            return result

    # ---- キックロジック（リトライ＋フォールバック） ----
    def _generate_with_retry(
        self,
        family: Literal["gemini", "deepseek", "openai", "local"],
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        chunk_callback: Optional[Callable[[str], None]],
    ) -> GenerateResult:
        families = [family, "openai", "deepseek", "local"] if family != "local" else ["local"]
        for attempt, fam in enumerate(families, 1):
            if fam == 'local' and not LOCAL_LLAMA: continue # ローカルモデルがなければスキップ
            try:
                logger.info(f"Attempt {attempt} with {fam}")
                if stream and fam != "local":
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
                time.sleep(1.5 ** attempt) # 指数バックオフ

    # ---- 生呼び出し ----
    def _raw_call(self, fam: str, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> tuple[str, dict[str, int]]:
        if fam == "gemini" and self.gemini_model:
            response = self.gemini_model.generate_content(prompt)
            usage = {"input": len(prompt) // 4, "output": len(response.text) // 4}
            return response.text, usage
        if fam == "deepseek" and self.deepseek_client:
            cmpl = self.deepseek_client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], tools=tools or [])
            txt = cmpl.choices[0].message.content or ""
            usage = cmpl.usage.model_dump() if cmpl.usage else {}
            return txt, usage
        if fam == "openai" and self.openai_client:
            cmpl = self.openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], tools=tools or [])
            txt = cmpl.choices[0].message.content or ""
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
        if fam == "openai" and self.openai_client:
            for chunk in self.openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], stream=True):
                delta = chunk.choices[0].delta.content or ""
                text += delta
                if chunk_callback: chunk_callback(delta)
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage
        raise RuntimeError(f"Streaming is only supported for OpenAI at the moment, but got {fam}")

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
