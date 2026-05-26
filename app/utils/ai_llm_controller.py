"""LLM controller: unified interface for Gemini / OpenAI / DeepSeek / local models."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from app.config.constants import (
    LLM_CACHE_TTL_SECONDS,
    LLM_LOCAL_CTX_SIZE,
    LLM_LOCAL_MAX_TOKENS,
    LLM_LOCAL_TEMPERATURE,
    LLM_RETRY_BACKOFF_BASE,
)
from app.models.result_models import GenerateResult

logger = logging.getLogger(__name__)

# ---------- Optional imports ----------
try:
    from opentelemetry import trace
    tracer = trace.get_tracer("ai_llm_controller")
except ImportError:
    class _NoopTracer:
        def start_as_current_span(self, name):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def set_attribute(self, *a):
            pass
    tracer = _NoopTracer()

try:
    import diskcache
    _CACHE_DIR = Path(__file__).resolve().parents[2] / "instance" / "llm_cache"
    _CACHE = diskcache.Cache(_CACHE_DIR, size_limit=500_000_000)
except Exception:
    _CACHE = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Model names ----------
GEMINI_MODEL = "gemini-1.5-flash-latest"
OPENAI_MODEL = "gpt-4o"
DEEPSEEK_MODEL = "deepseek-chat"
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "llama-7b-q4_0.gguf"

# ---------- Task routing ----------
TASK_TO_MODEL_FAMILY: dict[str, Literal["gemini", "deepseek", "openai", "local"]] = {
    "default": "openai",
    "analysis": "gemini",
    "code": "deepseek",
    "summarize": "openai",
}

MODEL_COST = {
    "gemini": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "openai": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "deepseek": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "local": {"input": 0, "output": 0},
}


def _load_config() -> dict[str, Any]:
    """設定を読み込む（Flask > AppConfig > Secrets > 環境変数）"""
    config_dict: dict[str, Any] = {}
    try:
        from flask import current_app
        config_dict.update(dict(current_app.config))
        return config_dict
    except Exception:
        pass
    try:
        from app.config.config import AppConfig
        for k in dir(AppConfig):
            if not k.startswith("_") and not callable(getattr(AppConfig, k)):
                config_dict[k] = getattr(AppConfig, k)
    except Exception:
        pass
    try:
        from app.config.secrets import Secrets
        for k in dir(Secrets):
            if not k.startswith("_") and not callable(getattr(Secrets, k)):
                val = getattr(Secrets, k)
                if val:
                    config_dict[k] = val
    except Exception:
        pass
    _ENV_KEYS = [
        "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_BASE", "LLM_ORDER_*", "CRAWLER_*",
        "FLASK_APP", "STAGE", "BUYANDSHIP_*",
    ]
    for k in _ENV_KEYS:
        # Support wildcard prefix matching (e.g. "LLM_ORDER_*")
        if k.endswith("*"):
            prefix = k.rstrip("*")
            for ek, ev in os.environ.items():
                if ek.startswith(prefix) and ev:
                    config_dict[ek] = ev
        else:
            val = os.environ.get(k)
            if val:
                config_dict[k] = val
    return config_dict


class AILlmController:
    _instance: AILlmController | None = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
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
        self._local_nlp = None
        self._local_llama = None
        self._init_clients()
        self._initialized = True

    def _init_clients(self) -> None:
        if genai:
            key = self.cfg.get("GEMINI_API_KEY")
            if key:
                genai.configure(api_key=key)
                self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        if OpenAI:
            d_key = self.cfg.get("DEEPSEEK_API_KEY")
            d_base = self.cfg.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            o_key = self.cfg.get("OPENAI_API_KEY")
            if d_key:
                self.deepseek_client = OpenAI(api_key=d_key, base_url=d_base)
            if o_key:
                self.openai_client = OpenAI(api_key=o_key)

    # ---- Public API ----

    def generate(
        self,
        prompt: str,
        task_type: str = "default",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> GenerateResult:
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("task.type", task_type)
            span.set_attribute("prompt.length", len(prompt))
            family = TASK_TO_MODEL_FAMILY.get(task_type, "default")
            cache_key = self._cache_key(family, prompt, tools)

            if _CACHE:
                cached = _CACHE.get(cache_key)
                if cached:
                    logger.info("Cache hit")
                    return GenerateResult(**cached, cached=True)

            result = self._generate_with_retry(family, prompt, tools, stream, chunk_callback)

            if _CACHE:
                _CACHE.set(cache_key, result.to_dict(), expire=LLM_CACHE_TTL_SECONDS)

            from app.utils.chat_history_saver import save_chat_history
            save_chat_history(prompt, result)

            return result

    @classmethod
    def quick(cls, prompt: str, task: str = "default") -> str:
        return cls().generate(prompt, task_type=task).text

    # ---- Retry + fallback ----

    def _generate_with_retry(
        self,
        family: Literal["gemini", "deepseek", "openai", "local"],
        prompt: str,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        chunk_callback: Callable[[str], None] | None,
    ) -> GenerateResult:
        families = [family, "openai", "deepseek", "local"] if family != "local" else ["local"]
        for attempt, fam in enumerate(families, 1):
            if fam == "local" and self._local_llama is None:
                if not LOCAL_MODEL_PATH.exists():
                    continue
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
                time.sleep(LLM_RETRY_BACKOFF_BASE**attempt)

    # ---- Provider dispatch ----

    def _raw_call(self, fam: str, prompt: str, tools: list[dict[str, Any]] | None) -> tuple[str, dict[str, int]]:
        handler = {
            "gemini": self._call_gemini,
            "deepseek": self._call_openai_compat,
            "openai": self._call_openai_compat,
            "local": self._call_local,
        }.get(fam)
        if not handler:
            raise RuntimeError(f"No client configured for {fam}")
        return handler(fam, prompt, tools)

    def _call_gemini(self, _fam: str, prompt: str, _tools: list[dict[str, Any]] | None) -> tuple[str, dict[str, int]]:
        if not self.gemini_model:
            raise RuntimeError("Gemini client not configured")
        response = self.gemini_model.generate_content(prompt)
        usage = {"input": len(prompt) // 4, "output": len(response.text) // 4}
        return response.text, usage

    def _call_openai_compat(self, fam: str, prompt: str, tools: list[dict[str, Any]] | None) -> tuple[str, dict[str, int]]:
        client = self.deepseek_client if fam == "deepseek" else self.openai_client
        if not client:
            raise RuntimeError(f"{fam} client not configured")
        model = DEEPSEEK_MODEL if fam == "deepseek" else OPENAI_MODEL
        cmpl = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], tools=tools or [],
        )
        txt = cmpl.choices[0].message.content or ""
        usage = cmpl.usage.model_dump() if cmpl.usage else {}
        return txt, usage

    def _call_local(self, _fam: str, prompt: str, _tools: list[dict[str, Any]] | None) -> tuple[str, dict[str, int]]:
        if self._local_llama is None:
            try:
                from llama_cpp import Llama
                if LOCAL_MODEL_PATH.exists():
                    self._local_llama = Llama(model_path=str(LOCAL_MODEL_PATH), n_ctx=LLM_LOCAL_CTX_SIZE)
            except Exception:
                pass
        if self._local_llama is None:
            raise RuntimeError("Local Llama model not available")
        out = self._local_llama(prompt, max_tokens=LLM_LOCAL_MAX_TOKENS, temperature=LLM_LOCAL_TEMPERATURE)
        txt = out["choices"][0]["text"]
        return txt, {"input": len(prompt) // 4, "output": len(txt) // 4}

    # ---- Streaming ----

    def _stream_call(self, fam: str, prompt: str, chunk_callback: Callable[[str], None]) -> tuple[str, dict[str, int]]:
        if fam == "openai" and self.openai_client:
            text = ""
            for chunk in self.openai_client.chat.completions.create(
                model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], stream=True,
            ):
                delta = chunk.choices[0].delta.content or ""
                text += delta
                if chunk_callback:
                    chunk_callback(delta)
            return text, {"input": len(prompt) // 4, "output": len(text) // 4}
        raise RuntimeError(f"Streaming is only supported for OpenAI at the moment, but got {fam}")

    # ---- Utilities ----

    def _cache_key(self, fam: str, prompt: str, tools: list[dict[str, Any]] | None) -> str:
        content = json.dumps({"fam": fam, "prompt": prompt, "tools": tools}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _calc_cost(self, fam: str, usage: dict[str, int]) -> float:
        inp = usage.get("prompt_tokens", usage.get("input", 0))
        out = usage.get("completion_tokens", usage.get("output", 0))
        c = MODEL_COST.get(fam, {"input": 0, "output": 0})
        return inp * c["input"] + out * c["output"]

    def _sentiment(self, text: str) -> Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if self._local_nlp is None:
            try:
                from transformers import pipeline
                self._local_nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except Exception:
                return "NEUTRAL"
        try:
            label = self._local_nlp(text[:512])[0]["label"]
            return "POSITIVE" if label == "POSITIVE" else "NEGATIVE"
        except Exception:
            return "NEUTRAL"
