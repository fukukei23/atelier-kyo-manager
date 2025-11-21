# ==============================================================================
# File: app/utils/ai_llm_controller.py
# Date & Time (JST): 2025-10-15 17:15
# Version: 6.1.0J (Production LLM Controller, Secrets-first, Safe Fallback)
#
# 目的
# - 本番LLMコントローラ。スタブ実装は一切含まない。
# - Secrets(.env→generate_secrets.py) を最優先で読み込み、未設定の
#   バックエンドは自動的に候補から除外。
# - OpenAI / Gemini / DeepSeek / Local(llama.cpp) に対応、順次フォールバック。
# - ディスクキャッシュ(5分TTL)で重複呼び出しを低減。
# - 返却は app.models.result_models.GenerateResult を統一使用。
#
# 前提
# - .env に API キーを記載 → `python app/config/generate_secrets.py` 実行済み
# - 使うプロバイダの依存:
#     OpenAI:             pip install openai
#     Gemini:             pip install google-generativeai
#     DeepSeek(API互換):  pip install openai
#     Local(任意):        pip install llama-cpp-python
#
# Secrets優先の読み取り順:
#   1) app.config.secrets.Secrets  ← 最優先
#   2) Flask current_app.config     (あれば)
#   3) os.environ                   (最後の砦)
# ==============================================================================

from __future__ import annotations
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Literal

from app.models.result_models import GenerateResult

logger = logging.getLogger(__name__)

# ---------------- OpenTelemetry (任意) ----------------
try:
    from opentelemetry import trace  # type: ignore
    tracer = trace.get_tracer("ai_llm_controller")
except Exception:
    class _Dummy:
        def start_as_current_span(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, *exc): return False
                def set_attribute(self, *_a, **_k): pass
            return _Ctx()
    tracer = _Dummy()

# ---------------- ディスクキャッシュ (任意) ----------------
try:
    import diskcache  # type: ignore
    CACHE_DIR = Path(__file__).resolve().parents[2] / "instance" / "llm_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE = diskcache.Cache(CACHE_DIR, size_limit=500_000_000)  # 500MB
except Exception:
    CACHE = None

# ---------------- LLM クライアント遅延ロード ----------------
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None


# ---------- 設定ローダ（Secrets優先） ----------
def _load_env_config() -> Dict[str, Any]:
    """
    優先順:
      1) app.config.secrets.Secrets  (.env → generate_secrets.py で生成)
      2) Flask current_app.config
      3) os.environ
    """
    # 1) Secrets 最優先
    try:
        from app.config.secrets import Secrets  # 自動生成ファイル
        cfg = {
            "OPENAI_API_KEY": getattr(Secrets, "OPENAI_API_KEY", None),
            "OPENAI_MODEL": getattr(Secrets, "OPENAI_MODEL", None) or "gpt-4o-mini",
            "GEMINI_API_KEY": getattr(Secrets, "GEMINI_API_KEY", None),
            "GEMINI_MODEL": getattr(Secrets, "GEMINI_MODEL", None) or "gemini-1.5-flash-latest",
            "DEEPSEEK_API_KEY": getattr(Secrets, "DEEPSEEK_API_KEY", None),
            "DEEPSEEK_MODEL": getattr(Secrets, "DEEPSEEK_MODEL", None) or "deepseek-chat",
            "DEEPSEEK_API_BASE": getattr(Secrets, "DEEPSEEK_API_BASE", None) or "https://api.deepseek.com",
            # Local llama.cpp（任意）
            "LOCAL_LLAMA_PATH": getattr(Secrets, "LOCAL_LLAMA_PATH", None),  # models/llama-*.gguf など
        }
        return {k: v for k, v in cfg.items() if v}
    except Exception:
        pass

    # 2) Flask config
    try:
        from flask import current_app  # type: ignore
        c = dict(current_app.config or {})
        if c:
            return c
    except Exception:
        pass

    # 3) 環境変数
    return dict(os.environ)


# ---------- タスク→モデルファミリ ----------
TASK_TO_MODEL_FAMILY: Dict[str, Literal["gemini", "openai", "deepseek", "local"]] = {
    "default": "openai",
    "analysis": "gemini",   # 解析系は Gemini を優先（未設定なら自動スキップ）
    "code": "openai",
    "summarize": "openai",
}

# ---------- コスト概算（任意） ----------
MODEL_COST = {
    "gemini":   {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
    "openai":   {"input": 5.00/1_000_000, "output": 15.00/1_000_000},
    "deepseek": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
    "local":    {"input": 0.0,            "output": 0.0},
}


class AILlmController:
    """プロダクション LLM コントローラ（シングルトン & Secrets-first）。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.cfg = _load_env_config()

        # --- クライアントの初期化（存在するものだけ） ---
        self.gemini_model = None
        if genai and self.cfg.get("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=self.cfg["GEMINI_API_KEY"])
                self.gemini_model = genai.GenerativeModel(self.cfg.get("GEMINI_MODEL", "gemini-1.5-flash-latest"))
            except Exception as e:
                logger.warning(f"[LLM] Gemini init failed: {e}")

        self.openai_client = None
        if OpenAI and self.cfg.get("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=self.cfg["OPENAI_API_KEY"])
            except Exception as e:
                logger.warning(f"[LLM] OpenAI init failed: {e}")

        self.deepseek_client = None
        if OpenAI and self.cfg.get("DEEPSEEK_API_KEY"):
            try:
                base = self.cfg.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
                self.deepseek_client = OpenAI(api_key=self.cfg["DEEPSEEK_API_KEY"], base_url=base)
            except Exception as e:
                logger.warning(f"[LLM] DeepSeek init failed: {e}")

        self.local_llama = None
        model_path = self.cfg.get("LOCAL_LLAMA_PATH")
        if Llama and model_path:
            try:
                self.local_llama = Llama(model_path=str(model_path), n_ctx=2048)
            except Exception as e:
                logger.warning(f"[LLM] Local llama init failed: {e}")

        self._initialized = True

    # ---------------- 公開API ----------------
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

            fam = TASK_TO_MODEL_FAMILY.get(task_type, "default")
            cache_key = self._cache_key(fam, prompt, tools)

            if CACHE is not None:
                cached = CACHE.get(cache_key)
                if cached:
                    logger.info("LLM cache hit")
                    return GenerateResult(**cached, cached=True)

            result = self._generate_with_retry(fam, prompt, tools, stream, chunk_callback)

            if CACHE is not None:
                CACHE.set(cache_key, result.to_dict(), expire=300)  # 5分TTL

            return result

    # ---------------- 実体（利用可能なバックエンドのみ試行） ----------------
    def _generate_with_retry(
        self,
        family: Literal["gemini", "openai", "deepseek", "local"],
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        chunk_callback: Optional[Callable[[str], None]],
    ) -> GenerateResult:

        def _has(fam: str) -> bool:
            return (
                (fam == "gemini"  and self.gemini_model is not None) or
                (fam == "openai"  and self.openai_client is not None) or
                (fam == "deepseek" and self.deepseek_client is not None) or
                (fam == "local"   and self.local_llama is not None)
            )

        # family を先頭に、他候補を追加しつつ未設定は除外
        pref = [family, "openai", "deepseek", "local"]
        candidates = []
        for fam in pref:
            if fam not in candidates and _has(fam):
                candidates.append(fam)

        if not candidates:
            raise RuntimeError(
                "No LLM backend is configured. "
                "Set OPENAI_API_KEY or GEMINI_API_KEY / DEEPSEEK_API_KEY, "
                "or configure LOCAL_LLAMA_PATH."
            )

        last_err: Optional[Exception] = None
        for attempt, fam in enumerate(candidates, 1):
            try:
                logger.info(f"[LLM] Attempt {attempt} with {fam}")
                if stream and fam in ("openai",):
                    text, usage = self._stream_call(fam, prompt, chunk_callback)
                else:
                    text, usage = self._raw_call(fam, prompt, tools)
                cost = self._calc_cost(fam, usage)
                return GenerateResult(
                    text=text, tokens=usage, cost_usd=cost, model_family=fam, sentiment=self._sentiment(text)
                )
            except Exception as e:
                last_err = e
                logger.warning(f"[LLM] {fam} failed: {e}")
                time.sleep(min(6, 1.5 ** attempt))

        raise RuntimeError(f"All LLM backends failed. last_error={last_err}")

    # ---------------- 個別バックエンド呼び出し ----------------
    def _raw_call(self, fam: str, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> tuple[str, Dict[str, int]]:
        if fam == "gemini" and self.gemini_model:
            res = self.gemini_model.generate_content(prompt)
            text = (getattr(res, "text", None) or "").strip()
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage

        if fam == "openai" and self.openai_client:
            model = self.cfg.get("OPENAI_MODEL", "gpt-4o-mini")
            cmpl = self.openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], tools=tools or []
            )
            text = (cmpl.choices[0].message.content or "").strip()
            usage = cmpl.usage.model_dump() if getattr(cmpl, "usage", None) else {}
            return text, usage

        if fam == "deepseek" and self.deepseek_client:
            model = self.cfg.get("DEEPSEEK_MODEL", "deepseek-chat")
            cmpl = self.deepseek_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], tools=tools or []
            )
            text = (cmpl.choices[0].message.content or "").strip()
            usage = cmpl.usage.model_dump() if getattr(cmpl, "usage", None) else {}
            return text, usage

        if fam == "local" and self.local_llama:
            out = self.local_llama(prompt, max_tokens=512, temperature=0.2)
            text = out["choices"][0]["text"].strip()
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage

        raise RuntimeError(f"No client configured for '{fam}'")

    def _stream_call(self, fam: str, prompt: str, chunk_callback: Optional[Callable[[str], None]]) -> tuple[str, Dict[str, int]]:
        if fam == "openai" and self.openai_client:
            model = self.cfg.get("OPENAI_MODEL", "gpt-4o-mini")
            text = ""
            for chunk in self.openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], stream=True
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    text += delta
                    if chunk_callback:
                        chunk_callback(delta)
            usage = {"input": len(prompt) // 4, "output": len(text) // 4}
            return text, usage

        raise RuntimeError(f"Streaming not supported for backend '{fam}'")

    # ---------------- 補助 ----------------
    def _cache_key(self, fam: str, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> str:
        content = json.dumps({"fam": fam, "prompt": prompt, "tools": tools}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _calc_cost(self, fam: str, usage: Dict[str, int]) -> float:
        inp = usage.get("prompt_tokens", usage.get("input", 0))
        out = usage.get("completion_tokens", usage.get("output", 0))
        c = MODEL_COST.get(fam, {"input": 0, "output": 0})
        return float(inp) * c["input"] + float(out) * c["output"]

    def _sentiment(self, text: str) -> Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        # 依存を増やさず簡易に。極端に短い/長い場合は NEUTRAL
        if not text or len(text) < 8:
            return "NEUTRAL"
        neg_cues = ("error", "failed", "cannot", "unable", "timeout", "invalid", "not found")
        pos = sum(w in text.lower() for w in ("success", "fix", "resolved", "passes"))
        neg = sum(w in text.lower() for w in neg_cues)
        if pos > neg and pos >= 1:
            return "POSITIVE"
        if neg > pos and neg >= 1:
            return "NEGATIVE"
        return "NEUTRAL"

    # おまけ
    @classmethod
    def quick(cls, prompt: str, task: str = "default") -> str:
        return cls().generate(prompt, task_type=task).text
