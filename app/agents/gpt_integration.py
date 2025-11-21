# -*- coding: utf-8 -*-
# ==============================================================================
# ファイル名: app/agents/gpt_integration.py
# 日付 (JST): 2025-11-05 13:00:00
# バージョン: 1.2.2-proposal-save-integrated
#
# 目的:
# - v1.2.1 の機能（リトライ、サニタイズ、スキーマ検証、save_proposal）を維持。
# - [v1.2.2] ★統合: propose_fixes 成功時に save_proposal を自動呼び出し。
# - [v1.2.2] ★修正: save_proposal で使用する 're' モジュールをインポート。
# ==============================================================================
import json
import time
import logging
import os  # ★ NEW
import re  # ★ NEW (v1.2.2: save_proposal のために必須)
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SAFE_OUT_KEYS = {
    "selectors_patch", "overrides_patch", "url_rules", "code_hints", "rationale", "risk"
}

class GPTIntegration:
    """
    LLM呼び出しの強化版ファサード。
      - AiLlmController の設定主導（overrides -> proxy_agent.llm.generation_config）
      - 出力 JSON スキーマ検証（余計なキーは除去）
      - 失敗時の指数バックオフ・再試行
      - PII/DOM の過剰流出防止（文字数制限）
    """

    def __init__(self, site_config: Dict[str, Any], generation_config: Optional[Dict[str, Any]] = None):
        # 遅延import（依存を薄く）
        from app.utils.ai_llm_controller import AILlmController  # type: ignore
        self.ctrl = AILlmController()
        self.site_config = site_config or {}
        self.generation_config = (
            generation_config
            or (self.site_config.get("proxy_agent", {}).get("llm", {}).get("generation_config") or {})
        )
        self.last_saved_proposal_path: Optional[str] = None
        # 安全・回復パラメータ
        self.max_blob = int(self.generation_config.get("max_context_blob_chars", 12000))
        self.max_retry = int(self.generation_config.get("max_retry", 2))
        self.retry_backoff = float(self.generation_config.get("retry_backoff", 2.0))

    # ---------------------------------------------------------------------
    def propose_fixes(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        入力: 失敗URL・エラー・（あれば）DOM抜粋 等
        出力: { selectors_patch, overrides_patch, url_rules, code_hints, rationale, risk }
        """
        self.last_saved_proposal_path = None
        fc = self._sanitize_context(failure_context)
        prompt = self._build_prompt(fc)

        tries = 0
        last_err: Optional[str] = None
        while tries <= self.max_retry:
            tries += 1
            res = self.ctrl.generate(
                prompt=prompt,
                task="analysis",
                use_cache=False,
                generation_config=self.generation_config
            )
            if getattr(res, "success", False) and getattr(res, "text", None):
                ok, data_or_err = self._parse_and_validate(res.text)
                if ok:
                    # --- ★統合(v1.2.2): 成功した提案をJSONに保存 ---
                    # diff の意図を、既存の save_proposal メソッド呼び出しで実現
                    self.last_saved_proposal_path = self.save_proposal(data_or_err)
                    # ----------------------------------------------
                    return data_or_err  # type: ignore[return-value]
                raw_dump = self._dump_raw_response(res.text, prefix="invalid_json")
                last_err = f"JSON schema invalid: {data_or_err} (dump={raw_dump})"
            else:
                err_detail = getattr(res, "error", "LLM failed")
                raw_dump = None
                text = getattr(res, "text", None)
                if text:
                    raw_dump = self._dump_raw_response(text, prefix="llm_error")
                last_err = f"{err_detail} (dump={raw_dump})"

            if tries <= self.max_retry:
                sleep_s = self.retry_backoff ** (tries - 1)
                logger.warning(f"[GPTIntegration] retry {tries}/{self.max_retry} after error: {last_err} (sleep {sleep_s:.1f}s)")
                time.sleep(sleep_s)

        raise RuntimeError(last_err or "LLM failed")

    # 互換エントリポイント (FailureAnalysisAgent が期待するシグネチャ)
    def run(
        self,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        failure_context: Dict[str, Any],
        run_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        FailureAnalysisAgent から呼ばれる互換メソッド。
        propose_fixes を呼び出し、提案と保存パスを返す。
        """
        proposal = self.propose_fixes(failure_context)
        return {
            "proposal": proposal,
            "proposal_path": self.last_saved_proposal_path,
            "site": site,
            "query": query,
        }

    # --- ★追加(v1.2.1): 提案JSONの保存ユーティリティ ---
    def save_proposal(self, proposal: Dict[str, Any], base_dir: Optional[str] = None) -> Optional[str]:
        """
        LLM提案の辞書をタイムスタンプ付きJSONファイルとして保存する。
        """
        try:
            out_dir = base_dir or os.environ.get("PROXY_OUTPUT_DIR", "exports")
            out_dir = os.path.join(out_dir, "llm_proposals")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            # proposal 辞書から site_key を安全に取得
            site_key_guess = (
                proposal.get("site")
                or proposal.get("site_key")
                or self.site_config.get("site_key") # __init__ の self.site_config からも参照
                or "site"
            )
            site = re.sub(r"[^a-z0-9_-]", "", str(site_key_guess).lower()) # ファイル名用にサニタイズ

            fname = f"{site}_{ts}_proposal.json"
            path = os.path.join(out_dir, fname)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(proposal, f, ensure_ascii=False, indent=2)
            logger.info(f"[GPTIntegration] wrote proposal JSON: {path}")
            return path
        except Exception as e:
            logger.warning(f"[GPTIntegration] failed to write proposal JSON: {e}")
            return None

    # ---------------------------------------------------------------------
    # Helpers
    def _sanitize_context(self, fc: Dict[str, Any]) -> Dict[str, Any]:
        """DOMやクッキーの機微情報は切り詰める。"""
        out = dict(fc or {})
        for k in ("dom_snapshot", "dom_excerpt", "html", "cookies", "headers"):
            if k in out and isinstance(out[k], str):
                s = out[k]
                if len(s) > self.max_blob:
                    out[k] = s[: self.max_blob] + f"\n"
        return out

    def _parse_and_validate(self, text: str) -> Tuple[bool, Any]:
        raw = self._extract_json(text)
        try:
            data = json.loads(raw)
        except Exception as e:
            return False, f"json parse error: {e}"
        if not isinstance(data, dict):
            return False, "root must be object"

        # 余計なキーは除去し、型を整形
        out = {k: data.get(k) for k in SAFE_OUT_KEYS}
        out["selectors_patch"] = out.get("selectors_patch") or {}
        out["overrides_patch"] = out.get("overrides_patch") or {}
        out["url_rules"] = out.get("url_rules") or {}
        out["code_hints"] = out.get("code_hints") or []
        out["rationale"] = out.get("rationale") or ""
        out["risk"] = (out.get("risk") or "low").lower()

        if not isinstance(out["selectors_patch"], dict): out["selectors_patch"] = {}
        if not isinstance(out["overrides_patch"], dict): out["overrides_patch"] = {}
        if not isinstance(out["url_rules"], dict): out["url_rules"] = {}
        if not isinstance(out["code_hints"], list): out["code_hints"] = []
        if not isinstance(out["rationale"], str): out["rationale"] = ""
        if out["risk"] not in ("low", "medium", "high"): out["risk"] = "low"

        return True, out

    def _build_prompt(self, fc: Dict[str, Any]) -> str:
        site = self.site_config.get("site_key") or self.site_config.get("name") or "TARGET"
        guidance = {
            "instruction": "あなたはECサイトのPLPスクレイピングを『安全に』自己進化させるエージェントです。",
            "rules": [
                "1) まず設定ファイル(overrides)で対処可能な案を最優先に出すこと。",
                "2) コード改修は『提案(diff)のみ』。自動適用は前提としない（人間承認）。",
                "3) 出力は JSON で、次のキーのみ: selectors_patch, overrides_patch, url_rules, code_hints, rationale, risk",
                "4) selectors_patch/overrides_patch は最小差分のみ。既存キーは壊さない。",
                "5) URLトラップ（/en-jp 等）や monclergroup.com への誤遷移を避ける正規化ルールを歓迎。",
                "6) リスク評価(risk: low/medium/high)を必ず含める。",
            ]
        }
        return json.dumps({
            "site": site,
            "failure_context": fc,
            "site_overrides": self.site_config,
            "guidance": guidance
        }, ensure_ascii=False, indent=2)

    def _extract_json(self, text: str) -> str:
        l = text.find("{")
        r = text.rfind("}")
        if l >= 0 and r >= 0 and r > l:
            return text[l:r+1]
        return "{}"

    def _dump_raw_response(self, text: str, prefix: str = "llm_raw") -> Optional[str]:
        try:
            base_dir = os.environ.get("PROXY_OUTPUT_DIR", "exports")
            out_dir = os.path.join(base_dir, "llm_proposals", "raw")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"{prefix}_{ts}.txt"
            path = os.path.join(out_dir, name)
            snippet = text if len(text) <= self.max_blob else text[: self.max_blob] + "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(snippet)
            logger.debug(f"[GPTIntegration] dumped raw LLM output -> {path}")
            return path
        except Exception as e:
            logger.warning(f"[GPTIntegration] failed to dump raw LLM output: {e}")
            return None
