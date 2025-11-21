# ==============================================================================
# ファイル名 (File Name): selector_repair_agent.py
# レジストリ (Registry): app/agents/selector_repair_agent.py
# 更新日時 (Date & Time JST): 2025-10-26 10:59
# バージョン (Version): 9.0.2J (堅牢性強化パッチ適用・構文修正済)
#
# 変更点:
# - AILlmController(本番) を採用 + run_in_executor による非同期安全呼び出し
# - JSONガード (Markdownフェンス/前置説明の除去) を実装
# - LLM呼び出しにタイムアウト (45s) + リトライ (1回)
# - 提案の正規化 (重複/空除去・最大3件)
# - モデル/コスト観測ログを追加
# ==============================================================================

from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from app.core.run_context import RunContext
from app.utils.ai_llm_controller import AILlmController

logger = logging.getLogger(__name__)

# ---------------------- ヘルパー（堅牢化） ----------------------

# 末尾側に存在する JSON オブジェクト（{ ... }）を大まかに拾う
_JSON_BLOCK_RX = re.compile(r"\{[\s\S]*\}$")

def _extract_json_str(text: str) -> str:
    """
    LLM が ```json フェンスや説明文を混ぜても、末尾の JSON ブロックだけを抽出する。
    """
    t = (text or "").strip()

    # ```json / ``` を剥がす（先頭・末尾とも）
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # 末尾の { ... } を強引に拾う（余計な前置文があってもOK）
    m = _JSON_BLOCK_RX.search(t)
    return m.group(0) if m else t

def _normalize_proposal(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    proposed_selectors を正規化:
      - list[str] 化
      - 空文字・重複除去
      - 上限3件
    rationale は str 以外なら空文字に。
    """
    sels = p.get("proposed_selectors", [])
    if isinstance(sels, str):
        sels = [sels]
    elif not isinstance(sels, list):
        sels = []

    seen: set[str] = set()
    norm: List[str] = []
    for s in sels:
        s = (s or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        norm.append(s)
        if len(norm) >= 3:
            break

    p["proposed_selectors"] = norm
    if not isinstance(p.get("rationale"), str):
        p["rationale"] = ""
    return p

# ---------------------- 本体 ----------------------

class SelectorRepairAgent:
    """壊れたセレクタをAIで修復・提案する本番版エージェント。"""

    def __init__(self):
        # AILlmController はシングルトン
        self.llm_controller = AILlmController()

    async def propose_fix(
        self,
        *,
        intent: str,
        failed_selectors: List[str],
        html_content: str,
        site: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
    ) -> Dict[str, Any]:
        """
        Returns:
            {"proposed_selectors": [...], "rationale": "..."} or {}
        """
        prompt = self._build_prompt(
            intent=intent,
            failed_selectors=failed_selectors,
            html_snippet=html_content,
            site=site,
            run_context=run_context,
        )

        logger.info(
            "[SelectorRepair] AIへ修復提案を要請 (RunID=%s, site=%s)",
            run_context.run_id, site
        )
        logger.debug("[SelectorRepair] Prompt (first 600 chars): %s...", prompt[:600])

        try:
            loop = asyncio.get_running_loop()

            async def _call_llm():
                # generate() は同期API → スレッド実行で非ブロッキング化
                return await loop.run_in_executor(
                    None, lambda: self.llm_controller.generate(prompt, task_type="code")
                )

            # タイムアウト 45s、失敗時 1 リトライ
            last_err: Exception | None = None
            for attempt in range(2):
                try:
                    result = await asyncio.wait_for(_call_llm(), timeout=45)
                    break
                except asyncio.TimeoutError as e:
                    last_err = e
                    if attempt == 0:
                        logger.warning("[SelectorRepair] LLM timeout -> retrying once...")
                        continue
                    raise
            else:
                if last_err:
                    raise last_err  # safety

            # 観測ログ（モデル・コスト）
            try:
                logger.info(
                    "[SelectorRepair] model=%s cost_usd=%.6f cached=%s",
                    getattr(result, "model_family", "?"),
                    float(getattr(result, "cost_usd", 0.0) or 0.0),
                    getattr(result, "cached", False),
                )
            except Exception:
                pass

            text = (getattr(result, "text", "") or "").strip()
            logger.debug("[SelectorRepair] LLM Raw Response (first 600 chars): %s...", text[:600])

            # JSONガード
            safe = _extract_json_str(text)
            proposal: Dict[str, Any] = json.loads(safe) if safe else {}

            # 正規化
            proposal = _normalize_proposal(proposal)

            # 保存
            try:
                run_context.save_json("selector_repair_proposal.json", proposal)
            except Exception as e:
                logger.warning("[SelectorRepair] 提案JSONの保存に失敗: %s (RunID=%s)", e, run_context.run_id)

            if not proposal.get("proposed_selectors"):
                logger.warning("[SelectorRepair] proposed_selectors が空。空提案として扱います。 (RunID=%s)", run_context.run_id)
                return {}

            logger.info(
                "[SelectorRepair] 提案を受領: %d件 (RunID=%s)",
                len(proposal["proposed_selectors"]), run_context.run_id
            )
            return proposal

        except json.JSONDecodeError as e:
            logger.error(
                "[SelectorRepair] LLM応答のJSONパースに失敗: %s. Response(head): %s... (RunID=%s)",
                e, text[:200] if 'text' in locals() else "", run_context.run_id
            )
            return {}
        except Exception as e:
            logger.error(
                "[SelectorRepair] LLM呼び出し/処理中に致命的なエラー: %s (RunID=%s)",
                e, run_context.run_id, exc_info=True
            )
            return {}

    def _build_prompt(
        self,
        *,
        intent: str,
        failed_selectors: List[str],
        html_snippet: str,
        site: str,
        run_context: RunContext,
    ) -> str:
        """AI に渡すプロンプトを構築。"""
        truncated_html = (html_snippet or "")[:8000]

        return f"""
# Role
あなたは、Web自動化とフロントエンド開発を専門とする世界トップクラスのAIエンジニアです。
# Goal
ウェブサイト「{site}」の自動化タスクが、古いCSSセレクタが原因で失敗しました。提供HTMLに基づき、失敗したタスクの「意図」を達成するための、より堅牢なCSSセレクタを提案してください。
# Context
- Website: {site}
- Intent: {intent}
- Previously Failed Selectors: {json.dumps(failed_selectors, ensure_ascii=False)}
- Run ID: {run_context.run_id}
# Evidence
- HTML Snippet:
```html
{truncated_html}
