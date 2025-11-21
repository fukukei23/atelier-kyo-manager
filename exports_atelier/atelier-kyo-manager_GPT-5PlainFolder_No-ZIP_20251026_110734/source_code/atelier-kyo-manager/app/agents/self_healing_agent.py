# ==============================================================================
# ファイル名 (File Name): app/agents/self_healing_agent.py
# レジストリ (Registry): app/agents/self_healing_agent.py
# 更新日時 (Date & Time JST): 2025-10-02 01:48:00
# バージョン (Version): 10.2.0J (Recovery→Repair, Dedup, Evidence)
#
# --- 本版での主なポイント ---
# - [API整合] execute(page, run_context, settings, attempt, failure_context) を厳密化。
# - [二段構え] 回復(PageRecoveryAgent)→修復(SelectorRepairAgent)の順で自己修復。
# - [安全な重複排除] 文字列/辞書混在の候補を _hashable_key でユニーク化。
# - [証跡] 開始時/修復開始時のスクショを RunContext に保存。提案は selector_repair_proposal.json にも保存。
# - [互換] 呼び出し側（SelectorDiscoveryAgent, BrowserUseAgent）が期待する
#          軽量 proposal 形式 {"proposed_selectors":[...]} を返却。
#
# --- 動作要件 ---
# - Python 3.10+ / Playwright (async)
# - 参照: app.core.run_context.RunContext
#         app.agents.page_recovery_agent.PageRecoveryAgent
#         app.agents.selector_repair_agent.SelectorRepairAgent
# ==============================================================================

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set, Union
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
try:
    APP_ROOT = Path(__file__).resolve().parents[2]
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
except NameError:
    if Path.cwd() not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

from playwright.async_api import Page
from app.agents.page_recovery_agent import PageRecoveryAgent
from app.agents.selector_repair_agent import SelectorRepairAgent
from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


def _hashable_key(sel: Any) -> Any:
    """
    セレクタ候補をハッシュ可能なキーに変換。
    - 文字列はそのまま
    - 辞書は (key, value) を key でソートしたタプルへ
    - それ以外は repr にフォールバック（念のため）
    """
    if isinstance(sel, dict):
        return tuple(sorted(sel.items()))
    try:
        hash(sel)  # 例外が出なければそのまま使える
        return sel
    except Exception:
        return repr(sel)


class SelfHealingAgent:
    """自己修復オペレーションの現場指揮官。回復→修復の二段構えで再起を図る。"""

    def __init__(self):
        self.recovery_agent = PageRecoveryAgent()
        self.repair_agent = SelectorRepairAgent()

    async def execute(
        self,
        *,
        page: Page,
        run_context: RunContext,
        settings: dict,
        attempt: int,
        failure_context: Dict[str, Any]
    ) -> dict:
        """
        自己修復の戦略的意思決定と実行を統括する。

        Parameters
        ----------
        page : Page
            Playwright page
        run_context : RunContext
            証跡保存・パス付与などのコンテキスト
        settings : dict
            サイト設定（overridesを含む可能性）
        attempt : int
            何回目の自己修復か（1から）
        failure_context : Dict[str, Any]
            失敗状況（intent, failed_selectors 等）

        Returns
        -------
        dict
            {"success": bool, "strategy": "recovery"|"repair_proposal"|"failed",
             "message": str, "proposal": {"proposed_selectors":[...]}}
        """
        logging.info(f"[SelfHealing] attempt={attempt} 開始")
        try:
            await run_context.take_screenshot(page, f"40_selfheal_strat_attempt_{attempt}_start")
        except Exception:
            pass

        # --- 戦略1: 物理的回復（PageRecoveryAgent） ---
        logging.info("[SelfHealing] Strategy#1: Physical recovery (PageRecoveryAgent)")
        try:
            recovery_result = await self.recovery_agent.execute(
                page=page, run_context=run_context, settings=settings, attempt=attempt
            )
        except Exception as e:
            logger.warning(f"[SelfHealing] Recovery agent raised error: {e}")
            recovery_result = {"success": False, "message": str(e)}

        if recovery_result.get("success"):
            msg = f"物理的回復に成功: {recovery_result.get('message')}"
            logging.info(f"[SelfHealing] {msg}")
            return {"success": True, "strategy": "recovery", "message": msg}

        logging.warning("[SelfHealing] Recovery failed. Strategy#2: Intelligent repair (SelectorRepairAgent)")
        try:
            await run_context.take_screenshot(page, f"60_repair_attempt_{attempt}_start")
        except Exception:
            pass

        # --- 戦略2: 知的修復（SelectorRepairAgent） ---
        try:
            html_content = await page.content()
        except Exception:
            html_content = ""

        intent = failure_context.get("intent", "不明な操作")
        failed_selectors = failure_context.get("failed_selectors", [])

        try:
            repair_proposal = await self.repair_agent.propose_fix(
                intent=intent,
                failed_selectors=failed_selectors,
                html_content=html_content,
                site=settings.get("name", "UnknownSite"),
                site_config=settings,
                run_context=run_context
            )
        except Exception as e:
            logger.error(f"[SelfHealing] SelectorRepairAgent.propose_fix failed: {e}", exc_info=True)
            repair_proposal = None

        # 提案が無い/空なら失敗扱い
        if not (repair_proposal and repair_proposal.get("proposed_selectors")):
            msg = "知的修復に失敗（提案なし）"
            logging.error(f"[SelfHealing] {msg}")
            return {"success": False, "strategy": "failed", "message": msg}

        # --- 既存×新規候補の重複排除（文字列/辞書混在安全） ---
        tried: Set[Any] = set()
        for sel in (failed_selectors or []):
            tried.add(_hashable_key(sel))

        unique_candidates: List[Any] = []
        seen = set(tried)
        for cand in repair_proposal.get("proposed_selectors", []):
            key = _hashable_key(cand)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(cand)

        if not unique_candidates:
            msg = "知的修復は実行されたが、新しい候補は得られなかった（全て既知）"
            logging.warning(f"[SelfHealing] {msg}")
            return {"success": False, "strategy": "failed", "message": msg}

        # 返却用 proposal（軽量）に整形
        proposal_out = {"proposed_selectors": unique_candidates}
        msg = f"知的修復に成功。新しい候補を {len(unique_candidates)} 件提案。"
        logging.info(f"[SelfHealing] {msg} -> {unique_candidates}")

        # 証跡として JSON も書き出しておく（任意、後段の overrides_store で利用可）
        try:
            run_context.save_json("selector_repair_proposal.json", {
                "proposed_selectors": unique_candidates,
                "intent": intent,
                "attempt": attempt
            })
        except Exception:
            pass

        return {
            "success": True,
            "strategy": "repair_proposal",
            "message": msg,
            "proposal": proposal_out
        }
