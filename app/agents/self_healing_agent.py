# ==============================================================================
# ファイル名 (File Name): self_healing_agent.py
# レジストリ (Registry): app/agents/self_healing_agent.py
# 更新日時 (Date & Time JST): 2025年09月21日 22:01:00
# バージョン (Version): 9.0.0J (Final Consolidated)
#
# --- 操作するソフト/前提 (Software & Prerequisites) ---
# - Python 3.10以上
# - 依存モジュール: PageRecoveryAgent, SelectorRepairAgent, RunContext
#
# --- 使用方法 (Usage) ---
# このエージェントは、SelectorDiscoveryAgentのような斥候部隊から、
# データ抽出の失敗時に呼び出されます。物理的回復と知的修復の
# 2段階の戦略を統括します。
#
# --- v9.0.0Jでの主な変更点 ---
# - [最終統合] これまでの対話で指摘された全てのバグ修正
#   （コンストラクタ引数の不整合、非同期`await`漏れ）を統合した最終版です。
# - [非同期安全] `SelectorRepairAgent`の非同期メソッド`propose_fix`を
#   `await`で正しく呼び出すように修正し、非同期処理の安全性を完全に保証します。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import sys
from pathlib import Path

# --- プロジェクトルートをPythonの検索パスに追加 ---
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from playwright.async_api import Page
from app.agents.page_recovery_agent import PageRecoveryAgent
from app.agents.selector_repair_agent import SelectorRepairAgent
from core.run_context import RunContext

logger = logging.getLogger(__name__)

class SelfHealingAgent:
    """自己修復オペレーションの現場指揮官。"""
    def __init__(self):
        self.recovery_agent = PageRecoveryAgent()
        self.repair_agent = SelectorRepairAgent()

    async def execute(self, *, page: Page, run_context: RunContext, settings: dict, attempt: int, failure_context: Dict[str, Any]) -> dict:
        """
        自己修復の戦略的意思決定と実行を統括する。
        """
        logging.info(f"SelfHealingAgent: {attempt}回目の自己修復戦略を開始...")
        await run_context.take_screenshot(page, f"40_selfheal_strat_attempt_{attempt}_start")

        # 戦略1: 物理的回復を試みる
        logging.info("戦略1: 物理的回復 (工兵部隊に出動を要請)")
        recovery_result = await self.recovery_agent.execute(
            page=page, run_context=run_context, settings=settings, attempt=attempt
        )
        if recovery_result.get("success"):
            message = f"物理的回復に成功しました。({recovery_result.get('message')})"
            logging.info(message)
            return {"success": True, "strategy": "recovery", "message": message}

        # 戦略2: 物理的回復が失敗した場合、知的修復を検討
        logging.warning("物理的回復に失敗。戦略2: 知的修復 (情報分析官に分析を要請)")
        await run_context.take_screenshot(page, f"60_repair_attempt_{attempt}_start")

        html_content = await page.content()

        # SelectorRepairAgent.propose_fixは非同期(async)メソッドなのでawaitで呼び出す
        repair_proposal = await self.repair_agent.propose_fix(
            intent=failure_context.get("intent", "不明な操作"),
            failed_selectors=failure_context.get("failed_selectors", []),
            html_content=html_content,
            site=settings.get("name", "UnknownSite"),
            site_config=settings,
            run_context=run_context
        )

        if repair_proposal and repair_proposal.get("proposed_selectors"):
            message = f"知的修復に成功。{len(repair_proposal['proposed_selectors'])}件の代替セレクタを提案します。"
            logging.info(message)
            # 提案の保存はrepair_agent自身が行う
            return {"success": True, "strategy": "repair_proposal", "message": message, "proposal": repair_proposal}

        message = "全ての自己修復戦略（物理的回復、知的修復）が失敗しました。"
        logging.error(message)
        return {"success": False, "strategy": "failed", "message": message}
