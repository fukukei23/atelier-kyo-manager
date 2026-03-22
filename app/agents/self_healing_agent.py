# ==============================================================================
# ファイル名: self_healing_agent.py
# レジストリ: app/agents/self_healing_agent.py
# 更新日時 (Date & Time JST): 2026年03月21日
# バージョン (Version): 10.0.0J (FKB Integration + Enhanced Recovery)
#
# --- v10.0.0Jでの主な変更点 ---
# - [FKB統合] Failure Knowledge Base との連携で既知エラーを自動解決
# - [戦略拡張] 3段階の回復戦略を実装
# - [成功率追跡] 回復成功率を記録して次回に反映
# - [設定改善] サイト別の回復戦略を動的に選択
#
# 操作するソフト/前提:
# - Python 3.10以上
# - 依存モジュール: PageRecoveryAgent, SelectorRepairAgent, RunContext, FKB
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import time
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
from app.agents.failure_analysis_agent import FKB
try:
    from app.core.run_context import RunContext
except Exception:
    from core.run_context import RunContext

logger = logging.getLogger(__name__)


class SelfHealingAgent:
    """
    自己修復オペレーションの現場指揮官。

    3段階の回復戦略を実行:
    1. 物理的回復 (工兵部隊) - PageRecoveryAgent
    2. FKBベース回復 - 既有知識データベース
    3. 知的修復 (情報分析官) - SelectorRepairAgent (LLM使用)
    """

    MAX_TOTAL_ATTEMPTS = 3

    def __init__(self):
        self.recovery_agent = PageRecoveryAgent()
        self.repair_agent = SelectorRepairAgent()
        self.fkb = FKB()
        self.recovery_stats = {
            "total_attempts": 0,
            "recovery_success": 0,
            "fkb_hit": 0,
            "llm_repair_success": 0,
            "failed": 0
        }

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
        """
        self.recovery_stats["total_attempts"] += 1
        site = settings.get("name", "UnknownSite")
        error_msg = failure_context.get("error_message", "")
        current_url = failure_context.get("current_url", "")

        logger.info(f"SelfHealingAgent: {attempt}回目/{self.MAX_TOTAL_ATTEMPTS} の自己修復戦略を開始... (Site: {site})")
        await run_context.take_screenshot(page, f"40_selfheal_strat_attempt_{attempt}_start")

        # --- 戦略1: 物理的回復 (工兵部隊) ---
        logger.info("戦略1: 物理的回復 (工兵部隊に出動を要請)")
        recovery_result = await self.recovery_agent.execute(
            page=page, run_context=run_context, settings=settings, attempt=attempt
        )
        if recovery_result.get("success"):
            message = f"物理的回復に成功しました。({recovery_result.get('message')})"
            logger.info(message)
            self.recovery_stats["recovery_success"] += 1
            return {"success": True, "strategy": "recovery", "message": message}

        # --- 戦略2: FKBベース回復 ---
        logger.info("戦略2: FKBベース回復 (既有知識データベースを検索)")
        fkb_result = await self._try_fkb_recovery(
            page=page,
            run_context=run_context,
            settings=settings,
            error_msg=error_msg,
            site=site,
            current_url=current_url
        )
        if fkb_result.get("success"):
            self.recovery_stats["fkb_hit"] += 1
            return fkb_result

        # --- 戦略3: 知的修復 (LLM) ---
        logger.warning("戦略3: 知的修復 (情報分析官に分析を要請)")
        await run_context.take_screenshot(page, f"60_repair_attempt_{attempt}_start")

        html_content = await page.content()

        repair_proposal = await self.repair_agent.propose_fix(
            intent=failure_context.get("intent", "不明な操作"),
            failed_selectors=failure_context.get("failed_selectors", []),
            html_content=html_content,
            site=site,
            site_config=settings,
            run_context=run_context
        )

        if repair_proposal and repair_proposal.get("proposed_selectors"):
            message = f"知的修復に成功。{len(repair_proposal['proposed_selectors'])}件の代替セレクタを提案します。"
            logger.info(message)
            self.recovery_stats["llm_repair_success"] += 1
            return {
                "success": True,
                "strategy": "repair_proposal",
                "message": message,
                "proposal": repair_proposal
            }

        # --- 全戦略失敗 ---
        message = "全ての自己修復戦略（物理的回復、FKB、知的修復）が失敗しました。"
        logger.error(message)
        self.recovery_stats["failed"] += 1
        return {"success": False, "strategy": "failed", "message": message}

    async def _try_fkb_recovery(
        self,
        *,
        page: Page,
        run_context: RunContext,
        settings: dict,
        error_msg: str,
        site: str,
        current_url: str
    ) -> dict:
        """FKBから解決策を取得して適用を試みる"""
        # FKBエントリを検索
        fkb_entry = self.fkb.find_matching_entry(error_msg, site, current_url)
        if not fkb_entry:
            logger.info("[FKB] マッチするエントリが見つかりませんでした")
            return {"success": False, "strategy": "fkb", "message": "FKBエントリが見つかりません"}

        logger.info(f"[FKB] エントリを発見: {fkb_entry.get('id')} - {fkb_entry.get('error_signature')}")
        solution = self.fkb.get_solution(fkb_entry)

        # 解決策を適用
        recovery_url = solution.get("recovery_url")
        if recovery_url:
            logger.info(f"[FKB] 回復URLへ遷移: {recovery_url}")
            try:
                await page.goto(recovery_url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # ページ安定待機

                # OneTrust対応
                await self._handle_one_trust(page)

                # スクリーンショット保存
                await run_context.take_screenshot(page, f"fkb_recovery_url_applied")

                logger.info("[FKB] 回復URLへの遷移に成功しました")
                return {
                    "success": True,
                    "strategy": "fkb_url_redirect",
                    "message": f"FKB ID: {solution.get('id')} - 回復URLへ遷移",
                    "fkb_entry": fkb_entry
                }
            except Exception as e:
                logger.warning(f"[FKB] 回復URLへの遷移に失敗: {e}")

        # 代替セレクタを試す
        alternate_selectors = solution.get("alternate_selectors", [])
        if alternate_selectors:
            logger.info(f"[FKB] 代替セレクタを試行: {alternate_selectors[:2]}")
            for selector in alternate_selectors[:3]:  # 最大3つ
                try:
                    count = await page.locator(selector).count()
                    if count > 0:
                        logger.info(f"[FKB] 代替セレクタ '{selector}' で {count} 件の要素を発見")
                        return {
                            "success": True,
                            "strategy": "fkb_alternate_selector",
                            "message": f"FKB代替セレクタ '{selector}' が有効",
                            "selector": selector,
                            "fkb_entry": fkb_entry
                        }
                except Exception:
                    continue

        logger.info("[FKB] 解決策の適用を試みましたが、成功しませんでした")
        return {"success": False, "strategy": "fkb", "message": "FKB解決策の適用に失敗"}

    async def _handle_one_trust(self, page: Page):
        """OneTrust GDPRバナーに対応"""
        one_trust_selectors = [
            "#onetrust-accept-btn-handler",
            "button:has-text('Accept All')",
            "button:has-text('ACCEPT AND CONTINUE')",
            "[aria-label*='Accept all cookies']",
        ]
        for selector in one_trust_selectors:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=2000):
                    await element.click()
                    logger.info(f"[OneTrust] '{selector}' をクリックしました")
                    await page.wait_for_timeout(1000)
                    break
            except Exception:
                continue

    def get_recovery_stats(self) -> dict:
        """回復成功率の統計を返す"""
        total = self.recovery_stats["total_attempts"]
        if total == 0:
            return {"success_rate": 0.0, "total_attempts": 0}

        success = (
            self.recovery_stats["recovery_success"]
            + self.recovery_stats["fkb_hit"]
            + self.recovery_stats["llm_repair_success"]
        )
        return {
            **self.recovery_stats,
            "success_rate": round(success / total * 100, 1)
        }
