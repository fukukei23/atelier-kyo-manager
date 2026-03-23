# ==============================================================================
# ファイル名 (File Name): self_healing_agent.py
# レジストリ (Registry): app/agents/self_healing_agent.py
# 更新日時 (Date & Time JST): 2026年03月23日
# バージョン (Version): 10.1.0J (Circuit Breaker + FKB Integration)
#
# --- v10.1.0Jでの主な変更点 ---
# - [Circuit Breaker] 連続失敗時に自動停止してリソース浪費を防止
# - [サイト別CB] サイトごとに独立したCBカウンターで特定のサイトに集中対策
# - [クールダウン] 一定時間後にCBをリセットして回復後の復帰を許可
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

    Circuit Breaker:
    - サイトごとに連続失敗回数を追跡
    - 閾値(5回)超過でサーキットオープン（自己修復をスキップ）
    - クールダウン(5分)後に полуоткрыто 状態で復帰を試行
    """

    MAX_TOTAL_ATTEMPTS = 3
    CB_THRESHOLD = 5
    CB_COOLDOWN_SEC = 300

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
        # Circuit Breaker: サイト別連続失敗カウンター
        self._cb_failures: Dict[str, int] = {}
        self._cb_opened_at: Dict[str, float] = {}

    async def execute(self, *, page: Page, run_context: RunContext, settings: dict, attempt: int, failure_context: Dict[str, Any]) -> dict:
        """
        自己修復の戦略的意思決定と実行を統括する。
        Circuit Breaker: サイト別連続失敗が閾値を超えると自己修復をスキップして失敗を返す。
        """
        self.recovery_stats["total_attempts"] += 1
        site = settings.get("name", "UnknownSite")
        error_msg = failure_context.get("error_message", "")
        current_url = failure_context.get("current_url", "")

        # Circuit Breaker チェック
        cb_status = self._check_circuit_breaker(site)
        if cb_status == "open":
            logger.warning(f"[CircuitBreaker] {site} のCBがオープン状態: 自己修復をスキップ")
            self._record_failure(site)
            return {
                "success": False,
                "strategy": "circuit_breaker_open",
                "message": f"CircuitBreaker open for {site} (failures={self._cb_failures.get(site, 0)}). Wait {self.CB_COOLDOWN_SEC}s.",
                "cb_status": "open"
            }

        logging.info(f"SelfHealingAgent: {attempt}回目/{self.MAX_TOTAL_ATTEMPTS} の自己修復戦略を開始... (Site: {site})")
        await run_context.take_screenshot(page, f"40_selfheal_strat_attempt_{attempt}_start")

        # 戦略1: 物理的回復を試みる
        logging.info("戦略1: 物理的回復 (工兵部隊に出動を要請)")
        recovery_result = await self.recovery_agent.execute(
            page=page, run_context=run_context, settings=settings, attempt=attempt
        )
        if recovery_result.get("success"):
            message = f"物理的回復に成功しました。({recovery_result.get('message')})"
            logging.info(message)
            self._record_success(site)
            return {"success": True, "strategy": "recovery", "message": message}

        # 戦略2: FKBベース回復
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
            self._record_success(site)
            return fkb_result

        # 戦略3: 知的修復
        logging.warning("戦略3: 知的修復 (情報分析官に分析を要請)")
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
            self._record_success(site)
            return {"success": True, "strategy": "repair_proposal", "message": message, "proposal": repair_proposal}

        message = "全ての自己修復戦略（物理的回復、FKB、知的修復）が失敗しました。"
        logging.error(message)
        self._record_failure(site)
        return {"success": False, "strategy": "failed", "message": message}
    
    async def handle_moncler_failure(self, failure_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        CR-ATELIER-002 Step 6-4: Moncler 専用の失敗ハンドリング
        
        Args:
            failure_payload: 失敗情報を含む辞書
                - site: str
                - url: str
                - failure_reason: str ("raw_zero", "rejected_all", "secondary_or_tertiary_used", "trap_detected", "locale_corrections_exceeded")
                - dom_snapshot_path: Optional[str]
                - layer_stats: Dict[str, Any]
                - rejection_stats: Dict[str, Any]
                - selectors_current: Dict[str, Any]
                - run_id: Optional[str]
                - timestamp: Optional[str]
        
        Returns:
            Dict[str, Any]: 分析結果
                - analysis: str
                - root_cause: str
                - suggested_actions: List[str]
                - confidence: float
        """
        logger.info(f"[SelfHealing][Moncler] Handling failure: reason={failure_payload.get('failure_reason')}")
        
        failure_reason = failure_payload.get("failure_reason", "unknown")
        layer_stats = failure_payload.get("layer_stats", {})
        rejection_stats = failure_payload.get("rejection_stats", {})
        
        # 失敗理由に基づいて分析
        analysis = ""
        root_cause = ""
        suggested_actions = []
        confidence = 0.0
        
        if failure_reason == "raw_zero":
            analysis = "セレクタが要素を見つけられていません。DOM構造が変化した可能性があります。"
            root_cause = "primary selector mismatch"
            suggested_actions = [
                "MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY に新しいセレクタを追加",
                "Secondary layer のセレクタを確認",
                "DOM スナップショットを分析してセレクタを再設計"
            ]
            confidence = 0.85
        elif failure_reason == "rejected_all":
            analysis = "セレクタは要素を見つけていますが、すべてのURLがバリデーションで拒否されています。"
            root_cause = "url validation too strict"
            suggested_actions = [
                "URLバリデーションルールを確認",
                "rejection_stats を分析して拒否理由を特定",
                "必要に応じてバリデーションルールを緩和"
            ]
            confidence = 0.75
        elif failure_reason == "secondary_or_tertiary_used":
            analysis = "Primary layer が失敗し、Secondary または Tertiary layer にフォールバックしました。"
            root_cause = "primary selector ineffective"
            suggested_actions = [
                "Primary layer のセレクタを更新",
                "Secondary layer のセレクタを Primary に昇格",
                "DOM スナップショットを分析して最適なセレクタを特定"
            ]
            confidence = 0.80
        elif failure_reason == "trap_detected":
            analysis = "Trap ページが検出されました。ロケール制御またはリダイレクトの問題の可能性があります。"
            root_cause = "trap page navigation"
            suggested_actions = [
                "LocaleGuard の動作を確認",
                "リダイレクト挙動を分析",
                "Trap ページの DOM スナップショットを確認"
            ]
            confidence = 0.90
        elif failure_reason == "locale_corrections_exceeded":
            analysis = "Locale補正が繰り返し発生しています。サーバ側のリダイレクトロジックが干渉している可能性があります。"
            root_cause = "locale redirect loop"
            suggested_actions = [
                "Cookie やセッション情報を確認",
                "リダイレクト挙動を分析",
                "LocaleGuard の再試行ロジックを調整"
            ]
            confidence = 0.70
        else:
            analysis = "不明な失敗理由です。詳細な分析が必要です。"
            root_cause = "unknown"
            suggested_actions = [
                "DOM スナップショットを分析",
                "ログを確認",
                "Telemetry データを確認"
            ]
            confidence = 0.50
        
        result = {
            "analysis": analysis,
            "root_cause": root_cause,
            "suggested_actions": suggested_actions,
            "confidence": confidence,
        }
        
        logger.info(
            f"[SelfHealing][Moncler] Analysis complete: root_cause={root_cause}, "
            f"confidence={confidence}, actions={len(suggested_actions)}"
        )
        
        # 提案をログに保存（将来の site_config パッチ作成に使用）
        # 実際の site_config 書き換えは Step7 以降の責務
        logger.info(
            f"[SelfHealing][Moncler] Suggested actions: {suggested_actions}"
        )

        return result

    # --- FKB ベース回復 ---

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
        fkb_entry = self.fkb.find_matching_entry(error_msg, site, current_url)
        if not fkb_entry:
            logger.info("[FKB] マッチするエントリが見つかりませんでした")
            return {"success": False, "strategy": "fkb", "message": "FKBエントリが見つかりません"}

        logger.info(f"[FKB] エントリを発見: {fkb_entry.get('id')} - {fkb_entry.get('error_signature')}")
        solution = self.fkb.get_solution(fkb_entry)

        recovery_url = solution.get("recovery_url")
        if recovery_url:
            logger.info(f"[FKB] 回復URLへ遷移: {recovery_url}")
            try:
                await page.goto(recovery_url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)
                await self._handle_one_trust(page)
                await run_context.take_screenshot(page, "fkb_recovery_url_applied")
                logger.info("[FKB] 回復URLへの遷移に成功しました")
                return {
                    "success": True,
                    "strategy": "fkb_url_redirect",
                    "message": f"FKB ID: {solution.get('id')} - 回復URLへ遷移",
                    "fkb_entry": fkb_entry
                }
            except Exception as e:
                logger.warning(f"[FKB] 回復URLへの遷移に失敗: {e}")

        alternate_selectors = solution.get("alternate_selectors", [])
        if alternate_selectors:
            logger.info(f"[FKB] 代替セレクタを試行: {alternate_selectors[:2]}")
            for selector in alternate_selectors[:3]:
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

    # --- Circuit Breaker ヘルパー ---

    def _check_circuit_breaker(self, site: str) -> str:
        """
        Circuit Breakerの状態を返す。

        Returns:
            "closed": 正常、回復処理を実行
            "half_open": クールダウン後、復帰試行OK
            "open": 停止中（閾値超過）
        """
        failures = self._cb_failures.get(site, 0)
        if failures < self.CB_THRESHOLD:
            return "closed"

        if site in self._cb_opened_at:
            elapsed = time.time() - self._cb_opened_at[site]
            if elapsed >= self.CB_COOLDOWN_SEC:
                logger.info(f"[CircuitBreaker] {site}: クールダウン完了({elapsed:.0f}s), half-open状態へ")
                return "half_open"
            logger.info(f"[CircuitBreaker] {site}: オープン中, {self.CB_COOLDOWN_SEC - elapsed:.0f}s後に再試行可能")
            return "open"

        self._cb_opened_at[site] = time.time()
        return "open"

    def _record_failure(self, site: str) -> None:
        """連続失敗を記録し、閾値超過でCBをオープン"""
        self._cb_failures[site] = self._cb_failures.get(site, 0) + 1
        self.recovery_stats["failed"] += 1

        if self._cb_failures[site] >= self.CB_THRESHOLD:
            if site not in self._cb_opened_at:
                self._cb_opened_at[site] = time.time()
                logger.warning(
                    f"[CircuitBreaker] {site}: 連続{self._cb_failures[site]}回失敗でオープン "
                    f"(閾値={self.CB_THRESHOLD}, クールダウン={self.CB_COOLDOWN_SEC}s)"
                )

    def _record_success(self, site: str) -> None:
        """成功を記録し、CBカウンターをリセット"""
        self._cb_failures[site] = 0
        if site in self._cb_opened_at:
            del self._cb_opened_at[site]
            logger.info(f"[CircuitBreaker] {site}: 回復成功でCBリセット")

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
            "cb_failures": dict(self._cb_failures),
            "cb_opened_sites": list(self._cb_opened_at.keys()),
            "success_rate": round(success / total * 100, 1)
        }