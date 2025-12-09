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
try:
    from app.core.run_context import RunContext
except Exception:
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