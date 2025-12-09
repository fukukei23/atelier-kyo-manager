# -*- coding: utf-8 -*-
"""
Moncler PLP Handler

CR-ATELIER-003 Phase B: Moncler 専用 PLP 処理を集約

このモジュールは、Moncler サイト専用の PLP 処理を担当します：
- PLP materialize
- PLP recover
- PLP → PDP 抽出
"""

import logging
from typing import Any, Dict, Optional

from playwright.async_api import Page

from app.agents.browser.navigation_driver import NavigationDriver, NavigationContext
from app.agents.browser_use_moncler_patch import moncler_plp_recovery

logger = logging.getLogger(__name__)


class MonclerPlpHandler:
    """
    Moncler 専用 PLP 処理ハンドラ
    
    CR-ATELIER-003 Phase B: BrowserUseAgent から Moncler 固有ロジックを分離
    """
    
    @staticmethod
    async def run(
        site_config: Dict[str, Any],
        nav_ctx: NavigationContext,
        *,
        page: Optional[Page] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Moncler 専用 PLP 処理を実行
        
        Args:
            site_config: サイト設定
            nav_ctx: NavigationContext
            page: Playwright Page オブジェクト（オプション）
            query: 検索クエリ（オプション）
        
        Returns:
            Dict[str, Any]: 処理結果
        """
        logger.info("[MonclerPlpHandler] Starting Moncler-specific PLP flow")
        
        # NavigationDriver を取得
        navigation_driver = nav_ctx.navigation_driver if hasattr(nav_ctx, "navigation_driver") else None
        if navigation_driver is None:
            # NavigationDriver が存在しない場合は作成
            from app.agents.browser.telemetry import TelemetryService
            telemetry_service = TelemetryService(nav_ctx.run_context) if nav_ctx.run_context else None
            navigation_driver = NavigationDriver(
                page=nav_ctx.page,
                telemetry=telemetry_service,
            )
        
        # Moncler 専用の PLP recovery を実行
        if page is None:
            page = nav_ctx.page
        
        if query is None:
            query = nav_ctx.query or ""
        
        # moncler_plp_recovery を呼び出す（BrowserUseAgent から移行）
        try:
            await moncler_plp_recovery(page, site_config, {"query": query, "shipTo": "GB"})
            logger.info("[MonclerPlpHandler] moncler_plp_recovery completed")
        except Exception as e:
            logger.warning(f"[MonclerPlpHandler] moncler_plp_recovery failed: {e}", exc_info=True)
            # エラーが発生しても続行
        
        # NavigationDriver の run_plp_flow を呼び出す前に、Moncler 専用の PDP 抽出を設定
        # CR-ATELIER-003 Phase B: Moncler 専用 PDP 抽出は MonclerPdpHandler 経由で実行
        from app.agents.moncler.moncler_pdp_handler import MonclerPdpHandler
        
        # NavigationDriver の collect_pdp_links を MonclerPdpHandler 経由で実行するように設定
        # ただし、NavigationDriver はブランド非依存のため、MonclerPdpHandler を直接呼び出す
        try:
            # Moncler 専用の PDP 抽出を実行
            pdp_result = await MonclerPdpHandler.extract_pdp_links(
                page=page,
                site_config=site_config,
                ctx=nav_ctx,
                max_links=50,
            )
            
            # NavigationDriver の run_plp_flow を呼び出す（PDP 抽出は既に完了）
            # ただし、NavigationDriver 内で Moncler 分岐を削除したため、汎用フローを実行
            outcome = await navigation_driver.run_plp_flow(nav_ctx)
            
            # Moncler 専用の PDP 抽出結果を使用
            if pdp_result.get("links"):
                outcome.pdp_links = pdp_result["links"]
                outcome.moncler_outcome = pdp_result.get("moncler_outcome_info", {})
            
            logger.info(f"[MonclerPlpHandler] NavigationDriver.run_plp_flow completed: {len(outcome.pdp_links)} PDP links")
            return {
                "success": True,
                "pdp_links": outcome.pdp_links,
                "outcome": outcome,
            }
        except Exception as e:
            logger.error(f"[MonclerPlpHandler] NavigationDriver.run_plp_flow failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "pdp_links": [],
            }

