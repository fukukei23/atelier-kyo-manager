# -*- coding: utf-8 -*-
"""
Moncler PDP Handler

CR-ATELIER-003 Phase B: Moncler 専用 PDP 抽出／検証を集約

このモジュールは、Moncler サイト専用の PDP 抽出ロジックを担当します。
"""

import logging
from typing import Any, Dict, List, Optional

from playwright.async_api import Page

from app.agents.browser.extractor import extract_moncler_pdp_links

logger = logging.getLogger(__name__)


class MonclerPdpHandler:
    """
    Moncler 専用 PDP 抽出ハンドラ
    
    CR-ATELIER-003 Phase B: BrowserUseAgent から Moncler 固有ロジックを分離
    """
    
    @staticmethod
    async def extract_pdp_links(
        page: Page,
        site_config: Dict[str, Any],
        *,
        ctx: Optional[Any] = None,
        max_links: int = 50,
    ) -> Dict[str, Any]:
        """
        Moncler 専用 PDP 抽出ロジック
        
        Args:
            page: Playwright Page オブジェクト
            site_config: サイト設定
            ctx: NavigationContext または RunContext（オプション）
            max_links: 最大抽出リンク数
        
        Returns:
            Dict[str, Any]: 抽出結果
                - links: List[str]: 有効な PDP URL のリスト
                - moncler_outcome_info: Dict[str, Any]: 詳細情報（raw_count, accepted_count, layer_stats, etc.）
        """
        logger.info("[MonclerPdpHandler] Starting Moncler-specific PDP extraction")
        
        # extract_moncler_pdp_links を呼び出す
        try:
            result = await extract_moncler_pdp_links(page, ctx or site_config, max_links=max_links)
            
            # extract_moncler_pdp_links の戻り値は Dict[str, Any] または List[str] の可能性がある
            if isinstance(result, dict):
                links = result.get("links", [])
                moncler_outcome_info = result.get("moncler_outcome_info", {})
            elif isinstance(result, list):
                links = result
                moncler_outcome_info = {
                    "raw_count": len(links),
                    "accepted_count": len(links),
                    "layer_stats": {},
                    "layers_used": [],
                    "rejection_stats": {},
                }
            else:
                links = []
                moncler_outcome_info = {}
            
            logger.info(
                f"[MonclerPdpHandler] Extracted {len(links)} PDP links "
                f"(raw={moncler_outcome_info.get('raw_count', 0)}, "
                f"accepted={moncler_outcome_info.get('accepted_count', len(links))})"
            )
            
            return {
                "links": links,
                "moncler_outcome_info": moncler_outcome_info,
            }
        except Exception as e:
            logger.error(f"[MonclerPdpHandler] PDP extraction failed: {e}", exc_info=True)
            return {
                "links": [],
                "moncler_outcome_info": {
                    "raw_count": 0,
                    "accepted_count": 0,
                    "layer_stats": {},
                    "layers_used": [],
                    "rejection_stats": {},
                    "error": str(e),
                },
            }

