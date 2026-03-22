# -*- coding: utf-8 -*-
# File: app/agents/plugins/prada_plp_v1.py
# Version: 0.1.0
# Purpose: PRADA サイト用スクレイピング戦略プラグイン

import logging
import re
from .base import StrategyPlugin

logger = logging.getLogger(__name__)


class PradaPLPStrategy(StrategyPlugin):
    """
    PRADA (prada.com) 用のPLPスクレイピング戦略。

    対応:
    - Cookie同意バナー (OneTrust)
    - ロケールトラップ回避 (/us/en/, /it/it/, etc.)
    - 产品价格表示待機
    - スクロールによるlazy load対応
    """
    site = "PRADA"
    _DEFAULT_LOCALE = "en-US"
    _HARD_PLP_URL = "https://www.prada.com/us/en/women/bags"
    _PLP_TILE_SELECTORS = (
        "div.product-item",
        "div[class*='product-item']",
        "a[href*='/product/']",
        "article[class*='product']",
        "div[data-testid='product-item']",
        "li.product-item",
    )

    def before_navigate(self, url: str, ctx) -> str:
        """URLを補正して locale トラップを回避"""
        url = self.strip_fragment(url)

        # パスを正規化
        path = self._path(url)
        if path and path.count('/') < 2:
            return self._HARD_PLP_URL

        # 製品詳細頁(PDP)のパスを含んでいた場合はPLPへ
        if '/product/' in url or '/p.' in url:
            return self._HARD_PLP_URL

        return url

    async def after_navigate(self, page, ctx) -> None:
        """Cookieバナー処理とページ安定待機"""
        await self.dismiss_consent(page)
        await page.wait_for_timeout(2000)  # ページ安定待機

    async def assert_plp(self, page, ctx) -> bool:
        """PLP頁面判定: 商品タイルが存在するか"""
        for sel in self._PLP_TILE_SELECTORS:
            try:
                count = await page.locator(sel).count()
                if count >= 3:
                    logger.info(f"[PRADA] PLP confirmed with selector {sel}, tile count: {count}")
                    return True
            except Exception:
                continue
        return False

    async def materialize(self, page, ctx) -> bool:
        """商品を読み込ませるためスクロール + 画像遅延読み込み対応"""
        try:
            await page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight / 4);
            """)
            await page.wait_for_timeout(1000)

            await page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight);
            """)
            await page.wait_for_timeout(1500)

            await page.evaluate("""
                window.scrollTo(0, 0);
            """)
            await page.wait_for_timeout(500)
            return True
        except Exception as e:
            logger.warning(f"[PRADA] Materialize scroll failed: {e}")
            return False

    def _path(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).path
