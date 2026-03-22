# -*- coding: utf-8 -*-
# File: app/agents/plugins/ssense_plp_v1.py
# Version: 0.1.0
# Purpose: SSENSE サイト用スクレイピング戦略プラグイン

import logging
import re
from .base import StrategyPlugin

logger = logging.getLogger(__name__)


class SSENSEPLPStrategy(StrategyPlugin):
    """
    SSENSE (ssense.com) 用のPLPスクレイピング戦略。

    対応:
    - Cookie同意バナー (OneTrust)
    - ロケールトラップ回避 (/en-us/, /en-gb/, etc.)
    - 产品价格表示待機
    """
    site = "SSENSE"
    _DEFAULT_LOCALE = "en-US"
    _DEFAULT_COUNTRY = "US"
    _HARD_PLP_URL = "https://www.ssense.com/en-us/women/outerwear"
    _PLP_TILE_SELECTORS = (
        "div.product-item",
        "div[class*='product-item']",
        "a[href*='/product/']",
        "article[class*='product']",
        "div[data-testid='product-item']",
    )

    def before_navigate(self, url: str, ctx) -> str:
        """URLを補正して locale トラップを回避"""
        url = self.strip_fragment(url)

        # ロケールが含まれているかチェック
        if re.search(r'/en-(us|gb|de|fr|it|es)/', url, re.IGNORECASE):
            # ロケールが正しければそのまま
            if '/en-us/' in url.lower():
                return url
            # 他ロケールなら強制的に en-us へ
            url = re.sub(r'/en-[a-z]{2}/', '/en-us/', url, flags=re.IGNORECASE)
            logger.info(f"[SSENSE] Locale corrected: {url}")

        # フィルタリングが深いURLを浅くする
        path = self._path(url)
        if path and path.count('/') > 4:
            # 深い階層の場合、ベーシックなPLPに飛ばす
            if 'outerwear' not in path.lower():
                url = self._HARD_PLP_URL
                logger.info("[SSENSE] Deep path detected, redirecting to base PLP")

        return url

    async def after_navigate(self, page, ctx) -> None:
        """Cookieバナー処理とページ安定待機"""
        await self.dismiss_consent(page)
        await page.wait_for_timeout(1500)  # ページ安定待機

    async def assert_plp(self, page, ctx) -> bool:
        """PLP頁面判定: 商品タイルが存在するか"""
        for sel in self._PLP_TILE_SELECTORS:
            try:
                count = await page.locator(sel).count()
                if count >= 3:
                    logger.info(f"[SSENSE] PLP confirmed with selector {sel}, tile count: {count}")
                    return True
            except Exception:
                continue
        return False

    async def materialize(self, page, ctx) -> bool:
        """商品を読み込ませるためスクロール"""
        try:
            await page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight / 3);
            """)
            await page.wait_for_timeout(1200)
            await page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight);
            """)
            await page.wait_for_timeout(800)
            return True
        except Exception as e:
            logger.warning(f"[SSENSE] Materialize scroll failed: {e}")
            return False

    def _path(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).path
