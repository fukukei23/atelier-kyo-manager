"""
Farfetch PLP Strategy Plugin
======================================================================
Version: 1.0.0
Site: FARFETCH
======================================================================
"""

import contextlib
import logging

from .base import StrategyPlugin, _apply_stealth

logger = logging.getLogger(__name__)

DEFAULT_STEALTH_TIMEOUT_MS = 2500
MAX_SCROLL_ITERATIONS = 30
SCROLL_BASE_DISTANCE = 300
SCROLL_INCREMENT_PER_ITERATION = 25


class FarfetchPlpStrategy(StrategyPlugin):
    """Farfetch 商品列表ページ (PLP) 用スクレイピング戦略"""

    site = "FARFETCH"

    async def after_navigate(self, page, ctx) -> None:
        """Farfetch遷移後の处理: Cookieconsent + stealth"""
        await _apply_stealth(page)
        await page.wait_for_timeout(DEFAULT_STEALTH_TIMEOUT_MS)

        # Cookie consent 対応
        await self.dismiss_consent(page)

    async def assert_plp(self, page, ctx) -> bool:
        """Farfetch PLP 判定: 商品カードまたは検索結果がればOK"""
        try:
            # 商品カードが存在するか
            card_selectors = ["article[data-testid='productCard']", "li.ProductCardWrapper", "a[href*='/pression/']"]
            for sel in card_selectors:
                count = await page.locator(sel).count()
                if count > 0:
                    logger.info(f"[Farfetch] PLP判定: {count}件のカードを発見 ({sel})")
                    return True

            # URLで判定
            url = page.url
            if "/shopping/" in url or "?q=" in url:
                logger.info(f"[Farfetch] URLでPLPと判定: {url}")
                return True

            return False
        except Exception as e:
            logger.warning(f"[Farfetch] assert_plp error: {e}")
            return False

    async def materialize(self, page, ctx) -> bool:
        """
        Farfetch PLP の商品カードをすべて読み込ませる。

        - 初期ロード待機
        - 段階的スクロール（最大30回）
        - 「Load More」ボタン対応（最大8回）
        """
        try:
            await page.wait_for_timeout(2000)

            # 商品カード数をカウント
            initial_count = await self._count_cards(page)
            logger.info(f"[Farfetch] materialize: 初期カード数={initial_count}")

            if initial_count == 0:
                logger.warning("[Farfetch] カード0件、スクロールを試行")

            # 段階的スクロール
            for i in range(MAX_SCROLL_ITERATIONS):
                prev = await self._count_cards(page)
                scroll_dist = SCROLL_BASE_DISTANCE + (i * SCROLL_INCREMENT_PER_ITERATION)
                await page.evaluate(f"window.scrollBy(0, {scroll_dist})")
                await page.wait_for_timeout(800)
                curr = await self._count_cards(page)

                logger.debug(f"[Farfetch] scroll #{i}: prev={prev} curr={curr}")
                if curr > prev:
                    continue
                # カード数が増えなければそれ以上読み込みなし
                if curr >= prev and i > 3:
                    logger.info(f"[Farfetch] スクロール完了: 最終={curr}件")
                    break

            # Load Moreボタン処理
            load_more_clicked = 0
            for _ in range(8):
                btn_selectors = [
                    "button:has-text('Load More')",
                    "button:has-text('LOAD MORE')",
                    "[data-testid='load-more-button']",
                    "button.LoadMore",
                ]
                clicked = False
                for sel in btn_selectors:
                    try:
                        btn = page.locator(sel).first
                        if await btn.is_visible(timeout=2000):
                            await btn.click(timeout=3000)
                            load_more_clicked += 1
                            await page.wait_for_timeout(1500)
                            clicked = True
                            logger.info(f"[Farfetch] Load More clicked ({load_more_clicked})")
                            break
                    except Exception:
                        continue
                if not clicked:
                    break

            final_count = await self._count_cards(page)
            logger.info(f"[Farfetch] materialize完了: 最終={final_count}件 (LoadMore={load_more_clicked}回)")
            return final_count > 0

        except Exception as e:
            logger.warning(f"[Farfetch] materialize error: {e}")
            return False

    async def _count_cards(self, page) -> int:
        """商品カード数をカウント"""
        selectors = ["article[data-testid='productCard']", "li.ProductCardWrapper", "a[href*='/pression/']"]
        total = 0
        for sel in selectors:
            with contextlib.suppress(Exception):
                total += await page.locator(sel).count()
        return total
