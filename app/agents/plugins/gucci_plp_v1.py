# -*- coding: utf-8 -*-
# File: app/agents/plugins/gucci_plp_v1.py
# Version: 0.2.0
# Purpose: GUCCI サイト用スクレイピング戦略プラグイン

import logging
import re
import json
from typing import List, Dict, Set, Optional
from urllib.parse import urlparse
from .base import StrategyPlugin

logger = logging.getLogger(__name__)


class GucciPLPStrategy(StrategyPlugin):
    """
    GUCCI (gucci.com) 用のPLPスクレイピング戦略。

    対応:
    - Cookie同意バナー (OneTrust)
    - ロケールトラップ回避 (/us/en/, /it/it/, etc.)
    - 遅延読み込み対応（段階的スクロール + Load Moreボタン）
    - JSON-LD製品抽出
    """
    site = "GUCCI"
    _DEFAULT_LOCALE = "en-US"
    _DEFAULT_COUNTRY = "US"
    _HARD_PLP_URL = "https://www.gucci.com/us/en/women/handbags"
    _PLP_TILE_SELECTORS = (
        "div.product-item",
        "div[class*='product-item']",
        "div[class*='ProductItem']",
        "a[href*='/product/']",
        "article[class*='product']",
        "li[class*='product']",
        "div[data-testid='product-item']",
    )

    def before_navigate(self, url: str, ctx) -> str:
        """URLを補正してlocaleトラップを回避"""
        url = self.strip_fragment(url)

        # ロケールトラップ検出
        path = self._path(url)
        if path:
            # ロケールと国の不一致を検出 (/us/en/, /it/it/, etc.)
            if re.search(r'/([a-z]{2})/([a-z]{2})/', path):
                # 正しいフォーマットに是正
                locale_match = re.search(r'/([a-z]{2})/([a-z]{2})/', path)
                if locale_match:
                    lang = locale_match.group(1)
                    country = locale_match.group(2)
                    if lang != country.lower():
                        # 不一致の場合はUS/enに強制
                        new_path = re.sub(r'/[a-z]{2}/[a-z]{2}/', '/us/en/', path)
                        url = url.replace(path, new_path)
                        logger.info(f"[GUCCI] Locale trap corrected: {url}")

        # 浅すぎるパスは正規PLPへ
        if path and path.count('/') < 3:
            return self._HARD_PLP_URL

        # PDPパスが含まれていたらPLPへ
        if '/product/' in url or '/p.' in url or '/p/' in url:
            return self._HARD_PLP_URL

        return url

    async def after_navigate(self, page, ctx) -> None:
        """Cookieバナー処理とページ安定待機"""
        await self.dismiss_consent(page)
        await page.wait_for_timeout(2500)

    async def materialize(self, page, ctx) -> bool:
        """
        商品を完全読み込ませる（段階的スクロール + Load Moreボタン）
        """
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

            all_product_urls: Set[str] = set()

            # === 手法1: 段階的スクロール ===
            logger.info("[GUCCI] Step 1: Progressive scrolling...")
            no_change_count = 0

            for scroll_idx in range(25):
                await page.evaluate(f"window.scrollBy(0, {300 + scroll_idx * 25})")
                await page.wait_for_timeout(400)

                current_urls = await self._get_product_urls(page)
                new_found = current_urls - all_product_urls

                if new_found:
                    all_product_urls.update(new_found)
                    no_change_count = 0
                    logger.info(f"[GUCCI] Scroll {scroll_idx + 1}: +{len(new_found)}, total: {len(all_product_urls)}")
                else:
                    no_change_count += 1

                if no_change_count >= 3:
                    logger.info("[GUCCI] No new products after 3 scrolls")
                    break

            # === 手法2: Load Moreボタン ===
            logger.info("[GUCCI] Step 2: Clicking Load More...")
            load_more_selectors = [
                "button:has-text('Load More')",
                "button:has-text('LOAD MORE')",
                "button:has-text('Show More')",
                "[data-testid='load-more']",
                ".load-more button",
                "button[class*='load-more']",
                "button[class*='LoadMore']",
            ]

            for _ in range(8):
                clicked = False
                for sel in load_more_selectors:
                    try:
                        btn = page.locator(sel)
                        if await btn.count() > 0 and await btn.is_visible():
                            await btn.scroll_into_view_if_needed()
                            await page.wait_for_timeout(300)
                            await btn.click()
                            await page.wait_for_timeout(1500)
                            logger.info("[GUCCI] Load More clicked")

                            current_urls = await self._get_product_urls(page)
                            new_found = current_urls - all_product_urls
                            if new_found:
                                all_product_urls.update(new_found)
                            clicked = True
                            break
                    except Exception:
                        pass

                if not clicked:
                    break

            # === 手法3: JSON-LD ===
            logger.info("[GUCCI] Step 3: Extracting JSON-LD...")
            jsonld_products = await self._extract_jsonld_products(page)
            if jsonld_products:
                for p in jsonld_products:
                    url = p.get("url", "")
                    if url:
                        if url.startswith("/"):
                            url = f"https://www.gucci.com{url}"
                        all_product_urls.add(url)
                logger.info(f"[GUCCI] JSON-LD products: {len(jsonld_products)}")

            final_count = len(all_product_urls)
            logger.info(f"[GUCCI] FINAL PRODUCT COUNT: {final_count}")

            if final_count > 0:
                for i, url in enumerate(sorted(all_product_urls)[:5], 1):
                    logger.info(f"[GUCCI] Product {i}: {url}")

            return final_count > 0

        except Exception as e:
            logger.warning(f"[GUCCI] Materialize error: {e}")
            return False

    async def _get_product_urls(self, page) -> Set[str]:
        """ページから商品URLを全て取得"""
        urls: Set[str] = set()
        try:
            links = await page.locator("a[href*='/product/']").all()
            for link in links:
                try:
                    href = await link.get_attribute("href")
                    if href:
                        if href.startswith("/"):
                            href = f"https://www.gucci.com{href}"
                        urls.add(href)
                except Exception:
                    pass
        except Exception:
            pass
        return urls

    async def _extract_jsonld_products(self, page) -> List[Dict]:
        """JSON-LDから製品情報を抽出"""
        products: List[Dict] = []
        try:
            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for script in scripts:
                try:
                    content = await script.inner_text()
                    data = json.loads(content)

                    items = []
                    if isinstance(data, dict) and "@graph" in data:
                        items = [x for x in data["@graph"] if x.get("@type") == "Product"]
                    elif isinstance(data, list):
                        items = [x for x in data if x.get("@type") == "Product"]
                    elif data.get("@type") == "Product":
                        items = [data]

                    for item in items:
                        products.append({
                            "name": item.get("name"),
                            "url": item.get("url"),
                            "price": None,
                            "brand": item.get("brand"),
                        })
                except Exception:
                    pass
        except Exception:
            pass
        return products

    async def assert_plp(self, page, ctx) -> bool:
        """PLP頁面判定"""
        for sel in self._PLP_TILE_SELECTORS:
            try:
                count = await page.locator(sel).count()
                if count >= 1:
                    logger.info(f"[GUCCI] PLP confirmed: {sel} x{count}")
                    return True
            except Exception:
                continue

        # JSON-LD check
        try:
            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for script in scripts:
                content = await script.inner_text()
                if '"@type":"Product"' in content or '"@type": "Product"' in content:
                    logger.info("[GUCCI] PLP confirmed via JSON-LD")
                    return True
        except Exception:
            pass

        return False

    def _path(self, url: str) -> str:
        return urlparse(url).path or "/"
