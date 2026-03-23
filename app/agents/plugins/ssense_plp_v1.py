# -*- coding: utf-8 -*-
# File: app/agents/plugins/ssense_plp_v1.py
# Version: 0.4.0
# Purpose: SSENSE サイト用スクレイピング戦略プラグイン

import logging
import re
import json
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from .base import StrategyPlugin, _apply_stealth

logger = logging.getLogger(__name__)

# Bot検出リスクが高いページをスキップするパターン
_SKIP_PATTERNS = [
    "/checkout/",
    "/cart/",
    "/account/",
    "/login/",
    "/search?q=",
]

# 商品候補になるパス
_PRODUCT_PATTERNS = [
    "/product/",
    "/p/",
    "/p-",
    "/products/",
]


class SSENSEPLPStrategy(StrategyPlugin):
    """
    SSENSE (ssense.com) 用のPLPスクレイピング戦略。

    対応:
    - Cookie同意バナー (OneTrust)
    - ロケールトラップ回避
    - 遅延読み込み対応（段階的スクロール + Load Moreボタン）
    - Stealthモード対応（Bot検出回避）
    - JSON-LD製品抽出

    注意: SSENSEは高度なBot検出を使用しています。
    完全な回避にはプロキシ/IP回転が必要な場合があります。
    """
    site = "SSENSE"
    _DEFAULT_LOCALE = "en-US"
    _DEFAULT_COUNTRY = "US"
    _HARD_PLP_URL = "https://www.ssense.com/en-us/women/outerwear"
    _PLP_TILE_SELECTORS = (
        "a[href*='/product/']",
        "div.product-item",
        "div[class*='product-item']",
        "article[class*='product']",
    )

    # Stealth適用済みフラグ
    _stealth_applied: bool = False

    def before_navigate(self, url: str, ctx) -> str:
        """URLを補正してlocaleトラップを回避"""
        url = self.strip_fragment(url)

        # リスクページパスはPLPに強制リダイレクト
        for skip in _SKIP_PATTERNS:
            if skip in url:
                logger.info(f"[SSENSE] Risky path detected ({skip}), redirecting to PLP")
                return self._HARD_PLP_URL

        if re.search(r'/en-(us|gb|de|fr|it|es)/', url, re.IGNORECASE):
            if '/en-us/' not in url.lower():
                url = re.sub(r'/en-[a-z]{2}/', '/en-us/', url, flags=re.IGNORECASE)
                logger.info(f"[SSENSE] Locale corrected: {url}")

        path = self._path(url)
        if path and path.count('/') > 4:
            if 'outerwear' not in path.lower():
                url = self._HARD_PLP_URL
                logger.info("[SSENSE] Deep path detected, redirecting to base PLP")

        return url

    async def after_navigate(self, page, ctx) -> None:
        """Cookieバナー処理とStealth適用"""
        # Stealthを初めて適用
        if not SSENSEPLPStrategy._stealth_applied:
            try:
                _apply_stealth(page)
                SSENSEPLPStrategy._stealth_applied = True
                logger.info("[SSENSE] Stealth mode applied")
            except Exception as e:
                logger.warning(f"[SSENSE] Stealth apply failed: {e}")

        await self.dismiss_consent(page)
        await page.wait_for_timeout(2000)

    async def materialize(self, page, ctx) -> bool:
        """
        商品を完全読み込ませる（段階的スクロール + Load Moreボタン）
        SSENSEはBot検出が厳しいため、限定的な取得成功率
        """
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)

            all_product_urls: Set[str] = set()

            # === 手法1: 段階的スクロール ===
            logger.info("[SSENSE] Step 1: Progressive scrolling...")
            no_change_count = 0

            for scroll_idx in range(25):
                await page.evaluate(f"window.scrollBy(0, {300 + scroll_idx * 20})")
                await page.wait_for_timeout(400)

                current_urls = await self._get_product_urls(page)
                new_found = current_urls - all_product_urls

                if new_found:
                    all_product_urls.update(new_found)
                    no_change_count = 0
                    logger.info(f"[SSENSE] Scroll {scroll_idx + 1}: +{len(new_found)}, total: {len(all_product_urls)}")
                else:
                    no_change_count += 1

                if no_change_count >= 3:
                    logger.info("[SSENSE] No new products after 3 scrolls")
                    break

            # === 手法2: Load Moreボタン ===
            logger.info("[SSENSE] Step 2: Clicking Load More...")
            load_more_selectors = [
                "button:has-text('Load More')",
                "button:has-text('LOAD MORE')",
                "[data-testid='load-more']",
                ".load-more button",
                "button[class*='load-more']",
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
                            await page.wait_for_timeout(1200)
                            logger.info("[SSENSE] Load More clicked")

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

            # === 手法3: JSON-LD製品 ===
            logger.info("[SSENSE] Step 3: Extracting JSON-LD...")
            jsonld_products = await self._extract_jsonld_products(page)
            if jsonld_products:
                for p in jsonld_products:
                    url = p.get("url", "")
                    if url:
                        if url.startswith("/"):
                            url = f"https://www.ssense.com{url}"
                        all_product_urls.add(url)

            final_count = len(all_product_urls)
            logger.info(f"[SSENSE] FINAL PRODUCT COUNT: {final_count}")

            if final_count > 0:
                for i, url in enumerate(sorted(all_product_urls)[:5], 1):
                    logger.info(f"[SSENSE] Product {i}: {url}")

            return final_count > 0

        except Exception as e:
            logger.warning(f"[SSENSE] Materialize error: {e}")
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
                            href = f"https://www.ssense.com{href}"
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
                            "brand": None,
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
                    logger.info(f"[SSENSE] PLP confirmed: {sel} x{count}")
                    return True
            except Exception:
                continue

        try:
            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for script in scripts:
                content = await script.inner_text()
                if '"@type":"Product"' in content or '"@type": "Product"' in content:
                    logger.info("[SSENSE] PLP confirmed via JSON-LD")
                    return True
        except Exception:
            pass

        return False

    def _path(self, url: str) -> str:
        return urlparse(url).path
