from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, BrowserContext, sync_playwright

logger = logging.getLogger(__name__)

PROXY_POOL_PATH = Path(__file__).resolve().parents[1] / "config" / "proxy_pool.json"

SUPPORTED_SITES = ["farfetch", "gucci_official", "prada_official", "ferragamo_official"]
SUPPORTED_BRANDS = ["Gucci", "Prada", "Ferragamo"]

_FARFETCH_BRAND_SLUGS = {
    "Gucci": "gucci",
    "Prada": "prada",
    "Ferragamo": "salvatore-ferragamo",
}

_OFFICIAL_SITE_URLS: dict[str, dict[str, str]] = {
    "Gucci": {
        "gucci_official": "https://www.gucci.com/jp/ja/ca/-c-women-handbags",
    },
    "Prada": {
        "prada_official": "https://www.prada.com/jp/ja/women/bags.html",
    },
    "Ferragamo": {
        "ferragamo_official": "https://www.ferragamo.com/shop/jpn/ja/sf/hug-bag-category",
    },
}

_CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _load_proxies() -> list[dict]:
    if not PROXY_POOL_PATH.exists():
        return []
    with open(PROXY_POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    proxies = []
    for key, entries in pool.items():
        if isinstance(entries, list):
            for e in entries:
                if e.get("active", True) and e.get("server"):
                    proxies.append(e)
        elif isinstance(entries, dict):
            for sub_key, sub_entries in entries.items():
                if isinstance(sub_entries, list):
                    for e in sub_entries:
                        if e.get("active", True) and e.get("server"):
                            proxies.append(e)
    return proxies


def _fetch_with_cffi(url: str, timeout: int = 25) -> str | None:
    from curl_cffi import requests as cffi_requests
    try:
        resp = cffi_requests.get(
            url, headers=_CHROME_HEADERS, impersonate="chrome124",
            timeout=timeout, allow_redirects=True,
        )
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"[cffi] {url} returned {resp.status_code}")
    except Exception as e:
        logger.error(f"[cffi] Request failed for {url}: {e}")
    return None


class BrandPriceScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.pw = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page = None

    def _init_browser(self, timeout_sec: int = 60):
        if self.page:
            return
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=self.headless)

        opts: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            "locale": "ja-JP",
            "timezone_id": "Asia/Tokyo",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        }

        proxies = _load_proxies()
        if proxies:
            proxy = random.choice(proxies)
            opts["proxy"] = {
                "server": proxy["server"],
                "username": proxy.get("username", ""),
                "password": proxy.get("password", ""),
            }

        self.context = self.browser.new_context(**opts)
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja', 'en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)
        self.page = self.context.new_page()
        self.page.set_default_timeout(timeout_sec * 1000)

    def _close_browser(self):
        for resource in [self.context, self.browser]:
            if resource:
                try:
                    resource.close()
                except Exception:
                    pass
        if self.pw:
            try:
                self.pw.stop()
            except Exception:
                pass
        self.page = None
        self.context = None
        self.browser = None
        self.pw = None

    def _scrape_farfetch(self, brand: str, item_limit: int = 10) -> list[dict]:
        slug = _FARFETCH_BRAND_SLUGS.get(brand)
        if not slug:
            return []

        url = f"https://www.farfetch.com/jp/shopping/women/{slug}/items.aspx"
        results = []

        try:
            logger.info(f"[farfetch] Navigating to {url}")
            self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)

            # Scroll to load lazy-rendered product cards
            for _ in range(15):
                self.page.evaluate("window.scrollBy(0, 600)")
                time.sleep(0.5)

            EXTRACT_JS = """
            (() => {
                const cards = document.querySelectorAll('[data-testid=product-card]');
                const results = [];
                for (const card of cards) {
                    try {
                        const link = card.querySelector("a[href*='item-']");
                        const href = link ? link.href : "";
                        const labelledby = link ? link.getAttribute("aria-labelledby") : "";
                        let brandName = "", productDesc = "";
                        if (labelledby) {
                            const ids = labelledby.split(" ");
                            for (const id of ids) {
                                const el = document.getElementById(id);
                                if (el) {
                                    const txt = el.innerText.trim();
                                    if (id.includes("brand")) brandName = txt;
                                    else if (id.includes("description")) productDesc = txt;
                                }
                            }
                        }
                        const text = card.innerText || "";
                        const priceMatch = text.match(/￥\\s*([\\d,]+)/);
                        const price = priceMatch ? parseInt(priceMatch[1].replace(/,/g, "")) : null;
                        if (price) {
                            results.push({
                                brand: brandName,
                                name: productDesc || brandName,
                                price: price,
                                url: href
                            });
                        }
                    } catch(e) {}
                }
                return results;
            })()
            """

            cards = self.page.evaluate(EXTRACT_JS) or []
            logger.info(f"[farfetch] Found {len(cards)} products for {brand}")

            for card in cards[:item_limit]:
                results.append({
                    "brand": brand,
                    "product_name": card["name"],
                    "source_site": "farfetch",
                    "source_url": card["url"],
                    "price_original": float(card["price"]),
                    "currency": "JPY",
                    "price_jpy": float(card["price"]),
                    "exchange_rate": 1.0,
                    "in_stock": True,
                    "size_available": "",
                    "scraped_at": datetime.utcnow().isoformat(),
                })

            logger.info(f"[farfetch] Extracted {len(results)} products for {brand}")
        except Exception as e:
            logger.error(f"[farfetch] Scraping failed: {e}")

        return results

    def _scrape_official_gucci(self, brand: str, item_limit: int = 10) -> list[dict]:
        urls = _OFFICIAL_SITE_URLS.get(brand, {})
        url = urls.get("gucci_official")
        if not url:
            return []

        html = _fetch_with_cffi(url)
        if not html:
            return []

        # Gucci uses: aria-label="ProductName, ￥Price"
        products = re.findall(
            r'aria-label="([^"]+?),\s*￥([\d,]+)"', html
        )
        logger.info(f"[gucci_official] Found {len(products)} products for {brand}")

        results = []
        for name, price_str in products[:item_limit]:
            price = float(price_str.replace(",", ""))
            results.append({
                "brand": brand,
                "product_name": name.strip(),
                "source_site": "gucci_official",
                "source_url": url,
                "price_original": price,
                "currency": "JPY",
                "price_jpy": price,
                "exchange_rate": 1.0,
                "in_stock": True,
                "size_available": "",
                "scraped_at": datetime.utcnow().isoformat(),
            })

        logger.info(f"[gucci_official] Extracted {len(results)} products for {brand}")
        return results

    def _scrape_official_prada(self, brand: str, item_limit: int = 10) -> list[dict]:
        urls = _OFFICIAL_SITE_URLS.get(brand, {})
        url = urls.get("prada_official")
        if not url:
            return []

        html = _fetch_with_cffi(url)
        if not html:
            return []

        # Prada uses: aria-label=" ProductName ¥ Price ..."
        products = re.findall(
            r'aria-label="\s*([^"]+?)\s*¥\s*([\d,]+)', html
        )
        logger.info(f"[prada_official] Found {len(products)} products for {brand}")

        results = []
        for name, price_str in products[:item_limit]:
            price = float(price_str.replace(",", ""))
            results.append({
                "brand": brand,
                "product_name": name.strip(),
                "source_site": "prada_official",
                "source_url": url,
                "price_original": price,
                "currency": "JPY",
                "price_jpy": price,
                "exchange_rate": 1.0,
                "in_stock": True,
                "size_available": "",
                "scraped_at": datetime.utcnow().isoformat(),
            })

        logger.info(f"[prada_official] Extracted {len(results)} products for {brand}")
        return results

    def _scrape_official_ferragamo(self, brand: str, item_limit: int = 10) -> list[dict]:
        urls = _OFFICIAL_SITE_URLS.get(brand, {})
        url = urls.get("ferragamo_official")
        if not url:
            return []

        from playwright_stealth import Stealth

        results = []
        try:
            with Stealth().use_sync(sync_playwright()) as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    locale="ja-JP",
                    timezone_id="Asia/Tokyo",
                    user_agent=_CHROME_HEADERS["User-Agent"],
                )
                page = ctx.new_page()

                logger.info(f"[ferragamo_official] Navigating to {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(8)

                for _ in range(10):
                    page.evaluate("window.scrollBy(0, 500)")
                    time.sleep(0.5)

                FERRAGAMO_EXTRACT_JS = """
                (() => {
                    const items = document.querySelectorAll('.product-list-r23__description');
                    const results = [];
                    items.forEach(item => {
                        const nameEl = item.querySelector('.product-list-r23__name');
                        const priceEl = item.querySelector('.product-list-r23__price');
                        if (nameEl && priceEl) {
                            const name = nameEl.innerText.trim();
                            const priceMatch = priceEl.innerText.match(/[¥￥]\\s*([\\d,]+)/);
                            const price = priceMatch ? parseInt(priceMatch[1].replace(/,/g, '')) : null;
                            if (price) results.push({name, price});
                        }
                    });
                    return results;
                })()
                """

                products = page.evaluate(FERRAGAMO_EXTRACT_JS) or []
                logger.info(f"[ferragamo_official] Found {len(products)} products for {brand}")

                for product in products[:item_limit]:
                    results.append({
                        "brand": brand,
                        "product_name": product["name"],
                        "source_site": "ferragamo_official",
                        "source_url": url,
                        "price_original": float(product["price"]),
                        "currency": "JPY",
                        "price_jpy": float(product["price"]),
                        "exchange_rate": 1.0,
                        "in_stock": True,
                        "size_available": "",
                        "scraped_at": datetime.utcnow().isoformat(),
                    })

                ctx.close()
                browser.close()

        except Exception as e:
            logger.error(f"[ferragamo_official] Scraping failed: {e}")

        logger.info(f"[ferragamo_official] Extracted {len(results)} products for {brand}")
        return results

    def scrape(
        self, brand: str, sites: list[str] | None = None, item_limit: int = 10
    ) -> list[dict]:
        if sites is None:
            sites = SUPPORTED_SITES

        all_results: list[dict] = []

        # Official sites use curl_cffi (no browser needed)
        if "gucci_official" in sites:
            results = self._scrape_official_gucci(brand, item_limit)
            all_results.extend(results)

        if "prada_official" in sites:
            results = self._scrape_official_prada(brand, item_limit)
            all_results.extend(results)

        # Ferragamo needs playwright-stealth for JS rendering
        if "ferragamo_official" in sites:
            results = self._scrape_official_ferragamo(brand, item_limit)
            all_results.extend(results)

        # Farfetch requires Playwright browser
        if "farfetch" in sites:
            try:
                self._init_browser()
                results = self._scrape_farfetch(brand, item_limit)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"[farfetch] Scraping error: {e}")
            finally:
                self._close_browser()

        logger.info(f"Total scraped: {len(all_results)} products for {brand}")
        return all_results
