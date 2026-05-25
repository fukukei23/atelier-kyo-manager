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

SUPPORTED_SITES = [
    "farfetch",
    "gucci_official", "prada_official", "ferragamo_official",
    "loewe_official", "balenciaga_official", "bottegaveneta_official",
]

SUPPORTED_BRANDS = [
    "Gucci", "Prada", "Ferragamo",
    "Loewe", "Balenciaga", "Bottega Veneta",
]

_FARFETCH_BRAND_SLUGS: dict[str, str] = {
    "Gucci": "gucci",
    "Prada": "prada",
    "Ferragamo": "salvatore-ferragamo",
    "Loewe": "loewe",
    "Balenciaga": "balenciaga",
    "Bottega Veneta": "bottega-veneta",
}

# ---------------------------------------------------------------------------
# Brand official site configs
# EU sites preferred for BUYMA resale margin
# ---------------------------------------------------------------------------
_OFFICIAL_CFFI: dict[str, dict[str, str]] = {
    "Gucci": {"gucci_official": "https://www.gucci.com/it/en/ca/-c-women-handbags"},
    "Prada": {"prada_official": "https://www.prada.com/it/en/women/bags.html"},
    "Loewe": {"loewe_official": "https://www.loewe.com/usa/en/women/bags"},
    "Balenciaga": {"balenciaga_official": "https://www.balenciaga.com/jp/ja/women/handbags"},
}

_OFFICIAL_STEALTH: dict[str, dict[str, str]] = {
    "Ferragamo": {
        "ferragamo_official": "https://www.ferragamo.com/shop/jpn/ja/sf/hug-bag-category",
    },
    "Bottega Veneta": {
        "bottegaveneta_official": "https://www.bottegaveneta.com/jp/ja/handbags",
    },
}

# Currency per brand (EU sites use EUR)
_CURRENCY_MAP: dict[str, str] = {
    "Gucci": "EUR",
    "Prada": "EUR",
    "Loewe": "JPY",
    "Balenciaga": "JPY",
    "Ferragamo": "JPY",
    "Bottega Veneta": "JPY",
}

# Extraction patterns per brand (curl_cffi)
_CFFI_PATTERNS: dict[str, str] = {
    "Gucci": r'aria-label="([^"]+?),\s*€\s*([\d.]+)"',
    "Prada": r'aria-label="\s*([^"]+?)\s*€\s*([\d.]+)',
    "Loewe": r'>([^<]{5,60}?)<.*?¥([\d,]+)',
    "Balenciaga": r'itemprop="price"\s+content="(\d+)"',
}

# Extraction JS per brand (playwright-stealth)
_STEALTH_JS: dict[str, str] = {
    "Ferragamo": """
    (() => {
        const items = document.querySelectorAll('.product-list-r23__description');
        const results = [];
        items.forEach(item => {
            const nameEl = item.querySelector('.product-list-r23__name');
            const priceEl = item.querySelector('.product-list-r23__price');
            if (nameEl && priceEl) {
                const name = nameEl.innerText.trim();
                const m = priceEl.innerText.match(/[¥￥]\\s*([\\d,]+)/);
                if (m) results.push({name, price: parseInt(m[1].replace(/,/g, ''))});
            }
        });
        return results;
    })()
    """,
    "Bottega Veneta": """
    (() => {
        const results = [];
        const priceEls = document.querySelectorAll('[itemprop="price"]');
        priceEls.forEach(priceEl => {
            const price = priceEl.getAttribute('content');
            if (!price) return;
            let el = priceEl;
            for (let i = 0; i < 15; i++) {
                el = el.previousElementSibling || el.parentElement;
                if (!el) break;
                const h = el.querySelector('h2, h3, h4') || el;
                const text = h.innerText.trim();
                if (text && text.length > 3 && text.length < 80 && !text.match(/^[¥￥\\d]/)) {
                    results.push({name: text.split('\\n')[0], price: parseInt(price)});
                    break;
                }
            }
        });
        const seen = new Set();
        return results.filter(r => { if (seen.has(r.name)) return false; seen.add(r.name); return true; });
    })()
    """,
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
    proxies: list[dict] = []
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
        if resp.status_code < 400 and len(resp.text) > 1000:
            return resp.text
        # Some sites return 404 but still include product data
        if resp.status_code == 404 and len(resp.text) > 10000:
            return resp.text
        logger.warning(f"[cffi] {url} returned {resp.status_code}")
    except Exception as e:
        logger.error(f"[cffi] Request failed for {url}: {e}")
    return None


def _make_item(brand: str, product_name: str, price: float, source_site: str, source_url: str, currency: str = "JPY") -> dict:
    price_jpy = price
    exchange_rate = 1.0

    if currency != "JPY":
        from app.utils.fx_utils import get_fx_table_jpy
        fx_table, _ = get_fx_table_jpy()
        rate = fx_table.get(currency)
        if rate:
            exchange_rate = rate
            price_jpy = price * rate
        else:
            logger.warning(f"No FX rate for {currency}, using raw price as JPY")

    return {
        "brand": brand,
        "product_name": product_name.strip(),
        "source_site": source_site,
        "source_url": source_url,
        "price_original": price,
        "currency": currency,
        "price_jpy": round(price_jpy),
        "exchange_rate": exchange_rate,
        "in_stock": True,
        "size_available": "",
        "scraped_at": datetime.utcnow().isoformat(),
    }


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
            "user_agent": _CHROME_HEADERS["User-Agent"],
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

    # ------------------------------------------------------------------
    # Farfetch (Playwright + proxy)
    # ------------------------------------------------------------------
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
                            results.push({brand: brandName, name: productDesc || brandName, price, url: href});
                        }
                    } catch(e) {}
                }
                return results;
            })()
            """

            cards = self.page.evaluate(EXTRACT_JS) or []
            logger.info(f"[farfetch] Found {len(cards)} products for {brand}")

            for card in cards[:item_limit]:
                results.append(_make_item(
                    brand, card["name"], float(card["price"]),
                    "farfetch", card["url"], currency="JPY",
                ))

            logger.info(f"[farfetch] Extracted {len(results)} products for {brand}")
        except Exception as e:
            logger.error(f"[farfetch] Scraping failed: {e}")

        return results

    # ------------------------------------------------------------------
    # curl_cffi official sites (Gucci, Prada, Loewe, Balenciaga)
    # ------------------------------------------------------------------
    def _scrape_cffi_official(self, brand: str, item_limit: int = 10) -> list[dict]:
        site_map = _OFFICIAL_CFFI.get(brand, {})
        site_key, url = next(iter(site_map.items()), (None, None))
        if not url:
            return []

        html = _fetch_with_cffi(url)
        if not html:
            return []

        pattern = _CFFI_PATTERNS.get(brand)
        if not pattern:
            return []

        results = []
        try:
            matches = re.findall(pattern, html)
            logger.info(f"[{site_key}] Found {len(matches)} matches for {brand}")

            for match in matches[:item_limit]:
                if brand == "Balenciaga":
                    price = float(match)
                    price_pos = html.find(f'content="{match}"')
                    if price_pos == -1:
                        continue
                    ctx = html[max(0, price_pos - 800):price_pos]
                    name_match = re.findall(r'<h\d[^>]*>\s*([^<]+?)\s*</h\d>', ctx)
                    name = name_match[-1].strip() if name_match else f"Balenciaga Item {match}"
                    results.append(_make_item(brand, name, price, site_key, url, currency=_CURRENCY_MAP.get(brand, "JPY")))
                elif brand == "Loewe":
                    name, price_str = match
                    price = float(price_str.replace(",", ""))
                    if len(name) > 60 or "css-" in name or "{" in name:
                        continue
                    results.append(_make_item(brand, name, price, site_key, url, currency=_CURRENCY_MAP.get(brand, "JPY")))
                else:
                    name, price_str = match
                    # EUR uses dot as thousands separator (€ 2.450 = 2450)
                    price = float(price_str.replace(".", "").replace(",", ""))
                    results.append(_make_item(brand, name, price, site_key, url, currency=_CURRENCY_MAP.get(brand, "JPY")))

        except Exception as e:
            logger.error(f"[{site_key}] Extraction failed: {e}")

        logger.info(f"[{site_key}] Extracted {len(results)} products for {brand}")
        return results

    # ------------------------------------------------------------------
    # playwright-stealth official sites (Ferragamo, Bottega Veneta)
    # ------------------------------------------------------------------
    def _scrape_stealth_official(self, brand: str, item_limit: int = 10) -> list[dict]:
        site_map = _OFFICIAL_STEALTH.get(brand, {})
        site_key, url = next(iter(site_map.items()), (None, None))
        if not url:
            return []

        extract_js = _STEALTH_JS.get(brand)
        if not extract_js:
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

                logger.info(f"[{site_key}] Navigating to {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(8)

                for _ in range(10):
                    page.evaluate("window.scrollBy(0, 500)")
                    time.sleep(0.3)

                products = page.evaluate(extract_js) or []
                logger.info(f"[{site_key}] Found {len(products)} products for {brand}")

                for product in products[:item_limit]:
                    results.append(_make_item(
                        brand, product["name"], float(product["price"]),
                        site_key, url, currency=_CURRENCY_MAP.get(brand, "JPY"),
                    ))

                ctx.close()
                browser.close()

        except Exception as e:
            logger.error(f"[{site_key}] Scraping failed: {e}")

        logger.info(f"[{site_key}] Extracted {len(results)} products for {brand}")
        return results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def scrape(
        self, brand: str, sites: list[str] | None = None, item_limit: int = 10
    ) -> list[dict]:
        if sites is None:
            sites = SUPPORTED_SITES

        all_results: list[dict] = []

        # curl_cffi sites (fast, no browser)
        if brand in _OFFICIAL_CFFI:
            site_key = next(iter(_OFFICIAL_CFFI[brand]))
            if site_key in sites:
                results = self._scrape_cffi_official(brand, item_limit)
                all_results.extend(results)

        # playwright-stealth sites (JS rendering)
        if brand in _OFFICIAL_STEALTH:
            site_key = next(iter(_OFFICIAL_STEALTH[brand]))
            if site_key in sites:
                results = self._scrape_stealth_official(brand, item_limit)
                all_results.extend(results)

        # Farfetch (Playwright + proxy)
        if "farfetch" in sites and brand in _FARFETCH_BRAND_SLUGS:
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
