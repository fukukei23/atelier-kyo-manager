from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from app.utils.fx_utils import get_fx_table_jpy

logger = logging.getLogger(__name__)

SITES_DIR = Path(__file__).resolve().parents[1] / "config" / "sites"
PROXY_POOL_PATH = Path(__file__).resolve().parents[1] / "config" / "proxy_pool.json"

SUPPORTED_SITES = ["farfetch", "mytheresa", "nap", "24s", "lvr"]
SUPPORTED_BRANDS = ["Gucci", "Prada", "Ferragamo"]
DEFAULT_GENDER = "women"


def _load_site_config(site_key: str) -> dict[str, Any]:
    path = SITES_DIR / f"{site_key}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_proxies() -> list[dict]:
    if not PROXY_POOL_PATH.exists():
        return []
    with open(PROXY_POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    proxies = []
    for entries in pool.values():
        if isinstance(entries, list):
            for e in entries:
                if e.get("active", True) and e.get("server"):
                    proxies.append(e)
    return proxies


class BrandPriceScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.pw = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._fx_table: dict[str, float] = {}
        self._fx_meta: dict = {}

    def _init_browser(self, timeout_sec: int = 45):
        if self.page:
            return
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=self.headless)

        launch_options: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            "locale": "en-US",
            "timezone_id": "Europe/London",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        }

        proxies = _load_proxies()
        if proxies:
            proxy = random.choice(proxies)
            launch_options["proxy"] = {
                "server": proxy["server"],
                "username": proxy.get("username", ""),
                "password": proxy.get("password", ""),
            }

        self.context = self.browser.new_context(**launch_options)
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'ja-JP', 'ja']});
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

    def _ensure_fx(self):
        if not self._fx_table:
            self._fx_table, self._fx_meta = get_fx_table_jpy()

    def _to_jpy(self, amount: float, currency: str) -> tuple[float, float]:
        if currency == "JPY":
            return amount, 1.0
        self._ensure_fx()
        rate = self._fx_table.get(currency.upper(), 0)
        if rate <= 0:
            return 0.0, 0.0
        return round(amount * rate, 0), rate

    def _dismiss_cookies(self, site_config: dict):
        cookie_sel = site_config.get("selectors", {}).get("cookie_consent", "")
        if not cookie_sel:
            return
        for sel in cookie_sel.split(","):
            sel = sel.strip()
            try:
                self.page.click(sel, timeout=3000)
                break
            except Exception:
                continue

    def _scrape_site(
        self, brand: str, site_key: str, site_config: dict, item_limit: int = 10
    ) -> list[dict]:
        results = []
        currency = site_config.get("currency", "EUR")
        search_url = site_config.get("search_template", "").format(
            q=quote_plus(brand), gender=DEFAULT_GENDER
        )
        if not search_url:
            return results

        try:
            logger.info(f"[{site_key}] Navigating to {search_url}")
            self.page.goto(search_url, wait_until="domcontentloaded")
            time.sleep(random.uniform(2.0, 4.0))
            self._dismiss_cookies(site_config)

            results_sel = site_config.get("selectors", {}).get("results_item", "")
            if results_sel:
                try:
                    self.page.wait_for_selector(results_sel, timeout=10000)
                except Exception:
                    logger.warning(f"[{site_key}] No results found for {brand}")
                    return results

            # Scroll to load more items
            for _ in range(3):
                self.page.evaluate("window.scrollBy(0, 800)")
                time.sleep(random.uniform(0.5, 1.0))

            # Extract product cards via JS
            link_sel = site_config.get("selectors", {}).get("plp_tile_link", "a")
            container_sel = site_config.get("selectors", {}).get("plp_tile_container", "")
            title_selectors = site_config.get("selectors", {}).get("pdp", {}).get("title", [])
            price_selectors = site_config.get("selectors", {}).get("pdp", {}).get("price", [])

            cards = self._extract_cards_js(container_sel, link_sel, title_selectors, price_selectors, currency)

            for card in cards[:item_limit]:
                price_jpy, rate = self._to_jpy(card["price_original"], card["currency"])
                if price_jpy <= 0:
                    continue
                results.append({
                    "brand": brand,
                    "product_name": card.get("name", ""),
                    "source_site": site_key,
                    "source_url": card.get("url", ""),
                    "price_original": card["price_original"],
                    "currency": card["currency"],
                    "price_jpy": price_jpy,
                    "exchange_rate": rate,
                    "in_stock": True,
                    "size_available": card.get("sizes", ""),
                    "scraped_at": datetime.utcnow().isoformat(),
                })

            logger.info(f"[{site_key}] Extracted {len(results)} products for {brand}")
        except Exception as e:
            logger.error(f"[{site_key}] Scraping failed: {e}")

        return results

    def _extract_cards_js(
        self,
        container_sel: str,
        link_sel: str,
        title_selectors: list[str],
        price_selectors: list[str],
        currency: str,
    ) -> list[dict]:
        price_sels_js = json.dumps(price_selectors)
        title_sels_js = json.dumps(title_selectors)
        script = f"""
        (() => {{
            const cards = [];
            const containers = document.querySelectorAll({container_sel!r});
            const priceSels = {price_sels_js};
            const titleSels = {title_sels_js};

            containers.forEach(c => {{
                try {{
                    const link = c.querySelector({link_sel!r});
                    if (!link) return;
                    const url = link.href || '';
                    let name = '';
                    for (const s of titleSels) {{
                        const el = c.querySelector(s) || link.querySelector(s);
                        if (el) {{ name = el.innerText.trim(); break; }}
                    }}
                    if (!name) name = link.innerText.trim().split('\\n')[0];

                    let price = null;
                    for (const s of priceSels) {{
                        const el = c.querySelector(s) || link.querySelector(s);
                        if (el) {{
                            const txt = el.innerText || el.textContent || '';
                            const m = txt.match(/[\\d,]+(?:\\.\\d+)?/);
                            if (m) {{ price = parseFloat(m[0].replace(/,/g, '')); break; }}
                        }}
                    }}
                    if (price) {{
                        cards.push({{name: name, url: url, price_original: price, currency: {currency!r}, sizes: ''}});
                    }}
                }} catch(e) {{}}
            }});
            return cards;
        }})()
        """
        try:
            return self.page.evaluate(script) or []
        except Exception as e:
            logger.warning(f"JS card extraction failed: {e}")
            return []

    def scrape(
        self, brand: str, sites: list[str] | None = None, item_limit: int = 10
    ) -> list[dict]:
        if sites is None:
            sites = SUPPORTED_SITES

        all_results = []
        self._ensure_fx()
        try:
            self._init_browser()
            for site_key in sites:
                if site_key not in SUPPORTED_SITES:
                    continue
                try:
                    site_config = _load_site_config(site_key)
                except FileNotFoundError:
                    logger.warning(f"Site config not found: {site_key}")
                    continue

                results = self._scrape_site(brand, site_key, site_config, item_limit)
                all_results.extend(results)

                if site_key != sites[-1]:
                    time.sleep(random.uniform(3.0, 5.0))

            logger.info(f"Total scraped: {len(all_results)} products for {brand}")
            return all_results
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return all_results
        finally:
            self._close_browser()
