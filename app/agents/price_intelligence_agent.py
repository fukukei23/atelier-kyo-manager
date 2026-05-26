# ==============================================================================
# ファイル名 (File Name): price_intelligence_agent.py
# レジストリ (Registry): app/agents/price_intelligence_agent.py
# 更新日時 (Date & Time JST): 2026-05-10
# バージョン (Version): 14.0.0J (Playwright Migration)
#
# --- v14.0.0Jでの主な変更点 (What's New in v14.0.0J) ---
# - [Playwright移行] Selenium → Playwright sync API に移行
# - selenium / selenium-stealth / webdriver-manager 依存を除去
# - chromium.launch() + new_context() パターンに統一
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from app.extractors.product_info_extractor import extract_product_info


USER_DATA_DIR = Path(__file__).resolve().parents[2] / "instance" / "pw_profile" / "price_intelligence"


class PriceIntelligenceAgent:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.pw = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

        base_dir = Path(__file__).resolve().parents[2]
        self.screenshot_dir = base_dir / "instance" / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _init_driver(self, timeout_sec: int):
        if self.page:
            return

        logging.info(f"Initializing Playwright chromium with {timeout_sec}s timeout...")

        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja', 'en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)
        self.page = self.context.new_page()
        self.page.set_default_timeout(timeout_sec * 1000)

    def _quit_driver(self):
        for resource in [self.context, self.browser]:
            if resource:
                try:
                    resource.close()
                except Exception as e:
                    logging.error(f"Error closing resource: {e}")
        if self.pw:
            try:
                self.pw.stop()
            except Exception as e:
                logging.error(f"Error stopping playwright: {e}")
        self.page = None
        self.context = None
        self.browser = None
        self.pw = None

    def _save_failure_screenshot(self, context_name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = self.screenshot_dir / f"failure_{context_name}_{ts}.png"
        try:
            self.page.screenshot(path=str(fp))
            logging.error(f"Saved failure screenshot to: {fp}")
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")

    def _perform_buyma_search(self, brand_name: str, site_config: dict[str, Any]) -> bool:
        try:
            logging.info(f"Navigating to BUYMA home page: {site_config['home_url']}")
            self.page.goto(site_config["home_url"])

            if cookie_selector := site_config.get("cookie_accept_selector"):
                with contextlib.suppress(Exception):
                    self.page.click(cookie_selector, timeout=3000)

            search_input = None
            for selector in site_config.get("selectors", {}).get("search_input_candidates", []):
                try:
                    self.page.wait_for_selector(selector, timeout=5000)
                    search_input = selector
                    break
                except Exception:
                    continue

            if not search_input:
                raise RuntimeError("Could not find any known search input field.")

            self.page.fill(search_input, "")
            for char in brand_name:
                self.page.type(search_input, char, delay=random.randint(50, 150))
            self.page.keyboard.press("Enter")

            self.page.wait_for_selector(site_config["selectors"]["results_item"])
            return True
        except Exception:
            logging.warning("Human-like search failed, falling back to direct search URL.")
            try:
                search_url = site_config["search_template"].format(q=quote_plus(brand_name))
                self.page.goto(search_url)
                self.page.wait_for_selector(site_config["selectors"]["results_item"])
                return True
            except Exception as e:
                logging.error(f"All search attempts for BUYMA failed: {e}")
                self._save_failure_screenshot(f"buyma_search_failed_{brand_name}")
                return False

    def _js_collect_item_anchors_buyma(self) -> list[dict]:
        script = """
        const out = []; const anchors = Array.from(document.querySelectorAll("a[href*='/item/']"));
        const yenRe = /[¥\\u00A5]\\s*([\\d,]+)|([\\d,]+)\\s*円/;
        for (const a of anchors){ try{ let root = a.closest("li, .product_HorizontalList_box"); if (!root) continue;
            const href = a.getAttribute('href'); const url = href && href.startsWith('http') ? href : (location.origin + (href || ""));
            const name = root.querySelector('.product_name')?.innerText.trim() || 'N/A';
            const priceText = root.querySelector('.product_price')?.innerText || '';
            const m = priceText.match(yenRe);
            out.push({ url, name, buyma_price_list_view: m ? parseInt((m[1] || m[2]).replace(/,/g,'')) : null });
        }catch(e){}} return out;
        """
        return self.page.evaluate(script)

    def _extract_from_pdp(self, site_config: dict[str, Any]) -> dict[str, Any]:
        html = self.page.content()
        try:
            data = extract_product_info(html, site_config=site_config)
            if data.get("price"):
                return data
        except Exception as e:
            logging.warning(f"Python extractor failed: {e}.")
        return {}

    def run(
        self, brand_name: str, item_limit: int, site_config: dict[str, Any], browser_name: str = "chrome"
    ) -> list[dict]:
        timeout_sec = site_config.get("discovery_settings", {}).get("timeout_sec", 20)

        try:
            self._init_driver(timeout_sec)
            if not self._perform_buyma_search(brand_name, site_config):
                return []

            anchors, seen = [], set()
            for i in range(6):
                self.page.evaluate("window.scrollBy(0, 900)")
                time.sleep(random.uniform(0.8, 1.2))
                batch = self._js_collect_item_anchors_buyma()
                new_items = 0
                for item in batch:
                    if (url := item.get("url")) and url not in seen:
                        seen.add(url)
                        anchors.append(item)
                        new_items += 1
                if len(anchors) >= item_limit or (new_items == 0 and i > 1):
                    break
            if not anchors:
                return []

            results = []
            for card in anchors[:item_limit]:
                pdp_data = {}
                try:
                    self.page.goto(card["url"])
                    time.sleep(random.uniform(1.4, 2.4))
                    pdp_data = self._extract_from_pdp(site_config)
                except Exception as e:
                    logging.warning(f"Failed to process PDP {card['url']}: {e}")

                if pdp_data and pdp_data.get("price"):
                    results.append(
                        {
                            "name": pdp_data.get("title") or card.get("name"),
                            "buyma_url": card["url"],
                            "buyma_price": pdp_data.get("price"),
                            "buyma_list_price": pdp_data.get("list_price"),
                            "buyma_discount_pct": pdp_data.get("discount_pct"),
                            "buyma_sale_until": pdp_data.get("sale_until"),
                        }
                    )
                elif card.get("buyma_price_list_view"):
                    results.append(
                        {
                            "name": card.get("name"),
                            "buyma_url": card["url"],
                            "buyma_price": card.get("buyma_price_list_view"),
                        }
                    )

            timestamp = datetime.now().isoformat()
            for res in results:
                res["captured_at"] = timestamp
                res["brand"] = brand_name

            logging.info(f"Intelligence extracted for {len(results)} products from BUYMA.")
            return results
        except Exception as e:
            logging.exception(f"A critical error occurred in PriceIntelligenceAgent: {e}")
            return []
        finally:
            self._quit_driver()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price Intelligence Agent for BUYMA (v14.0J)")
    parser.add_argument("brand", help="The brand name to research on BUYMA")
    parser.add_argument("--items", type=int, default=3)
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()

    try:
        from app.config.config import Config

        buyma_config = Config.SITES.get("BUYMA")
        if not buyma_config:
            raise RuntimeError("BUYMA config not found.")
    except Exception as e:
        logging.fatal(f"Could not load config for standalone test: {e}")
        exit(1)

    agent = PriceIntelligenceAgent(headless=not args.headful)
    results = agent.run(brand_name=args.brand, item_limit=args.items, site_config=buyma_config)
    print("\n--- Price Intelligence Agent Results ---")
    print(json.dumps(results, ensure_ascii=False, indent=2))
