# ==============================================================================
# ファイル名 (File Name): price_intelligence_agent.py
# レジストリ (Registry): app/agents/price_intelligence_agent.py
# 更新日時 (Date & Time JST): 2025-09-19 22:11:00
# バージョン (Version): 13.0.0J (Config-Driven Timeout)
#
# --- v13.0.0Jでの主な変更点 (What's New in v13.0.0J) ---
# - [設定駆動タイムアウト] `run` メソッドが `site_config` を受け取り、
#   SeleniumのWebDriverWaitの待機時間を `timeout_sec` (デフォルト20秒) に
#   基づいて動的に設定するように変更。
# - [即時撤退思想の導入] これにより、Orchestratorから渡される「即時撤退」
#   ポリシーに準拠し、応答の遅いサイトでの無駄な待機を削減します。
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

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium_stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

try:
    from app.extractors.product_info_extractor import extract_product_info

    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    logging.error("Fatal: product_info_extractor.pyが見つかりません。")

    def extract_product_info(html: str, site_config: dict) -> dict:
        return {}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PriceIntelligenceAgent:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver: webdriver.Chrome | webdriver.Edge | None = None
        self.wait: WebDriverWait | None = None

        base_dir = Path(__file__).resolve().parents[2]
        self.screenshot_dir = base_dir / "instance" / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _init_driver(self, browser_name: str, timeout_sec: int):
        if self.driver:
            return

        logging.info(f"Initializing {browser_name.capitalize()} driver with {timeout_sec}s timeout...")

        if browser_name == "chrome":
            options = webdriver.ChromeOptions()
            profile_path = Path.home() / "AppData/Local/Google/Chrome/SeleniumProfile"
            options.add_argument(f"--user-data-dir={profile_path}")
            options.add_argument("--profile-directory=Default")
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            try:
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except Exception as e:
                logging.error(f"Failed to initialize Chrome Driver: {e}", exc_info=True)
                raise
        elif browser_name == "edge":
            options = webdriver.EdgeOptions()
            profile_path = Path.home() / "AppData/Local/Microsoft/Edge/SeleniumProfile"
            options.add_argument(f"user-data-dir={profile_path}")
            options.add_argument("profile-directory=Default")
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            try:
                service = EdgeService(EdgeChromiumDriverManager().install())
                self.driver = webdriver.Edge(service=service, options=options)
            except Exception as e:
                logging.error(f"Failed to initialize Edge Driver: {e}", exc_info=True)
                raise
        else:
            raise ValueError(f"Unsupported browser: {browser_name}")

        stealth(self.driver, languages=["ja-JP", "ja"], vendor="Google Inc.", platform="Win32")
        self.wait = WebDriverWait(self.driver, timeout_sec)

    def _quit_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logging.error(f"Error quitting driver: {e}")
            finally:
                self.driver = None

    def _save_failure_screenshot(self, context_name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = self.screenshot_dir / f"failure_{context_name}_{ts}.png"
        try:
            self.driver.save_screenshot(str(fp))
            logging.error(f"Saved failure screenshot to: {fp}")
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")

    def _perform_buyma_search(self, brand_name: str, site_config: dict[str, Any]) -> bool:
        try:
            logging.info(f"Navigating to BUYMA home page: {site_config['home_url']}")
            self.driver.get(site_config["home_url"])

            if cookie_selector := site_config.get("cookie_accept_selector"):
                with contextlib.suppress(TimeoutException):
                    self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_selector))).click()

            search_input = None
            for selector in site_config.get("selectors", {}).get("search_input_candidates", []):
                try:
                    search_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    if search_input:
                        break
                except TimeoutException:
                    continue

            if not search_input:
                raise TimeoutException("Could not find any known search input field.")

            search_input.clear()
            for char in brand_name:
                search_input.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
            search_input.send_keys(Keys.RETURN)

            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, site_config["selectors"]["results_item"])))
            return True
        except Exception:
            logging.warning("Human-like search failed, falling back to direct search URL.")
            try:
                search_url = site_config["search_template"].format(q=quote_plus(brand_name))
                self.driver.get(search_url)
                self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, site_config["selectors"]["results_item"]))
                )
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
        return self.driver.execute_script(script)

    def _extract_from_pdp(self, site_config: dict[str, Any]) -> dict[str, Any]:
        html = self.driver.page_source
        if EXTRACTOR_AVAILABLE:
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
        if not EXTRACTOR_AVAILABLE:
            return []

        timeout_sec = site_config.get("discovery_settings", {}).get("timeout_sec", 20)

        try:
            self._init_driver(browser_name, timeout_sec)
            if not self._perform_buyma_search(brand_name, site_config):
                return []

            anchors, seen = [], set()
            for i in range(6):
                self.driver.execute_script("window.scrollBy(0, 900);")
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
                    self.driver.get(card["url"])
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
    parser = argparse.ArgumentParser(description="Price Intelligence Agent for BUYMA (v13.0J)")
    parser.add_argument("brand", help="The brand name to research on BUYMA")
    parser.add_argument("--items", type=int, default=3)
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome", help="Browser to use")
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
    results = agent.run(
        brand_name=args.brand, item_limit=args.items, site_config=buyma_config, browser_name=args.browser
    )
    print("\n--- Price Intelligence Agent Results ---")
    print(json.dumps(results, ensure_ascii=False, indent=2))
