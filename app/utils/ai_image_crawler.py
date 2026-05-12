# ======================================================================
# ファイル名: app/utils/ai_image_crawler.py
# 役割:
#   - Playwright で対象サイト（config.pyの定義）を検索
#   - 商品ページから画像URLを収集
# ======================================================================

import json
import logging
from typing import Any
from urllib.parse import quote_plus

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CrawlerService:
    """対象サイトから商品画像URLを収集するクローラー"""

    def __init__(self, sites_config: dict[str, dict[str, Any]], headless: bool = True, wait_time: int = 25):
        self.sites_config = sites_config
        self.headless = headless
        self.wait_time = wait_time
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pw = None
        self._browser: Browser | None = None

    def _ensure_browser(self) -> Browser:
        if self._browser is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=self.headless,
                args=["--disable-gpu", "--no-sandbox"],
            )
        return self._browser

    def _new_context(self) -> BrowserContext:
        browser = self._ensure_browser()
        return browser.new_context(
            viewport={"width": 1200, "height": 1600},
        )

    def close(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw:
            self._pw.stop()
            self._pw = None

    def _accept_cookie_if_any(self, page: Page, selector: str | None) -> None:
        if not selector:
            return
        try:
            locator = page.locator(selector)
            locator.first.click(timeout=5000)
            self.logger.info("Cookie consent accepted.")
        except Exception:
            self.logger.info("No cookie consent (or not clickable).")

    def _extract_images_from_jsonld(self, json_text: str) -> list[str]:
        """JSON-LD（<script type='application/ld+json'>）から image を抽出"""
        out: list[str] = []
        try:
            data = json.loads(json_text)
        except Exception:
            return out

        def _normalize_images(img_field) -> list[str]:
            if not img_field:
                return []
            if isinstance(img_field, str):
                return [img_field]
            if isinstance(img_field, list):
                urls = []
                for it in img_field:
                    if isinstance(it, str):
                        urls.append(it)
                    elif isinstance(it, dict):
                        if it.get("contentUrl"):
                            urls.append(it["contentUrl"])
                        elif "@id" in it:
                            urls.append(it["@id"])
                return urls
            if isinstance(img_field, dict):
                if img_field.get("contentUrl"):
                    return [img_field["contentUrl"]]
                if "@id" in img_field:
                    return [img_field["@id"]]
            return []

        images = []
        if isinstance(data, dict):
            if "image" in data:
                images = _normalize_images(data.get("image"))
            elif "@graph" in data and isinstance(data["@graph"], list):
                for node in data["@graph"]:
                    if not isinstance(node, dict):
                        continue
                    if node.get("@type") in ("Product", ["Product"]):
                        images = _normalize_images(node.get("image"))
                        if images:
                            break
        elif isinstance(data, list):
            for node in data:
                if isinstance(node, dict) and node.get("@type") in ("Product", ["Product"]):
                    images = _normalize_images(node.get("image"))
                    if images:
                        break

        return [u for u in images if isinstance(u, str)]

    def search_and_collect_images(self, site_key: str, query: str, max_results: int = 6) -> list[str]:
        """指定サイトで検索し、商品ページから画像URLを取得"""
        if site_key not in self.sites_config:
            raise ValueError(f"Site '{site_key}' is not defined in configuration.")

        site_conf = self.sites_config[site_key]
        search_url = site_conf["search_url_template"].format(query=quote_plus(query))

        context = self._new_context()
        page = context.new_page()
        page.set_default_timeout((max(30, self.wait_time + 5)) * 1000)

        try:
            self.logger.info(f"Open search URL: {search_url}")
            page.goto(search_url, wait_until="domcontentloaded")
            self._accept_cookie_if_any(page, site_conf.get("cookie_accept_selector"))

            # 検索結果リンク要素の取得
            try:
                locator = page.locator(site_conf["search_result_link_selector"])
                locator.first.wait_for(timeout=self.wait_time * 1000)
                link_elements = locator.all()
            except Exception:
                self.logger.error("Search results not found within wait time.")
                return []

            product_links = []
            for link_elem in link_elements[: max(1, max_results * 3)]:
                href = link_elem.get_attribute("href")
                if href:
                    product_links.append(href)

            self.logger.info(f"Candidate product links: {len(product_links)}")

            # 各商品ページから JSON-LD を読み取って画像を集める
            image_urls: list[str] = []
            for href in product_links[:max_results]:
                product_page = context.new_page()
                try:
                    product_page.goto(href, wait_until="domcontentloaded")
                    self._accept_cookie_if_any(product_page, site_conf.get("cookie_accept_selector"))

                    scripts = product_page.locator(site_conf["structured_data_selector"]).all()
                    got = False
                    for sc in scripts:
                        text = sc.inner_text()
                        urls = self._extract_images_from_jsonld(text or "")
                        if urls:
                            image_urls.extend(urls)
                            got = True
                    if not got:
                        self.logger.warning(f"No JSON-LD images on: {href}")

                except Exception as e:
                    self.logger.warning(f"Fail on product page {href}: {e}")
                finally:
                    product_page.close()

            # 重複除去
            deduped = list(dict.fromkeys(image_urls))
            self.logger.info(f"Collected image URLs: {len(deduped)}")
            return deduped
        finally:
            context.close()
