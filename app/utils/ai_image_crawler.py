# ======================================================================
# ファイル名: app/utils/ai_image_crawler.py
# 役割:
#   - Playwright で対象サイト（config.pyの定義）を検索
#   - 商品ページから画像URLを収集
#   - 自己テスト用エントリーポイントを搭載
#   - 自己テスト結果を PNG 画像として保存（コンタクトシート + メタ情報）
# ======================================================================

import datetime
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from app.config.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =========================
# Crawler
# =========================
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


# =========================
# Self-test Report (PNG)
# =========================
def _fetch_image(url: str, timeout: int = 20, max_bytes: int = 5_000_000) -> Image.Image | None:
    """URL から画像を取得（簡易サイズ制限つき）"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content if len(r.content) <= max_bytes else r.raw.read(max_bytes)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return img
    except Exception:
        return None


def _text_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """簡易テキスト折り返し"""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if draw.textlength(cur + " " + w, font=font) <= max_width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def save_selftest_report_png(
    outfile: Path,
    *,
    site: str,
    query: str,
    image_urls: list[str],
    thumb_cols: int = 3,
    thumb_size: int = 256,
) -> Path:
    """
    収集結果を 1 枚の PNG にまとめて保存（メタ情報 + サムネイル格子）
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_text = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    padding = 24
    meta_height = 220
    bg = Image.new("RGB", (thumb_cols * (thumb_size + padding) + padding, meta_height), "white")
    d = ImageDraw.Draw(bg)

    title = "Crawler Self Test Report"
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d.text((padding, padding), title, fill="black", font=font_title)
    d.text((padding, padding + 44), f"Site: {site}", fill="black", font=font_text)
    d.text((padding, padding + 70), f"Query: {query}", fill="black", font=font_text)
    d.text((padding, padding + 96), f"Collected: {len(image_urls)} images", fill="black", font=font_text)
    d.text((padding, padding + 122), f"Generated at: {dt}", fill="black", font=font_text)

    max_show = 3
    y = padding + 150
    w = bg.size[0] - padding * 2
    if image_urls:
        for i, u in enumerate(image_urls[:max_show], 1):
            lines = _text_wrap(d, f"{i}. {u}", font_small, w)
            for line in lines:
                d.text((padding, y), line, fill="gray20", font=font_small)
                y += 18
    else:
        d.text((padding, y), "(No image URLs were collected)", fill="gray30", font=font_small)

    max_thumbs = thumb_cols * thumb_cols
    grid_urls = image_urls[:max_thumbs]
    rows = math.ceil(len(grid_urls) / thumb_cols)
    grid_height = rows * (thumb_size + padding) + padding if rows > 0 else padding

    canvas = Image.new("RGB", (bg.size[0], bg.size[1] + grid_height), "white")
    canvas.paste(bg, (0, 0))
    d2 = ImageDraw.Draw(canvas)

    _x0, y0 = padding, bg.size[1] + padding
    for idx, url in enumerate(grid_urls):
        col = idx % thumb_cols
        row = idx // thumb_cols
        x = padding + col * (thumb_size + padding)
        y = y0 + row * (thumb_size + padding)

        thumb = _fetch_image(url)
        if thumb is None:
            ph = Image.new("RGB", (thumb_size, thumb_size), "#e6e6e6")
            dph = ImageDraw.Draw(ph)
            msg = "Failed\nto load"
            wmsg, hmsg = dph.textlength("Failed", font=font_small), 2 * 18
            dph.multiline_text(
                ((thumb_size - wmsg) / 2 - 6, (thumb_size - hmsg) / 2),
                msg,
                fill="gray40",
                font=font_small,
                align="center",
            )
            thumb = ph
        else:
            thumb.thumbnail((thumb_size, thumb_size))
            sq = Image.new("RGB", (thumb_size, thumb_size), "white")
            ox = (thumb_size - thumb.size[0]) // 2
            oy = (thumb_size - thumb.size[1]) // 2
            sq.paste(thumb, (ox, oy))
            thumb = sq

        canvas.paste(thumb, (x, y))
        d2.rectangle([x, y, x + thumb_size, y + thumb_size], outline="lightgray", width=1)

    canvas.save(outfile, "PNG")
    return outfile


# =========================
# 自己テスト用エントリーポイント
# =========================
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)
    sites = app.config.get("CRAWLER_TARGET_SITES", {})

    if not sites:
        logging.error("No sites configuration found. Check config.py or crawler_sites.json.")
        raise SystemExit(1)

    test_query = os.getenv("CRAWLER_TEST_QUERY", "バッグ")
    test_site = os.getenv("CRAWLER_TEST_SITE", "farfetch")
    max_results = int(os.getenv("CRAWLER_TEST_MAX", "6"))

    crawler = CrawlerService(
        sites_config=sites,
        headless=Config.CRAWLER_HEADLESS,
        wait_time=Config.CRAWLER_DEFAULT_WAIT_TIME,
    )

    logging.info(f"[SELF-TEST] site='{test_site}', query='{test_query}', max={max_results}")
    t0 = time.time()
    try:
        urls = crawler.search_and_collect_images(site_key=test_site, query=test_query, max_results=max_results)
    except Exception as e:
        logging.exception(f"Self-test failed: {e}")
        urls = []
    finally:
        crawler.close()
    t1 = time.time()

    logging.info(f"Collected {len(urls)} image URLs in {t1 - t0:.2f}s")
    for u in urls[:10]:
        logging.info(f"  - {u}")
    if len(urls) > 10:
        logging.info(f"  ... (+{len(urls) - 10} more)")

    # 画像レポートを保存
    report_dir = Path("selftest_reports")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"selftest_{test_site}_{ts}.png"

    try:
        outpath = save_selftest_report_png(
            report_file,
            site=test_site,
            query=test_query,
            image_urls=urls,
            thumb_cols=3,
            thumb_size=256,
        )
        logging.info(f"[SELF-TEST] Report saved: {outpath.resolve()}")
    except Exception as e:
        logging.exception(f"Failed to save self-test report: {e}")
        try:
            fallback = Image.new("RGB", (1000, 360), "white")
            d = ImageDraw.Draw(fallback)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
                title = ImageFont.truetype("arial.ttf", 26)
            except Exception:
                font = ImageFont.load_default()
                title = ImageFont.load_default()
            d.text((24, 24), "Crawler Self Test Report (Fallback)", fill="black", font=title)
            d.text((24, 70), f"Site: {test_site}", fill="black", font=font)
            d.text((24, 94), f"Query: {test_query}", fill="black", font=font)
            d.text((24, 118), f"Collected: {len(urls)} images", fill="black", font=font)
            d.text((24, 142), "No image grid generated due to an error.", fill="gray35", font=font)
            report_dir.mkdir(parents=True, exist_ok=True)
            fallback.save(report_file, "PNG")
            logging.info(f"[SELF-TEST] Fallback report saved: {report_file.resolve()}")
        except Exception as ee:
            logging.exception(f"Fallback image save also failed: {ee}")
            raise

    logging.info("Self-test finished.")
