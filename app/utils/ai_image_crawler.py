# ======================================================================
# ファイル名: app/utils/ai_image_crawler.py
# 役割:
#   - Selenium で対象サイト（config.pyの定義）を検索
#   - 商品ページから画像URLを収集
#   - 自己テスト用エントリーポイントを搭載
#   - 自己テスト結果を PNG 画像として保存（コンタクトシート + メタ情報）
# ======================================================================

import os
import io
import json
import math
import time
import logging
import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Flask 経由でなく単体実行も可能にするための config 読み込み
try:
    from config import Config
except ImportError:
    raise ImportError("config.py が見つかりません。プロジェクトルートに config.py を配置してください。")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Crawler
# =========================
class CrawlerService:
    """対象サイトから商品画像URLを収集するクローラー"""

    def __init__(self, sites_config: Dict[str, Dict[str, Any]], headless: bool = True, wait_time: int = 25):
        self.sites_config = sites_config
        self.headless = headless
        self.wait_time = wait_time
        self.logger = logging.getLogger(self.__class__.__name__)

    def _init_driver(self) -> WebDriver:
        """Chrome WebDriver の初期化"""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1200,1600")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(max(30, self.wait_time + 5))
        return driver

    def _accept_cookie_if_any(self, driver: WebDriver, selector: Optional[str]) -> None:
        if not selector:
            return
        try:
            btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            btn.click()
            self.logger.info("Cookie consent accepted.")
        except Exception:
            self.logger.info("No cookie consent (or not clickable).")

    def _extract_images_from_jsonld(self, json_text: str) -> List[str]:
        """JSON-LD（<script type='application/ld+json'>）から image を抽出"""
        out: List[str] = []
        try:
            data = json.loads(json_text)
        except Exception:
            return out

        # Product 直下 or @graph 配下に対応
        def _normalize_images(img_field) -> List[str]:
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
            # まれに配列で来る
            for node in data:
                if isinstance(node, dict) and node.get("@type") in ("Product", ["Product"]):
                    images = _normalize_images(node.get("image"))
                    if images:
                        break

        return [u for u in images if isinstance(u, str)]

    def search_and_collect_images(self, site_key: str, query: str, max_results: int = 6) -> List[str]:
        """指定サイトで検索し、商品ページから画像URLを取得"""
        if site_key not in self.sites_config:
            raise ValueError(f"Site '{site_key}' is not defined in configuration.")

        site_conf = self.sites_config[site_key]
        search_url = site_conf["search_url_template"].format(query=quote_plus(query))

        driver = self._init_driver()
        self.logger.info(f"Open search URL: {search_url}")
        driver.get(search_url)
        self._accept_cookie_if_any(driver, site_conf.get("cookie_accept_selector"))

        # 検索結果リンク要素の取得
        try:
            result_links = WebDriverWait(driver, self.wait_time).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, site_conf["search_result_link_selector"])
                )
            )
        except TimeoutException:
            self.logger.error("Search results not found within wait time.")
            driver.quit()
            return []

        product_links = []
        for link_elem in result_links[: max(1, max_results * 3)]:  # 多少多めに拾っておく
            href = link_elem.get_attribute("href")
            if href:
                product_links.append(href)

        self.logger.info(f"Candidate product links: {len(product_links)}")

        # 各商品ページから JSON-LD を読み取って画像を集める
        image_urls: List[str] = []
        for href in product_links[:max_results]:
            try:
                driver.execute_script("window.open(arguments[0], '_blank');", href)
                driver.switch_to.window(driver.window_handles[-1])
                self._accept_cookie_if_any(driver, site_conf.get("cookie_accept_selector"))

                # JSON-LD をすべて拾う（複数あるケースあり）
                scripts = driver.find_elements(By.CSS_SELECTOR, site_conf["structured_data_selector"])
                got = False
                for sc in scripts:
                    text = sc.get_attribute("innerText") or sc.get_attribute("innerHTML")
                    urls = self._extract_images_from_jsonld(text or "")
                    if urls:
                        image_urls.extend(urls)
                        got = True
                if not got:
                    self.logger.warning(f"No JSON-LD images on: {href}")

            except Exception as e:
                self.logger.warning(f"Fail on product page {href}: {e}")
            finally:
                # タブを閉じて検索結果に戻る
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

        driver.quit()

        # 重複除去
        deduped = list(dict.fromkeys(image_urls))
        self.logger.info(f"Collected image URLs: {len(deduped)}")
        return deduped


# =========================
# Self-test Report (PNG)
# =========================
def _fetch_image(url: str, timeout: int = 20, max_bytes: int = 5_000_000) -> Optional[Image.Image]:
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


def _text_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """簡易テキスト折り返し"""
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
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
    image_urls: List[str],
    thumb_cols: int = 3,
    thumb_size: int = 256,
) -> Path:
    """
    収集結果を 1 枚の PNG にまとめて保存（メタ情報 + サムネイル格子）
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # メタ情報描画用フォント（非依存でOKなデフォルト）
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_text = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # まず上部にメタ情報、その下にグリッド
    padding = 24
    meta_height = 220  # タイトル & URL一覧領域（必要に応じて増減）
    bg = Image.new("RGB", (thumb_cols * (thumb_size + padding) + padding, meta_height), "white")
    d = ImageDraw.Draw(bg)

    # タイトル
    title = "Crawler Self Test Report"
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d.text((padding, padding), title, fill="black", font=font_title)
    d.text((padding, padding + 44), f"Site: {site}", fill="black", font=font_text)
    d.text((padding, padding + 70), f"Query: {query}", fill="black", font=font_text)
    d.text((padding, padding + 96), f"Collected: {len(image_urls)} images", fill="black", font=font_text)
    d.text((padding, padding + 122), f"Generated at: {dt}", fill="black", font=font_text)

    # URL の一部を表示（長すぎると溢れるので3件まで）
    max_show = 3
    y = padding + 150
    w = bg.size[0] - padding * 2
    if image_urls:
        for i, u in enumerate(image_urls[:max_show], 1):
            # 1行に収まらない時は折返し
            lines = _text_wrap(d, f"{i}. {u}", font_small, w)
            for line in lines:
                d.text((padding, y), line, fill="gray20", font=font_small)
                y += 18
    else:
        d.text((padding, y), "(No image URLs were collected)", fill="gray30", font=font_small)

    # 画像を最大 9 枚（3x3 デフォルト）まで並べる
    max_thumbs = thumb_cols * thumb_cols
    grid_urls = image_urls[:max_thumbs]
    rows = math.ceil(len(grid_urls) / thumb_cols)
    grid_height = rows * (thumb_size + padding) + padding if rows > 0 else padding

    canvas = Image.new("RGB", (bg.size[0], bg.size[1] + grid_height), "white")
    canvas.paste(bg, (0, 0))
    d2 = ImageDraw.Draw(canvas)

    # サムネイル描画
    x0, y0 = padding, bg.size[1] + padding
    for idx, url in enumerate(grid_urls):
        col = idx % thumb_cols
        row = idx // thumb_cols
        x = padding + col * (thumb_size + padding)
        y = y0 + row * (thumb_size + padding)

        thumb = _fetch_image(url)
        if thumb is None:
            # 取得失敗時は灰色のプレースホルダ
            ph = Image.new("RGB", (thumb_size, thumb_size), "#e6e6e6")
            dph = ImageDraw.Draw(ph)
            msg = "Failed\nto load"
            wmsg, hmsg = dph.textlength("Failed", font=font_small), 2 * 18
            dph.multiline_text(((thumb_size - wmsg) / 2 - 6, (thumb_size - hmsg) / 2), msg, fill="gray40", font=font_small, align="center")
            thumb = ph
        else:
            # 余白あり縮小（サムネイル化）
            thumb.thumbnail((thumb_size, thumb_size))

            # 正方形カンバスに中央寄せ
            sq = Image.new("RGB", (thumb_size, thumb_size), "white")
            ox = (thumb_size - thumb.size[0]) // 2
            oy = (thumb_size - thumb.size[1]) // 2
            sq.paste(thumb, (ox, oy))
            thumb = sq

        canvas.paste(thumb, (x, y))
        # 枠線
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
            thumb_cols=3,       # 3x3 グリッド
            thumb_size=256,
        )
        logging.info(f"[SELF-TEST] Report saved: {outpath.resolve()}")
    except Exception as e:
        logging.exception(f"Failed to save self-test report: {e}")
        # 画像がゼロの場合など、最低限のテキストだけでも PNG にして残す
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

    # プロセス終了コード: 画像 0件でも成功扱い（テスト容易性のため）
    logging.info("Self-test finished.")
