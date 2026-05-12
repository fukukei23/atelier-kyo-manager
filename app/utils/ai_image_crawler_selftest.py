# ======================================================================
# ファイル名: app/utils/ai_image_crawler_selftest.py
# 役割:
#   - CrawlerService の自己テストエントリーポイント
#   - テスト結果を PNG コンタクトシートとして保存
# ======================================================================

import datetime
import io
import logging
import math
import os
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from app.config.config import Config
from app.utils.ai_image_crawler import CrawlerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    """収集結果を 1 枚の PNG にまとめて保存（メタ情報 + サムネイル格子）"""
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
