"""Phase -1 検証B: メルカリ安値出品の滞留時間 実測モニタ（使い捨て）.

why: 「相場より安い出品が人間に買われる前に生き残る時間」を実測し、
設計書 §7 検証B の通過判定（購入間に合い率・月間粗利）の材料にする。

方式:
  1) 発見: 各キーワードを新着順で検索し、price_ceiling 以下の出品を安値候補として登録。
  2) SOLD確定: 登録済み候補の個別ページを毎周ポーリングし、売り切れ表示で滞留時間を確定。
     （新着リストから消えた=売れた、は新出品に押し出される交絡があるため使わない。）

これはエンジン本体ではなく計測専用の使い捨てスクリプト。相場判定は行わず、
「price_ceiling 未満か」だけで安値候補にする（相場はキーワード単位で人が決める）。

使い方:
  ../../venv/bin/python mercari_monitor.py --config monitor_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from tracker import ListingTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("mercari_monitor")

SEARCH_URL = (
    "https://jp.mercari.com/search?keyword={kw}"
    "&status=on_sale&sort=created_time&order=desc"
)
ITEM_URL = "https://jp.mercari.com/item/{item_id}"
PRICE_RE = re.compile(r"[\d,]+")


def _human_pause(base: float = 1.5, span: float = 1.0) -> None:
    """ボット検知緩和のためのランダム待機."""
    time.sleep(base + random.uniform(0, span))


def _apply_stealth(page: Page) -> None:
    """price_intelligence_agent 準拠の最小stealth注入."""
    page.context.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        "Object.defineProperty(navigator,'languages',"
        "{get:()=>['ja-JP','ja','en-US','en']});"
        "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
    )


def discover_candidates(
    page: Page,
    tracker: ListingTracker,
    keyword: str,
    price_ceiling: int,
    now: datetime,
) -> int:
    """新着検索で price_ceiling 以下の出品を安値候補として登録し, 新規数を返す."""
    page.goto(SEARCH_URL.format(kw=keyword), wait_until="domcontentloaded")
    try:
        page.wait_for_selector('li[data-testid="item-cell"]', timeout=15000)
    except Exception:
        logger.warning(f"[{keyword}] item-cell が出ない（レイアウト変更/0件）")
        return 0

    cells = page.query_selector_all('li[data-testid="item-cell"]')
    new_count = 0
    for cell in cells:
        anchor = cell.query_selector('a[href^="/item/"]')
        if anchor is None:
            continue
        href = anchor.get_attribute("href") or ""
        item_id = href.rsplit("/", 1)[-1]
        if not item_id:
            continue
        price = _extract_price(cell.inner_text())
        if price is None or price > price_ceiling:
            continue
        title = _extract_title(cell)
        if tracker.observe_candidate(now, item_id, keyword, title, price):
            new_count += 1
            logger.info(f"[{keyword}] 新規安値候補 {item_id} ¥{price} {title}")
    return new_count


def _extract_title(cell) -> str:  # noqa: ANN001
    """タイル内サムネイル img の alt から商品名を取る（末尾の定型語を除去）."""
    img = cell.query_selector("img")
    alt = (img.get_attribute("alt") or "") if img else ""
    return alt.replace("のサムネイル", "").strip()[:60]


def _extract_price(text: str) -> int | None:
    """タイル内テキストから最初の価格らしい数値を抜く."""
    for token in PRICE_RE.findall(text):
        value = int(token.replace(",", ""))
        if value >= 300:  # メルカリ最低価格帯のノイズ除去
            return value
    return None


def check_sold(page: Page, item_id: str) -> bool:
    """個別商品ページが売り切れ表示なら True."""
    page.goto(ITEM_URL.format(item_id=item_id), wait_until="domcontentloaded")
    _human_pause(0.8, 0.6)
    body = page.inner_text("body")
    return ("売り切れました" in body) or ("SOLD OUT" in body.upper())


def run_once(page: Page, tracker: ListingTracker, watches: list[dict]) -> None:
    """1周: 全キーワードの発見 → 既存候補のSOLD確定."""
    now = datetime.now()
    for w in watches:
        added = discover_candidates(
            page, tracker, w["keyword"], int(w["price_ceiling"]), now
        )
        logger.info(f"[{w['keyword']}] 発見周: 新規{added}件")
        _human_pause()

    for item_id in tracker.active_ids():
        try:
            if check_sold(page, item_id):
                rec = tracker.mark_sold(datetime.now(), item_id)
                if rec:
                    logger.info(
                        f"SOLD確定 {item_id} 滞留{rec['residence_min']}分"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SOLDチェック失敗 {item_id}: {exc}")
        _human_pause()


def dump_results(tracker: ListingTracker, out_dir: Path) -> None:
    """売却記録CSVとサマリーJSONを書き出す."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sold_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "item_id",
                "keyword",
                "title",
                "price",
                "first_seen",
                "sold_at",
                "residence_min",
            ],
        )
        writer.writeheader()
        writer.writerows(tracker.sold_records)
    (out_dir / "summary.json").write_text(
        json.dumps(tracker.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"結果を書き出し: {csv_path} / summary.json")


def main() -> None:
    """設定を読み, 指定回数/間隔でポーリングして結果を保存する."""
    parser = argparse.ArgumentParser(description="メルカリ安値滞留モニタ")
    parser.add_argument("--config", required=True, help="監視設定JSON")
    parser.add_argument("--headless", action="store_true", help="ヘッドレス起動")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    watches = cfg["watches"]
    interval_sec = int(cfg.get("interval_sec", 900))
    rounds = int(cfg.get("rounds", 96))  # 15分×96 = 24時間
    out_dir = Path(cfg.get("out_dir", "results"))

    tracker = ListingTracker()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        _apply_stealth(page)
        try:
            for r in range(rounds):
                logger.info(f"=== round {r + 1}/{rounds} ===")
                run_once(page, tracker, watches)
                dump_results(tracker, out_dir)  # 毎周保存（中断耐性）
                if r < rounds - 1:
                    time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("中断: 現時点の結果を保存します")
        finally:
            dump_results(tracker, out_dir)
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
