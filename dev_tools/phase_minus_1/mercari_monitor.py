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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mercari_monitor")

SEARCH_URL = "https://jp.mercari.com/search?keyword={kw}&status=on_sale&sort=created_time&order=desc"
ITEM_URL = "https://jp.mercari.com/item/{item_id}"
PRICE_RE = re.compile(r"[\d,]+")


def _human_pause(base: float = 1.5, span: float = 1.0) -> None:
    """ボット検知緩和のためのランダム待機."""
    time.sleep(base + random.uniform(0, span))


def _apply_stealth(page: Page) -> None:
    """app/agents/browser/stealth.py と同一の init script を注入.

    why: 簡易stealth(webdriver秘匿のみ)では個別商品ページが
    ヘッダー/フッターのみの空コンテンツで返り、SOLD判定が常に
    Falseになる不具合が実測で判明した(2026-07-03)。
    Canvas/WebGL/Audioノイズまで含むフル版を適用すると正しく
    描画されることを確認済み。
    """
    page.context.add_init_script(
        "try { localStorage.setItem('a11y-contrast','off'); "
        "localStorage.setItem('high-contrast','off'); } catch(e){} "
        "Object.defineProperty(navigator, 'language', {get: () => 'en-GB'}); "
        "Object.defineProperty(navigator, 'languages', "
        "{get: () => ['en-GB','en']}); "
        "(function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions; "
        "Intl.DateTimeFormat.prototype.resolvedOptions=function(){"
        "const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();"
    )
    page.context.add_init_script(r"""
      (() => {
        try {
          const nav = navigator;
          if (nav) {
            const lang = (nav.language || 'en-GB');
            const langs = Array.isArray(nav.languages) && nav.languages.length ? nav.languages : ['en-GB','en'];
            Object.defineProperty(nav, 'webdriver', { get: () => undefined });
            Object.defineProperty(nav, 'hardwareConcurrency', { get: () => 8 });
            Object.defineProperty(nav, 'deviceMemory', { get: () => 8 });
            Object.defineProperty(nav, 'language', { get: () => lang });
            Object.defineProperty(nav, 'languages', { get: () => langs });
            Object.defineProperty(nav, 'maxTouchPoints', { get: () => 0 });
            Object.defineProperty(nav, 'platform', { get: () => 'Win32' });
          }

          const patchCanvas = (proto) => {
            if (!proto) return;
            const toDataURL = proto.toDataURL;
            proto.toDataURL = function(...args) {
              const ctx = this.getContext && this.getContext('2d');
              if (ctx) {
                const shift = () => (Math.random() - 0.5) * 2;
                ctx.fillStyle = `rgba(${128+shift()},${128+shift()},${128+shift()},0.01)`;
                ctx.fillRect(0, 0, 2, 2);
              }
              return toDataURL.apply(this, args);
            };
          };
          if (typeof HTMLCanvasElement !== 'undefined' && HTMLCanvasElement.prototype) {
            patchCanvas(HTMLCanvasElement.prototype);
          }
          if (typeof OffscreenCanvas !== 'undefined' && OffscreenCanvas.prototype) {
            patchCanvas(OffscreenCanvas.prototype);
          }

          const patchWebGL = (proto) => {
            if (!proto) return;
            const getParameter = proto.getParameter;
            proto.getParameter = function(param) {
              const VENDOR = 0x1F00, RENDERER = 0x1F01;
              if (param === VENDOR) {
                const v = getParameter.call(this, param);
                return typeof v === 'string' ? v.replace(/Google Inc\./, 'Google LLC') : v;
              }
              if (param === RENDERER) {
                const r = getParameter.call(this, param);
                return typeof r === 'string' ? r.replace(/ANGLE \(|\)/g, '') : r;
              }
              return getParameter.call(this, param);
            };
          };
          if (typeof WebGLRenderingContext !== 'undefined' && WebGLRenderingContext.prototype) {
            patchWebGL(WebGLRenderingContext.prototype);
          }
          if (typeof WebGL2RenderingContext !== 'undefined' && WebGL2RenderingContext.prototype) {
            patchWebGL(WebGL2RenderingContext.prototype);
          }

          const patchAudio = (Cls) => {
            if (!Cls || !Cls.prototype) return;
            const getFloatFrequencyData = Cls.prototype.getFloatFrequencyData;
            if (getFloatFrequencyData) {
              Cls.prototype.getFloatFrequencyData = function(arr) {
                const res = getFloatFrequencyData.call(this, arr);
                for (let i = 0; i < arr.length; i += Math.floor(arr.length / 8) || 1) {
                  arr[i] = arr[i] * (0.99 + Math.random() * 0.02);
                }
                return res;
              };
            }
          };
          if (typeof AnalyserNode !== 'undefined') {
            patchAudio(AnalyserNode);
          }
        } catch (e) { /* swallow */ }
      })();
    """)


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


RENDERED_BODY_MIN_LEN = 1500  # 実測: 未描画=~700字, 描画完了後=数千字


def check_sold(page: Page, item_id: str) -> bool:
    """個別商品ページが売り切れ表示なら True.

    why: 商品詳細は domcontentloaded 後もクライアント側で遅延描画され、
    実測で3〜4秒かかる（それまでは本文がヘッダー/フッターのみ）。
    固定秒数の決め打ちは環境差・回線変動で再び偽陰性化しうるため、
    本文長で描画完了を検知するまで適応的にポーリングする。
    """
    page.goto(ITEM_URL.format(item_id=item_id), wait_until="domcontentloaded")
    body = ""
    for _ in range(8):  # 最大 約0.8〜1.4s x 8 ≈ 6〜11秒まで許容
        _human_pause(0.8, 0.6)
        body = page.inner_text("body")
        if len(body) >= RENDERED_BODY_MIN_LEN:
            break
    return ("売り切れました" in body) or ("SOLD OUT" in body.upper())


SOLD_SEARCH_URL = "https://jp.mercari.com/search?keyword={kw}&status=sold_out&sort=created_time&order=desc"


def _find_known_sold_items(page: Page, keyword: str, limit: int = 3) -> list[str]:
    """自己診断用: そのキーワードで確実に売り切れ済みの商品IDを複数取得.

    why: 1件だけで判定すると、その1件がたまたま「取引中」等の紛らわしい
    状態だったり、単発の描画ブレに当たった場合、実際は壊れていないのに
    誤って自己診断NGにしてしまう（過検知で24h走行を無駄に止める）。
    複数候補を見て多数決に近い形で判定する方が頑健。
    """
    page.goto(SOLD_SEARCH_URL.format(kw=keyword), wait_until="domcontentloaded")
    try:
        page.wait_for_selector('li[data-testid="item-cell"]', timeout=10000)
    except Exception:
        return []
    anchors = page.query_selector_all('li[data-testid="item-cell"] a[href^="/item/"]')
    ids: list[str] = []
    for anchor in anchors[:limit]:
        href = anchor.get_attribute("href") or ""
        item_id = href.rsplit("/", 1)[-1]
        if item_id:
            ids.append(item_id)
    return ids


def self_check_sold_detection(page: Page, watches: list[dict]) -> bool:
    """SOLD判定ロジック自体が機能しているかを実データで自己検証する.

    why: 2026-07-03に描画待機不足でSOLD判定が常にFalseを返す不具合が
    6.5時間ユーザーが確認するまで気づかれなかった。同種の回帰
    （サイト側レイアウト変更・タイミング変化）をユーザーの目視確認に
    頼らず、起動時と実行中に自動で検知するためのセルフテスト。
    watch対象キーワードで実際に売り切れ済みの商品を複数見つけ、
    check_sold() が**そのうち1件でも**正しく True と判定できれば
    ロジックは生きていると判断する（単発の描画ブレによる過検知を防ぐ）。
    """
    for w in watches:
        item_ids = _find_known_sold_items(page, w["keyword"])
        if not item_ids:
            continue
        results = {item_id: check_sold(page, item_id) for item_id in item_ids}
        if any(results.values()):
            ok_id = next(i for i, ok in results.items() if ok)
            logger.info(f"自己診断OK（{w['keyword']}の既知SOLD {ok_id} を正しく検知）")
            return True
        logger.critical(
            f"自己診断失敗: 既知SOLD商品{list(results.keys())}を"
            "いずれも売切と判定できず。SOLD検知ロジックが壊れている疑いあり"
            "（描画タイミング変化/DOM変更等）。"
        )
        return False
    logger.warning("自己診断: 検証用のSOLD済み商品が見つからず判定不能（続行）")
    return True  # 判定材料が無いだけなので、無理に失敗扱いにしない


def run_once(page: Page, tracker: ListingTracker, watches: list[dict]) -> None:
    """1周: 全キーワードの発見 → 既存候補のSOLD確定."""
    now = datetime.now()
    for w in watches:
        try:
            added = discover_candidates(page, tracker, w["keyword"], int(w["price_ceiling"]), now)
            logger.info(f"[{w['keyword']}] 発見周: 新規{added}件")
        except Exception as exc:  # noqa: BLE001
            # ネットワーク一時障害等で1キーワード失敗しても24h走行を止めない
            logger.warning(f"[{w['keyword']}] 発見周失敗（継続）: {exc}")
        _human_pause()

    for item_id in tracker.active_ids():
        try:
            if check_sold(page, item_id):
                rec = tracker.mark_sold(datetime.now(), item_id)
                if rec:
                    logger.info(f"SOLD確定 {item_id} 滞留{rec['residence_min']}分")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SOLDチェック失敗 {item_id}: {exc}")
        _human_pause()


def dump_results(tracker: ListingTracker, out_dir: Path, self_check_ok: bool | None) -> None:
    """売却記録CSVとサマリーJSONを書き出す.

    self_check_ok: 直近の自己診断結果（True=正常/False=SOLD検知が壊れている疑い/
    None=まだ診断していない）。summary.json に含め、ユーザーがログを見に行かなくても
    「この結果は信用してよいか」を一目で判断できるようにする。
    """
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
    summary = tracker.summary()
    summary["self_check_ok"] = self_check_ok
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
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

        self_check_ok: bool | None = None
        try:
            logger.info("起動時セルフチェック: SOLD判定ロジックの健全性を検証中...")
            self_check_ok = self_check_sold_detection(page, watches)
            if not self_check_ok:
                logger.critical(
                    "起動時セルフチェックに失敗。SOLD検知が壊れた状態で走らせても"
                    "無意味なデータしか溜まらないため、実行を中止します。"
                )
                dump_results(tracker, out_dir, self_check_ok)
                return

            for r in range(rounds):
                logger.info(f"=== round {r + 1}/{rounds} ===")
                run_once(page, tracker, watches)

                # 定期セルフチェック: サイト側の変更等による回帰をユーザーの
                # 目視確認任せにせず、実行中も自動で検知する（約3時間毎）。
                if r > 0 and r % 12 == 0:
                    logger.info(f"定期セルフチェック（round {r + 1}）...")
                    self_check_ok = self_check_sold_detection(page, watches)
                    if not self_check_ok:
                        logger.critical(
                            "定期セルフチェックに失敗。SOLD検知が壊れた疑いがある"
                            "ため、これ以上データを溜めず実行を停止します。"
                        )
                        dump_results(tracker, out_dir, self_check_ok)
                        return

                dump_results(tracker, out_dir, self_check_ok)  # 毎周保存（中断耐性）
                if r < rounds - 1:
                    time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("中断: 現時点の結果を保存します")
        finally:
            dump_results(tracker, out_dir, self_check_ok)
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
