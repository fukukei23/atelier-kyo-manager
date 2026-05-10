# ==============================================================================
# ファイル名 (File Name): browser_use_moncler_patch.py
# レジストリ (Registry): app/agents/browser_use_moncler_patch.py
# バージョン (Version): 2.2.0 (Stealth 共通化対応)
# 更新日時 (Date & Time JST): 2025年12月3日
#
# 変更要旨:
# - (v2.2.0) ★ Stealth 共通化対応: UA / viewport / navigator.webdriver などの
#   指紋対策は scraping/stealth.py に移行。このパッチは Moncler サイト固有の
#   UI 操作（Cookie注入、ロケールモーダル処理、URL正規化）のみを担当。
# - (v2.1.0) ★ 強化: `moncler-shipping-country` / `moncler-shipping-language` Cookieを
#   overrides.local.json の設定値に基づいて強制的に注入するロジックを強化。
# - (v2.1.0) ★ 強化: `en-de` (ドイツ) などの意図しないロケールへのリダイレクトを検知し、
#   Cookie再設定後に `en-int` へ強制送還するループを追加。
# - (v2.1.0) ★ 修正: `moncler_plp_recovery` の引数 `query` の型ヒントを `Any` に維持。
#
# 役割分担:
# - scraping/stealth.py: UA / viewport / navigator.webdriver / permissions.query など
#   汎用的な Bot 検知回避（すべてのサイトで共通）
# - browser_use_moncler_patch.py: Moncler サイト固有の UI 操作
#   - Cookie 注入（moncler-shipping-country, moncler-shipping-language）
#   - LocalStorage 設定
#   - OneTrust バナーのクリック
#   - ロケーションモーダルの処理
#   - URL 正規化（en-de → en-int など）
#
# 使い方 (Usage):
# browser_use_agent.py (v85+) の PLP 突入前に以下のように1回呼ぶ:
#    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
#    if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None:
#        try:
#            await moncler_plp_recovery(page, site_config, query_context)
#        except Exception as _e:
#            logger.warning(f"[MonclerPatch] skipped: {_e}")
# ==============================================================================

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page

try:
    # ロガーはプロジェクト側のものがあれば使う（無いときは簡易版）
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except Exception:  # pragma: no cover

    class _Dummy:  # type: ignore
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    logger = _Dummy()


# ---------------- ユーティリティ ----------------

_CARD_SELECTORS: list[str] = [
    # Moncler のPLP/検索結果でよく見る構造を広めにマッチ
    "[data-testid*='product' i]",
    "[data-qa*='product' i]",
    "[data-test*='product' i]",
    "[class*='product' i]",
    "[class*='Product' i]",
    "article, li, div",
]

_ANCHOR_HINTS: list[str] = [
    "a[href*='/p/']",
    "a[href*='/product']",
    "a[href*='/products/']",
    "a[href^='/en-']",  # ロケール付き相対
    "a[href^='/it-']",
    "a[href^='/fr-']",
    "a[href^='/de-']",
    "a[href^='/']",
]

_DATA_URL_ATTRS: list[str] = [
    "data-url",
    "data-href",
    "data-product-url",
    "data-plp-url",
    "data-link",
]

_TEXT_URL_ATTRS: list[str] = [
    "onclick",
    "data-ga",
    "data-gtm",
    "data-analytics",
]

_URL_RE = re.compile(r"https?://[^\s'\"<>]+|/(?:[^\s'\"<>]+)")


async def _sleep(ms: int) -> None:
    await asyncio.sleep(ms / 1000.0)


async def _collect_candidate_urls(page: Page) -> list[str]:
    """
    画面内から PDP らしき URL 候補をできるだけ拾う。
    - a[href] はもちろん、data-* や onclick の中の URL 断片も抽出
    - 絶対URLへ正規化、重複除去
    """
    js = """
    ([cardSels, anchorSels, dataAttrs, textAttrs]) => {
      const uniq = new Set();
      const push = (u) => { if (u && typeof u === 'string' && u.trim()) uniq.add(u.trim()); };

      // 1) a[href] の総当たり
      for (const sel of anchorSels) {
        for (const a of document.querySelectorAll(sel)) {
          const href = a.getAttribute('href');
          if (href) push(href);
          const dataHref = a.dataset && (a.dataset.url || a.dataset.href || a.dataset.productUrl);
          if (dataHref) push(dataHref);
        }
      }

      // 2) カード要素を起点に data-* を洗う
      const cards = [];
      for (const cs of cardSels) {
        for (const el of document.querySelectorAll(cs)) {
          // タイルっぽい最低条件
          const t = (el.innerText || '').toLowerCase();
          if (!/jacket|coat|moncler|down|women|men|price|€|¥|£|\\$/.test(t)) continue;
          cards.push(el);
        }
      }

      for (const el of cards) {
        for (const name of dataAttrs) {
          const v = el.getAttribute(name);
          if (v) push(v);
        }
        // 埋め込みの a をもう一度
        for (const a of el.querySelectorAll('a[href]')) {
          const href = a.getAttribute('href');
          if (href) push(href);
        }
        // text 属性に URL 断片がある場合
        for (const name of textAttrs) {
          const v = el.getAttribute(name);
          if (v) {
            const m = v.match(/https?:\\/\\/[^\\s'\"<>]+|\\/(?:[^\\s'\"<>]+)/g);
            if (m) m.forEach(s => push(s));
          }
        }
      }

      // 3) 画像から近傍の a を辿る（画像だけでリンクされるパターン）
      for (const img of document.querySelectorAll('img')) {
        let p = img;
        for (let i = 0; i < 4 && p; i++) {
          if (p.tagName === 'A') {
            const href = p.getAttribute('href');
            if (href) push(href);
            break;
          }
          p = p.parentElement;
        }
      }

      return Array.from(uniq);
    }
    """
    raw_args = [_CARD_SELECTORS, _ANCHOR_HINTS, _DATA_URL_ATTRS, _TEXT_URL_ATTRS]
    raw: list[str] = await page.evaluate(js, raw_args)  # type: ignore

    # 絶対URL化 & ノイズ除去
    origin = page.url
    out: list[str] = []
    seen: set[str] = set()
    for u in raw:
        if u.startswith("javascript:") or "doubleclick.net" in u or "criteo.com" in u:
            continue
        absu = urljoin(origin, u)
        # PDPらしさの軽いヒューリスティクス
        if (
            any(k in absu for k in ("/product", "/products/", "/p/")) or urlparse(absu).path.count("/") >= 3
        ) and absu not in seen:
            seen.add(absu)
            out.append(absu)
    return out[:50]


async def _try_router_click(page: Page) -> bool:
    """
    最前面の“商品カードっぽい”要素に対して、自然クリック→合成クリック→dispatchEvent を順に試す。
    クリックが阻害されるケース向け。
    """
    js_pick = """
    (cardSels) => {
      // ビューポート内のカードをスコアリングして一枚返す
      const vw = window.innerWidth, vh = window.innerHeight;
      let best = null, bestScore = -1;
      const inView = (el) => {
        const r = el.getBoundingClientRect();
        return r.width > 80 && r.height > 120 && r.top < vh && r.bottom > 0 && r.left < vw && r.right > 0;
      };
      const score = (el) => {
        let s = 0;
        const t = (el.innerText || "").toLowerCase();
        if (/down|jacket|coat|moncler/.test(t)) s += 3;
        if (/price|€|¥|£|\\$/.test(t)) s += 2;
        const r = el.getBoundingClientRect();
        s += Math.min(3, Math.floor(r.width/200)) + Math.min(3, Math.floor(r.height/200));
        return s;
      };
      for (const cs of cardSels) {
        for (const el of document.querySelectorAll(cs)) {
          if (!inView(el)) continue;
          const sc = score(el);
          if (sc > bestScore) { bestScore = sc; best = el; }
        }
      }
      if (!best) return null;
      best.scrollIntoView({block:"center", inline:"center"});
      return best;
    }
    """
    el = await page.evaluate_handle(js_pick, _CARD_SELECTORS)  # type: ignore
    if not el:  # pragma: no cover
        return False
    try:
        # a 要素があれば優先してクリック
        a = await el.query_selector("a[href]")
        if a:
            try:
                await a.click(timeout=3000)
                return True
            except Exception:
                try:
                    await a.dispatch_event("click")
                    return True
                except Exception:
                    pass
        # カード本体へイベント連打
        try:
            await el.click(timeout=3000)
            return True
        except Exception:
            try:
                await el.dispatch_event("click")
                return True
            except Exception:
                return False
    finally:
        await el.dispose()


async def _goto_first_working(page: Page, urls: list[str]) -> bool:
    for u in urls:
        try:
            await page.goto(u, wait_until="domcontentloaded", timeout=15000)
            # PDP らしさの簡易判定（タイトルや価格っぽいもの）
            ok = await page.evaluate("""
              () => {
                const hasH1 = !!document.querySelector('h1, [itemprop="name"]');
                const hasPrice = !!document.querySelector('[itemprop="price"], [class*="price" i], [data-testid*="price" i]');
                const bodyText = (document.body && document.body.innerText || '').toLowerCase();
                const money = /\\b(usd|eur|gbp|jpy)\\b|[€¥£$]/i.test(bodyText);
                return hasH1 || (hasPrice && money);
              }
            """)
            if ok:
                return True
        except Exception:
            continue
    return False


# ---------------- ロケーションモーダル対応 ----------------


async def _select_preferred_locale(page: Page) -> bool:
    """
    UK/EN のロケールリンク/ボタンを優先的にクリックする。
    """
    selectors: list[str] = [
        "[data-country-code='GB']",
        "[data-country='GB']",
        "button[data-value='GB']",
        "button[data-locale*='en-gb' i]",
        "a[data-locale*='en-gb' i]",
        "button:has-text('United Kingdom')",
        "a:has-text('United Kingdom')",
        "button:has-text('United Kingdom EN')",
    ]
    for sel in selectors:
        try:
            locator = page.locator(sel).first
            if await locator.count() > 0 and await locator.is_visible():
                await locator.click(timeout=2500)
                await page.wait_for_timeout(300)
                logger.info("[MonclerPatch] Locale modal: selected United Kingdom / EN")
                return True
        except Exception:
            continue
    try:
        clicked = await page.evaluate(
            """(labels) => {
                const targets = Array.from(document.querySelectorAll("button, a, [role='button']"));
                for (const label of labels) {
                    const target = targets.find(el => (el.innerText || "").toLowerCase().includes(label));
                    if (target) { target.click(); return true; }
                }
                return false;
            }""",
            ["united kingdom en", "united kingdom", "english (uk)", "gb en"],
        )
        if clicked:
            await page.wait_for_timeout(300)
            logger.info("[MonclerPatch] Locale modal: selected via JS fallback")
            return True
    except Exception:
        pass
    return False


async def _handle_location_modal(page: Page) -> None:
    """
    Moncler の 「Select your location」 モーダルを閉じる軽量ハンドラ。
    既に閉じていれば何もしない。最大10回/合計5秒で諦める（無限待ち防止）。
    """
    start = time.monotonic()
    attempts = 0
    try:
        while True:
            attempts += 1
            modal_title = page.locator("text=Select your location").first
            if await modal_title.count() == 0 or not await modal_title.is_visible():
                return
            if await _select_preferred_locale(page):
                return
            close = page.locator(
                "button[aria-label*='close' i], "
                "button:has-text('Close'), "
                "button:has-text('×'), "
                ".modal__close, .c-modal__close"
            ).first
            if await close.count() > 0:
                try:
                    await close.click(timeout=2500)
                    await page.wait_for_timeout(300)
                    logger.info("[MonclerPatch] Location modal closed (close button).")
                    return
                except Exception:
                    pass
            if attempts >= 10 or (time.monotonic() - start) > 5:
                raise RuntimeError("Locale modal not dismissed within 10 attempts/5s")
            await page.wait_for_timeout(300)
    except Exception as e:
        logger.warning(f"[MonclerPatch] Location modal handling failed: {e}")
        raise


# ---------------- メイン: リカバリー ----------------


async def moncler_plp_recovery(page: Page, site_config: dict[str, Any] | None, query: Any = None) -> None:
    """
    1) ロケール強制(Cookie/localStorage)と同意バナー(OneTrust)クリック
    2) URL正規化 (page.goto) - en-de 等からの脱出
    3) 商品リンク候補を総ざらい
    4) 直接遷移（page.goto）で PDP へ
    5) 失敗時はカードクリックの強制発火
    6) 最後の手段として window.location で直叩き
    """

    # --- Config読み込み ---
    cfg = site_config or {}
    ds = cfg.get("discovery_settings") or {}

    # 設定からターゲットを取得 (デフォルトは GB / en-int)
    ship_to: str = cfg.get("shipToCountry") or ds.get("shipToCountry") or "GB"
    force_locale: str = cfg.get("force_locale") or ds.get("forceLocale") or "en-int"

    # Cookie用の言語コード (例: en-int -> en)
    lang_code = force_locale.split("-")[0] if "-" in force_locale else "en"

    logger.info(f"[MonclerPatch] Starting recovery. Target: {force_locale} / {ship_to}")

    # --- 1. Cookie & LocalStorage Injection (最優先) ---
    # サイトが参照する正規のCookieをセットする
    try:
        # ドメインは .moncler.com で広域にセット
        cookies = [
            {"name": "moncler-shipping-country", "value": ship_to, "domain": ".moncler.com", "path": "/"},
            {"name": "moncler-shipping-language", "value": lang_code, "domain": ".moncler.com", "path": "/"},
            {"name": "store-country-code", "value": ship_to, "domain": ".moncler.com", "path": "/"},  # 追加
            {"name": "store-language-code", "value": lang_code, "domain": ".moncler.com", "path": "/"},  # 追加
        ]
        await page.context.add_cookies(cookies)
        logger.info("[MonclerPatch] Injected shipping cookies.")

        # LocalStorage も念のため
        await page.evaluate(
            """(data) => {
              try {
                localStorage.setItem('moncler-shipping-country', data.country);
                localStorage.setItem('moncler-shipping-language', data.lang);
                localStorage.setItem('akm.forceLocale', data.locale);
              } catch(e) {}
            }""",
            {"country": ship_to, "lang": lang_code, "locale": force_locale},
        )
    except Exception as e:
        logger.warning(f"[MonclerPatch] Cookie/Storage injection failed: {e}")

    # --- 2. OneTrust等の同意バナーを潰す ---
    async def _try_click(sel: str) -> bool:
        try:
            await page.locator(sel).click(timeout=1000)
            return True
        except Exception:
            return False

    try:
        clicked_once = False
        for sel in (
            "#onetrust-accept-btn-handler",
            "button#onetrust-accept-btn-handler",
            "button[aria-label*='Accept'][id*='onetrust']",
            "button[aria-label*='agree']",
            "[id^='onetrust-'] button",
        ):
            if await _try_click(sel):
                clicked_once = True
                break

        if clicked_once:
            await asyncio.sleep(0.5)
            # 念押し
            await _try_click("#onetrust-accept-btn-handler")
    except Exception as e:
        logger.warning(f"[MonclerPatch] OneTrust click failed: {e}")

    # --- 3. ロケーションモーダル対応 ---
    try:
        await _handle_location_modal(page)
    except Exception as e:
        logger.warning(f"[MonclerPatch] location modal handling failed: {e}")
        # タイムアウト時は動画を確定させ、早期に落とす
        try:
            Path("instance/logs").mkdir(parents=True, exist_ok=True)
            stuck_path = Path("instance/logs/moncler_gate_stuck.png")
            await page.screenshot(path=str(stuck_path), full_page=True)
            logger.warning(f"[MonclerPatch] Gate screenshot saved: {stuck_path}")
        except Exception:
            pass
        with contextlib.suppress(Exception):
            await page.context.close()
        raise RuntimeError("Gate dismissal failed -> aborting session") from None

    # --- 4. URL正規化 & リダイレクト脱出 ---
    # 現在のURLがターゲットロケールと異なる場合、強制的に遷移する
    try:
        url = page.url

        # 悪いロケール (例: /en-de/) を検知
        # ターゲットが en-int なのに en-de, en-fr などにいる場合
        current_path = urlparse(url).path

        is_wrong_locale = False
        # 例: /en-de/ が含まれていて、かつ force_locale が en-int の場合
        if (
            f"/{force_locale}/" not in current_path
            and re.search(r"/en-[a-z]{2}/", current_path)
            and force_locale not in current_path
        ):
            is_wrong_locale = True
            logger.warning(f"[MonclerPatch] Wrong locale detected: {current_path} (Target: {force_locale})")

        # パラメータ不足の検知
        has_params = f"forceLocale={force_locale}" in url and f"shipToCountry={ship_to}" in url

        if is_wrong_locale or not has_params:
            # URLを書き換え
            # 1. 既存のロケール部分を force_locale に置換
            new_path = re.sub(r"/en-[a-z]{2}/", f"/{force_locale}/", current_path)
            if new_path == current_path and f"/{force_locale}/" not in new_path:
                # ロケールがない場合は挿入 (簡易的)
                new_path = f"/{force_locale}" + current_path

            base = urljoin(url, new_path).split("?")[0].rstrip("/") + "/"
            params = f"forceLocale={force_locale}&shipToCountry={ship_to}"
            new_url = base + ("?" if "?" not in url else "&") + params

            logger.info(f"[MonclerPatch] Forcing navigation to: {new_url}")
            await page.goto(new_url, wait_until="domcontentloaded", timeout=20000)
            await _sleep(1000)  # 遷移後の安定待ち

            # 遷移後に再度Cookieをセット (リダイレクトで消されることがあるため)
            await page.context.add_cookies(cookies)

    except Exception as e:
        logger.warning(f"[MonclerPatch] URL normalization failed: {e}")

    # PLP のロード完了を待つ
    with contextlib.suppress(Exception):
        await page.wait_for_selector("img, [data-testid]", timeout=8000)

    # --- 5. URL候補収集 ---
    urls = await _collect_candidate_urls(page)
    logger.info(f"[MonclerPatch] PLP prepared (tiles={len(urls)})")

    # tiles=0 なら即 PDP直行を試みる
    if not urls:
        logger.warning("[MonclerPatch] tiles=0 -> direct PDP hop attempts")
        # --- クリック強制 ---
        if await _try_router_click(page):
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("domcontentloaded", timeout=8000)
            return
        # --- DOM 断片直叩き ---
        try:
            pattern = _URL_RE.pattern
            fragment = await page.evaluate(
                """
              (pat) => {
                try {
                  const re = new RegExp(pat, 'g');
                  const html = (document.body && document.body.innerHTML) || '';
                  const m = html.match(re);
                  return m ? m[0] : null;
                } catch (e) { return null; }
              }
            """,
                pattern,
            )
            if fragment:
                await page.goto(urljoin(page.url, fragment), wait_until="domcontentloaded", timeout=15000)
                return
        except Exception:
            pass
        raise RuntimeError("Moncler PLP recovery: no tiles and no fallback URL found.")

    # --- 6. 直接遷移 ---
    if urls:
        ok = await _goto_first_working(page, urls)
        if ok:
            return

    # --- 7. クリック強制 ---
    if await _try_router_click(page):
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
        return

    # --- 8. 最後の手段：DOM から URL 断片を直叩き ---
    try:
        pattern = _URL_RE.pattern
        fragment = await page.evaluate(
            """
          (pat) => {
            try {
              const re = new RegExp(pat, 'g');
              const html = (document.body && document.body.innerHTML) || '';
              const m = html.match(re);
              return m ? m[0] : null;
            } catch (e) { return null; }
          }
        """,
            pattern,
        )
        if fragment:
            await page.goto(urljoin(page.url, fragment), wait_until="domcontentloaded", timeout=15000)
            return
    except Exception:
        pass

    # ここまで来たら諦め
    raise RuntimeError("Moncler PLP recovery could not navigate to any PDP.")
