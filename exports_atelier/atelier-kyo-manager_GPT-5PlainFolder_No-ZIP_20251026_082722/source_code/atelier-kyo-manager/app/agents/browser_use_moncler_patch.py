# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : app/agents/browser_use_moncler_patch.py
# Version   : 2.0.0 (PLPリカバリ + ロケール/同意バナーパッチ)
# Date (JST): 2025年10月25日 18:31
# ------------------------------------------------------------------------------
# 変更要旨
#  - (v1) PLPのリンク深掘り収集・強制遷移リカバリ
#  - ★ 統合 (v2):
#    - moncler_plp_recovery の冒頭で、localStorage へのロケール強制注入 (evaluate) を追加。
#    - OneTrust 同意バナーを強制クリックで排除するロジックを追加。
#
# 使い方:
# browser_use_agent.py (v85+) の PLP 突入前に以下のように1回呼ぶ:
#    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
#    if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None:
#        try:
#            await moncler_plp_recovery(page, site_config, query)
#        except Exception as _e:
#            logger.warning(f"[MonclerPatch] skipped: {_e}")
# ==============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import asyncio
import re

from playwright.async_api import Page

try:
    # ロガーはプロジェクト側のものがあれば使う（無いときは簡易版）
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except Exception:  # pragma: no cover
    class _Dummy:  # type: ignore
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    logger = _Dummy()


# ---------------- ユーティリティ ----------------

_CARD_SELECTORS: List[str] = [
    # Moncler のPLP/検索結果でよく見る構造を広めにマッチ
    "[data-testid*='product' i]",
    "[data-qa*='product' i]",
    "[data-test*='product' i]",
    "[class*='product' i]",
    "[class*='Product' i]",
    "article, li, div",
]

_ANCHOR_HINTS: List[str] = [
    "a[href*='/p/']",
    "a[href*='/product']",
    "a[href*='/products/']",
    "a[href^='/en-']",   # ロケール付き相対
    "a[href^='/it-']",
    "a[href^='/fr-']",
    "a[href^='/de-']",
    "a[href^='/']",
]

_DATA_URL_ATTRS: List[str] = [
    "data-url", "data-href", "data-product-url", "data-plp-url", "data-link",
]

_TEXT_URL_ATTRS: List[str] = [
    "onclick", "data-ga", "data-gtm", "data-analytics",
]

_URL_RE = re.compile(r"https?://[^\s'\"<>]+|/(?:[^\s'\"<>]+)")

async def _sleep(ms: int) -> None:
    await asyncio.sleep(ms / 1000.0)

async def _collect_candidate_urls(page: Page) -> List[str]:
    """
    画面内から PDP らしき URL 候補をできるだけ拾う。
    - a[href] はもちろん、data-* や onclick の中の URL 断片も抽出
    - 絶対URLへ正規化、重複除去
    """
    js = """
    (cardSels, anchorSels, dataAttrs, textAttrs) => {
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
    raw: List[str] = await page.evaluate(js, _CARD_SELECTORS, _ANCHOR_HINTS, _DATA_URL_ATTRS, _TEXT_URL_ATTRS)  # type: ignore
    # 絶対URL化 & ノイズ除去
    origin = page.url
    out: List[str] = []
    seen: set = set()
    for u in raw:
        if u.startswith("javascript:") or "doubleclick.net" in u or "criteo.com" in u:
            continue
        absu = urljoin(origin, u)
        # PDPらしさの軽いヒューリスティクス
        if any(k in absu for k in ("/product", "/products/", "/p/")) or urlparse(absu).path.count("/") >= 3:
            if absu not in seen:
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


async def _goto_first_working(page: Page, urls: List[str]) -> bool:
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


# ---------------- メイン: リカバリー ----------------

async def moncler_plp_recovery(page: Page, site_config: Optional[Dict[str, Any]], query: str) -> None:
    """
    1) ★ 統合: ロケール強制(localStorage)と同意バナー(OneTrust)クリック
    2) 既存: URL正規化 (page.goto)
    3) 既存: 商品リンク候補を総ざらい
    4) 既存: 直接遷移（page.goto）で PDP へ
    5) 既存: 失敗時はカードクリックの強制発火
    6) 既存: 最後の手段として window.location で直叩き
    """

    # === ★ 統合 (v2 / diff): ロケール強制 (localStorage) + 同意バナークリック ===
    force_locale: str = "en-int"
    ship_to: str = "GB"

    # 1. 互換のため evaluate は1引数のオブジェクトだけを渡す
    # (diffの add_init_script は context がないと呼べないため、evaluate のみ実行)
    try:
        await page.evaluate(
            """(cfg) => {
              try {
                window.__AKM__ = cfg;
                localStorage.setItem("akm.forceLocale", cfg.forceLocale);
                localStorage.setItem("akm.shipToCountry", cfg.shipTo);
              } catch(e) {}
            }""",
            {"forceLocale": force_locale, "shipTo": ship_to, "source": "MonclerPatch"},
        )
    except Exception as e:
        logger.warning(f"[MonclerPatch] evaluate(localStorage) failed: {e}")

    # 2. OneTrust等の同意バナーを最初に潰す（失敗してもスルー）
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
        ):
            if await _try_click(sel):
                clicked_once = True
                break

        if clicked_once:
            # たまに Shadow DOM / 遅延描画なので少しだけ待ってもう一度
            await asyncio.sleep(0.3)
            for sel in (
                "#onetrust-accept-btn-handler",
                "button#onetrust-accept-btn-handler",
            ):
                await _try_click(sel)
    except Exception as e:
        logger.warning(f"[MonclerPatch] OneTrust click failed: {e}")
    # === ★ 統合 (v2) ここまで ===


    # === 既存 (v1) ロジック ===

    # まず “en-int + shipToCountry” を URL上で正規化（何度呼ばれても安全）
    try:
        url = page.url
        if "forceLocale=en-int" not in url or "shipToCountry=" not in url:
            base = url.split("?")[0].rstrip("/") + "/"
            params = "forceLocale=en-int&shipToCountry=GB"
            new_url = base + ("?" if "?" not in url else "&") + params
            await page.goto(new_url, wait_until="domcontentloaded", timeout=15000)
            logger.info("[MonclerPatch] URL normalized to en-int with params")
            await _sleep(300)
    except Exception:
        pass

    # PLP のロード完了を軽く待つ
    try:
        await page.wait_for_selector("img, [data-testid]", timeout=8000)
    except Exception:
        pass

    # 1) URL候補を最大限収集
    urls = await _collect_candidate_urls(page)
    logger.info(f"[MonclerPatch] PLP prepared (tiles={len(urls)})")

    # 2) 直接遷移（Router の邪魔を受けない）
    if urls:
        ok = await _goto_first_working(page, urls)
        if ok:
            return

    # 3) クリック強制（Router発火）
    if await _try_router_click(page):
        # 遷移が走るまで少し待つ
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        return

    # 4) 最後の手段：DOM から URL 断片をもう一度拾って直叩き
    try:
        fragment = await page.evaluate("""
          (re) => {
            const m = (document.body && document.body.innerHTML || '').match(re);
            return m ? m[0] : null;
          }
        """, _URL_RE)
        if fragment:
            await page.goto(urljoin(page.url, fragment), wait_until="domcontentloaded", timeout=15000)
            return
    except Exception:
        pass

    # ここまで来たら諦め（呼び出し側のフォールバックに戻す）
    raise RuntimeError("Moncler PLP recovery could not navigate to any PDP.")
