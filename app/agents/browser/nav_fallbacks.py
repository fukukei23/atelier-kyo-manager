# ==============================================================================
# File: app/agents/browser/nav_fallbacks.py
# Version: 1.0.0 (P1-1 Phase 5: Fallback methods extracted from navigation_driver.py)
# Purpose: FallbackMixin - PLP fallback navigation methods
# ==============================================================================
"""
P1-1 Phase 5: NavigationDriver からフォールバック系メソッドを抽出した Mixin。

抽出対象メソッド:
- run_deep_extraction
- _run_deep_extraction_phase2
- _click_and_capture_navigation
- header_search_fallback
- click_first_card_or_link
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from playwright.async_api import BrowserContext, ElementHandle, Page

from app.agents.browser.nav_types import NavigationContext

if TYPE_CHECKING:
    pass

from app.agents.browser.extractor import (
    VISIBLE_PRICE_SELECTORS,
    extract_moncler_pdp_links,  # noqa: F401
)

logger = logging.getLogger(__name__)


# ヘルパー関数（モジュールレベル）
def _dedupe_keep_order(items: list[str]) -> list[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


async def click_and_capture_nav(
    click_coro,
    page: Page,
    context: BrowserContext,
    *,
    url_regex: re.Pattern | None = None,
    wait_state: str = "domcontentloaded",
    timeout_ms: int = 5000,
) -> Page | None:
    """Click and capture navigation (popup / same-tab / SPA race)."""
    if url_regex is None:
        url_regex = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

    popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
    same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))
    spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms)) if url_regex else None
    sel_spa = (
        ", ".join(VISIBLE_PRICE_SELECTORS)
        if VISIBLE_PRICE_SELECTORS
        else "[itemprop=price],[class*=price],[data-testid*=price]"
    )
    spa_price_task = asyncio.create_task(page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms))

    try:
        await click_coro()
    except Exception:
        for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
            if t and not t.done():
                t.cancel()
        return None

    tasks = {popup_task, same_tab_nav_task, spa_price_task}
    if spa_url_task:
        tasks.add(spa_url_task)

    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout_ms / 1000, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if not done:
            return None
        winner = next(iter(done))
        new_page = winner.result() if winner is popup_task else page
        log_msg = (
            "popup"
            if winner is popup_task
            else "framenav"
            if winner is same_tab_nav_task
            else "SPA URL"
            if winner is spa_url_task
            else "SPA Price"
        )
        logger.debug(f"[click_and_capture_nav] {log_msg}")
        try:
            if new_page.url == "about:blank":
                await new_page.wait_for_load_state("domcontentloaded", timeout=1500)
        except Exception as e_blank:
            logger.debug(f"[click_and_capture_nav] Wait for about:blank failed: {e_blank}")
        with contextlib.suppress(Exception):
            await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10))
        if url_regex:
            try:
                await new_page.wait_for_url(url_regex, timeout=max(1000, timeout_ms // 4))
            except Exception as e_url_final:
                logger.debug(f"[click_and_capture_nav] Final wait_for_url failed: {e_url_final}")
        return new_page
    except Exception as e_wait:
        logger.warning(f"[click_and_capture_nav] Nav race failed: {e_wait}")
        return None
    finally:
        for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
            if t and not t.done():
                t.cancel()


class FallbackMixin:
    """
    PLP フォールバックナビゲーションメソッドの Mixin。

    NavigationDriver に組み込んで使用する。
    `page: Page` 属性へのアクセスを前提とする。
    """

    page: Page

    async def run_deep_extraction(
        self,
        page: Page,
        site_config: dict[str, Any],
    ) -> list[str]:
        """
        深い抽出を実行する（骨組みのみ）

        Stage 3A-1: このステップでは、メソッドシグネチャのみ定義。
        実際のロジック移動は Stage 3A-2 で行います。

        Args:
            page: Playwright Page オブジェクト
            site_config: サイト設定

        Returns:
            List[str]: 抽出されたリンクのリスト（現時点では空リストを返す）
        """
        # Stage 3A-1: スタブ実装
        logger.debug("[NavigationDriver] run_deep_extraction (stub): returning empty list")
        return []

    async def _run_deep_extraction_phase2(self, page: Page, site_config: dict[str, Any]) -> list[str]:
        """
        Stage 3A-2-1:
        旧 BrowserUseAgent._run_deep_extraction_phase2 のロジックをここに移す。
        挙動・ログ・例外の流れはそのまま維持すること。

        Deep Extraction Phase 2: JSON-LD, onclick, data-* 属性からリンクを抽出する
        """
        logger.debug("[Phase 2] Running deep extraction (JSON-LD, onclick, data-*, ...)")
        # ★ 88.6.2: (BugFix) 括弧が過剰だった SyntaxError を修正
        container_sels: list[str] = ((site_config.get("selectors") or {}).get("pdp") or {}).get(
            "plp_container_selectors", []
        ) or []
        for cont in container_sels or []:
            await self.safe_wait_selector(page, cont, timeout_ms=1000, state="visible")
        try:
            for _ in range(2):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(200)
        except Exception:
            pass

        # V86.0: Strict mode violation prevention + V88.2: Get ElementHandle
        # Stage 4: タイムアウト処理改善（CancelledErrorを適切に処理）
        scope = page.locator("main, [role='main'], #main, #app")
        handle: ElementHandle | None = None
        try:
            # タイムアウトを短くして、CancelledErrorを避ける
            # ただし、asyncio.wait_forでラップすると、タイムアウト時にCancelledErrorが発生する可能性がある
            # そのため、直接wait_forを呼び出し、タイムアウトエラーをキャッチする
            try:
                await scope.first.wait_for(state="attached", timeout=4000)
                handle = await scope.first.element_handle(timeout=4000)
            except asyncio.CancelledError:
                logger.debug("[Phase 2] Cancelled while waiting for scope")
                raise
            except Exception as e_handle:
                # (V88.6.2: ログレベルは warning のまま)
                logger.warning(
                    f"[Phase 2] Could not get element handle for scope: {e_handle}. Falling back to page evaluate."
                )
                handle = None  # Ensure handle is None if getting it failed
        except asyncio.CancelledError:
            logger.debug("[Phase 2] Cancelled in outer try block")
            raise
        except Exception as e_outer:
            logger.warning(f"[Phase 2] Outer exception: {e_outer}. Falling back to page evaluate.")
            handle = None

        # JS Script that takes an optional node context
        _js_script = """
          (node) => {
            const area = node || document;
            const out = [];
            const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
            area.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
              const a = el.closest("a") || el;
              const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
              if (cand) push(cand);
            });
            area.querySelectorAll("[onclick]").forEach(el => {
              const oc = el.getAttribute("onclick") || "";
              const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
              if (m1 && m1[1]) push(m1[1]);
              const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
              if (m2 && m2[1]) push(m2[1]);
            });
            area.querySelectorAll("script[type='application/ld+json']").forEach(s => {
              try {
                const data = JSON.parse(s.textContent || "null");
                const arr = Array.isArray(data) ? data : [data];
                const pushAny = (v) => { if (v && typeof v === "string") push(v); };
                arr.forEach(d => {
                  if (!d || typeof d !== "object") return;
                  pushAny(d.url || d['@id']);
                  if (Array.isArray(d.offers)) {
                    d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                  } else if (d.offers && typeof d.offers === "object") {
                    pushAny(d.offers.url || d.offers['@id']);
                  }
                  if (d.itemListElement && Array.isArray(d.itemListElement)) {
                    d.itemListElement.forEach(it => {
                      if (it && it.item && (it.item.url || it.item['@id'])) {
                        pushAny(it.item.url || it.item['@id']);
                      }
                    });
                  }
                });
              } catch(e) {}
            });
            return out.filter(Boolean);
          }
        """

        hrefs: list[str] = []
        try:
            if handle:
                # Execute JS within the specific element context
                hrefs = await handle.evaluate(_js_script)
                logger.debug("[Phase 2] Deep extraction performed using element handle.")
            else:
                # --- V88.3.0: Safer Fallback ---
                # ドキュメント全体を対象にした無引数版を直接渡す
                hrefs = await page.evaluate("""
                  () => {
                    const out = [];
                    const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
                    document.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
                      const a = el.closest("a") || el;
                      const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
                      if (cand) push(cand);
                    });
                    document.querySelectorAll("[onclick]").forEach(el => {
                      const oc = el.getAttribute("onclick") || "";
                      const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
                      if (m1 && m1[1]) push(m1[1]);
                      const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
                      if (m2 && m2[1]) push(m2[1]);
                    });
                    document.querySelectorAll("script[type='application/ld+json']").forEach(s => {
                      try {
                        const data = JSON.parse(s.textContent || "null");
                        const arr = Array.isArray(data) ? data : [data];
                        const pushAny = (v) => { if (v && typeof v === "string") push(v); };
                        arr.forEach(d => {
                          if (!d || typeof d !== "object") return;
                          pushAny(d.url || d['@id']);
                          if (Array.isArray(d.offers)) {
                            d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                          } else if (d.offers && typeof d.offers === "object") {
                            pushAny(d.offers.url || d.offers['@id']);
                          }
                          if (d.itemListElement && Array.isArray(d.itemListElement)) {
                            d.itemListElement.forEach(it => {
                              if (it && it.item && (it.item.url || it.item['@id'])) {
                                pushAny(it.item.url || it.item['@id']);
                            }
                            });
                          }
                        });
                      } catch(e) {}
                    });
                    return out.filter(Boolean);
                  }
                """)
                logger.debug("[Phase 2] Deep extraction performed using page evaluate (fallback).")
                # --- V88.3.0 修正ここまで ---
        except Exception as e:
            logger.warning(f"[Phase 2] Deep extraction evaluate failed: {e}")
            hrefs = []  # Ensure hrefs is a list even on error

        return _dedupe_keep_order(hrefs)

    async def _click_and_capture_navigation(
        self,
        click_coro: Callable,
        page: Page,
        context: BrowserContext,
        *,
        url_regex: re.Pattern | None = None,
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Click and capture navigation — delegates to module-level helper."""
        return await click_and_capture_nav(
            click_coro,
            page,
            context,
            url_regex=url_regex,
            wait_state=wait_state,
            timeout_ms=timeout_ms,
        )

    async def header_search_fallback(self, ctx: NavigationContext) -> bool:
        """
        Stage 3A-2-4:
        旧 BrowserUseAgent._plp_header_search_fallback のロジックをここに移行。
        PLP の検索UIを使って query を再投入し、PLP を再構成する fallback。
        成功したら True, 変化なしなら False。
        """
        page = self.page
        site_config = ctx.site_config
        query = ctx.query
        start_t = ctx.start_t
        budget_ms = ctx.budget_ms

        # Stage 3A-2-5: site_config["navigation"]["header_search"] から取得
        nav_cfg = site_config.get("navigation", {}) or {}
        hs_cfg = nav_cfg.get("header_search", {}) or {}

        # フォールバック: 既存の ui 構造もサポート
        ui = (site_config.get("selectors", {}) or {}).get("ui", {}) or {}

        # Stage 4: 文字列とリストの両方に対応（site_configで文字列が設定されている場合がある）
        def _ensure_list(value):
            """文字列をリストに変換、リストはそのまま、None/空は空リスト"""
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, list):
                return value
            return []

        sel_open = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("search_open_selector"))
            + _ensure_list(ui.get("search_open"))
            + [
                "button[aria-label='Search']",
                "[aria-label*='Search' i]",
            ]
        )
        sel_input = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("search_input_selector"))
            + _ensure_list(ui.get("search_input"))
            + [
                "form[role='search'] input",
                "input[type='search']",
                "input[name='q']",
                "[data-testid*='search' i] input",
                "[role='search'] input",
                "dialog input[type='search']",
            ]
        )
        sel_submit = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("submit_selector"))
            + _ensure_list(ui.get("search_submit"))
            + ["form[role='search'] button[type='submit']"]
        )
        clear_before_type = hs_cfg.get("clear_before_type", True)

        try:
            opened = False
            for s in sel_open:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0:
                    await el.click(timeout=3000)
                    opened = True
                    await asyncio.sleep(0.2)
                    await self.safe_wait_selector(
                        page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible"
                    )
                    logger.debug(f"[Fallback] opened search with '{s}'")
                    break
            if not opened:
                await page.keyboard.press("/")
                await self.safe_wait_selector(page, "input[type='search']", timeout_ms=5000, state="visible")
            found_input = False
            for s in sel_input:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_visible():
                    # Stage 3A-2-5: clear_before_type が True の場合は先にクリア
                    if clear_before_type:
                        await el.clear(timeout=2000)
                    await el.fill(query, timeout=8000)
                    found_input = True
                    logger.debug(f"[Fallback] filled '{query}' into '{s}'")
                    break
            if not found_input:
                raise ValueError("Input field not found")
            submitted = False
            for s in sel_submit:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_enabled():
                    await el.click(timeout=5000)
                    submitted = True
                    logger.debug(f"[Fallback] submitted with '{s}'")
                    break
            if not submitted:
                await page.keyboard.press("Enter")
                logger.debug("[Fallback] submitted with Enter key.")
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    logger.debug("[Fallback] Optional main wait timed out.")
            return True
        except Exception:
            logger.warning("[Fallback] UI search failed. Trying direct search URL.")
            try:
                # Stage 4: site_configから検索URLテンプレートを取得（既存設定との互換性を確保）
                url_template = hs_cfg.get("url_template", "")
                base_url_key = hs_cfg.get("base_url", "home_url")

                # url_templateがリストの場合は最初の要素を使用
                if isinstance(url_template, list):
                    url_template = url_template[0] if url_template else ""

                if not url_template:
                    # フォールバック: discovery_settingsから取得（既存設定との互換性）
                    ds = site_config.get("discovery_settings", {}) or {}
                    url_templates = ds.get("url_templates", {}) or {}
                    url_template = url_templates.get("search", "")
                    # リストの場合は最初の要素を使用
                    if isinstance(url_template, list):
                        url_template = url_template[0] if url_template else ""

                if not url_template or not isinstance(url_template, str):
                    logger.warning("[Fallback] No valid search URL template found in site_config")
                    return False

                # ベースURLを取得
                if base_url_key == "home_url":
                    base_url = site_config.get("home_url", "")
                else:
                    base_url = site_config.get(base_url_key, site_config.get("home_url", ""))

                # base_urlがリストの場合は最初の要素を使用
                if isinstance(base_url, list):
                    base_url = base_url[0] if base_url else ""

                if not base_url or not isinstance(base_url, str):
                    logger.warning("[Fallback] No valid base URL found in site_config")
                    return False

                # URLテンプレートのプレースホルダを置換
                # {query} を置換
                search_url = url_template.replace("{query}", quote_plus(query))

                # {locale} を置換（locale.preferを使用）
                locale_cfg = site_config.get("locale", {}) or {}
                prefer_locale = locale_cfg.get("prefer", "")
                if "{locale}" in search_url and prefer_locale:
                    search_url = search_url.replace("{locale}", prefer_locale)

                # 相対URLの場合はbase_urlと結合
                if not search_url.startswith("http"):
                    from urllib.parse import urljoin

                    search_url = urljoin(base_url, search_url)

                logger.info(f"[Fallback] Using search URL from site_config: {search_url}")
                await page.goto(url=search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    logger.debug("[Fallback] Optional main wait (URL) timed out.")
                return True
            except Exception as final_e:
                logger.error(f"[Fallback] Direct search URL failed: {final_e}")
                return False

    async def click_first_card_or_link(self, ctx: NavigationContext) -> str | None:
        """
        Stage 3A-2-4:
        旧 BrowserUseAgent._click_first_card_or_link のロジックをここに移行。
        PLP 上の最初のカード/リンクをクリックして PDP へ遷移し、遷移先 URL を返す。ダメなら None。
        """
        page = self.page
        site_config = ctx.site_config
        context = ctx.context

        if context is None:
            logger.warning("[Fallback:click-card] BrowserContext is not available")
            return None

        # Stage 3A-2-5: site_config["navigation"]["fallback"]["click_first_card"] から取得
        nav_cfg = site_config.get("navigation", {}) or {}
        fb_cfg = nav_cfg.get("fallback", {}) or {}
        click_cfg = fb_cfg.get("click_first_card", {}) or {}

        # フォールバック: 既存の pdp 構造もサポート
        pdp = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}

        # enabled が False の場合はスキップ
        if not click_cfg.get("enabled", True):
            logger.debug("[Fallback:click-card] click_first_card is disabled in site_config")
            return None

        # card_selectors を取得（優先順位: navigation.fallback.click_first_card.card_selectors > selectors.plp.card_selectors > selectors.pdp.pdp_link_selectors）
        link_sel = _dedupe_keep_order(
            (click_cfg.get("card_selectors", []) or [])
            + (plp_selectors.get("card_selectors", []) or [])
            + (pdp.get("pdp_link_selectors", []) or [])
        )

        plp_boxes = _dedupe_keep_order(
            (plp_selectors.get("container_selectors", []) or [])
            + (pdp.get("plp_container_selectors", []) or [])
            + (["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"])
        )
        block_ng = set(
            (click_cfg.get("blocklist_href_substrings", []) or [])
            + (pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
        )
        url_pat = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

        if link_sel:
            for s in link_sel:
                try:
                    loc = page.locator(s)
                    count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i)
                        href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed()
                            newp = await self._click_and_capture_navigation(
                                lambda el=el: el.click(timeout=5000), page, context, url_regex=url_pat
                            )
                            if newp:
                                return newp.url
                except Exception:
                    continue

        # CR-E2E-003B拡張: MonclerPLPStrategyが検出できているセレクタを優先
        # site_configからtile_selectorsを取得（Moncler向けに最適化）
        tile_selectors_from_config = plp_selectors.get("tile_selectors", []) or []
        tile_selectors = tile_selectors_from_config + [
            "[data-qa='product-tile']",
            "[data-testid='product-tile']",
            "div[data-testid='product-tile']",
            "div[class*='product-tile' i]",
            "div.product-tile",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]",
        ]
        # 重複を除去
        tile_selectors = _dedupe_keep_order(tile_selectors)

        # まず、box スコープなしで直接 tile_selectors を試す
        for tile_sel in tile_selectors:
            try:
                card = page.locator(tile_sel).first
                # 要素が存在するか確認（CR-E2E-003B拡張: タイムアウトを5000msに延長）
                try:
                    await card.wait_for(state="attached", timeout=5000)
                except Exception as e:
                    logger.debug(f"[Fallback:click-card] wait_for failed for '{tile_sel}': {e}")
                    continue

                count = await card.count()
                if count > 0:
                    await card.scroll_into_view_if_needed(timeout=3000)
                    newp = await self._click_and_capture_navigation(
                        lambda card=card: card.click(timeout=5000), page, context, url_regex=url_pat
                    )
                    if newp:
                        return newp.url
            except Exception as e:
                logger.debug(f"[Fallback:click-card] Selector '{tile_sel}' failed: {e}")
                continue

        # box スコープ付きで試す（フォールバック）
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    # セレクタを組み立て
                    selector = f"{box} {tile_sel}".strip()
                    card = page.locator(selector).first

                    # 要素が存在するか確認（タイムアウトを短く設定）
                    # Stage 4: タイムアウトエラーを適切に処理（CancelledErrorを避ける）
                    try:
                        count = await card.count()
                        if count == 0:
                            continue
                        # 短いタイムアウトで確認（要素が存在する場合のみ）
                        await asyncio.wait_for(card.wait_for(state="attached", timeout=2000), timeout=2.5)
                    except asyncio.CancelledError:
                        logger.debug(f"[Fallback:click-card] Cancelled for '{tile_sel}'")
                        raise
                    except asyncio.TimeoutError:
                        logger.debug(f"[Fallback:click-card] Timeout for '{tile_sel}'")
                        continue
                    except Exception as e:
                        logger.debug(f"[Fallback:click-card] wait_for failed for '{tile_sel}': {e}")
                        continue

                    count = await card.count()
                    if count > 0:
                        await card.scroll_into_view_if_needed(timeout=3000)
                        newp = await self._click_and_capture_navigation(
                            lambda card=card: card.click(timeout=5000), page, context, url_regex=url_pat
                        )
                        if newp:
                            return newp.url
                except Exception as e:
                    logger.debug(f"[Fallback:click-card] Selector '{selector}' failed: {e}")
                    continue

        logger.warning("[Fallback:click-card] Could not find any clickable link or card.")
        return None
