# ==============================================================================
# File: app/agents/browser/nav_plp_materializer.py
# Purpose: NavPlpMaterializerMixin - PLP materialization, trap detection, recovery
# ==============================================================================
"""
NavigationDriver から PLP マテリアライズ・トラップ検出・リカバリを抽出した Mixin。

抽出対象メソッド:
- ensure_plp_materialized
- recover_plp
- _detect_trap_page
- _click_first_card
- safe_wait_selector
- _time_left_ms
- _accept_cookies_if_present
- _kill_overlays
- _click_continue_shopping_if_present
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from playwright.async_api import Page

from app.agents.browser.nav_types import _LOCALE_SEG_RE, NavigationContext
from app.agents.browser.ui_helpers import (
    accept_cookies_if_present as ui_accept_cookies_if_present,
)
from app.agents.browser.ui_helpers import (
    click_continue_shopping_if_present as ui_click_continue_shopping_if_present,
)
from app.agents.browser.ui_helpers import (
    kill_overlays as ui_kill_overlays,
)
from app.agents.browser.ui_helpers import (
    safe_wait_selector as ui_safe_wait_selector,
)

logger = logging.getLogger(__name__)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


class NavPlpMaterializerMixin:
    """PLP マテリアライズ・トラップ検出・リカバリを担当する Mixin。"""

    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        """セレクタが出現するまで安全に待機する。ui_helpers に委譲。"""
        if ui_safe_wait_selector is not None:
            return await ui_safe_wait_selector(page, selector, timeout_ms=timeout_ms, state=state)
        if not page or page.is_closed():
            return False
        try:
            await asyncio.wait_for(
                page.wait_for_selector(selector, state=state, timeout=timeout_ms), timeout=(timeout_ms / 1000.0) + 0.5
            )
            return True
        except asyncio.CancelledError:
            logger.debug(f"[safe_wait_selector] Cancelled for '{selector}'")
            raise
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[safe_wait_selector] Timeout/Error for '{selector}': {e}")
            return False

    def _time_left_ms(self, start_t: float, budget_ms: int) -> int:
        """残り時間をミリ秒で返す"""
        used = int((time.monotonic() - start_t) * 1000)
        return max(0, budget_ms - used)

    async def _accept_cookies_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        """Cookie 同意バナーがあればクリックする。ui_helpers に委譲。"""
        if ui_accept_cookies_if_present is not None:
            return await ui_accept_cookies_if_present(page, site_config)
        return False

    async def _kill_overlays(self, page: Page, site_config: dict[str, Any] | None = None) -> None:
        """オーバーレイを削除する。ui_helpers に委譲。"""
        if ui_kill_overlays is not None:
            await ui_kill_overlays(page)
            return
        with contextlib.suppress(Exception):
            await page.evaluate("""
              (() => {
                const sels = ['.overlay','.backdrop','.modal-backdrop','#onetrust-banner-sdk','.cookie-banner','[aria-modal="true"]','.cmp-ui-overlay','.cmp-modal','.drawer--open'];
                document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                const b = document.body; if (b) { b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }
                const html=document.documentElement; if (html) { html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }
              })();
            """)

    async def _click_continue_shopping_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        """CONTINUE SHOPPING ボタンがあればクリックする。ui_helpers に委譲。"""
        if ui_click_continue_shopping_if_present is not None:
            return await ui_click_continue_shopping_if_present(page, site_config)
        return False

    async def recover_plp(self, ctx: NavigationContext) -> bool:
        """
        PLP を回復する（強制的に PLP URL にナビゲート）。

        Returns:
            bool: 成功したら True / 失敗したら False
        """
        page: Page = self.page  # type: ignore[attr-defined]
        site_config = ctx.site_config
        target_url = ctx.entry_url

        try:
            await self._force_plp_recover(page, site_config, target_url)  # type: ignore[attr-defined]
            if self._looks_like_trap_or_legal(page.url, site_config):  # type: ignore[attr-defined]
                logger.warning(f"[recover_plp] Still trap-like after recovery: {page.url}")
                return False
            return True
        except Exception as e:
            logger.debug(f"[recover_plp] Recovery failed: {e}")
        return False

    async def _detect_trap_page(self, ctx: NavigationContext) -> dict[str, str] | None:
        """
        PLP ではない状態（404 / location gate / 想定外ロケール＋検索ページ）を検出する。

        Returns:
            trap を検出した場合は dict（type, reason を含む）、検出されなければ None
        """
        page: Page = self.page  # type: ignore[attr-defined]
        site_config = ctx.site_config
        current_url = page.url or ""

        try:
            # 1. 404 ページの検出
            try:
                h1_text = await page.locator("h1").first.inner_text(timeout=2000)
                if h1_text and "It's not here" in h1_text:
                    return {
                        "type": "404",
                        "reason": f"h1 contains 'It's not here': {h1_text[:50]}",
                    }
            except Exception:
                pass

            url_lower = current_url.lower()
            if "/404" in url_lower or "not-found" in url_lower:
                return {
                    "type": "404",
                    "reason": f"URL contains /404 or not-found: {current_url}",
                }

            # 2. Location gate の検出
            try:
                body_text = await page.locator("body").first.inner_text(timeout=2000)
                if body_text and "Select your location" in body_text:
                    plp_cfg = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
                    pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

                    product_selectors = [
                        "[data-component='ProductCard']",
                        "[data-testid*='product']",
                        "article[data-product-id]",
                        ".product-card",
                        ".c-product-card",
                        "a[href*='/products/']",
                    ]
                    product_selectors.extend(
                        (plp_cfg.get("tile_selectors", []) or []) + (pdp_cfg.get("pdp_link_selectors", []) or [])
                    )

                    product_found = False
                    for sel in product_selectors[:5]:
                        try:
                            count = await page.locator(sel).count()
                            if count > 0:
                                product_found = True
                                break
                        except Exception:
                            continue

                    if not product_found:
                        return {
                            "type": "location_gate",
                            "reason": "Contains 'Select your location' but no product cards found",
                        }
            except Exception:
                pass

            # 3. 想定外ロケール＋検索ページの検出
            locale_cfg = site_config.get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "en-int")
            target_locale_path = f"/{prefer_locale}/"

            unexpected_locale_patterns = [
                "/en-lt/",
                "/en-de/",
                "/en-fr/",
                "/en-jp/",
                "/en-us/",
            ]

            if "/search" in url_lower:
                for pattern in unexpected_locale_patterns:
                    if pattern in url_lower and target_locale_path not in url_lower:
                        return {
                            "type": "unexpected_locale_search",
                            "reason": f"URL contains unexpected locale '{pattern}' and '/search': {current_url}",
                        }

            if target_locale_path in url_lower and "/search" in url_lower:
                for pattern in unexpected_locale_patterns:
                    if pattern in url_lower:
                        return {
                            "type": "unexpected_locale_search",
                            "reason": f"URL contains double locale pattern '{pattern}' and '/search': {current_url}",
                        }

            return None

        except Exception as e:
            logger.debug(f"[TrapDetector] Error during trap page detection: {e}", exc_info=True)
            return None

    async def _click_first_card(
        self,
        page: Page,
        site_config: dict[str, Any],
    ) -> Page | None:
        """
        最初のカードをクリックする（骨組みのみ）。

        Returns:
            Optional[Page]: クリック後の新しい Page（存在する場合）、または None
        """
        logger.debug("[NavigationDriver] _click_first_card (stub): returning None")
        return None

    async def ensure_plp_materialized(self, ctx: NavigationContext) -> bool:
        """
        PLP をスクロールしてタイルが十分に出るまで待つ処理。

        Returns:
            bool: タイルが十分に見つかったら True
        """
        page: Page = self.page  # type: ignore[attr-defined]
        site_config = ctx.site_config
        settings = ctx.settings
        start_t = ctx.start_t
        budget_ms = ctx.budget_ms
        target_url = ctx.entry_url

        plp_cfg = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

        tile_selectors = _dedupe_keep_order(
            (plp_cfg.get("tile_selectors", []) or [])
            + (plp_cfg.get("pdp_link_selectors", []) or [])
            + (pdp_cfg.get("pdp_link_selectors", []) or [])
            + (pdp_cfg.get("plp_container_selectors", []) or [])
            + [
                "a[data-product-url]",
                "[data-product-url]",
                "[data-qa='product-tile']",
                ".product-card",
                ".c-product-card",
                ".c-product-tile",
                "[data-testid*='product' i]",
            ]
        )
        tile_selector_str = ", ".join(tile_selectors)
        target_min_tiles = 8
        max_scroll_attempts = int(max(settings.get("plp_scroll_rounds", 10), 10))
        run_ctx = ctx.run_context

        locale_recover_attempts = 0
        locale_recover_max = int(settings.get("locale_recover_max", 5))

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                logger.warning("[Materialize] Timed out.")
                return False

            if attempt == 0:
                try:
                    cookie_closed = await self._accept_cookies_if_present(page, site_config)
                    if cookie_closed:
                        logger.info("[Materialize] Cookie banner closed, waiting for content to load...")
                        await page.wait_for_timeout(1000)
                except Exception as e:
                    logger.debug(f"[Materialize] Cookie banner handling failed: {e}")
            else:
                with contextlib.suppress(Exception):
                    await self._accept_cookies_if_present(page, site_config)

            try:
                try:
                    await asyncio.wait_for(self._dismiss_geo_modal(page, site_config), timeout=10.0)  # type: ignore[attr-defined]
                except asyncio.TimeoutError:
                    logger.debug("[Materialize] Geo modal dismissal timed out (non-fatal), continuing")
                except Exception as geo_e:
                    logger.debug(f"[Materialize] Geo modal dismissal failed (non-fatal): {geo_e}")
            except Exception as geo_e:
                logger.debug(f"[Materialize] Geo modal dismissal failed (non-fatal): {geo_e}")
                pass
            with contextlib.suppress(Exception):
                await self._kill_overlays(page, site_config)

            # ロケールリダイレクトの検出
            current_url = (page.url or "").lower()
            locale_cfg = site_config.get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "")
            allowed_domain = site_config.get("allowed_domain", "")

            if prefer_locale and allowed_domain:
                target_locale_path = f"/{prefer_locale}/"
                if (
                    allowed_domain.lower() in current_url
                    and target_locale_path not in current_url
                    and _LOCALE_SEG_RE.search(current_url)
                ):
                    logger.warning(
                        f"[Materialize] Detected locale redirect away from {prefer_locale} mid-attempt: {current_url}"
                    )
                    if locale_recover_attempts >= locale_recover_max:
                        logger.error("[Materialize] Locale recovery exceeded max attempts. Aborting.")
                        return False
                    locale_recover_attempts += 1
                    if target_url:
                        await self._force_plp_recover(page, site_config, target_url)  # type: ignore[attr-defined]
                        await page.wait_for_timeout(800)
                        try:
                            await self._ensure_expected_locale(ctx)  # type: ignore[attr-defined]
                        except Exception as locale_e:
                            logger.warning(
                                f"[Materialize] Locale Guard after locale redirect recovery failed: {locale_e}",
                                exc_info=True,
                            )
                        continue

            if run_ctx is not None and hasattr(run_ctx, "take_screenshot") and attempt < 3:
                try:
                    await run_ctx.take_screenshot(page, f"30_plp_materialize_attempt_{attempt + 1:02d}")
                except Exception as ss_e:
                    logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=800)
            except Exception as e:
                logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}")
                break

            # ロケールゲートが途中で出た場合に備えて閉じておく
            try:
                modal_title = page.locator("text=Select your location").first
                if await modal_title.count() > 0:
                    logger.info("[GeoModal] Locale gate header detected during PLP materialization.")
                    close_btn = page.locator(
                        "button[aria-label*='close' i], "
                        "button:has-text('Close'), "
                        "button:has-text('×'), "
                        ".modal__close, .c-modal__close"
                    ).first
                    if await close_btn.count() > 0:
                        await close_btn.click(timeout=3000)
                        await page.wait_for_timeout(500)
                        logger.info("[GeoModal] Locale gate closed.")
            except Exception as e:
                logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

            try:
                count = await page.locator(tile_selector_str).count()
                logger.info(f"[Materialize] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")

                if count == 0 and attempt < 2:
                    try:
                        banner_container = page.locator("#onetrust-banner-sdk, #onetrust-pc-sdk")
                        if await banner_container.count() > 0:
                            is_visible = await banner_container.first.is_visible()
                            if is_visible:
                                logger.warning(
                                    "[Materialize] Cookie banner still visible, attempting to close again..."
                                )
                                await self._accept_cookies_if_present(page, site_config)
                                await page.wait_for_timeout(1500)
                                count = await page.locator(tile_selector_str).count()
                                logger.info(f"[Materialize] After closing banner, found {count} tiles.")
                    except Exception as e:
                        logger.debug(f"[Materialize] Cookie banner check failed: {e}")

                if count >= target_min_tiles:
                    logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
                if count < 4 and attempt >= 1:
                    logger.warning(
                        f"[Materialize] Low tiles ({count}) after {attempt + 1} attempts, forcing recovery hop."
                    )
                    if target_url:
                        try:
                            await self._force_plp_recover(page, site_config, target_url)  # type: ignore[attr-defined]
                            await page.wait_for_timeout(500)
                            try:
                                await self._ensure_expected_locale(ctx)  # type: ignore[attr-defined]
                            except Exception as locale_e:
                                logger.warning(
                                    f"[Materialize] Locale Guard after recovery hop failed: {locale_e}", exc_info=True
                                )
                            rec_count = await page.locator(tile_selector_str).count()
                            logger.info(f"[Materialize] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return True
                        except Exception as rec_e:
                            logger.warning(f"[Materialize] Recovery hop failed: {rec_e}")
                    return False
            except asyncio.CancelledError:
                logger.warning("[Materialize] Cancelled during tile count.")
                return False
            except Exception as e:
                logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
        if final_count > 0:
            logger.warning(
                f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty."
            )
            return True
        logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
        return False
