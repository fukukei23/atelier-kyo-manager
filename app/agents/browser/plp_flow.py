"""PLP flow Mixin: product listing page exploration."""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.extractor import (
    VISIBLE_PRICE_SELECTORS,
    looks_like_product_url,
)
from app.agents.browser.navigation_driver import (
    _dedupe_keep_order,
    is_same_origin,
)
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import count_selectors, save_dom

logger = logging.getLogger(__name__)


class PlpFlowMixin:
    """BrowserUseAgent の PLP 探索フローを担当。"""

    _page: Page | None
    run_context: Any
    logger: logging.Logger

    async def _run_plp_flow(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        target_url: str,
        *,
        start_t: float,
        budget_ms: int,
        skip_materialize: bool = False,
    ) -> DiscoveryResult:
        attempted_recover = False
        if self._looks_like_trap_or_legal(page.url):
            self.logger.warning("[_looks_like_trap] initial trap-like url: %s", page.url)
            attempted_recover = True
            await self._force_plp_recover(page, site_config, target_url)
            if self._looks_like_trap_or_legal(page.url):
                raise ValueError(f"Landing page looks like legal/trap (even after recovery attempt): {page.url}")
            self.logger.info("[_looks_like_trap] Recovery navigation seems successful.")

        await self._pause_for_operator(page, run_context, "before_plp_materialize")

        if skip_materialize:
            ok_materialized = True
        else:
            ok_materialized = await self._ensure_plp_materialized(
                page, site_config, settings, start_t=start_t, budget_ms=budget_ms, target_url=target_url
            )

        if not ok_materialized:
            raise ValueError(f"PLP did not materialize (no product tiles). URL={page.url}")

        if self._looks_like_trap_or_legal(page.url):
            if not attempted_recover:
                self.logger.warning("[_looks_like_trap] trap-like url after materialize: %s", page.url)
                await self._force_plp_recover(page, site_config, target_url)
                if self._looks_like_trap_or_legal(page.url):
                    raise ValueError(
                        f"After materialize still on legal/trap page (even after recovery attempt): {page.url}"
                    )
                self.logger.info("[_looks_like_trap] Recovery navigation (post-materialize) seems successful.")
            else:
                raise ValueError(f"After materialize, bounced back to legal/trap page: {page.url}")

        try:
            await save_dom(run_context, page, "plp_dom_initial_materialized")
            pdp_cfg_a = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            await count_selectors(
                run_context, page,
                (pdp_cfg_a.get("pdp_link_selectors") or []) + (pdp_cfg_a.get("plp_container_selectors") or []),
                name="selector_counts_plp_initial",
            )
        except Exception as e:
            logger.warning(f"[Hook A1] Failed: {e}")

        pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)

        if not pdp_links:
            if not pdp_links and self._looks_like_trap_or_legal(page.url):
                raise ValueError(f"No PDP links and URL looks like trap/legal page: {page.url}")

            try:
                self.logger.debug("[Fallback] Trying header search UI...")
                did_search = await self._plp_header_search_fallback(
                    page, query, site_config, settings, run_context, context, start_t=start_t, budget_ms=budget_ms
                )
                if did_search:
                    await self._click_continue_shopping_if_present(page, site_config)
                    try:
                        anchors = await page.locator("a[href*='/p/'], a[href*='/product/']").count()
                    except Exception:
                        anchors = 0
                    if anchors < 6:
                        self.logger.debug(f"[Fallback] Materializing after search (anchors={anchors}<6)")
                        await self._ensure_plp_materialized(
                            page, site_config, settings, start_t=start_t, budget_ms=budget_ms, target_url=target_url
                        )
                    try:
                        await save_dom(run_context, page, "plp_dom_search_fallback")
                        pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                        await count_selectors(
                            run_context, page,
                            (pdp_cfg_a2.get("pdp_link_selectors") or [])
                            + (pdp_cfg_a2.get("plp_container_selectors") or []),
                            name="selector_counts_after_search_fallback",
                        )
                    except Exception as e:
                        logger.warning(f"[Hook A3] Failed: {e}")
                    pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)

                    if not pdp_links:
                        self.logger.warning("[Fallback] No hrefs after search. Clicking first card...")
                        new_page = await self._click_first_card_or_link(page, site_config, settings, context)
                        if new_page:
                            return await self._run_pdp_flow(
                                new_page or page, site, query, settings, run_context, site_config
                            )
                        raise ValueError("No PDP links and click fallback failed (gave up early for speed).")

            except Exception as _e:
                self.logger.warning(f"[Fallback:header-search] failed or gave up early: {_e}", exc_info=True)
                pass

        if pdp_links:
            prepare_hook = self._build_pdp_prepare_hook(
                site_config=site_config, settings=settings, run_context=run_context
            )
            return await self.extraction_service.extract_from_pdp_list(
                page=page, context=context, site=site, query=query,
                pdp_links=pdp_links, site_config=site_config,
                settings=settings, run_context=run_context,
                start_t=start_t, budget_ms=budget_ms, prepare_page=prepare_hook,
            )
        raise ValueError("All PDP attempts failed after all recovery attempts.")

    async def _ensure_plp_materialized(
        self,
        page: Page,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        *,
        start_t: float,
        budget_ms: int,
        target_url: str | None = None,
    ) -> bool:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        tile_selectors = _dedupe_keep_order(
            (pdp_cfg.get("pdp_link_selectors") or [])
            + (pdp_cfg.get("plp_container_selectors") or [])
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
        run_ctx = getattr(self, "run_context", None)

        locale_recover_attempts = 0
        locale_recover_max = int(settings.get("locale_recover_max", 5))

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                self.logger.warning("[Materialize] Timed out.")
                return False

            with contextlib.suppress(Exception):
                await self._accept_cookies_if_present(page, site_config)
            with contextlib.suppress(Exception):
                await self._dismiss_geo_modal(page)
            with contextlib.suppress(Exception):
                await self._kill_overlays(page)

            current_url = (page.url or "").lower()
            if "moncler.com/en-gb" in current_url:
                self.logger.warning("[Materialize] Detected EN-GB redirect mid-attempt.")
                if locale_recover_attempts >= locale_recover_max:
                    self.logger.error("[Materialize] Locale recovery exceeded max attempts. Aborting.")
                    return False
                locale_recover_attempts += 1
                if target_url:
                    await self._force_plp_recover(page, site_config, target_url)
                    await page.wait_for_timeout(800)
                    continue

            if run_ctx is not None and attempt < 3:
                try:
                    await run_ctx.take_screenshot(page, f"30_plp_materialize_attempt_{attempt + 1:02d}")
                except Exception as ss_e:
                    self.logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=800)
            except Exception as e:
                self.logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}")
                break

            try:
                modal_title = page.locator("text=Select your location").first
                if await modal_title.count() > 0:
                    close_btn = page.locator(
                        "button[aria-label*='close' i], "
                        "button:has-text('Close'), "
                        "button:has-text('×'), "
                        ".modal__close, .c-modal__close"
                    ).first
                    if await close_btn.count() > 0:
                        await close_btn.click(timeout=3000)
                        await page.wait_for_timeout(500)
                        self.logger.info("[GeoModal] Locale gate closed.")
            except Exception as e:
                self.logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

            try:
                count = await page.locator(tile_selector_str).count()
                self.logger.info(f"[Materialize] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")
                if count >= target_min_tiles:
                    self.logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
                if count < 4 and attempt >= 1:
                    self.logger.warning(
                        f"[Materialize] Low tiles ({count}) after {attempt + 1} attempts, forcing recovery hop."
                    )
                    if target_url:
                        try:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(500)
                            rec_count = await page.locator(tile_selector_str).count()
                            self.logger.info(f"[Materialize] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return True
                        except Exception as rec_e:
                            self.logger.warning(f"[Materialize] Recovery hop failed: {rec_e}")
                    return False
            except Exception as e:
                self.logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
        if final_count > 0:
            self.logger.warning(
                f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty."
            )
            return True
        self.logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
        return False

    async def _plp_header_search_fallback(
        self,
        page,
        query: str,
        site_config,
        settings,
        run_context,
        context: BrowserContext,
        *,
        start_t: float,
        budget_ms: int,
    ) -> bool:
        from urllib.parse import quote_plus

        ui = (site_config.get("selectors") or {}).get("ui") or {}
        sel_open = _dedupe_keep_order(
            ui.get("search_open", []) + ["button[aria-label='Search']", "[aria-label*='Search' i]"]
        )
        sel_input = _dedupe_keep_order(
            ui.get("search_input", [])
            + [
                "form[role='search'] input",
                "input[type='search']",
                "input[name='q']",
                "[data-testid*='search' i] input",
                "[role='search'] input",
                "dialog input[type='search']",
            ]
        )
        sel_submit = _dedupe_keep_order(ui.get("search_submit", []) + ["form[role='search'] button[type='submit']"])
        try:
            opened = False
            for s in sel_open:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0:
                    await el.click(timeout=3000)
                    opened = True
                    import asyncio
                    await asyncio.sleep(0.2)
                    await self.safe_wait_selector(
                        page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible"
                    )
                    self.logger.debug(f"[Fallback] opened search with '{s}'")
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
                    await el.fill(query, timeout=8000)
                    found_input = True
                    self.logger.debug(f"[Fallback] filled '{query}' into '{s}'")
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
                    self.logger.debug(f"[Fallback] submitted with '{s}'")
                    break
            if not submitted:
                await page.keyboard.press("Enter")
                self.logger.debug("[Fallback] submitted with Enter key.")
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main wait timed out.")
            return True
        except Exception:
            self.logger.warning("[Fallback] UI search failed. Trying direct search URL.")
            try:
                search_url = f"https://www.moncler.com/en-int/search?q={quote_plus(query)}&forceLocale=en-int"
                await page.goto(url=search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main wait (URL) timed out.")
                return True
            except Exception as final_e:
                self.logger.error(f"[Fallback] Direct search URL failed: {final_e}")
                return False
