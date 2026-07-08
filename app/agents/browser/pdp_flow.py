"""PDP flow Mixin: product detail page extraction and navigation."""

from __future__ import annotations

import asyncio
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
from app.utils.observability import save_raw_hrefs

logger = logging.getLogger(__name__)

PRICE_SELECTORS = VISIBLE_PRICE_SELECTORS


class PdpFlowMixin:
    """BrowserUseAgent の PDP 抽出・ナビゲーションを担当。"""

    _page: Page | None
    _context: BrowserContext | None
    run_context: Any
    logger: logging.Logger

    async def _run_pdp_flow(
        self,
        page: Page,
        site: str,
        query: str,
        settings: dict,
        run_context: RunContext,
        site_config: dict[str, Any],
    ) -> None:
        logger.info("[Mode] PDP (detail)")
        prepare_hook = self._build_pdp_prepare_hook(site_config=site_config, settings=settings, run_context=run_context)
        return await self.extraction_service.extract_single_pdp(
            page=page,
            context=self._context,
            site=site,
            query=query,
            settings=settings,
            run_context=run_context,
            site_config=site_config,
            target_url=page.url,
            prepare_page=prepare_hook,
        )

    async def _read_price_or_none(self, page: Page) -> str | None:
        try:
            for sel in PRICE_SELECTORS:
                loc = page.locator(sel)
                count = await loc.count()
                if count == 0:
                    continue
                for i in range(count):
                    el = loc.nth(i)
                    try:
                        tag = (await el.evaluate("e => e && e.tagName") or "").lower()
                        if not tag:
                            continue
                        content = await (el.get_attribute("content") if tag == "meta" else el.inner_text())
                    except Exception:
                        continue
                    if content:
                        s = content.strip()
                        if s and re.search(r"\d", s):
                            self.logger.debug(f"Price found via selector '{sel}' (nth={i}): {s}")
                            return s
        except Exception as e:
            self.logger.warning(f"Error: Exception during _read_price_or_none (outer loop): {e}")
        self.logger.debug("Price string not found (_read_price_or_none).")
        return None

    def _normalize_abs_url(self, base_url: str, href: str) -> str:
        from app.agents.browser.navigation_helpers import normalize_abs_url

        return normalize_abs_url(base_url, href)

    async def _collect_pdp_links(
        self, page: Page, site_config: dict, settings: dict, run_context: RunContext
    ) -> list[str]:
        target_url = page.url
        found_links: set[str] = set()

        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
        except Exception as e:
            logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}")
            raw_hrefs = []
        pdp_rx = re.compile(r"/(products?|p)/", re.I)
        for href in raw_hrefs:
            if pdp_rx.search(href):
                norm_url = self._normalize_abs_url(target_url, href)
                if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                    found_links.add(norm_url)
        if found_links:
            logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")

        selectors_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        PLP_PDP_LINK_SELECTORS = _dedupe_keep_order(
            (selectors_cfg.get("pdp_link_selectors", []) or [])
            + [
                "a[href*='/products/']",
                "a[href*='/product/']",
                "a[href*='/p/']",
                "[data-component*='ProductCard'] a[href]",
                "[class*='product-card'] a[href]",
                "article [data-testid*='product']:is(a, * a)",
                "[data-testid*='card'] a[href]",
                "[data-testid*='product-card'] a[href]",
                "a[data-product-url]",
                "[data-qa='product-tile'] a[href]",
            ]
        )
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
                    if not href:
                        continue
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                        found_links.add(norm_url)
                        matched_count += 1
                if matched_count > 0:
                    logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
            except Exception as e:
                logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")

        if not found_links:
            logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
            try:
                deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)
                for href in deep_hrefs:
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                        found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
            except Exception as e:
                logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")

        links = sorted(list(found_links))
        if not links:
            logger.warning("[PLP→PDP] No PDP hrefs found after all phases.")
            return []

        cleaned: list[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u):
                cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")
        try:
            sample = cleaned[:20]
            logger.debug(f"[PLP→PDP] sample={sample}")
            if self.run_context:
                self.run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
            try:
                if callable(save_raw_hrefs) and run_context:
                    res = save_raw_hrefs(run_context, cleaned, name="raw_hrefs_final_cleaned")
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass
        except Exception:
            pass
        return cleaned

    async def _run_deep_extraction_phase2(self, page: Page, site_config: dict) -> list[str]:
        from app.agents.browser.deep_extraction import run_deep_extraction_phase2

        return await run_deep_extraction_phase2(page, site_config, self.safe_wait_selector)

    async def _click_and_capture_navigation(
        self,
        click_coro,
        page: Page,
        context: BrowserContext,
        *,
        url_regex: re.Pattern | None = re.compile(r"/product[s]?/|/p/|/pp/", re.I),
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Detect navigation after click — delegates to shared helper."""
        from app.agents.browser.nav_fallbacks import click_and_capture_nav

        return await click_and_capture_nav(
            click_coro,
            page,
            context,
            url_regex=url_regex,
            wait_state=wait_state,
            timeout_ms=timeout_ms,
        )

    async def _click_first_card_or_link(
        self, page: Page, site_config: dict, settings: dict, context: BrowserContext
    ) -> Page | None:
        pdp = site_config.get("selectors", {}).get("pdp") or {}
        link_sel = pdp.get("pdp_link_selectors", [])
        plp_boxes = pdp.get(
            "plp_container_selectors",
            ["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"],
        )
        block_ng = set(pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
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
                                return newp
                except Exception:
                    continue
        tile_selectors = [
            "[data-qa='product-tile']",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]",
        ]
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    card = page.locator(f"{box} {tile_sel}").first
                    await card.scroll_into_view_if_needed()
                    if await card.count() > 0:
                        newp = await self._click_and_capture_navigation(
                            lambda card=card: card.click(timeout=5000), page, context, url_regex=url_pat
                        )
                        if newp:
                            return newp
                except Exception:
                    continue
        self.logger.warning("[Fallback:click-card] Could not find any clickable link or card.")
        return None
