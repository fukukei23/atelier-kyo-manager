from __future__ import annotations

import asyncio
import logging
import re
from re import Pattern
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

from app.agents.browser.plp_config import get_plp_config


class PlpNavigationMixin:
    """PLP → PDP navigation mixin for PlpDriver."""

    page: Page
    context: BrowserContext
    site_config: dict[str, Any]
    logger: logging.Logger

    def _get_plp_config(self) -> dict[str, Any]:
        return get_plp_config(self.site_config)

    async def _click_tile_and_navigate(
        self,
        *,
        plp_config: dict[str, Any],
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Click a tile and navigate to PDP."""
        return await self._click_tile_and_navigate_to_pdp(
            url_regex=None,
            timeout_ms=timeout_ms,
        )

    async def _click_tile_and_navigate_to_pdp(
        self,
        *,
        url_regex: Pattern[str] | None = None,
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Click a product tile/link and navigate to PDP via race detection."""
        if url_regex is None:
            url_regex = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

        plp_config = self._get_plp_config()
        pdp_cfg = (self.site_config.get("selectors", {}) or {}).get("pdp") or {}

        product_link = plp_config.get("product_link", [])
        container = plp_config.get("container", [])
        click_strategy = plp_config.get("click_strategy", "link")

        link_sel = product_link or pdp_cfg.get("pdp_link_selectors", [])
        plp_boxes = container or pdp_cfg.get(
            "plp_container_selectors",
            ["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"],
        )
        block_ng = set(pdp_cfg.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))

        # Try link selectors
        if link_sel and click_strategy in ("link", "both"):
            for s in link_sel:
                try:
                    loc = self.page.locator(s)
                    count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i)
                        href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed()
                            new_page = await self._click_and_wait_for_navigation(
                                lambda el=el: el.click(timeout=5000),
                                url_regex=url_regex,
                                timeout_ms=timeout_ms,
                            )
                            if new_page:
                                return new_page
                except Exception:
                    continue

        # Try tile selectors
        if click_strategy in ("tile", "both"):
            product_tiles = plp_config.get("product_tiles", [])
            tile_selectors = product_tiles or [
                "[data-qa='product-tile']",
                ".c-product-tile",
                ".product-card",
                "[data-testid*='product-card']",
                "article[data-product-id]",
            ]
            for box in plp_boxes:
                for tile_sel in tile_selectors:
                    try:
                        card = self.page.locator(f"{box} {tile_sel}").first
                        await card.scroll_into_view_if_needed()
                        if await card.count() > 0:
                            new_page = await self._click_and_wait_for_navigation(
                                lambda card=card: card.click(timeout=5000),
                                url_regex=url_regex,
                                timeout_ms=timeout_ms,
                            )
                            if new_page:
                                return new_page
                    except Exception:
                        continue

        self.logger.warning("[PlpDriver] Could not find any clickable link or card.")
        return None

    async def _click_and_wait_for_navigation(
        self,
        click_coro,
        *,
        url_regex: Pattern[str] | None,
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Wait for navigation after click (delegates to _click_and_capture_navigation)."""
        return await self._click_and_capture_navigation(
            click_coro=click_coro,
            url_regex=url_regex,
            timeout_ms=timeout_ms,
        )

    async def _click_and_capture_navigation(
        self,
        click_coro,
        *,
        url_regex: Pattern[str] | None,
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Page | None:
        """Detect navigation after tile click — delegates to shared helper."""
        from app.agents.browser.nav_fallbacks import click_and_capture_nav
        return await click_and_capture_nav(
            click_coro, self.page, self.context,
            url_regex=url_regex, wait_state=wait_state, timeout_ms=timeout_ms,
        )
