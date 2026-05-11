from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.agents.browser.plp_config import get_plp_config
from app.agents.browser.plp_config import get_trap_config as _get_trap_config


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys([i for i in (items or []) if i]))


class PlpTileMaterializerMixin:
    """PLP tile materialization mixin for PlpDriver."""

    page: Page
    site_config: dict[str, Any]
    logger: logging.Logger

    def _get_plp_config(self) -> dict[str, Any]:
        return get_plp_config(self.site_config)

    async def _materialize_tiles(
        self,
        *,
        plp_config: dict[str, Any],
        start_t: float,
        budget_ms: int,
        target_url: str | None = None,
    ) -> int:
        """Materialize PLP tiles (delegates to _materialize_plp_tiles)."""
        return await self._materialize_plp_tiles(
            start_t=start_t,
            budget_ms=budget_ms,
            target_url=target_url,
        )

    async def _materialize_plp_tiles(
        self,
        *,
        start_t: float,
        budget_ms: int,
        target_url: str | None = None,
    ) -> int:
        """Scroll PLP and wait for product tiles to materialize."""
        plp_config = self._get_plp_config()

        product_tiles = plp_config.get("product_tiles", [])
        product_link = plp_config.get("product_link", [])
        container = plp_config.get("container", [])

        tile_selectors = _dedupe_keep_order(
            product_link
            + product_tiles
            + container
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
        tile_selector_str = ", ".join(tile_selectors) if tile_selectors else "a[href*='/product'], a[href*='/p/']"

        target_min_tiles = plp_config.get("min_tiles", 8)
        max_scroll_attempts = int(plp_config.get("max_scroll_rounds", 10))
        scroll_pause_ms = plp_config.get("scroll_pause_ms", 160)
        target_load_state = plp_config.get("target_load_state", "networkidle")
        wait_for_selectors = plp_config.get("wait_for_selectors", [])

        locale_recover_attempts = 0
        locale_recover_max = 5

        for sel in wait_for_selectors:
            with contextlib.suppress(Exception):
                await self.page.wait_for_selector(sel, state="visible", timeout=2000)

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                self.logger.warning("[PlpDriver] Materialize timed out.")
                break

            overlays_handled: list[str] = []
            await self._handle_overlays(overlays_handled)

            if target_url and self._detected_locale_redirect():
                if locale_recover_attempts >= locale_recover_max:
                    self.logger.error("[PlpDriver] Locale recovery exceeded max attempts.")
                    break
                locale_recover_attempts += 1
                trap_config = _get_trap_config(self.site_config)
                await self._recover_from_trap(target_url=target_url, trap_config=trap_config)
                await self.page.wait_for_timeout(800)
                continue

            try:
                scroll_steps = 6
                for _ in range(scroll_steps):
                    await self.page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await self.page.wait_for_timeout(scroll_pause_ms)
                with contextlib.suppress(Exception):
                    await self.page.wait_for_load_state(target_load_state, timeout=800)
            except Exception as e:
                self.logger.warning(f"[PlpDriver] Scroll failed on attempt {attempt + 1}: {e}")
                break

            try:
                count = await self.page.locator(tile_selector_str).count()
                self.logger.info(f"[PlpDriver] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")
                if count >= target_min_tiles:
                    self.logger.info(f"[PlpDriver] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return count
                if count < 4 and attempt >= 1:
                    self.logger.warning(f"[PlpDriver] Low tiles ({count}) after {attempt + 1} attempts.")
                    if target_url:
                        try:
                            trap_config = _get_trap_config(self.site_config)
                            await self._recover_from_trap(target_url=target_url, trap_config=trap_config)
                            await self.page.wait_for_timeout(500)
                            rec_count = await self.page.locator(tile_selector_str).count()
                            self.logger.info(f"[PlpDriver] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return rec_count
                        except Exception as rec_e:
                            self.logger.warning(f"[PlpDriver] Recovery hop failed: {rec_e}")
                    return count if count > 0 else 0
            except asyncio.CancelledError:
                self.logger.warning("[PlpDriver] Cancelled during tile count.")
                break
            except Exception as e:
                self.logger.warning(f"[PlpDriver] Could not count tiles on attempt {attempt + 1}: {e}")

        try:
            final_count = await self.page.locator(tile_selector_str).count()
            if final_count > 0:
                self.logger.warning(
                    f"[PlpDriver] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding."
                )
                return final_count
        except Exception:
            pass

        self.logger.error("[PlpDriver] Failed: No product tiles found after all scroll attempts.")

        await self._save_materialize_failure_evidence(tile_selector_str, plp_config)

        return 0
