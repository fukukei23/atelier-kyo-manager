from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.agents.browser.plp_config import get_overlay_config


class PlpOverlayMixin:
    """Overlay handling mixin for PlpDriver."""

    page: Page
    site_config: dict[str, Any]
    logger: logging.Logger

    def _get_overlay_config(self) -> dict[str, Any]:
        return get_overlay_config(self.site_config)

    async def _handle_overlays(self, overlays_handled: list[str]) -> None:
        """Handle Cookie, Geo, and other overlays."""
        overlay_config = self._get_overlay_config()

        if await self._handle_cookie_banner(overlay_config["cookie"]):
            overlays_handled.append("cookie")

        if await self._handle_geo_modal(overlay_config["geo"]):
            overlays_handled.append("geo")

        await self._remove_generic_overlays(overlay_config["other"])

    async def _handle_cookie_banner(self, config: dict[str, Any]) -> bool:
        """Handle cookie banner with progressive fallback."""
        try:
            selectors = config.get("selectors", [])
            wait_ms = config.get("wait_after_click_ms", 500)

            if not selectors:
                selectors = [
                    "#accept-recommended-btn-handler",
                    "button#accept-recommended-btn-handler",
                    "#onetrust-accept-btn-handler",
                    "button:has-text('ACCEPT ALL')",
                    "button[id*='accept'][id*='btn']",
                    "button:has-text('CONTINUE WITHOUT ACCEPTING')",
                    "button[aria-label*='Accept' i]",
                ]

            for sel in selectors:
                try:
                    loc = self.page.locator(sel).first
                    count = await loc.count()

                    if count == 0:
                        continue

                    try:
                        is_visible = await loc.is_visible()
                        if is_visible:
                            await loc.click(timeout=2000)
                            await self.page.wait_for_timeout(wait_ms)
                            self.logger.info(f"[CookieBanner] Accepted via visible click: {sel}")
                            return True
                    except Exception as e_visible:
                        self.logger.debug(f"[CookieBanner] Visible click failed for {sel}: {e_visible}")

                    try:
                        await loc.evaluate("el => el.click()")
                        await self.page.wait_for_timeout(wait_ms)
                        self.logger.info(f"[CookieBanner] Accepted via evaluate click: {sel}")
                        return True
                    except Exception as e_eval:
                        self.logger.debug(f"[CookieBanner] Evaluate click failed for {sel}: {e_eval}")

                    try:
                        await loc.click(force=True, timeout=1500)
                        await self.page.wait_for_timeout(wait_ms)
                        self.logger.info(f"[CookieBanner] Accepted via force click: {sel}")
                        return True
                    except Exception as e_force:
                        self.logger.debug(f"[CookieBanner] Force click failed for {sel}: {e_force}")

                    try:
                        if sel.startswith("#"):
                            element_id = sel.lstrip("#").split()[0]
                            clicked = await self.page.evaluate(
                                """((elementId) => {
                                    try {
                                        const btn = document.getElementById(elementId) ||
                                                   document.querySelector('button#' + elementId);
                                        if (btn) { btn.click(); return true; }
                                        return false;
                                    } catch (e) { return false; }
                                })""",
                                element_id,
                            )
                            if clicked:
                                await self.page.wait_for_timeout(wait_ms)
                                self.logger.info(f"[CookieBanner] Accepted via DOM click (id): {element_id}")
                                return True
                    except Exception as e_dom:
                        self.logger.debug(f"[CookieBanner] DOM click failed for {sel}: {e_dom}")

                except Exception as e:
                    self.logger.debug(f"[CookieBanner] Selector evaluation failed for {sel}: {e}")
                    continue

            self.logger.warning(
                "[CookieBanner] Could not close cookie banner with any method. "
                "Continuing execution (banner may interfere with navigation)."
            )
            return False

        except asyncio.CancelledError:
            self.logger.warning("[CookieBanner] Cancelled during cookie banner handling. Continuing execution.")
            return False
        except Exception as e:
            exc_type = type(e).__name__
            self.logger.warning(f"[CookieBanner] Could not accept; continue. reason={exc_type}: {e}")
            return False

    async def _handle_geo_modal(self, config: dict[str, Any]) -> bool:
        """Handle geo modal popup."""
        selectors = config.get("selectors", [])
        wait_ms = config.get("wait_after_click_ms", 500)

        if not selectors:
            selectors = [
                "button:has-text('Select your location')",
                "button[aria-label*='close' i]",
                "button:has-text('Close')",
                ".modal__close",
                ".c-modal__close",
            ]

        for sel in selectors:
            try:
                loc = self.page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=3000)
                    await self.page.wait_for_timeout(wait_ms)
                    self.logger.info(f"[PlpDriver] Geo modal dismissed via: {sel}")
                    return True
            except Exception:
                continue
        return False

    async def _remove_generic_overlays(self, config: dict[str, Any]) -> None:
        """Remove generic overlay elements via JS."""
        remove_selectors = config.get("remove_selectors", [])
        remove_body_classes = config.get("remove_body_classes", [])

        if not remove_selectors:
            remove_selectors = [
                ".overlay",
                ".backdrop",
                ".modal-backdrop",
                "#onetrust-banner-sdk",
                ".cookie-banner",
                '[aria-modal="true"]',
                ".cmp-ui-overlay",
                ".cmp-modal",
                ".drawer--open",
            ]

        if not remove_selectors and not remove_body_classes:
            try:
                await self.page.evaluate("""
                    (() => {
                        const sels = [
                            '.overlay', '.backdrop', '.modal-backdrop',
                            '#onetrust-banner-sdk', '.cookie-banner',
                            '[aria-modal="true"]', '.cmp-ui-overlay',
                            '.cmp-modal', '.drawer--open'
                        ];
                        document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                    })()
                """)
            except Exception as e:
                self.logger.debug(f"[PlpDriver] Default overlay removal failed: {e}")
            return

        js_code = f"""
        (() => {{
            const sels = {json.dumps(remove_selectors)};
            document.querySelectorAll(sels.join(',')).forEach(el => el.remove());

            const bodyClasses = {json.dumps(remove_body_classes)};
            const body = document.body;
            if (body) {{
                bodyClasses.forEach(cls => body.classList.remove(cls));
                body.style.overflow = '';
            }}

            const html = document.documentElement;
            if (html) {{
                html.style.overflow = '';
                bodyClasses.forEach(cls => html.classList.remove(cls));
            }}
        }})();
        """

        try:
            await self.page.evaluate(js_code)
        except Exception as e:
            self.logger.debug(f"[PlpDriver] Overlay removal failed: {e}")

    # Legacy compatibility wrappers

    async def _accept_cookies_if_present(self) -> bool:
        overlay_config = self._get_overlay_config()
        return await self._handle_cookie_banner(overlay_config["cookie"])

    async def _dismiss_geo_modal(self) -> None:
        overlay_config = self._get_overlay_config()
        await self._handle_geo_modal(overlay_config["geo"])

    async def _kill_overlays(self) -> None:
        overlay_config = self._get_overlay_config()
        await self._remove_generic_overlays(overlay_config["other"])
