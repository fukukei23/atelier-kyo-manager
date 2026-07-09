"""Session lifecycle Mixin: open/close/bootstrap sessions."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.session_manager import SessionManager
from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


class SessionLifecycleMixin:
    """BrowserUseAgent のセッション開閉・ブートストラップを担当。"""

    _session_manager: SessionManager | None
    _context: BrowserContext | None
    _page: Page | None
    runtime_kwargs: dict[str, Any]
    logger: logging.Logger

    def _attach_session(self, session: SessionManager) -> None:
        self._session_manager = session
        self._context = session.context
        self._page = session.page

    def _detach_session(self) -> None:
        self._session_manager = None
        self._context = None
        self._page = None

    async def _bootstrap_session_page(
        self,
        *,
        page: Page,
        site: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
        target_url: str,
        likely_plp: bool = False,
    ) -> Page:
        if not page or page.is_closed():
            raise ValueError("BrowserUseAgent: Session page is not available or already closed.")
        if not target_url:
            raise ValueError("BrowserUseAgent: target_url が指定されていません。起点URLが必要です。")

        await page.goto(url=target_url, wait_until="domcontentloaded")
        await self._accept_cookies_if_present(page, site_config)
        await self._dismiss_geo_modal(page)
        await self._kill_overlays(page)
        await self._click_continue_shopping_if_present(page, site_config)

        if settings.get("enable_human_like"):
            try:
                await self._human_like_mouse_move(page)
                await self._human_like_scroll(page)
            except Exception as e:
                self.logger.debug(f"[HumanLike] skipped: {e}")

        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=800)
        await page.wait_for_timeout(120)

        if site.upper() == "MONCLER_OFFICIAL" and not likely_plp:
            try:
                gate_links_count = await page.evaluate("() => document.querySelectorAll(\"a[href*='/en-']\").length")
                if gate_links_count and gate_links_count >= 10:
                    self.logger.warning(
                        f"[Moncler] Locale gate detected ({gate_links_count} links). Forcing navigation to PLP."
                    )
                    fixed_url = (
                        "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
                        "?forceLocale=en-int&shipToCountry=GB"
                    )
                    await page.goto(url=fixed_url, wait_until="domcontentloaded")
                    await self._click_continue_shopping_if_present(page, site_config)
                    with contextlib.suppress(Exception):
                        await page.wait_for_load_state("networkidle", timeout=2000)
                    await self._accept_cookies_if_present(page, site_config)
            except Exception as gate_e:
                self.logger.warning(f"[Moncler] Gate detection failed: {gate_e}")

            if settings.get("enable_locale_escape"):
                await self._force_en_int(page)
                await self._click_continue_shopping_if_present(page, site_config)
                await run_context.take_screenshot(page, "12_after_locale_escape")

            try:
                if "monclergroup.com" in (page.url or "").lower():
                    fixed_url = (
                        "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
                        "?forceLocale=en-int&shipToCountry=GB"
                    )
                    self.logger.warning(f"[Moncler] Bounced to corporate. Forcing back to PLP: {fixed_url}")
                    await page.goto(url=fixed_url, wait_until="domcontentloaded")
                    await self._accept_cookies_if_present(page, site_config)
                    await self._click_continue_shopping_if_present(page, site_config)
            except Exception:
                pass

        return page

    async def _open_session(
        self,
        *,
        site: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
        target_url: str,
        timeout_ms: int,
        likely_plp: bool = False,
    ) -> Page:
        if self._session_manager and self._page and not self._page.is_closed():
            self.logger.warning("[SessionManager] Existing session detected. Reusing current page.")
            return self._page

        session = SessionManager(
            site=site,
            site_config=site_config,
            run_context=run_context,
            settings=settings,
            target_url=target_url,
            timeout_ms=timeout_ms,
            likely_plp=likely_plp,
            runtime_kwargs=self.runtime_kwargs,
            logger=self.logger,
            url_normalizer=self._normalize_to_en_int_url,
        )
        await session.open()
        self._attach_session(session)

        page = session.page
        if page is None:
            raise ValueError("SessionManager failed to provision a Playwright page.")

        return await self._bootstrap_session_page(
            page=page,
            site=site,
            site_config=site_config,
            run_context=run_context,
            settings=settings,
            target_url=target_url,
            likely_plp=likely_plp,
        )

    async def _close_session(
        self,
        run_context: RunContext | None,
        settings: dict[str, Any],
    ) -> None:
        if not self._session_manager:
            self._detach_session()
            return
        try:
            await self._session_manager.close()
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to close browser session: {e}")
        finally:
            self._detach_session()
