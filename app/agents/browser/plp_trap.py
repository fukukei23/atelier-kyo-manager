from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.agents.browser.plp_config import get_trap_config


class PlpTrapMixin:
    """Trap detection and recovery mixin for PlpDriver."""

    page: Page
    site_config: dict[str, Any]
    logger: logging.Logger
    _navigation_driver: Any | None

    def _get_trap_config(self) -> dict[str, Any]:
        return get_trap_config(self.site_config)

    def _is_trap_page(self, url: str, trap_config: dict[str, Any]) -> bool:
        """Check if URL is a trap/legal page."""
        if self._navigation_driver:
            return self._navigation_driver._looks_like_trap_or_legal(url, self.site_config)
        else:
            from app.agents.browser.navigation_driver import NavigationDriver

            driver = NavigationDriver(page=None)  # type: ignore
            return driver._looks_like_trap_or_legal(url, self.site_config)

    def _looks_like_trap_or_legal(self, url: str) -> bool:
        """Legacy wrapper for trap detection."""
        trap_config = self._get_trap_config()
        return self._is_trap_page(url, trap_config)

    async def _recover_from_trap(
        self,
        *,
        target_url: str,
        trap_config: dict[str, Any],
    ) -> bool:
        """Recover from a trap page using site_config recovery actions."""
        recovery_actions = trap_config.get("recovery_actions", [])

        for action_cfg in recovery_actions:
            action = action_cfg.get("action")
            max_attempts = action_cfg.get("max_attempts", 1)

            for attempt in range(max_attempts):
                try:
                    if action == "go_back":
                        await self.page.go_back(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            self.logger.info("[PlpDriver] Recovery successful via 'go_back'")
                            return True

                    elif action == "goto_target":
                        target = (
                            target_url or self.site_config.get(action_cfg.get("target_url_key", "seed_plp_url")) or ""
                        )
                        if target:
                            await self.page.goto(url=target, wait_until="domcontentloaded")
                            await self.page.wait_for_timeout(1000)
                            if not self._is_trap_page(self.page.url, trap_config):
                                self.logger.info(f"[PlpDriver] Recovery successful via 'goto_target': {target}")
                                return True

                    elif action == "reload":
                        await self.page.reload(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            self.logger.info("[PlpDriver] Recovery successful via 'reload'")
                            return True

                except Exception as e:
                    self.logger.warning(
                        f"[PlpDriver] Recovery action '{action}' (attempt {attempt + 1}/{max_attempts}) failed: {e}"
                    )
                    continue

        # Fallback: NavigationDriver._force_plp_recover
        try:
            from app.agents.browser.navigation_driver import NavigationDriver

            driver = NavigationDriver(page=self.page)
            await driver._force_plp_recover(self.page, self.site_config, target_url)
            if not self._is_trap_page(self.page.url, trap_config):
                self.logger.info("[PlpDriver] Recovery successful via NavigationDriver._force_plp_recover")
                return True
        except Exception as e:
            self.logger.warning(f"[PlpDriver] NavigationDriver recovery fallback failed: {e}")

        return False
