# ==============================================================================
# File: app/agents/browser/plp_driver.py
# Version: 3.0.0 (Stage 4 + P2-5 refactoring: Mixin-based split)
# Purpose: PLP → PDP ナビゲーションロジックの汎用ドライバ（コア）
# ==============================================================================
"""
Stage 4: 汎用 PLP Driver 拡張版 (Mixin分割後)

PLP → PDP のナビゲーションロジックを完全に site_config ベースで動作する
汎用的なドライバに拡張。Mixin分割により各責務を分離:

- plp_config: 設定取得 (pure functions)
- plp_overlay: Overlay処理 (PlpOverlayMixin)
- plp_trap: Trap検出/回復 (PlpTrapMixin)
- plp_tile_materializer: タイルマテリアライズ (PlpTileMaterializerMixin)
- plp_navigation: PDP遷移 (PlpNavigationMixin)
- plp_evidence: 証跡保存 (独立関数)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.plp_config import get_trap_config
from app.agents.browser.plp_evidence import save_materialize_failure_evidence
from app.agents.browser.plp_navigation import PlpNavigationMixin
from app.agents.browser.plp_overlay import PlpOverlayMixin
from app.agents.browser.plp_tile_materializer import PlpTileMaterializerMixin
from app.agents.browser.plp_trap import PlpTrapMixin

if TYPE_CHECKING:
    from app.agents.browser.telemetry import TelemetryClient
    from app.core.run_context import RunContext

logger = logging.getLogger(__name__)

_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)


@dataclass
class PlpNavigationResult:
    """PLP → PDP ナビゲーションの結果."""

    pdp_url: str
    pdp_opened_in_new_tab: bool
    plp_url: str
    tiles_seen: int
    trap_detected: bool
    trap_reason: str | None = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    overlays_handled: list[str] = field(default_factory=list)
    navigation_method: str | None = None
    errors: list[str] = field(default_factory=list)


class PlpDriver(PlpOverlayMixin, PlpTrapMixin, PlpTileMaterializerMixin, PlpNavigationMixin):
    """汎用 PLP Driver - site_configベースで動作（Mixin分割版）."""

    def __init__(
        self,
        page: Page,
        context: BrowserContext,
        *,
        site_config: dict[str, Any],
        run_context: RunContext,
        logger: logging.Logger | None = None,
        telemetry: TelemetryClient | None = None,
    ) -> None:
        self.page = page
        self.context = context
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        self.telemetry = telemetry
        self._navigation_driver: Any | None = None

    async def navigate_to_pdp(
        self,
        *,
        target_url: str | None = None,
        timeout_ms: int = 60000,
        start_t: float | None = None,
        budget_ms: int | None = None,
    ) -> PlpNavigationResult:
        """PLP → PDP ナビゲーションを実行する（メインエントリーポイント）."""
        if start_t is not None and budget_ms is not None:
            actual_start_t = start_t
            actual_budget_ms = budget_ms
        else:
            actual_start_t = time.time()
            actual_budget_ms = timeout_ms

        plp_url = self.page.url
        errors: list[str] = []
        overlays_handled: list[str] = []

        try:
            # 0. trap/locale early detection
            trap_config = get_trap_config(self.site_config)
            trap_reason_early = None

            if self._is_trap_page(plp_url, trap_config):
                trap_reason_early = f"trap_url_detected: {plp_url}"
                self.logger.warning(f"[PlpDriver] {trap_reason_early}")
                self._log_diagnostic_info(plp_url, "Early trap detection")
                errors.append(trap_reason_early)
                return PlpNavigationResult(
                    pdp_url="",
                    pdp_opened_in_new_tab=False,
                    plp_url=plp_url,
                    tiles_seen=0,
                    trap_detected=True,
                    trap_reason=trap_reason_early,
                    recovery_attempted=False,
                    recovery_successful=False,
                    overlays_handled=overlays_handled,
                    navigation_method=None,
                    errors=errors,
                )

            if target_url and self._detected_locale_redirect():
                error_msg = f"locale_redirect_loop: Current URL={plp_url}, Target URL={target_url}"
                self._log_diagnostic_info(plp_url, "Early locale redirect loop detection", target_url=target_url)
                errors.append(error_msg)
                return PlpNavigationResult(
                    pdp_url="",
                    pdp_opened_in_new_tab=False,
                    plp_url=plp_url,
                    tiles_seen=0,
                    trap_detected=False,
                    trap_reason=None,
                    recovery_attempted=False,
                    recovery_successful=False,
                    overlays_handled=overlays_handled,
                    navigation_method=None,
                    errors=errors,
                )

            # 1. Overlay handling
            await self._handle_overlays(overlays_handled)

            # 2. Tile materialization
            plp_config = self._get_plp_config()
            tiles_seen = await self._materialize_tiles(
                plp_config=plp_config,
                start_t=actual_start_t,
                budget_ms=actual_budget_ms,
                target_url=target_url,
            )

            if tiles_seen == 0:
                await self._save_materialize_failure_evidence(
                    ", ".join(plp_config.get("product_tiles", []) + plp_config.get("product_link", [])),
                    plp_config,
                )
                error_msg = f"PLP did not materialize (no product tiles). URL={plp_url}"
                errors.append(error_msg)
                raise ValueError(error_msg)

            # 3. Trap detection and recovery
            trap_detected = False
            trap_reason = None
            recovery_attempted = False
            recovery_successful = False

            if self._is_trap_page(self.page.url, trap_config):
                trap_detected = True
                trap_reason = f"Trap/legal page detected: {self.page.url}"
                self.logger.warning(f"[PlpDriver] {trap_reason}")

                if target_url:
                    recovery_attempted = True
                    recovery_successful = await self._recover_from_trap(
                        target_url=target_url,
                        trap_config=trap_config,
                    )
                    if recovery_successful:
                        if self._is_trap_page(self.page.url, trap_config):
                            error_msg = f"Still on trap/legal page after recovery: {self.page.url}"
                            errors.append(error_msg)
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"Trap recovery failed. URL={self.page.url}"
                        errors.append(error_msg)
                        raise ValueError(error_msg)

            # 4. Click tile → PDP navigation
            pdp_page = await self._click_tile_and_navigate(
                plp_config=plp_config,
                timeout_ms=min(10000, actual_budget_ms // 6),
            )

            if pdp_page is None:
                error_msg = "Failed to navigate to PDP from PLP tile"
                errors.append(error_msg)
                raise ValueError(error_msg)

            pdp_url = pdp_page.url
            pdp_opened_in_new_tab = pdp_page is not self.page
            navigation_method = "new_tab" if pdp_opened_in_new_tab else "same_tab" if pdp_url != plp_url else "spa"

            if pdp_opened_in_new_tab:
                self.page = pdp_page

            return PlpNavigationResult(
                pdp_url=pdp_url,
                pdp_opened_in_new_tab=pdp_opened_in_new_tab,
                plp_url=plp_url,
                tiles_seen=tiles_seen,
                trap_detected=trap_detected,
                trap_reason=trap_reason,
                recovery_attempted=recovery_attempted,
                recovery_successful=recovery_successful,
                overlays_handled=overlays_handled,
                navigation_method=navigation_method,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"[PlpDriver] Navigation failed: {e}", exc_info=True)
            raise

    # === Helpers ===

    def _detected_locale_redirect(self) -> bool:
        """Detect locale redirect loop."""
        current_url = (self.page.url or "").lower()
        locale_cfg = self.site_config.get("locale", {}) or {}
        prefer_locale = locale_cfg.get("prefer", "")
        allowed_domain = self.site_config.get("allowed_domain", "")

        if prefer_locale and allowed_domain:
            target_locale_path = f"/{prefer_locale}/"
            if (
                allowed_domain.lower() in current_url
                and target_locale_path not in current_url
                and _LOCALE_SEG_RE.search(current_url)
            ):
                return True
        return False

    def _time_left_ms(self, start_t: float, budget_ms: int) -> float:
        """Calculate remaining budget time."""
        elapsed_ms = (time.time() - start_t) * 1000
        return max(0, budget_ms - elapsed_ms)

    async def _save_materialize_failure_evidence(
        self,
        tile_selector_str: str,
        plp_config: dict[str, Any],
    ) -> None:
        """Delegate to plp_evidence module."""
        await save_materialize_failure_evidence(
            page=self.page,
            run_context=self.run_context,
            tile_selector_str=tile_selector_str,
            plp_config=plp_config,
            logger=self.logger,
        )

    def _log_diagnostic_info(
        self,
        plp_url: str,
        context: str,
        target_url: str | None = None,
    ) -> None:
        """Log diagnostic info for trap/locale detection (fire-and-forget)."""
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(plp_url)
            path_ok = "/en-int/" in parsed_url.path
            country_ok = "en-lt" not in parsed_url.path and "en-int" in parsed_url.path

            proxy_info = "unknown"
            try:
                if hasattr(self.context, "_proxy_entry") and self.context._proxy_entry:
                    proxy_info = f"proxy={self.context._proxy_entry.server}"
                elif hasattr(self.context, "proxy") and self.context.proxy:
                    proxy_info = f"proxy={self.context.proxy.get('server', 'unknown')}"
                else:
                    proxy_info = "proxy=none"
            except Exception:
                proxy_info = "proxy=unknown"

            self.logger.warning(
                f"[PlpDriver] {context} - Diagnostic info: "
                f"current_url={plp_url}, "
                + (f"target_url={target_url}, " if target_url else "")
                + f"path_ok={path_ok}, country_ok={country_ok}, {proxy_info}"
            )
        except Exception as e_diag:
            self.logger.debug(f"[PlpDriver] Failed to collect diagnostic info: {e_diag}")
