"""Failure handling Mixin: error processing and VRT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import write_fail_snapshot

logger = logging.getLogger(__name__)


class FailureHandlerMixin:
    """BrowserUseAgent の障害処理を担当。"""

    _page: Page | None
    logger: logging.Logger

    async def _handle_run_failure(
        self, e: Exception, site: str, query: str, site_config: dict, run_context: RunContext, page: Page | None
    ) -> DiscoveryResult:
        logger.error(f"Browser task failed (RunID: {run_context.run_id}): {e}", exc_info=True)
        final_url_on_fail = None

        active_page = page or self._page

        try:
            await self._pause_for_operator(active_page, run_context, "failure_inspection")
            if active_page and not active_page.is_closed():
                final_url_on_fail = active_page.url
        except Exception:
            pass

        dom_path_str = None
        try:
            await write_fail_snapshot(run_context, active_page, final_url_on_fail, e, site_config)
            dom_guess = run_context.get_path("failure_dom.html")
            if dom_guess and Path(dom_guess).exists():
                dom_path_str = str(dom_guess)
        except Exception as hook_e:
            logger.warning(f"[Hook C2] write_fail_snapshot failed: {hook_e}")

        screenshots_list = []
        if hasattr(run_context, "screenshots") and isinstance(run_context.screenshots, list):
            screenshots_list = run_context.screenshots
        elif hasattr(run_context, "get_path"):
            try:
                run_dir = Path(run_context.run_path)
                recent_pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                screenshots_list = [str(p) for p in recent_pngs[:6]]
            except Exception:
                pass

        failure_context = {
            "final_url": final_url_on_fail,
            "dom_snapshot_path": dom_path_str,
            "errors": [str(e)],
            "screenshots": screenshots_list,
            "intent_description": (
                "Goal: reach a product listing page (PLP) and extract product cards "
                "and individual PDP links, then read price/title from PDP. "
                "We expected to see product tiles and extract price. "
                "Instead we hit an unexpected layout / modal / redirect."
            ),
            "selectors_tried_hint": "See site_config['selectors']['pdp'] and wait_for_selectors in settings.",
        }

        return DiscoveryResult(
            ok=False, site=site, query=query,
            message=str(e),
            evidence={
                "final_url": final_url_on_fail,
                "failure_context": failure_context,
            },
        )

    async def _perform_vrt(self, page: Page, scope: str, settings: dict[str, Any]):
        from app.utils.visual_regression import perform_vrt
        site_name = self.runtime_kwargs.get("site") or "GENERIC"
        await perform_vrt(page, scope, settings, site_name, self.logger)

    def _build_pdp_prepare_hook(
        self,
        *,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        run_context: RunContext,
    ):
        async def _prepare(page: Page):
            await self._kill_overlays(page)
            await self._click_continue_shopping_if_present(page, site_config)
            await self._dismiss_geo_modal(page)
            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "pdp", settings)

        return _prepare
