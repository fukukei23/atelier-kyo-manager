from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.agent_failure import FailureHandlerMixin
from app.agents.browser.extractor import (
    BrowserExtractionService,
)
from app.agents.browser.learning_flow import LearningMixin
from app.agents.browser.moncler_locale import MonclerLocaleMixin
from app.agents.browser.navigation_driver import (
    NavigationContext,
    NavigationDriver,
)
from app.agents.browser.pdp_flow import PdpFlowMixin
from app.agents.browser.plp_flow import PlpFlowMixin
from app.agents.browser.repair_orchestrator import RepairOrchestratorMixin
from app.agents.browser.session_lifecycle import SessionLifecycleMixin
from app.agents.browser.session_manager import SessionManager
from app.agents.browser.settings import (
    resolve_run_settings as settings_resolve_run_settings,
)
from app.agents.browser.settings import (
    time_left_ms as settings_time_left_ms,
)
from app.agents.browser.ui_helpers import (
    accept_cookies_if_present as ui_accept_cookies_if_present,
)
from app.agents.browser.ui_helpers import (
    click_continue_shopping_if_present as ui_click_continue_shopping_if_present,
)
from app.agents.browser.ui_helpers import (
    dismiss_geo_modal as ui_dismiss_geo_modal,
)
from app.agents.browser.ui_helpers import (
    human_like_mouse_move as ui_human_like_mouse_move,
)
from app.agents.browser.ui_helpers import (
    human_like_pause as ui_human_like_pause,
)
from app.agents.browser.ui_helpers import (
    human_like_scroll as ui_human_like_scroll,
)
from app.agents.browser.ui_helpers import (
    kill_overlays as ui_kill_overlays,
)
from app.agents.browser.ui_helpers import (
    pause_for_operator as ui_pause_for_operator,
)
from app.agents.browser.ui_helpers import (
    safe_wait_selector as ui_safe_wait_selector,
)
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

try:
    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
except Exception:
    moncler_plp_recovery = None


logger = logging.getLogger(__name__)

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000


# ==============================================================================
# BrowserUseAgent Class — Mixin Composition
# ==============================================================================


class BrowserUseAgent(
    SessionLifecycleMixin,
    RepairOrchestratorMixin,
    PlpFlowMixin,
    PdpFlowMixin,
    MonclerLocaleMixin,
    LearningMixin,
    FailureHandlerMixin,
):
    """
    Playwright を駆動して PLP/PDP を探索するメインエージェント。

    各責務は Mixin に分割済み:
    - SessionLifecycleMixin: セッション開閉・ブートストラップ
    - RepairOrchestratorMixin: 自己修復スクレイピングフロー
    - PlpFlowMixin: PLP 探索フロー
    - PdpFlowMixin: PDP 抽出・ナビゲーション
    - MonclerLocaleMixin: Moncler 固有ロケール処理
    - LearningMixin: 学習フロー
    - FailureHandlerMixin: 障害処理・VRT
    """

    def __init__(self, runtime_kwargs: dict[str, Any] | None = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.logger = logger
        self.run_context: RunContext | None = None

        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._session_manager: SessionManager | None = None
        self.extraction_service = BrowserExtractionService(self.logger, self.runtime_kwargs)

    # --- Settings Resolution ---
    def _resolve_run_settings(self, site_config: dict[str, Any]) -> dict[str, Any]:
        return settings_resolve_run_settings(site_config, self.runtime_kwargs, self.logger)

    @staticmethod
    def _start_watchdog(budget_ms: int) -> tuple[float, int]:
        return time.monotonic(), int(budget_ms)

    @staticmethod
    def _time_left_ms(start_t: float, budget_ms: int) -> int:
        return settings_time_left_ms(start_t, budget_ms)

    @staticmethod
    def _slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
        return max(500, min(left_ms, cap_ms))

    # --- UI Helpers (thin delegation to ui_helpers) ---
    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        return await ui_safe_wait_selector(page, selector, timeout_ms=timeout_ms, state=state)

    async def _kill_overlays(self, page: Page) -> None:
        await ui_kill_overlays(page)

    async def _click_continue_shopping_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        return await ui_click_continue_shopping_if_present(page, site_config)

    async def _pause_for_operator(self, page: Page | None, run_context: RunContext | None, label: str) -> None:
        await ui_pause_for_operator(page, run_context, label, self.runtime_kwargs, self.logger)

    async def _accept_cookies_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        return await ui_accept_cookies_if_present(page, site_config)

    async def _dismiss_geo_modal(self, page: Page) -> None:
        await ui_dismiss_geo_modal(page, self.logger)

    # --- Human-like interaction helpers ---
    async def _human_like_pause(self, page: Page, *, min_ms: int = 400, max_ms: int = 900):
        await ui_human_like_pause(page, min_ms=min_ms, max_ms=max_ms)

    async def _human_like_mouse_move(self, page: Page):
        await ui_human_like_mouse_move(page)

    async def _human_like_scroll(self, page: Page):
        await ui_human_like_scroll(page)

    async def _setup_init_scripts(self, context: BrowserContext):
        from app.agents.browser.stealth import setup_stealth_init_scripts

        await setup_stealth_init_scripts(context)

    # --- Browser Context Setup ---
    def _build_context_options(self, settings: dict[str, Any], run_context: RunContext) -> dict[str, Any]:
        from app.agents.browser.session_config import build_context_options

        return build_context_options(settings, run_context.get_path, self.logger)

    async def _setup_routes(self, context: BrowserContext, settings: dict[str, Any]):
        from app.agents.browser.route_setup import setup_routes

        await setup_routes(context, settings, self._normalize_to_en_int_url, self.logger)

    def _get_session_file(self, site: str, site_config: dict[str, Any]) -> Path:
        from app.agents.browser.session_config import get_session_file

        return get_session_file(site, site_config)

    async def _apply_saved_session(
        self, context: BrowserContext, page: Page, site: str, site_config: dict[str, Any]
    ) -> None:
        from app.agents.browser.session_config import apply_saved_session, get_session_file

        sess_file = get_session_file(site, site_config)
        await apply_saved_session(context, sess_file, self.logger)

    # --- Main Run Logic ---
    async def run(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
    ) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        self.run_context = run_context
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        from app.agents.browser.plugins import get_plugin_registry
        from app.agents.browser.repair_orchestrator import _merge_learned_selectors

        registry = get_plugin_registry()
        plugin = registry.get(site.upper())
        plugin_ctx = {"query": query, "site_config": site_config}

        nav_url = target_url
        if plugin:
            try:
                nav_url = plugin.before_navigate(target_url, plugin_ctx) or target_url
                self.logger.info(f"[Plugin:{site}] before_navigate => {nav_url}")
            except Exception as plugin_e:
                nav_url = target_url
                self.logger.warning(f"[Plugin:{site}] before_navigate failed: {plugin_e}")

        _merge_learned_selectors(site, site_config, run_context)

        page: Page | None = None

        try:
            async with SessionManager(
                site=site,
                site_config=site_config,
                run_context=run_context,
                settings=settings,
                target_url=nav_url,
                timeout_ms=timeout_ms,
                likely_plp=likely_plp,
                runtime_kwargs=self.runtime_kwargs,
                logger=self.logger,
                url_normalizer=self._normalize_to_en_int_url,
            ) as session:
                self._attach_session(session)
                page = session.page
                if page is None:
                    raise ValueError("SessionManager failed to provision a Playwright page.")

                page = await self._bootstrap_session_page(
                    page=page,
                    site=site,
                    site_config=site_config,
                    run_context=run_context,
                    settings=settings,
                    target_url=nav_url,
                    likely_plp=likely_plp,
                )

                if plugin:
                    try:
                        await plugin.after_navigate(page, plugin_ctx)
                    except Exception as plugin_e:
                        self.logger.warning(f"[Plugin:{site}] after_navigate failed: {plugin_e}")

                await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
                if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                    await self._perform_vrt(page, "plp", settings)

                if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None and plugin is None:
                    try:
                        await moncler_plp_recovery(page, site_config, query)
                    except Exception as _e:
                        self.logger.warning(f"[MonclerPatch] skipped: {_e}")

                start_t, budget_ms = self._start_watchdog(
                    settings.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)
                )

                context = session.context
                if context is None:
                    raise ValueError("SessionManager did not provide a BrowserContext.")

                if mode == "learn":
                    return await self._run_learning_flow(
                        page, context, site, site_config, settings, run_context, start_t=start_t, budget_ms=budget_ms
                    )

                if likely_plp:
                    if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None and plugin is None:
                        try:
                            await moncler_plp_recovery(page, site_config, query)
                        except Exception as _e:
                            self.logger.warning(f"[MonclerPatch] skipped: {_e}")
                    if plugin:
                        plp_ctx = {"plp_min_cards": 24, "query": query}
                        try:
                            materialized = await plugin.materialize(page, plp_ctx)
                            if materialized:
                                self.logger.info(f"[Plugin:{site}] PLP materialized successfully.")
                            else:
                                self.logger.warning(
                                    f"[Plugin:{site}] materialize failed (fallback to default PLP flow)."
                                )
                        except Exception as plugin_e:
                            self.logger.warning(
                                f"[Plugin:{site}] materialize raised {plugin_e!r} (ignored, continue default PLP flow)."
                            )
                        try:
                            asserted = await plugin.assert_plp(page, plp_ctx)
                        except Exception as plugin_e:
                            self.logger.warning(
                                f"[Plugin:{site}] assert_plp raised {plugin_e!r} (ignored, continue default PLP flow)."
                            )
                        else:
                            if not asserted:
                                self.logger.warning(
                                    f"[Plugin:{site}] assert_plp=False (continue with default PLP flow)."
                                )

                    nav_ctx = NavigationContext(
                        site=site,
                        query=query,
                        site_config=site_config,
                        settings=settings,
                        run_context=run_context,
                        start_t=start_t,
                        budget_ms=budget_ms,
                        entry_url=nav_url,
                    )
                    navigation_driver = NavigationDriver(
                        page=page,
                        ensure_plp_materialized=lambda pg, scfg, stg, s_t, b_ms: self._ensure_plp_materialized(
                            pg,
                            scfg,
                            stg,
                            start_t=s_t,
                            budget_ms=b_ms,
                            target_url=nav_url,
                        ),
                        trap_checker=None,
                        telemetry=None,
                        strategy=plugin,
                    )
                    try:
                        nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
                    except Exception as nav_e:
                        self.logger.debug(f"[NavigationDriver] Stage3A-2 failed (fallback to legacy): {nav_e}")
                        nav_outcome = None

                    skip_materialize = bool(getattr(nav_outcome, "plp_materialized", False))

                    return await self._run_plp_flow(
                        page,
                        context,
                        site,
                        query,
                        site_config,
                        settings,
                        run_context,
                        target_url=nav_url,
                        start_t=start_t,
                        budget_ms=budget_ms,
                        skip_materialize=skip_materialize,
                    )

                return await self._run_pdp_flow(page, site, query, settings, run_context, site_config)

        except Exception as e:
            return await self._handle_run_failure(e, site, query, site_config, run_context, page or self._page)
        finally:
            self._detach_session()
            if hasattr(self, "run_context"):
                del self.run_context
