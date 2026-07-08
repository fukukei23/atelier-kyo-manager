"""BrowserOrchestrator Mixin: PLP→PDP flow orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.extractor import BrowserExtractionService
from app.agents.browser.navigation_driver import (
    NavigationContext,
    NavigationDriver,
    NavigationOutcome,
    TrapPageDetected,
)
from app.agents.browser.plp_driver import PlpDriver, PlpNavigationResult
from app.agents.browser.telemetry import TelemetryClient
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

logger = logging.getLogger(__name__)


class PlpPdpFlowMixin:
    """PLP→PDP flow: run_plp_to_pdp, run_pdp, _run_full."""

    log: logging.Logger
    runtime_kwargs: dict[str, Any]

    async def run_plp_to_pdp(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        nav_outcome: NavigationOutcome | None = None,
        trap_checker: Callable[[str], bool] | None = None,
        telemetry: Any | None = None,
        plugin: Any | None = None,
        navigation_driver: NavigationDriver | None = None,
        extraction_service: BrowserExtractionService | None = None,
    ) -> PlpNavigationResult | DiscoveryResult:
        self.log.info(f"[Debug][Telemetry] telemetry value={telemetry}, type={type(telemetry)}")

        nav_ctx = NavigationContext(
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            start_t=start_t,
            budget_ms=budget_ms,
            entry_url=target_url,
            context=context,
        )

        if telemetry is None:
            telemetry = TelemetryClient(run_context=run_context)

        if navigation_driver is None:
            navigation_driver = NavigationDriver(
                page=page,
                trap_checker=trap_checker,
                telemetry=telemetry,
                strategy=plugin,
            )

        try:
            nav_outcome = await self._run_navigation_phase(
                page,
                navigation_driver,
                nav_ctx,
                nav_outcome,
                telemetry,
                site,
                query,
                site_config,
                run_context,
                target_url,
            )
        except TrapPageDetected:
            raise
        except Exception as nav_e:
            self.log.debug(f"[Orchestrator] NavigationDriver.run_plp_flow failed: {nav_e}")
            nav_outcome = None
            if telemetry:
                try:
                    failure_ctx = self._build_failure_context(
                        site=site,
                        query=query,
                        final_url=page.url,
                        error_type="navigation_failed",
                        error_class=type(nav_e).__name__,
                        error_message=str(nav_e),
                        site_config=site_config,
                        run_context=run_context,
                    )
                    await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record navigation failure: {te}", exc_info=True)

        if nav_outcome and nav_outcome.trap_detected:
            return await self._handle_trap_recovery(
                page,
                nav_outcome,
                site,
                query,
                site_config,
                run_context,
            )

        pdp_links = nav_outcome.pdp_links if nav_outcome else []

        if not pdp_links:
            return await self._handle_no_pdp_links(
                page,
                context,
                site,
                query,
                site_config,
                settings,
                run_context,
                target_url,
                start_t,
                budget_ms,
                telemetry,
                plp_driver=None,
            )

        if pdp_links:
            result = await self._extract_from_pdp_links(
                page,
                context,
                site,
                query,
                site_config,
                settings,
                run_context,
                target_url,
                start_t,
                budget_ms,
                pdp_links,
                telemetry,
                extraction_service=extraction_service,
            )
            if result is not None:
                return result

        return await self._build_no_pdp_result(page, site, query, site_config, run_context)

    # ------------------------------------------------------------------ #
    # run_pdp
    # ------------------------------------------------------------------ #

    async def run_pdp(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        run_context: RunContext,
        target_url: str | None = None,
        extraction_service: BrowserExtractionService | None = None,
        telemetry: Any | None = None,
    ) -> DiscoveryResult:
        if extraction_service is None:
            extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)

        telemetry_client = telemetry
        if telemetry_client is None:
            try:
                if TelemetryClient:
                    telemetry_client = TelemetryClient(run_context=run_context)
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to create TelemetryClient: {te}", exc_info=True)

        prepare_hook = self._build_prepare_hook(
            page,
            site,
            query,
            site_config,
            settings,
            run_context,
            telemetry_client,
        )

        try:
            return await extraction_service.extract_single_pdp(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                prepare_page=prepare_hook,
            )
        except Exception as e:
            self.log.warning(f"[Orchestrator] extract_single_pdp failed: {e}")
            return await self._build_extraction_failure_result(
                page,
                site,
                query,
                site_config,
                run_context,
                e,
                "pdp_extraction_failed",
            )

    # ------------------------------------------------------------------ #
    # _run_full
    # ------------------------------------------------------------------ #

    async def _run_full(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        nav_outcome: NavigationOutcome | None = None,
        trap_checker: Callable[[str], bool] | None = None,
        telemetry: Any | None = None,
        plugin: Any | None = None,
    ) -> DiscoveryResult:
        result_plp = await self.run_plp_to_pdp(
            page=page,
            context=context,
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            target_url=target_url,
            start_t=start_t,
            budget_ms=budget_ms,
            nav_outcome=nav_outcome,
            trap_checker=trap_checker,
            telemetry=telemetry,
            plugin=plugin,
        )

        if isinstance(result_plp, PlpNavigationResult):
            if result_plp.pdp_url:
                result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=site_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=result_plp.pdp_url,
                )
            else:
                result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={"error": "No PDP URL found", "plp_result": result_plp},
                )
        else:
            result = result_plp

        return self._enrich_result_with_success_stage(
            result=result,
            run_context=run_context,
            nav_outcome=nav_outcome,
            page=page,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_prepare_hook(
        self,
        page: Page,
        site: str,
        query: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        telemetry_client: Any,
    ):
        log = self.log

        async def prepare_hook(inner_page: Page) -> None:
            if telemetry_client and hasattr(telemetry_client, "record_plp_state"):
                try:
                    await telemetry_client.record_plp_state(
                        inner_page,
                        name="pdp_dom_before_extract",
                        site_config=site_config,
                    )
                except Exception as te:
                    log.warning(f"[Orchestrator] Failed to record PDP DOM: {te}", exc_info=True)
            try:
                from app.agents.browser import ui_helpers

                if ui_helpers.kill_overlays:
                    await ui_helpers.kill_overlays(inner_page)
                if ui_helpers.click_continue_shopping_if_present:
                    await ui_helpers.click_continue_shopping_if_present(inner_page, site_config)
                if ui_helpers.dismiss_geo_modal:
                    await ui_helpers.dismiss_geo_modal(inner_page, log)
            except Exception as e:
                log.warning(f"[Orchestrator] UI helpers failed: {e}", exc_info=True)

            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                try:
                    from app.utils.visual_regression import compare_and_maybe_update

                    await compare_and_maybe_update(inner_page, "pdp", settings)
                except Exception as e:
                    log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)

        return prepare_hook

    async def _run_navigation_phase(
        self,
        page: Page,
        navigation_driver: NavigationDriver,
        nav_ctx: NavigationContext,
        nav_outcome: NavigationOutcome | None,
        telemetry: Any,
        site: str,
        query: str,
        site_config: dict,
        run_context: RunContext,
        target_url: str,
    ) -> NavigationOutcome:
        if nav_outcome is not None:
            return nav_outcome

        if telemetry and hasattr(telemetry, "record_plp_state"):
            try:
                await telemetry.record_plp_state(page, name="plp_dom_initial", site_config=site_config)
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to record PLP initial state: {te}", exc_info=True)

        nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
        self.log.debug(f"[Orchestrator] NavigationDriver.run_plp_flow called: entry_url={nav_outcome.entry_url}")

        if telemetry and hasattr(telemetry, "save_json"):
            try:
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_after_nav")
                await telemetry.save_json(
                    "plp_navigation_outcome",
                    {
                        "entry_url": nav_outcome.entry_url,
                        "pdp_links_count": len(nav_outcome.pdp_links) if nav_outcome else 0,
                        "trap_detected": nav_outcome.trap_detected if nav_outcome else False,
                        "recovered": nav_outcome.recovered if nav_outcome else False,
                    },
                    tctx,
                )
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to record PLP navigation outcome: {te}", exc_info=True)

        return nav_outcome

    async def _handle_trap_recovery(
        self,
        page: Page,
        nav_outcome: NavigationOutcome,
        site: str,
        query: str,
        site_config: dict,
        run_context: RunContext,
    ) -> DiscoveryResult:
        if nav_outcome.recovered:
            self.log.debug("[Orchestrator] NavigationDriver already handled trap detection and recovery")
            return nav_outcome  # type: ignore[return-value]

        failure_ctx = self._build_failure_context(
            site=site,
            query=query,
            final_url=page.url,
            error_type="trap_recovery_failed",
            error_class="TrapPageDetected",
            error_message=nav_outcome.trap_reason or "Trap page detected but recovery failed",
            site_config=site_config,
            run_context=run_context,
        )
        analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
        evidence: dict[str, Any] = {"failure_context": failure_ctx}
        if analysis:
            evidence["failure_analysis"] = analysis
            patch = await self._maybe_build_patch_candidate(
                failure_context=failure_ctx,
                failure_analysis=analysis,
                site_config=site_config,
                run_context=run_context,
                page=page,
            )
            if patch:
                evidence["self_healing_patch_candidate"] = patch
        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message=f"Landing page looks like legal/trap (NavigationDriver recovery failed): {nav_outcome.trap_reason}",
            evidence=evidence,
        )

    async def _handle_no_pdp_links(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        telemetry: Any,
        plp_driver: PlpDriver | None = None,
    ) -> PlpNavigationResult | DiscoveryResult:
        self.log.warning("[Orchestrator] No PDP links found. Clicking first card using PlpDriver...")
        if telemetry and hasattr(telemetry, "save_json"):
            try:
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_no_pdp_links")
                await telemetry.save_json(
                    "plp_no_pdp_links",
                    {
                        "entry_url": target_url,
                        "current_url": page.url,
                        "pdp_links_count": 0,
                    },
                    tctx,
                )
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to record no PDP links: {te}", exc_info=True)

        try:
            if telemetry is None:
                telemetry = TelemetryClient(run_context=run_context)

            if plp_driver is None:
                plp_driver = PlpDriver(
                    page=page,
                    context=context,
                    site_config=site_config,
                    run_context=run_context,
                    logger=self.log,
                    telemetry=telemetry,
                )
            default_timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
            timeout_ms = min(budget_ms, default_timeout_ms) if budget_ms is not None else default_timeout_ms

            nav_result = await plp_driver.navigate_to_pdp(
                target_url=target_url,
                timeout_ms=timeout_ms,
                start_t=start_t,
                budget_ms=budget_ms,
            )

            if nav_result.trap_detected:
                self.log.warning(f"[Orchestrator] Trap detected in PlpDriver. Early exit: {nav_result.trap_reason}")
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="trap_detected",
                    error_class="PlpDriverTrapDetection",
                    error_message=f"Trap/locale detected: {nav_result.trap_reason}",
                    site_config=site_config,
                    run_context=run_context,
                )
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message=failure_ctx.get("error_message", f"Trap/locale detected: {nav_result.trap_reason}"),
                    evidence=failure_ctx,
                )
            return nav_result

        except Exception as plp_e:
            self.log.warning(f"[Orchestrator] PlpDriver failed: {plp_e}", exc_info=True)
            return await self._build_extraction_failure_result(
                page,
                site,
                query,
                site_config,
                run_context,
                plp_e,
                "plp_driver_failed",
            )

    async def _extract_from_pdp_links(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        pdp_links: list[str],
        telemetry: Any,
        extraction_service: BrowserExtractionService | None = None,
    ) -> DiscoveryResult | None:
        self.log.debug(f"[Orchestrator] Found {len(pdp_links)} PDP links, extracting from PDP list...")
        if telemetry and hasattr(telemetry, "save_json"):
            try:
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp_pdp_links_found")
                await telemetry.save_json(
                    "plp_pdp_links",
                    {
                        "entry_url": target_url,
                        "current_url": page.url,
                        "pdp_links_count": len(pdp_links),
                        "pdp_links": pdp_links[:10],
                    },
                    tctx,
                )
            except Exception as te:
                self.log.warning(f"[Orchestrator] Failed to record PDP links: {te}", exc_info=True)

        try:
            if extraction_service is None:
                extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)
            prepare_hook = self._build_prepare_hook(
                page,
                site,
                query,
                site_config,
                settings,
                run_context,
                None,
            )
            return await extraction_service.extract_from_pdp_list(
                page=page,
                context=context,
                site=site,
                query=query,
                pdp_links=pdp_links,
                site_config=site_config,
                settings=settings,
                run_context=run_context,
                start_t=start_t,
                budget_ms=budget_ms,
                prepare_page=prepare_hook,
            )
        except Exception as extract_e:
            self.log.error(f"[Orchestrator] extract_from_pdp_list failed: {extract_e}", exc_info=True)
            return await self._build_extraction_failure_result(
                page,
                site,
                query,
                site_config,
                run_context,
                extract_e,
                "pdp_extraction_failed",
            )

    async def _build_extraction_failure_result(
        self,
        page: Page,
        site: str,
        query: str,
        site_config: dict,
        run_context: RunContext,
        error: Exception,
        error_type: str,
    ) -> DiscoveryResult:
        failure_ctx = self._build_failure_context(
            site=site,
            query=query,
            final_url=page.url,
            error_type=error_type,
            error_class=type(error).__name__,
            error_message=str(error),
            site_config=site_config,
            run_context=run_context,
        )
        analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
        evidence: dict[str, Any] = {"failure_context": failure_ctx}
        if analysis:
            evidence["failure_analysis"] = analysis
            patch = await self._maybe_build_patch_candidate(
                failure_context=failure_ctx,
                failure_analysis=analysis,
                site_config=site_config,
                run_context=run_context,
                page=page,
            )
            if patch:
                evidence["self_healing_patch_candidate"] = patch
        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message=f"{error_type}: {str(error)}",
            evidence=evidence,
        )

    async def _build_no_pdp_result(
        self,
        page: Page,
        site: str,
        query: str,
        site_config: dict,
        run_context: RunContext,
    ) -> DiscoveryResult:
        self.log.error("[Orchestrator] No PDP links found and PlpDriver fallback was not called")
        failure_ctx = self._build_failure_context(
            site=site,
            query=query,
            final_url=page.url,
            error_type="no_pdp_links",
            error_class="NoPdpLinksError",
            error_message="No PDP links found and all fallback strategies failed",
            site_config=site_config,
            run_context=run_context,
        )
        analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
        evidence: dict[str, Any] = {"failure_context": failure_ctx}
        if analysis:
            evidence["failure_analysis"] = analysis
            patch = await self._maybe_build_patch_candidate(
                failure_context=failure_ctx,
                failure_analysis=analysis,
                site_config=site_config,
                run_context=run_context,
                page=page,
            )
            if patch:
                evidence["self_healing_patch_candidate"] = patch
        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message="No PDP links found and all fallback strategies failed",
            evidence=evidence,
        )
