"""BrowserOrchestrator Mix-in: コアフロー（PLP/PDP/成功段階）"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
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
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

from .config_and_metrics import ConfigAndMetricsMixin
from .self_healing import SelfHealingMixin

from app.agents.browser.telemetry import TelemetryClient
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
from app.agents.healing.failure_analysis_agent import FailureAnalysisAgent
from app.agents.healing.self_healing_patch_agent import SelfHealingPatchAgent
from app.agents.healing.self_healing_sandbox import SelfHealingSandbox
from app.agents.healing.self_healing_patch_applier import SelfHealingPatchApplier

# self_healing_policy は未実装モジュール（optional import）
try:
    from app.agents.healing.self_healing_policy import SelfHealingPolicy
except ImportError:
    SelfHealingPolicy = None  # type: ignore
from app.agents.healing.selector_repair_agent import SelectorRepairAgent

# e2e_success_stage は未実装モジュール（optional import）
try:
    from app.utils.e2e_success_stage import collect_run_artifacts, compute_success_stage
except ImportError:
    compute_success_stage = None  # type: ignore
    collect_run_artifacts = None  # type: ignore

logger = logging.getLogger(__name__)


class BrowserOrchestrator(SelfHealingMixin, ConfigAndMetricsMixin):
    """
    BrowserUseAgent と NavigationDriver/PlpDriver/SelectorDiscoveryAgent の間に立ち、
    PLP→PDP フロー全体の状態遷移とエラー処理を一元管理するオーケストレータ。
    """

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any] | None = None,
        analysis_agent: Any | None = None,
        discovery_agent: Any | None = None,
        patch_agent: Any | None = None,
        sandbox: Any | None = None,
        policy: Any | None = None,
        patch_applier: Any | None = None,
        selector_repair_agent: Any | None = None,
        llm_client: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.runtime_kwargs: dict[str, Any] = runtime_kwargs or {}
        self.log = log or logger

        self.analysis_agent = analysis_agent or FailureAnalysisAgent(runtime_kwargs=self.runtime_kwargs)
        self.discovery_agent = discovery_agent or SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.patch_agent = patch_agent or SelfHealingPatchAgent(runtime_kwargs=self.runtime_kwargs)
        self.sandbox = sandbox or SelfHealingSandbox()
        self.policy = policy or (
            SelfHealingPolicy.from_file(Path("app/config/self_healing_policy.json")) if SelfHealingPolicy else None
        )
        self.patch_applier = patch_applier or SelfHealingPatchApplier()
        self.selector_repair_agent = selector_repair_agent or SelectorRepairAgent(llm_client=llm_client)

        self._overrides_path = Path("app/config/sites/overrides.local.json")

    # ------------------------------------------------------------------ #
    # PLP → PDP flow
    # ------------------------------------------------------------------ #

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

        navigation_driver = NavigationDriver(
            page=page,
            trap_checker=trap_checker,
            telemetry=telemetry,
            strategy=plugin,
        )

        try:
            if nav_outcome is None:
                if telemetry and hasattr(telemetry, "record_plp_state"):
                    try:
                        await telemetry.record_plp_state(
                            page,
                            name="plp_dom_initial",
                            site_config=site_config,
                        )
                    except Exception as te:
                        self.log.warning(f"[Orchestrator] Failed to record PLP initial state: {te}", exc_info=True)

                nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
                self.log.debug(
                    f"[Orchestrator] NavigationDriver.run_plp_flow called: entry_url={nav_outcome.entry_url}"
                )

                if telemetry and hasattr(telemetry, "save_json"):
                    try:
                        from app.agents.browser.telemetry import TelemetryContext

                        tctx = TelemetryContext(
                            site=site, query=query, run_id=run_context.run_id, stage="plp_after_nav"
                        )
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
        except TrapPageDetected as trap_e:
            self.log.warning(
                f"[Orchestrator] Trap page detected by NavigationDriver: "
                f"type={trap_e.trap_type}, reason={trap_e.reason}, URL={trap_e.url}"
            )
            if telemetry:
                try:
                    failure_ctx = self._build_failure_context(
                        site=site,
                        query=query,
                        final_url=trap_e.url,
                        error_type="trap_detected",
                        error_class=type(trap_e).__name__,
                        error_message=str(trap_e),
                        site_config=site_config,
                        run_context=run_context,
                    )
                    await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record trap detection: {te}", exc_info=True)
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
            if nav_outcome.recovered:
                self.log.debug("[Orchestrator] NavigationDriver already handled trap detection and recovery")
            else:
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

        pdp_links = nav_outcome.pdp_links if nav_outcome else []

        if not pdp_links:
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
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="plp_driver_failed",
                    error_class=type(plp_e).__name__,
                    error_message=str(plp_e),
                    site_config=site_config,
                    run_context=run_context,
                )
                analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                evidence = {"failure_context": failure_ctx}
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
                    message=f"PlpDriver failed: {str(plp_e)}",
                    evidence=evidence,
                )

        if pdp_links:
            self.log.debug(f"[Orchestrator] Found {len(pdp_links)} PDP links, extracting from PDP list...")
            if telemetry and hasattr(telemetry, "save_json"):
                try:
                    from app.agents.browser.telemetry import TelemetryContext

                    tctx = TelemetryContext(
                        site=site, query=query, run_id=run_context.run_id, stage="plp_pdp_links_found"
                    )
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
                extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)

                async def prepare_hook(page: Page):
                    try:
                        from app.agents.browser import ui_helpers

                        if ui_helpers.kill_overlays:
                            await ui_helpers.kill_overlays(page)
                        if ui_helpers.click_continue_shopping_if_present:
                            await ui_helpers.click_continue_shopping_if_present(page, site_config)
                        if ui_helpers.dismiss_geo_modal:
                            await ui_helpers.dismiss_geo_modal(page, self.log)
                    except Exception as e:
                        self.log.warning(f"[Orchestrator] UI helpers failed: {e}", exc_info=True)

                    if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                        try:
                            from app.utils.visual_regression import compare_and_maybe_update

                            await compare_and_maybe_update(page, "pdp", settings)
                        except Exception as e:
                            self.log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)

                result = await extraction_service.extract_from_pdp_list(
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

                return result

            except Exception as extract_e:
                self.log.error(f"[Orchestrator] extract_from_pdp_list failed: {extract_e}", exc_info=True)
                failure_ctx = self._build_failure_context(
                    site=site,
                    query=query,
                    final_url=page.url,
                    error_type="pdp_extraction_failed",
                    error_class=type(extract_e).__name__,
                    error_message=str(extract_e),
                    site_config=site_config,
                    run_context=run_context,
                )
                analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
                evidence = {"failure_context": failure_ctx}
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
                    message=f"extract_from_pdp_list failed: {str(extract_e)}",
                    evidence=evidence,
                )

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
        evidence = {"failure_context": failure_ctx}
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

    # ------------------------------------------------------------------ #
    # PDP extraction
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
    ) -> DiscoveryResult:
        extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)

        telemetry_client = None
        try:
            if TelemetryClient:
                telemetry_client = TelemetryClient(run_context=run_context)
        except Exception as te:
            self.log.warning(f"[Orchestrator] Failed to create TelemetryClient: {te}", exc_info=True)

        async def prepare_hook(inner_page: Page) -> None:
            if telemetry_client and hasattr(telemetry_client, "record_plp_state"):
                try:
                    await telemetry_client.record_plp_state(
                        inner_page,
                        name="pdp_dom_before_extract",
                        site_config=site_config,
                    )
                except Exception as te:
                    self.log.warning(f"[Orchestrator] Failed to record PDP DOM: {te}", exc_info=True)
            try:
                from app.agents.browser import ui_helpers

                if ui_helpers.kill_overlays:
                    await ui_helpers.kill_overlays(inner_page)
                if ui_helpers.click_continue_shopping_if_present:
                    await ui_helpers.click_continue_shopping_if_present(inner_page, site_config)
                if ui_helpers.dismiss_geo_modal:
                    await ui_helpers.dismiss_geo_modal(inner_page, self.log)
            except Exception as e:
                self.log.warning(f"[Orchestrator] UI helpers failed: {e}", exc_info=True)

            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                try:
                    from app.utils.visual_regression import compare_and_maybe_update

                    vrt_result = await compare_and_maybe_update(inner_page, "pdp", settings)
                    if vrt_result and telemetry_client and hasattr(telemetry_client, "save_json"):
                        try:
                            from app.agents.browser.telemetry import TelemetryContext

                            tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="pdp_vrt")
                            await telemetry_client.save_json(
                                "pdp_vrt_result",
                                {"vrt_diff_pixels": getattr(vrt_result, "diff_pixels", None)},
                                tctx,
                            )
                        except Exception as te:
                            self.log.warning(f"[Orchestrator] Failed to record VRT result: {te}", exc_info=True)
                except Exception as e:
                    self.log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)

        try:
            result = await extraction_service.extract_single_pdp(
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
            return result
        except Exception as e:
            self.log.warning(f"[Orchestrator] extract_single_pdp failed: {e}")
            failure_ctx = self._build_failure_context(
                site=site,
                query=query,
                final_url=page.url,
                error_type="pdp_extraction_failed",
                error_class=type(e).__name__,
                error_message=str(e),
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
                message=f"extract_single_pdp failed: {str(e)}",
                evidence=evidence,
            )

    # ------------------------------------------------------------------ #
    # _run_full helper
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
                    evidence={
                        "error": "No PDP URL found",
                        "plp_result": result_plp,
                    },
                )
        else:
            result = result_plp

        result = self._enrich_result_with_success_stage(
            result=result,
            run_context=run_context,
            nav_outcome=nav_outcome,
            page=page,
        )

        return result

    # ------------------------------------------------------------------ #
    # Success stage enrichment (CR-E2E-001)
    # ------------------------------------------------------------------ #

    def _enrich_result_with_success_stage(
        self,
        result: DiscoveryResult,
        run_context: RunContext,
        nav_outcome: NavigationOutcome | None = None,
        page: Page | None = None,
    ) -> DiscoveryResult:
        if not compute_success_stage or not collect_run_artifacts:
            return result

        try:
            artifacts = collect_run_artifacts(
                result=result,
                run_context=run_context,
                nav_outcome=nav_outcome,
                page=page,
            )
            success_stage, criteria = compute_success_stage(
                run_artifacts=artifacts,
                run_context=run_context,
            )
            if result.evidence is None:
                result.evidence = {}
            result.evidence["success_stage"] = success_stage
            result.evidence["criteria"] = criteria
            result.evidence["urls"] = {
                "plp_url": artifacts.get("plp_url"),
                "pdp_url_sample": artifacts.get("pdp_url_sample"),
            }
            result.evidence["extracted_fields"] = artifacts.get("extracted_fields", [])
            result.evidence["screenshots"] = artifacts.get("screenshots", [])
            result.evidence["dom_snapshots"] = artifacts.get("dom_snapshots", [])

            if run_context and hasattr(run_context, "get_path"):
                try:
                    validation_report_path = run_context.get_path("pdp_link_validation_report.json")
                    if validation_report_path.exists():
                        import json

                        with open(validation_report_path, encoding="utf-8") as f:
                            validation_report = json.load(f)
                        result.evidence["link_collection"] = {
                            "total_candidates": validation_report.get("total_candidates", 0),
                            "total_valid": validation_report.get("total_valid", 0),
                            "top_reject_reasons": validation_report.get("top_reject_reasons", {}),
                            "sample_candidates": validation_report.get("sample_rejected", [])[:10],
                        }
                except Exception as e:
                    self.log.debug(f"[Orchestrator] Failed to load pdp_link_validation_report: {e}")

            from app.utils.e2e_success_stage import should_result_be_ok

            result.ok = should_result_be_ok(success_stage)

            self.log.info(
                f"[Orchestrator] Success stage computed: {success_stage}, criteria={criteria}, result.ok={result.ok}"
            )
        except Exception as e:
            self.log.warning(f"[Orchestrator] Failed to compute success stage: {e}", exc_info=True)

        return result
