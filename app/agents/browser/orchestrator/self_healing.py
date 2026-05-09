# -*- coding: utf-8 -*-
"""BrowserOrchestrator Mix-in: Self-Healing エンジン（診断→パッチ→再実行）"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from playwright.async_api import Page, BrowserContext

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.agents.browser.plp_driver import PlpNavigationResult

logger = logging.getLogger(__name__)


class SelfHealingMixin:
    """失敗分析・パッチ生成・selector修復・再実行ループ。"""

    # ---- Failure context builder (CR-ATELIER-003 Phase D-4) ----

    def _build_failure_context(
        self,
        *,
        site: str,
        query: str,
        final_url: str,
        error_type: str,
        error_class: str,
        error_message: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
    ) -> Dict[str, Any]:
        failure_ctx: Dict[str, Any] = {
            "final_url": final_url,
            "error_type": error_type,
            "error_class": error_class,
            "error_message": error_message,
            "site": site,
            "query": query,
            "run_id": getattr(run_context, "run_id", "unknown"),
        }
        try:
            if hasattr(run_context, "get_path"):
                dom_path = run_context.get_path("failure_dom.html")
                if dom_path:
                    failure_ctx["dom_snapshot_path"] = str(dom_path)
        except Exception:
            pass
        try:
            if hasattr(run_context, "screenshots") and isinstance(run_context.screenshots, list):
                failure_ctx["screenshot_paths"] = run_context.screenshots[:6]
            elif hasattr(run_context, "get_path"):
                run_dir = Path(getattr(run_context, "run_path", "."))
                if run_dir.exists():
                    recent_pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    failure_ctx["screenshot_paths"] = [str(p) for p in recent_pngs[:6]]
        except Exception:
            pass
        try:
            site_summary: Dict[str, Any] = {
                "site_code": site_config.get("site_code") or site_config.get("site") or site,
            }
            selectors = site_config.get("selectors", {})
            if selectors:
                site_summary["has_plp_selectors"] = bool(selectors.get("plp"))
                site_summary["has_pdp_selectors"] = bool(selectors.get("pdp"))
            failure_ctx["site_config_summary"] = site_summary
        except Exception:
            pass
        return failure_ctx

    # ---- Failure analysis (CR-ATELIER-003 Phase D-5) ----

    async def _maybe_analyze_failure(
        self,
        failure_context: Dict[str, Any],
        *,
        page: Optional[Page] = None,
        run_context: Optional[RunContext] = None,
    ) -> Optional[Dict[str, Any]]:
        self.log.info(
            f"[Orchestrator][SelfHealing] Failure detected: "
            f"type={failure_context.get('error_type')}, "
            f"class={failure_context.get('error_class')}, "
            f"url={failure_context.get('final_url')}"
        )
        if self.analysis_agent is None:
            self.log.debug("[Orchestrator][SelfHealing] AnalysisAgent is not available, skipping analysis")
            return None
        try:
            if hasattr(self.analysis_agent, 'analyze_failure_context'):
                analysis = await self.analysis_agent.analyze_failure_context(
                    failure_context,
                    run_context=run_context,
                )
                self.log.info(
                    f"[Orchestrator][SelfHealing] Analysis completed: "
                    f"summary={analysis.get('summary', 'N/A')[:100]}, "
                    f"confidence={analysis.get('confidence', 0.0)}"
                )
                return analysis
            else:
                self.log.warning(
                    "[Orchestrator][SelfHealing] AnalysisAgent does not have analyze_failure_context method, "
                    "falling back to legacy analyze method"
                )
                return None
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelfHealing] Failure analysis failed: {e}",
                exc_info=True
            )
            return None

    # ---- Patch candidate generation (CR-ATELIER-003 Phase D-6, D-10) ----

    async def _maybe_build_patch_candidate(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        page: Optional[Page] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.patch_agent is None:
            self.log.debug("[Orchestrator][SelfHealing] PatchAgent is not available, skipping patch generation")
            return None
        try:
            selector_repair_result = None
            if self.selector_repair_agent and self.policy:
                if self.policy.selector_auto_healing_enabled():
                    selector_repair_result = await self._maybe_repair_selectors(
                        failure_context=failure_context,
                        failure_analysis=failure_analysis,
                        site_config=site_config,
                        run_context=run_context,
                        page=page,
                    )
            patch_candidate = await self.patch_agent.build_patch_candidate(
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                site_config=site_config,
                run_context=run_context,
                selector_repair_result=selector_repair_result,
            )
            if patch_candidate:
                self.log.info(
                    f"[Orchestrator][SelfHealing] Patch candidate generated: "
                    f"run_id={patch_candidate.get('run_id')}, "
                    f"changes_count={len(patch_candidate.get('changes', []))}"
                )
            return patch_candidate
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelfHealing] Patch candidate generation failed: {e}",
                exc_info=True
            )
            return None

    # ---- Selector repair (CR-ATELIER-003 Phase D-10) ----

    async def _maybe_repair_selectors(
        self,
        *,
        failure_context: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        site_config: Dict[str, Any],
        run_context: RunContext,
        page: Optional[Page] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.selector_repair_agent is None:
            return None
        try:
            site = failure_context.get("site", "unknown")
            error_type = failure_context.get("error_type", "")
            error_message = failure_context.get("error_message", "")
            is_selector_error = (
                "selector" in error_message.lower() or
                "not found" in error_message.lower() or
                error_type in ("pdp_extraction_failed", "plp_extraction_failed")
            )
            if not is_selector_error:
                self.log.debug(
                    f"[Orchestrator][SelectorRepair] Skipping selector repair: "
                    f"error_type={error_type} is not selector-related"
                )
                return None
            page_type = "pdp" if "pdp" in error_type.lower() else "plp"
            dom_snapshot_html = ""
            dom_snapshot_path = failure_context.get("dom_snapshot_path")
            if dom_snapshot_path:
                try:
                    dom_path = Path(dom_snapshot_path)
                    if dom_path.exists():
                        dom_snapshot_html = dom_path.read_text(encoding="utf-8")
                except Exception as e:
                    self.log.warning(f"[Orchestrator][SelectorRepair] Failed to read DOM snapshot: {e}")
            if not dom_snapshot_html and page and not page.is_closed():
                try:
                    dom_snapshot_html = await page.content()
                except Exception as e:
                    self.log.warning(f"[Orchestrator][SelectorRepair] Failed to get page content: {e}")
            if not dom_snapshot_html:
                self.log.warning("[Orchestrator][SelectorRepair] DOM snapshot not available, skipping selector repair")
                return None
            current_selectors = site_config.get("selectors", {}).get(page_type, {})
            previous_successes: list[Any] = []
            previous_failures: list[Any] = []
            if self.sandbox:
                try:
                    previous_successes = self.sandbox.get_previous_successes(
                        site=site,
                        kind="pdp_link",
                    )
                    previous_failures = self.sandbox.get_previous_failures(
                        site=site,
                        kind="pdp_link",
                    )
                    self.log.debug(
                        f"[Orchestrator][SelectorFeedback] Loaded feedback: "
                        f"successes={len(previous_successes)}, failures={len(previous_failures)}"
                    )
                except Exception as e:
                    self.log.warning(
                        f"[Orchestrator][SelectorFeedback] Failed to load feedback: {e}",
                        exc_info=True
                    )
            selector_repair_result = await self.selector_repair_agent.propose_selector_patches(
                site=site,
                page_type=page_type,
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                dom_snapshot_html=dom_snapshot_html,
                current_selectors=current_selectors,
                site_config=site_config,
                previous_successes=previous_successes,
                previous_failures=previous_failures,
            )
            if selector_repair_result and selector_repair_result.get("candidates"):
                self.log.info(
                    f"[Orchestrator][SelectorRepair] Generated {len(selector_repair_result['candidates'])} "
                    f"selector candidates for {site}/{page_type}"
                )
            return selector_repair_result
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelectorRepair] Selector repair failed: {e}",
                exc_info=True
            )
            return None

    # ---- Self-Healing v1: single retry (CR-ATELIER-003 Phase D-8) ----

    async def run_with_self_healing_once(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        overrides_json: Dict[str, Any],
        nav_outcome: Optional[Any] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self.log.info("[Orchestrator][SelfHealing] Starting initial run...")

        initial_result_plp = await self.run_plp_to_pdp(
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

        if isinstance(initial_result_plp, PlpNavigationResult):
            if initial_result_plp.pdp_url:
                initial_result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=site_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=initial_result_plp.pdp_url,
                )
            else:
                initial_result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={
                        "error": "No PDP URL found",
                        "plp_result": initial_result_plp,
                    },
                )
        else:
            initial_result = initial_result_plp

        if initial_result.ok:
            self.log.info("[Orchestrator][SelfHealing] Initial run succeeded. Self-Healing not needed.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": None,
                "sandbox_config": None,
            }

        self.log.info("[Orchestrator][SelfHealing] Initial run failed. Generating patch candidate...")
        failure_context = self._build_failure_context(
            site=site,
            query=query,
            error_type="pdp_extraction_failed",
            error_class=type(initial_result.evidence.get("error", Exception())).__name__,
            error_message=str(initial_result.evidence.get("error", "Unknown error")),
            final_url=page.url,
            run_context=run_context,
            site_config=site_config,
        )
        failure_analysis = initial_result.evidence.get("failure_analysis")
        if not failure_analysis:
            failure_analysis = {
                "summary": "Initial run failed",
                "root_causes": [],
                "suggested_fixes": [],
            }

        patch_candidate = None
        if self.patch_agent:
            patch_candidate = await self._maybe_build_patch_candidate(
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                site_config=site_config,
                run_context=run_context,
                page=None,
            )

        if not patch_candidate:
            self.log.warning("[Orchestrator][SelfHealing] Patch candidate generation failed or skipped.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": None,
                "sandbox_config": None,
            }

        if not self.sandbox:
            self.log.warning("[Orchestrator][SelfHealing] Sandbox not available. Skipping Self-Healing.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": patch_candidate,
                "sandbox_config": None,
            }

        self.log.info("[Orchestrator][SelfHealing] Applying patch in sandbox...")
        sandbox_overrides = self.sandbox.apply_patch_in_memory(
            overrides_json=overrides_json,
            patch_candidate=patch_candidate,
        )
        sandbox_config = self.sandbox.get_site_config_from_overrides(
            overrides_json=sandbox_overrides,
            site_code=site,
        )
        if not sandbox_config:
            self.log.warning(f"[Orchestrator][SelfHealing] Site '{site}' not found in sandbox overrides.")
            return {
                "initial": initial_result,
                "after_patch": None,
                "self_healing_success": False,
                "patch_candidate": patch_candidate,
                "sandbox_config": None,
            }

        self.log.info("[Orchestrator][SelfHealing] Re-running with sandbox config...")
        after_patch_result_plp = await self.run_plp_to_pdp(
            page=page,
            context=context,
            site=site,
            query=query,
            site_config=sandbox_config,
            settings=settings,
            run_context=run_context,
            target_url=target_url,
            start_t=start_t,
            budget_ms=budget_ms,
            nav_outcome=None,
            trap_checker=trap_checker,
            telemetry=telemetry,
            plugin=plugin,
        )

        if isinstance(after_patch_result_plp, PlpNavigationResult):
            if after_patch_result_plp.pdp_url:
                after_patch_result = await self.run_pdp(
                    page=page,
                    context=context,
                    site=site,
                    query=query,
                    site_config=sandbox_config,
                    settings=settings,
                    run_context=run_context,
                    target_url=after_patch_result_plp.pdp_url,
                )
            else:
                after_patch_result = DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    evidence={
                        "error": "No PDP URL found after patch",
                        "plp_result": after_patch_result_plp,
                    },
                )
        else:
            after_patch_result = after_patch_result_plp

        self_healing_success = after_patch_result.ok

        if self.sandbox and patch_candidate:
            self._record_selector_usage(
                site=site,
                patch_candidate=patch_candidate,
                success=self_healing_success,
                result=after_patch_result,
            )

        if self_healing_success:
            self.log.info("[Orchestrator][SelfHealing] Self-Healing succeeded!")
        else:
            self.log.warning("[Orchestrator][SelfHealing] Self-Healing failed. Manual intervention may be needed.")

        return {
            "initial": initial_result,
            "after_patch": after_patch_result,
            "self_healing_success": self_healing_success,
            "patch_candidate": patch_candidate,
            "sandbox_config": sandbox_config,
        }

    # ---- Self-Healing v2: multi-attempt loop (CR-ATELIER-003 Phase D-9) ----

    async def run_with_self_healing_loop(
        self,
        *,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        start_t: float,
        budget_ms: int,
        max_attempts: Optional[int] = None,
        nav_outcome: Optional[Any] = None,
        trap_checker: Optional[Callable[[str], bool]] = None,
        telemetry: Optional[Any] = None,
        plugin: Optional[Any] = None,
    ) -> DiscoveryResult:
        if not self.policy:
            self.log.warning("[Orchestrator][SelfHealingLoop] Policy not available. Falling back to normal execution.")
            return await self._run_full(
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

        attempt = 0
        max_attempts_value = max_attempts or self.policy.max_attempts()
        current_site_config = site_config
        auto_patches_applied = 0
        patch_backups: List[str] = []

        self.log.info(
            f"[Orchestrator][SelfHealingLoop] Starting Self-Healing Loop "
            f"(max_attempts={max_attempts_value})"
        )

        while attempt < max_attempts_value:
            attempt += 1
            self.log.info(f"[Orchestrator][SelfHealingLoop] Attempt {attempt}/{max_attempts_value}")

            result = await self._run_full(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=current_site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                start_t=start_t,
                budget_ms=budget_ms,
                nav_outcome=nav_outcome if attempt == 1 else None,
                trap_checker=trap_checker,
                telemetry=telemetry,
                plugin=plugin,
            )

            if result.ok:
                self.log.info(
                    f"[Orchestrator][SelfHealingLoop] Success on attempt {attempt}"
                )
                result.evidence["self_healing_attempts"] = attempt
                result.evidence["auto_patches_applied"] = auto_patches_applied
                result.evidence["patch_backups"] = patch_backups
                await self._record_self_healing_metrics(
                    run_id=run_context.run_id,
                    site=site,
                    self_healing_attempts=attempt,
                    auto_patches_applied=auto_patches_applied,
                    patch_backups=patch_backups,
                    final_status="success",
                    run_context=run_context,
                )
                return result

            patch_candidate = result.evidence.get("self_healing_patch_candidate")
            if not patch_candidate:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] No patch_candidate found. "
                    "Stopping Self-Healing loop."
                )
                break

            if not self.policy.enabled():
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Policy disabled. Stopping Self-Healing loop."
                )
                break

            if not self.policy.is_allowed_site(site):
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Site '{site}' not allowed. "
                    "Stopping Self-Healing loop."
                )
                break

            if not self.policy.safe_change(patch_candidate):
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Unsafe change detected. "
                    "Stopping Self-Healing loop."
                )
                break

            if not self.policy.can_apply_today(site):
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Daily limit reached for '{site}'. "
                    "Stopping Self-Healing loop."
                )
                break

            if not self.sandbox:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Sandbox not available. "
                    "Stopping Self-Healing loop."
                )
                break

            overrides_json = self._load_overrides_json()
            if not overrides_json:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Failed to load overrides_json. "
                    "Stopping Self-Healing loop."
                )
                break

            self.log.info("[Orchestrator][SelfHealingLoop] Applying patch in sandbox...")
            sandbox_overrides = self.sandbox.apply_patch_in_memory(
                overrides_json=overrides_json,
                patch_candidate=patch_candidate,
            )
            sandbox_site_config = self.sandbox.get_site_config_from_overrides(
                overrides_json=sandbox_overrides,
                site_code=site,
            )
            if not sandbox_site_config:
                self.log.warning(
                    f"[Orchestrator][SelfHealingLoop] Site '{site}' not found in sandbox overrides. "
                    "Stopping Self-Healing loop."
                )
                break

            sandbox_result = await self._run_full(
                page=page,
                context=context,
                site=site,
                query=query,
                site_config=sandbox_site_config,
                settings=settings,
                run_context=run_context,
                target_url=target_url,
                start_t=start_t,
                budget_ms=budget_ms,
                nav_outcome=None,
                trap_checker=trap_checker,
                telemetry=telemetry,
                plugin=plugin,
            )

            if not sandbox_result.ok:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] Sandbox execution failed. "
                    "Not applying patch to production."
                )
                break

            if not self.patch_applier:
                self.log.warning(
                    "[Orchestrator][SelfHealingLoop] PatchApplier not available. "
                    "Stopping Self-Healing loop."
                )
                break

            candidate_path = run_context.run_path / "patch_candidate_self_healing.json"
            backup_suffix = f".bak-selfheal-{attempt}"

            try:
                apply_meta = self.patch_applier.apply_patch_candidate(
                    candidate_path=candidate_path,
                    overrides_path=self._overrides_path,
                    backup_suffix=backup_suffix,
                )
                if apply_meta and apply_meta.get("success"):
                    auto_patches_applied += 1
                    backup_path = apply_meta.get("backup_path")
                    if backup_path:
                        patch_backups.append(str(backup_path))
                    self.log.info(
                        f"[Orchestrator][SelfHealingLoop] Patch applied successfully "
                        f"(attempt {attempt})"
                    )
                    self.policy.increment_daily_counter(site)
                    current_site_config = self._load_site_config_from_overrides(site)
                    if not current_site_config:
                        self.log.warning(
                            f"[Orchestrator][SelfHealingLoop] Failed to reload site_config for '{site}'. "
                            "Stopping Self-Healing loop."
                        )
                        break
                else:
                    self.log.warning(
                        "[Orchestrator][SelfHealingLoop] Patch application failed. "
                        "Stopping Self-Healing loop."
                    )
                    break
            except Exception as e:
                self.log.error(
                    f"[Orchestrator][SelfHealingLoop] Exception during patch application: {e}",
                    exc_info=True
                )
                break

        result.evidence["self_healing_attempts"] = attempt
        result.evidence["auto_patches_applied"] = auto_patches_applied
        result.evidence["patch_backups"] = patch_backups

        final_status = "success" if result.ok else "failure"
        result = self._enrich_result_with_success_stage(
            result=result,
            run_context=run_context,
            nav_outcome=None,
            page=None,
        )
        await self._record_self_healing_metrics(
            run_id=run_context.run_id,
            site=site,
            self_healing_attempts=attempt,
            auto_patches_applied=auto_patches_applied,
            patch_backups=patch_backups,
            final_status=final_status,
            run_context=run_context,
        )
        return result
