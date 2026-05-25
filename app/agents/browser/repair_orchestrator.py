"""Repair orchestrator Mixin: self-healing scraping flow."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.extractor import BrowserExtractionService
from app.agents.browser.session_manager import SessionManager
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

logger = logging.getLogger(__name__)

try:
    from app.agents.interactive_repair_session import InteractiveRepairSession
except Exception as _e:
    logger.warning("InteractiveRepairSession not available: %s", _e)
    InteractiveRepairSession = None

try:
    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
except Exception as _e:
    logger.warning("moncler_plp_recovery not available: %s", _e)
    moncler_plp_recovery = None

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000

_SITE_RECOVERY_PATCHES: dict[str, Any] = {}
if moncler_plp_recovery is not None:
    _SITE_RECOVERY_PATCHES["MONCLER_OFFICIAL"] = moncler_plp_recovery


def _merge_learned_selectors(site: str, site_config: dict[str, Any], run_context: RunContext) -> None:
    try:
        instance_dir = Path(run_context.run_path).parent.parent
        learned_path = instance_dir / "sites" / site.upper() / "learned_selectors.json"
        if learned_path.exists():
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
            sc_sel = site_config.setdefault("selectors", {}).setdefault("pdp", {})
            for k in ("price_selectors", "title_selectors", "pdp_link_selectors"):
                v = learned.get(k)
                if v:
                    sc_sel[k] = list(dict.fromkeys(list(v) + list(sc_sel.get(k, []))))
            logger.info(f"[LEARN] loaded and merged selectors: {learned_path}")
    except Exception as e:
        logger.warning(f"[LEARN] merge skipped: {e}")


class RepairOrchestratorMixin:
    """BrowserUseAgent の自己修復スクレイピングフローを担当。"""

    _session_manager: SessionManager | None
    _context: BrowserContext | None
    _page: Page | None
    runtime_kwargs: dict[str, Any]
    logger: logging.Logger
    run_context: RunContext | None
    extraction_service: BrowserExtractionService

    # DI: 外部からfactoryを注入可能（テスト用）。None時は遅延importで生成。
    _interactive_repair_session_factory: Any | None = None
    _llm_controller_factory: Any | None = None

    async def run_with_repair(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
        max_steps: int = 5,
        repair_budget_ms: int = 60000,
    ) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        self.run_context = run_context
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        _merge_learned_selectors(site, site_config, run_context)

        base_result = await self._repair_initial_run(
            site=site, query=query, site_config=site_config,
            run_context=run_context, target_url=target_url,
            likely_plp=likely_plp, settings=settings,
        )

        if getattr(base_result, "ok", False):
            try:
                await self._close_session(run_context, settings)
            finally:
                if hasattr(self, "run_context"):
                    del self.run_context
            return base_result

        return await self._run_interactive_repair(
            base_result=base_result, site=site, query=query,
            site_config=site_config, run_context=run_context,
            settings=settings, max_steps=max_steps,
            repair_budget_ms=repair_budget_ms, target_url=target_url,
        )

    async def _repair_initial_run(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
        settings: dict[str, Any],
    ) -> DiscoveryResult:
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        try:
            page = await self._open_session(
                site=site, site_config=site_config, run_context=run_context,
                settings=settings, target_url=target_url,
                timeout_ms=timeout_ms, likely_plp=likely_plp,
            )

            await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
            if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "plp", settings)

            site_patch = _SITE_RECOVERY_PATCHES.get(site.upper())
            if site_patch is not None and not likely_plp:
                try:
                    await site_patch(page, site_config, {"query": query, "shipTo": "GB"})
                except Exception as _e:
                    self.logger.warning(f"[SitePatch] {site} recovery skipped: {_e}")

            context = self._context
            if context is None:
                raise ValueError("BrowserContext was not initialized by _open_session")

            start_t, budget_ms = self._start_watchdog(
                settings.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)
            )

            if mode == "learn":
                raise ValueError("mode='learn' is not supported in run_with_repair. Use run() instead.")

            if likely_plp:
                return await self._run_plp_flow(
                    page, context, site, query, site_config,
                    settings, run_context, target_url=target_url,
                    start_t=start_t, budget_ms=budget_ms,
                )
            else:
                return await self._run_pdp_flow(page, site, query, settings, run_context, site_config)

        except Exception as e:
            return await self._handle_run_failure(e, site, query, site_config, run_context, self._page)

    async def _run_interactive_repair(
        self,
        *,
        base_result: DiscoveryResult,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
        max_steps: int,
        repair_budget_ms: int,
        target_url: str,
    ) -> DiscoveryResult:
        self.logger.warning(
            f"[run_with_repair] Initial run failed. Entering guided repair loop. Reason: {base_result.message}"
        )

        if InteractiveRepairSession is None:
            self.logger.error("[run_with_repair] InteractiveRepairSession not available, cannot self-heal.")
            await self._close_session_safely(run_context, settings)
            return base_result

        try:
            if self._llm_controller_factory is not None:
                llm_ctrl = self._llm_controller_factory()
            else:
                from app.utils.ai_llm_controller import AiLlmController
                llm_ctrl = AiLlmController(mode="Chat/Default")
        except Exception as e:
            self.logger.error(f"[run_with_repair] Failed to instantiate AiLlmController: {e}. Aborting repair.")
            await self._close_session_safely(run_context, settings)
            return base_result

        failure_ctx = self._build_repair_failure_context(base_result, site_config)

        repair_out, repair_status, healed_result = await self._execute_repair_loop(
            site=site, query=query, run_context=run_context,
            max_steps=max_steps, repair_budget_ms=repair_budget_ms,
            base_result=base_result, failure_ctx=failure_ctx, llm_ctrl=llm_ctrl,
        )

        if healed_result is None:
            healed_result = self._evaluate_repair_result(
                repair_out=repair_out, repair_status=repair_status,
                site=site, query=query, site_config=site_config,
                run_context=run_context, target_url=target_url,
                base_result=base_result,
            )

        await self._close_session_safely(run_context, settings)
        return healed_result

    async def _close_session_safely(self, run_context: RunContext, settings: dict[str, Any]) -> None:
        try:
            await self._close_session(run_context, settings)
        finally:
            if hasattr(self, "run_context"):
                del self.run_context

    @staticmethod
    def _build_repair_failure_context(base_result: DiscoveryResult, site_config: dict[str, Any]) -> dict[str, Any]:
        failure_ev = base_result.evidence or {}
        failure_ctx_from_run = failure_ev.get("failure_context", {})
        return {
            "final_url": failure_ev.get("final_url"),
            "page_html_path": failure_ctx_from_run.get("dom_snapshot_path"),
            "screenshot_path": (failure_ctx_from_run.get("screenshots") or [None])[0],
            "exception_message": (failure_ctx_from_run.get("errors") or [""])[0],
            "selectors_used": {
                "pdp": (site_config.get("selectors", {}) or {}).get("pdp", {}),
            },
            "intent_description": failure_ctx_from_run.get(
                "intent_description", "Goal: Extract PLP items or PDP price. Initial attempt failed."
            ),
        }

    async def _execute_repair_loop(
        self,
        *,
        site: str,
        query: str,
        run_context: RunContext,
        max_steps: int,
        repair_budget_ms: int,
        base_result: DiscoveryResult,
        failure_ctx: dict[str, Any],
        llm_ctrl: Any,
    ) -> tuple[Any, str, DiscoveryResult | None]:
        repair_out = None
        repair_status = "unknown_error"
        healed_result: DiscoveryResult | None = None

        try:
            session_cls = self._interactive_repair_session_factory or InteractiveRepairSession
            if session_cls is None:
                raise RuntimeError("InteractiveRepairSession not available")
            repair_session = session_cls(
                ai_controller=llm_ctrl, run_context=run_context, max_steps=max_steps,
            )
            maybe_coro = repair_session.run_repair_loop(
                page=self._page, site_key=site,
                intent="Collect PLP items and PDP prices",
                initial_failure=failure_ctx,
            )
            repair_budget_sec = max(5.0, float(repair_budget_ms) / 1000.0)

            if asyncio.iscoroutine(maybe_coro):
                self.logger.info(f"[run_with_repair] Waiting for async repair loop (budget: {repair_budget_sec}s)")
                repair_out = await asyncio.wait_for(maybe_coro, timeout=repair_budget_sec)
            else:
                self.logger.warning("[run_with_repair] Running sync repair loop (cannot enforce budget)")
                repair_out = maybe_coro
            repair_status = "completed"

        except asyncio.TimeoutError:
            self.logger.error(f"[run_with_repair] InteractiveRepairSession timed out after {repair_budget_sec}s.")
            repair_status = "timeout_exceeded"
            healed_result = DiscoveryResult(
                ok=False, site=site, query=query,
                message=f"Repair loop timed out after {repair_budget_sec}s",
                evidence={"status": "timeout_exceeded", "initial_failure": base_result.evidence},
            )
        except Exception as repair_e:
            self.logger.error(f"[run_with_repair] InteractiveRepairSession failed: {repair_e}", exc_info=True)
            repair_status = "catastrophic_failure"
            healed_result = DiscoveryResult(
                ok=False, site=site, query=query,
                message=f"Repair loop failed: {repair_e}",
                evidence={"status": "catastrophic_failure", "initial_failure": base_result.evidence},
            )

        return repair_out, repair_status, healed_result

    def _evaluate_repair_result(
        self,
        *,
        repair_out: Any,
        repair_status: str,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        base_result: DiscoveryResult,
    ) -> DiscoveryResult:
        if repair_status == "completed" and repair_out:
            selectors_update = repair_out.get("selectors_update")
            code_patch = repair_out.get("code_patch", "")
            self._merge_repair_selectors(selectors_update, site, run_context)

            if code_patch:
                self.logger.info("[run_with_repair] Received code patch suggestion. Saving to artifacts.")
                run_context.save_text("browser_use_agent.patch", code_patch)

            if selectors_update or code_patch:
                self.logger.info("[run_with_repair] Repair loop completed and produced artifacts.")
                return DiscoveryResult(
                    ok=True, site=site, query=query,
                    message="Recovered via InteractiveRepairSession",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "selectors_update": selectors_update, "code_patch": code_patch,
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []), "status": "recovered",
                    },
                )
            else:
                self.logger.warning("[run_with_repair] Repair completed but produced no artifacts.")
                status_from_repair = repair_out.get("status", "exhausted_steps")
                return DiscoveryResult(
                    ok=False, site=site, query=query,
                    message=f"Repair loop finished without artifacts (Status: {status_from_repair})",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []),
                        "status": status_from_repair, "initial_failure": base_result.evidence,
                    },
                )

        self.logger.error(
            f"[run_with_repair] Repair loop finished abnormally (Status: {repair_status}, repair_out: {repair_out})"
        )
        return DiscoveryResult(
            ok=False, site=site, query=query,
            message=f"Repair loop failed with unknown status: {repair_status}",
            evidence={"status": repair_status, "initial_failure": base_result.evidence},
        )

    def _merge_repair_selectors(self, selectors_update: dict | None, site: str, run_context: RunContext) -> None:
        if not selectors_update:
            return
        try:
            from app.utils.overrides_store import update_site_selectors
            site_block = selectors_update.get(site.upper()) or selectors_update.get(site) or {}
            new_sels = site_block.get("selectors") or {}
            if new_sels:
                overrides_path = "app/config/sites/overrides.local.json"
                updated, diff_txt = update_site_selectors(
                    site=site.upper(), new_selectors=new_sels, overrides_path=overrides_path,
                )
                if updated and diff_txt:
                    self.logger.info(f"[run_with_repair] Merged selectors_update to {overrides_path}")
                    run_context.save_text("repair_selector_diff.patch", diff_txt)
                else:
                    self.logger.info("[run_with_repair] Selectors update did not result in changes.")
            else:
                self.logger.info("[run_with_repair] Selectors update was empty.")
        except ImportError:
            self.logger.error("[run_with_repair] `app.utils.overrides_store` not found. Cannot merge selectors.")
        except Exception as merge_e:
            self.logger.warning(f"[run_with_repair] could not merge selectors_update: {merge_e}")
