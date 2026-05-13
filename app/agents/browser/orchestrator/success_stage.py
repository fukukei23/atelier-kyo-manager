"""BrowserOrchestrator Mixin: Success stage enrichment (CR-E2E-001)."""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import Page

from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult

# Optional import
try:
    from app.utils.e2e_success_stage import collect_run_artifacts, compute_success_stage
except ImportError:
    compute_success_stage = None  # type: ignore
    collect_run_artifacts = None  # type: ignore


class SuccessStageMixin:
    """Enriches DiscoveryResult with success stage computation."""

    log: logging.Logger

    def _enrich_result_with_success_stage(
        self,
        result: DiscoveryResult,
        run_context: RunContext,
        nav_outcome: Any | None = None,
        page: Page | None = None,
    ) -> DiscoveryResult:
        if not compute_success_stage or not collect_run_artifacts:
            return result

        try:
            artifacts = collect_run_artifacts(
                result=result, run_context=run_context, nav_outcome=nav_outcome, page=page,
            )
            success_stage, criteria = compute_success_stage(
                run_artifacts=artifacts, run_context=run_context,
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
