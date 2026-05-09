# -*- coding: utf-8 -*-
"""BrowserOrchestrator Mix-in: 設定ロード + メトリクス記録"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigAndMetricsMixin:
    """overrides.local.json のロード、selector 使用記録、self-healing メトリクス保存。"""

    # --- Config helpers ---

    def _load_overrides_json(self) -> Dict[str, Any]:
        if not self._overrides_path.exists():
            self.log.warning(
                f"[Orchestrator] overrides.local.json not found at {self._overrides_path}. "
                "Returning empty dict."
            )
            return {}
        try:
            with open(self._overrides_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.log.error(
                f"[Orchestrator] Failed to load overrides.local.json: {e}",
                exc_info=True
            )
            return {}

    def _load_site_config_from_overrides(self, site: str) -> Optional[Dict[str, Any]]:
        overrides_json = self._load_overrides_json()
        if not overrides_json:
            return None
        sites = overrides_json.get("sites", {})
        site_config = sites.get(site)
        if not site_config:
            self.log.warning(
                f"[Orchestrator] Site '{site}' not found in overrides.local.json"
            )
            return None
        return site_config

    # --- Selector usage recording (CR-ATELIER-003 Phase D-10.1) ---

    def _record_selector_usage(
        self,
        *,
        site: str,
        patch_candidate: Dict[str, Any],
        success: bool,
        result: Any,
    ) -> None:
        if not self.sandbox:
            return
        changes = patch_candidate.get("changes", [])
        selector_patches = [
            change for change in changes
            if change.get("action") == "selector_patch"
        ]
        if not selector_patches:
            return
        for change in selector_patches:
            new_selector = change.get("new_value")
            if not new_selector:
                continue
            if isinstance(new_selector, list):
                if not new_selector:
                    continue
                selector = new_selector[0]
            else:
                selector = str(new_selector)
            reason = "PDP extraction succeeded" if success else "PDP extraction failed"
            if not success and result:
                error_msg = result.evidence.get("error", "")
                if error_msg:
                    reason = f"Failed: {error_msg[:100]}"
            self.sandbox.record_selector_result(
                site=site,
                selector=selector,
                success=success,
                kind="pdp_link",
                reason=reason,
            )
            self.log.debug(
                f"[Orchestrator][SelectorFeedback] Recorded selector result: "
                f"site={site}, selector={selector[:50]}, success={success}"
            )

    # --- Self-Healing metrics (CR-ATELIER-003 Phase D-9/D-12) ---

    async def _record_self_healing_metrics(
        self,
        *,
        run_id: str,
        site: str,
        self_healing_attempts: int,
        auto_patches_applied: int,
        patch_backups: List[str],
        final_status: str,
        run_context: Optional[Any] = None,
    ) -> None:
        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "site": site,
            "self_healing_attempts": self_healing_attempts,
            "auto_patches_applied": auto_patches_applied,
            "patch_backups": patch_backups,
            "final_status": final_status,
            "timestamp": datetime.now().isoformat(),
            "policy": {},
        }
        if run_context:
            if hasattr(run_context, "run_type") and run_context.run_type:
                metrics["run_type"] = run_context.run_type
            if hasattr(run_context, "scenario_name") and run_context.scenario_name:
                metrics["scenario_name"] = run_context.scenario_name
        if self.policy and hasattr(self.policy, "max_attempts") and hasattr(self.policy, "data"):
            try:
                metrics["policy"] = {
                    "max_attempts": self.policy.max_attempts(),
                    "max_auto_applies_per_day": self.policy.data.get("max_auto_applies_per_day", 0),
                }
            except Exception:
                metrics["policy"] = {}
        metrics_path = Path("docs/reports/self_healing_metrics.jsonl")
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            self.log.info(
                f"[Orchestrator][SelfHealingLoop] Metrics saved to {metrics_path}"
            )
        except Exception as e:
            self.log.error(
                f"[Orchestrator][SelfHealingLoop] Failed to save metrics: {e}",
                exc_info=True
            )
