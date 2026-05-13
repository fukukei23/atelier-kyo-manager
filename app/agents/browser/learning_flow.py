"""Learning flow Mixin: selector discovery and persistence."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.navigation_driver import _dedupe_keep_order
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import save_dom

logger = logging.getLogger(__name__)


class LearningMixin:
    """BrowserUseAgent の学習フローを担当。"""

    logger: logging.Logger
    discovery_agent: Any

    async def _run_learning_flow(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        *,
        start_t: float,
        budget_ms: int,
    ) -> DiscoveryResult:
        self.logger.info(f"[LEARN] Starting learning flow for site: {site}")
        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)
        await save_dom(run_context, page, "learn_plp_dom_for_discovery")
        try:
            discovered_selectors = await self.discovery_agent.discover(
                page=page, context=context, run_context=run_context
            )
            if not discovered_selectors:
                raise ValueError("SelectorDiscoveryAgent returned no selectors.")
            self.logger.info(f"[LEARN] Discovered selectors: {json.dumps(discovered_selectors, indent=2)}")
        except Exception as e:
            self.logger.error(f"[LEARN] Selector discovery failed: {e}", exc_info=True)
            return await self._handle_run_failure(
                e, site, "(learning)", site_config, run_context, page
            )
        try:
            await self._save_learned_selectors(site, discovered_selectors, run_context)
        except Exception as e:
            self.logger.error(f"[LEARN] Failed to save learned selectors: {e}", exc_info=True)
            return DiscoveryResult(
                ok=False, site=site, query="(learning)",
                message=f"Failed to save: {e}",
                evidence={"learned_selectors": discovered_selectors},
            )
        return DiscoveryResult(
            ok=True, site=site, query="(learning)",
            message="Successfully learned and saved.",
            evidence={"learned_selectors": discovered_selectors},
        )

    async def _save_learned_selectors(
        self, site: str, new_selectors: dict[str, list[str]], run_context: RunContext
    ) -> None:
        try:
            instance_dir = Path(run_context.run_path).parent.parent
            site_dir = instance_dir / "sites" / site.upper()
            site_dir.mkdir(parents=True, exist_ok=True)
            learned_path = site_dir / "learned_selectors.json"
        except Exception:
            learned_path = Path(f"instance/sites/{site.upper()}/learned_selectors.json")
        learned_path.parent.mkdir(parents=True, exist_ok=True)
        existing_selectors = {}
        if learned_path.exists():
            with contextlib.suppress(Exception):
                existing_selectors = json.loads(learned_path.read_text(encoding="utf-8")) or {}
        merged_selectors = {}
        all_keys = set(existing_selectors.keys()) | set(new_selectors.keys())
        for key in all_keys:
            merged_list = _dedupe_keep_order(new_selectors.get(key, []) + existing_selectors.get(key, []))
            if merged_list:
                merged_selectors[key] = merged_list
        learned_path.write_text(json.dumps(merged_selectors, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info(f"[LEARN] Saved merged selectors to: {learned_path}")
