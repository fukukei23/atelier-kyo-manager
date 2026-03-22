# -*- coding: utf-8 -*-
# ==============================================================================
# File: navigation_driver.py
# Registry: app/agents/browser/navigation_driver.py
# Date & Time (JST): 2026-03-21
# Version: 1.1J
# Purpose: Navigation driver for PLP materialize and trap detection
# ==============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Awaitable

from playwright.async_api import Page

logger = logging.getLogger(__name__)

# BrowserUseAgent 側の _ensure_plp_materialized を受け取るための型
EnsurePlpMaterializedFn = Callable[
    [Page, Dict[str, Any], Dict[str, Any], float, int],
    Awaitable[bool],
]

TrapCheckerFn = Callable[
    [Page, Dict[str, Any], Dict[str, Any]],
    Awaitable[Optional[str]],
]


@dataclass
class NavigationContext:
    site: str
    query: str
    site_config: Dict[str, Any]
    settings: Dict[str, Any]
    run_context: Any
    start_t: float
    budget_ms: int
    entry_url: Optional[str] = None


@dataclass
class NavigationOutcome:
    entry_url: str
    plp_materialized: bool = False
    trap_detected: bool = False
    trap_reason: Optional[str] = None
    retry_count: int = 0
    elapsed_ms: float = 0.0
    error_message: Optional[str] = None


class NavigationDriver:
    """
    Stage 3A-2:
      - PLP materialize を BrowserUseAgent から一部移管
      - trap 判定の拡張
      - リトライロジックの追加
    """

    MAX_RETRIES: int = 2

    def __init__(
        self,
        page: Page,
        *,
        ensure_plp_materialized: EnsurePlpMaterializedFn,
        trap_checker: Optional[TrapCheckerFn] = None,
        telemetry: Any = None,
        strategy: Any = None,
    ) -> None:
        self.page = page
        self.ensure_plp_materialized = ensure_plp_materialized
        self.trap_checker = trap_checker
        self.telemetry = telemetry
        self.strategy = strategy

    async def run_plp_flow(self, ctx: NavigationContext) -> NavigationOutcome:
        import time
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)
        start_time = time.monotonic()

        try:
            # --- trap チェック ---
            if self.trap_checker:
                reason = await self._check_trap(ctx)
                if reason:
                    outcome.trap_detected = True
                    outcome.trap_reason = reason
                    outcome.elapsed_ms = (time.monotonic() - start_time) * 1000
                    return outcome

            # --- PLP materialize 本体（リトライ付き） ---
            ok = await self._ensure_plp_with_retry(ctx, outcome)
            outcome.plp_materialized = bool(ok)

        except Exception as e:
            logger.warning(f"[NavigationDriver] Unexpected error: {e}")
            outcome.error_message = str(e)

        outcome.elapsed_ms = (time.monotonic() - start_time) * 1000
        return outcome

    async def _check_trap(self, ctx: NavigationContext) -> Optional[str]:
        """Trap チェックを実行"""
        try:
            reason = await self.trap_checker(
                self.page, ctx.site_config, ctx.settings
            )
            if reason:
                logger.info(f"[NavigationDriver] Trap detected: {reason}")
            return reason
        except Exception as e:
            logger.warning(f"[NavigationDriver] Trap check failed: {e}")
            return None

    async def _ensure_plp_with_retry(
        self, ctx: NavigationContext, outcome: NavigationOutcome
    ) -> bool:
        """PLP materialize をリトライ付きで実行"""
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            outcome.retry_count = attempt
            try:
                ok = await self.ensure_plp_materialized(
                    self.page,
                    ctx.site_config,
                    ctx.settings,
                    ctx.start_t,
                    ctx.budget_ms,
                )
                if ok:
                    if attempt > 0:
                        logger.info(
                            f"[NavigationDriver] PLP materialized on retry {attempt}"
                        )
                    return True
                last_error = "PLP did not materialize"
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[NavigationDriver] PLP materialize attempt {attempt} failed: {e}"
                )

        outcome.error_message = last_error
        logger.warning(
            f"[NavigationDriver] PLP materialize failed after {self.MAX_RETRIES + 1} attempts"
        )
        return False

    def _save_navigation_log(self, ctx: NavigationContext, outcome: NavigationOutcome):
        """ナビゲーション結果をログに保存"""
        try:
            ctx.run_context.save_json(
                "navigation_plp_initial.json",
                {
                    "entry_url": outcome.entry_url,
                    "plp_materialized": outcome.plp_materialized,
                    "trap_detected": outcome.trap_detected,
                    "trap_reason": outcome.trap_reason,
                    "retry_count": outcome.retry_count,
                    "elapsed_ms": outcome.elapsed_ms,
                    "error_message": outcome.error_message,
                },
            )
        except Exception as e:
            logger.debug(f"[NavigationDriver] Failed to save navigation log: {e}")
