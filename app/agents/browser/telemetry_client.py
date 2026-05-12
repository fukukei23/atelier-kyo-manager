"""
TelemetryClient — BrowserUseAgent / NavigationDriver から利用される Telemetry Facade

telemetry.py から分離。TelemetryService のラッパーとして機能し、よりシンプルなインターフェースを提供する。
データクラス（RunPhase, FailureContext, TelemetryContext）もここに配置。
"""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

    from app.core.run_context import RunContext


class RunPhase(str, Enum):
    """実行フェーズを表すEnum"""

    PLP_DISCOVERY = "plp_discovery"
    PDP_EXTRACT = "pdp_extract"
    RECOVERY = "recovery"
    LEARNING = "learning"
    MATERIALIZE = "materialize"


@dataclass
class FailureContext:
    """失敗時のコンテキスト情報"""

    site_code: str
    url: str
    phase: RunPhase
    exception: Exception | None = None
    retry_count: int = 0
    query: str | None = None
    site_config: dict[str, Any] | None = None
    intent_description: str | None = None


@dataclass
class TelemetryContext:
    """Telemetry 実行時のコンテキスト情報"""

    site: str
    query: str
    run_id: str | None = None
    stage: str | None = None


class TelemetryClient:
    """
    Telemetry Facade — TelemetryService のラッパー。

    BrowserUseAgent / NavigationDriver から利用されるシンプルなインターフェースを提供。
    """

    def __init__(self, run_context: Any, base_dir: str = "") -> None:
        self.run_context = run_context
        self.base_dir = base_dir
        # 遅延importで循環参照回避
        self._service = None
        self.logger = logging.getLogger(__name__)

    @property
    def service(self):
        if self._service is None:
            from app.agents.browser.telemetry import TelemetryService
            self._service = TelemetryService(run_context=self.run_context)
        return self._service

    async def save_dom(self, page: Any, name: str, tctx: TelemetryContext) -> None:
        await self.service.save_dom(page, name)

    async def save_json(self, name: str, payload: dict[str, Any], tctx: TelemetryContext) -> None:
        await self.service._save_json(payload, name)

    async def save_screenshot(self, page: Any, name: str, tctx: TelemetryContext) -> None:
        if hasattr(self.run_context, "take_screenshot"):
            try:
                await self.run_context.take_screenshot(page, name)
            except Exception as e:
                self.logger.warning(f"[TelemetryClient] Failed to save screenshot '{name}': {e}")
        else:
            self.logger.debug("[TelemetryClient] RunContext does not support take_screenshot")

    async def record_plp_state(
        self,
        page: Any,
        *,
        name: str = "plp_dom_initial_materialized",
        selectors: list[str] | None = None,
        site_config: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self.service.record_plp_state(
                page=page, name=name, selectors=selectors, site_config=site_config,
            )
        except Exception as e:
            self.logger.warning(f"[TelemetryClient] Failed to record PLP state '{name}': {e}", exc_info=True)

    async def write_fail_snapshot(
        self,
        page: Any,
        reason: str,
        tctx: TelemetryContext,
        extra: dict[str, Any] | None = None,
    ) -> None:
        site_config = extra.get("site_config", {}) if extra else {}

        class ErrorWrapper(Exception):
            def __init__(self, message: str):
                self.message = message
                super().__init__(message)

        error = ErrorWrapper(reason)
        final_url = None
        if page and not page.is_closed():
            with contextlib.suppress(Exception):
                final_url = page.url

        await self.service.write_fail_snapshot(
            page=page, final_url=final_url, error=error, site_config=site_config,
        )
