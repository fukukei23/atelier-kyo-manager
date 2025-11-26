# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Awaitable

from playwright.async_api import Page

# BrowserUseAgent 側の _ensure_plp_materialized を受け取るための型
EnsurePlpMaterializedFn = Callable[
    [Page, Dict[str, Any], Dict[str, Any], float, int],
    Awaitable[bool],  # ← bool を返す（_ensure_plp_materialized と合わせる）
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


class NavigationDriver:
    """
    Stage 3A-2:
      - PLP materialize を BrowserUseAgent から一部移管
      - trap 判定はまだ optional
      - PDPリンク収集までは触らない
    """

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
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)

        # --- trap チェック（まだあまり使わないが型だけ確保） ---
        if self.trap_checker:
            try:
                reason = await self.trap_checker(self.page, ctx.site_config, ctx.settings)
            except Exception:
                reason = None
            if reason:
                outcome.trap_detected = True
                outcome.trap_reason = reason
                return outcome

        # --- PLP materialize 本体 ---
        ok = await self.ensure_plp_materialized(
            self.page,
            ctx.site_config,
            ctx.settings,
            ctx.start_t,
            ctx.budget_ms,
        )
        outcome.plp_materialized = bool(ok)

        # --- 軽い観測ログ（失敗しても無視） ---
        try:
            ctx.run_context.save_json(
                "navigation_plp_initial.json",
                {
                    "entry_url": entry,
                    "plp_materialized": bool(ok),
                    "trap_detected": outcome.trap_detected,
                    "trap_reason": outcome.trap_reason,
                },
            )
        except Exception:
            pass

        return outcome
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from playwright.async_api import Page


@dataclass
class NavigationContext:
    site_code: str
    entry_url: str
    plp_url: Optional[str] = None
    run_id: Optional[str] = None
    retry_count: int = 0
    likely_plp: bool = True
    plugin_name: Optional[str] = None


@dataclass
class NavigationOutcome:
    ctx: NavigationContext
    pdp_urls: List[str] = field(default_factory=list)
    last_url: str = ""
    recovered: bool = False
    errors: List[str] = field(default_factory=list)


class NavigationDriver:
    """
    Stage 3A skeleton: 管理対象は「PLP→PDP ナビゲーション」だけ。
    まだ実ロジックは BrowserUseAgent 側に残し、このクラスは器として定義しておく。
    """

    def __init__(self, page: Page, telemetry: Optional[Any] = None, strategy: Optional[Any] = None) -> None:
        self.page = page
        self.telemetry = telemetry
        self.strategy = strategy

    async def run_plp_flow(self, ctx: NavigationContext) -> NavigationOutcome:
        """
        Stage 3A ではまだスタブ。今後 `_run_plp_flow` の実装を徐々に移していく。
        """
        last_url = self.page.url if self.page else ctx.entry_url
        return NavigationOutcome(ctx=ctx, last_url=last_url)

    async def collect_pdp_links(self, page: Page, *args, **kwargs) -> List[str]:
        """
        `_collect_pdp_links` を移設予定のスタブ。
        """
        return []

    async def run_deep_extraction(self, page: Page, *args, **kwargs) -> List[str]:
        """
        `_run_deep_extraction_phaseX` を集約するためのスタブ。
        """
        return []

    async def recover_plp(self, page: Page, ctx: NavigationContext) -> bool:
        """
        `_force_plp_recover` や専用復旧ロジックをまとめる予定。
        """
        return False

    async def _looks_like_trap_or_legal(self, url: str) -> bool:
        """
        trap検知のロジックを移すための仮メソッド。
        """
        return False
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Awaitable

from playwright.async_api import Page

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


class NavigationDriver:
    """
    Stage 3A-2:
      - PLP materialize を BrowserUseAgent から一部移管
      - trap 判定はまだ optional
      - PDPリンク収集までは触らない
    """

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
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)

        # --- trap チェック（まだあまり使わないが型だけ確保） ---
        if self.trap_checker:
            try:
                reason = await self.trap_checker(self.page, ctx.site_config, ctx.settings)
            except Exception:
                reason = None
            if reason:
                outcome.trap_detected = True
                outcome.trap_reason = reason
                return outcome

        # --- PLP materialize 本体 ---
        ok = await self.ensure_plp_materialized(
            self.page,
            ctx.site_config,
            ctx.settings,
            ctx.start_t,
            ctx.budget_ms,
        )
        outcome.plp_materialized = bool(ok)

        # --- 軽い観測ログ（失敗しても無視） ---
        try:
            ctx.run_context.save_json(
                "navigation_plp_initial.json",
                {
                    "entry_url": entry,
                    "plp_materialized": bool(ok),
                    "trap_detected": outcome.trap_detected,
                    "trap_reason": outcome.trap_reason,
                },
            )
        except Exception:
            pass

        return outcome
