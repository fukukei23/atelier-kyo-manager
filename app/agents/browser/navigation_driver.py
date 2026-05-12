# ==============================================================================
# File: app/agents/browser/navigation_driver.py
# Version: 2.0.0 (Phase 3-1: Mixin分割完了)
# Purpose: NavigationDriver エントリポイント - PLP フロー制御
# ==============================================================================
"""
NavigationDriver のエントリポイント。

Mixin 構成:
- MonclerNavMixin: Moncler専用ナビゲーション
- LocaleMixin: ロケール管理
- FallbackMixin: フォールバックナビゲーション
- NavPdpCollectorMixin: PDP リンク収集・証跡保存
- NavPlpMaterializerMixin: PLP マテリアライズ・トラップ検出・リカバリ
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from playwright.async_api import Page

# データクラス・型を nav_types から re-export
from app.agents.browser.nav_types import (
    _LOCALE_SEG_RE,
    LinkCandidate,
    NavigationContext,
    NavigationOutcome,
    RejectReason,
    TrapCheckerFn,
    TrapPageDetected,
)

if TYPE_CHECKING:
    from app.agents.browser.telemetry import TelemetryClient

from app.agents.browser.extractor import (
    VISIBLE_PRICE_SELECTORS,
    extract_moncler_pdp_links,  # noqa: F401
    looks_like_product_url,
)

from app.agents.browser.navigation_helpers import (
    normalize_abs_url as nav_normalize_abs_url,
)
from app.agents.browser.navigation_helpers import (
    normalize_url as nav_normalize_url,
)

# URL検証 pure functions を url_rules から import
from app.agents.browser.url_rules import (
    check_origin_allowed,
    classify_candidate,
    extract_etld_plus_one,
    extract_origin,
    is_same_site,
    normalize_candidate_url,
    validate_candidate_url,
)

# Mixin imports
from app.agents.browser.moncler_nav import MonclerNavMixin
from app.agents.browser.locale_manager import LocaleMixin
from app.agents.browser.nav_fallbacks import FallbackMixin
from app.agents.browser.nav_pdp_collector import NavPdpCollectorMixin
from app.agents.browser.nav_plp_materializer import NavPlpMaterializerMixin

logger = logging.getLogger(__name__)


def is_same_origin(url: str, base: str) -> bool:
    """URL が同じオリジンかどうかを判定する"""
    try:
        u = urlparse(url)
        b = urlparse(base)
        return (u.scheme, u.netloc) == (b.scheme, b.netloc)
    except Exception:
        return False


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


class NavigationDriver(
    NavPdpCollectorMixin,
    NavPlpMaterializerMixin,
    MonclerNavMixin,
    LocaleMixin,
    FallbackMixin,
):
    """
    PLP ナビゲーションロジックを BrowserUseAgent から分離するためのクラス。

    MRO順序: NavPdpCollectorMixin → NavPlpMaterializerMixin → MonclerNavMixin → LocaleMixin → FallbackMixin
    NavPdpCollectorMixin は FallbackMixin._run_deep_extraction_phase2 を使用するため、
    FallbackMixin より先に解決される必要がある。
    """

    def __init__(
        self,
        page: Page,
        *,
        trap_checker: TrapCheckerFn | None = None,
        telemetry: TelemetryClient | None = None,
        strategy: Any = None,
    ) -> None:
        self.page = page
        self.trap_checker = trap_checker
        self.telemetry = telemetry
        self.strategy = strategy

    async def run_plp_flow(self, ctx: NavigationContext) -> NavigationOutcome:
        """
        PLP フローを実行する

        CR-ATELIER-002 Step 4: 実ブラウザ検証と最終修正

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            NavigationOutcome: ナビゲーション結果
        """
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)
        locale_correction_count = 0

        # trap 判定・復旧ロジック
        url = self.page.url or entry
        attempted_recover = False

        if self.trap_checker:
            trap_check_fn = self.trap_checker
        else:
            def trap_check_with_config(url: str) -> bool:
                return self._looks_like_trap_or_legal(url, ctx.site_config)

            trap_check_fn = trap_check_with_config

        if trap_check_fn and url:
            try:
                if trap_check_fn(url):
                    outcome.trap_detected = True
                    outcome.trap_reason = f"initial_url={url}"
                    logger.warning("[NavigationDriver] trap-like url detected: %s", url)

                    attempted_recover = True
                    recovered = await self.recover_plp(ctx)
                    outcome.recovered = recovered
                    if recovered:
                        logger.info("[NavigationDriver] recovered from trap-like url")
                    else:
                        logger.warning("[NavigationDriver] failed to recover from trap-like url")
            except Exception as e:
                logger.debug(f"[NavigationDriver] trap check failed: {e}")

        # ensure_plp_materialized
        if not outcome.trap_detected or outcome.recovered:
            try:
                materialized = await self.ensure_plp_materialized(ctx)
                outcome.materialized = materialized
                if not materialized:
                    logger.warning("[NavigationDriver] PLP materialization failed")
            except Exception as e:
                logger.error(f"[NavigationDriver] PLP materialization error: {e}")
                outcome.materialized = False

        # collect_pdp_links
        if outcome.materialized or (outcome.trap_detected and outcome.recovered):
            try:
                links = await self.collect_pdp_links(ctx)
                outcome.pdp_links = links
                outcome.links_count = len(links)
            except Exception as e:
                logger.error(f"[NavigationDriver] PDP link collection error: {e}")
                outcome.pdp_links = []
                outcome.links_count = 0

        # CR-ATELIER-002 Step 6: Locale補正の回数をログ出力
        if locale_correction_count > 0:
            logger.info(f"[NavigationDriver] Locale correction applied {locale_correction_count} times during PLP flow")

        # Stage 3A-3: 観測フック（挙動は変更しない）
        if hasattr(ctx, "on_navigation_complete") and callable(ctx.on_navigation_complete):
            try:
                await ctx.on_navigation_complete(outcome)
            except Exception as e:
                logger.debug(f"[NavigationDriver] on_navigation_complete hook failed: {e}")

        # Telemetry: ナビゲーション結果を保存
        if self.telemetry and ctx.run_context:
            try:
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(
                    site=ctx.site, query=ctx.query, run_id=getattr(ctx.run_context, "run_id", None), stage="plp_flow"
                )
                await self.telemetry.save_json(
                    "navigation_outcome",
                    {
                        "entry_url": entry,
                        "trap_detected": outcome.trap_detected,
                        "trap_reason": outcome.trap_reason,
                        "recovered": outcome.recovered,
                        "materialized": outcome.materialized,
                        "links_count": outcome.links_count,
                        "locale_correction_count": locale_correction_count,
                    },
                    tctx,
                )
            except Exception as e:
                logger.debug(f"[NavigationDriver] Telemetry save failed: {e}")

        return outcome


# 後方互換 re-export
__all__ = [
    "NavigationDriver",
    "NavigationContext",
    "NavigationOutcome",
    "LinkCandidate",
    "RejectReason",
    "TrapPageDetected",
    "TrapCheckerFn",
    "is_same_site",
    "extract_etld_plus_one",
    "extract_origin",
    "check_origin_allowed",
    "normalize_candidate_url",
    "validate_candidate_url",
]
