# ==============================================================================
# File: app/agents/browser/navigation_driver.py
# Version: 1.0.4 (Stage 3A-2-1: _collect_pdp_links の移行)
# Purpose: NavigationDriver の骨組み - メソッドシグネチャのみ実装
# ==============================================================================
"""
Stage 3A-1: NavigationDriver 骨組み
Stage 3A-3: trap 判定の観測フック追加
Stage 3A-2-1: _collect_pdp_links の移行

このステップでは、クラス定義とメソッドシグネチャのみを作成し、
実際のロジック移動は Stage 3A-2 で行います。
Stage 3A-3 では、trap 判定の観測フックを追加します（挙動は変更しない）。
Stage 3A-2-1 では、BrowserUseAgent._collect_pdp_links のロジックをここに移行します。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, quote_plus, urljoin, urlparse, urlsplit, urlunsplit

from playwright.async_api import BrowserContext, ElementHandle, Locator, Page

# P1-1 Phase 1: データクラス・型を nav_types から re-export
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

# Stage 3A-2-1: extractor モジュールから looks_like_product_url を import
try:
    from app.agents.browser.extractor import (
        VISIBLE_PRICE_SELECTORS,
        extract_moncler_pdp_links,  # noqa: F401
        looks_like_product_url,
    )
except ImportError:

    def looks_like_product_url(url: str) -> bool:
        return True

    VISIBLE_PRICE_SELECTORS = ["[itemprop=price]", "[class*=price]", "[data-testid*=price]"]

# 責務分割: UI 操作・URL 正規化は ui_helpers / navigation_helpers に委譲
try:
    from app.agents.browser.navigation_helpers import (
        normalize_abs_url as nav_normalize_abs_url,
    )
    from app.agents.browser.navigation_helpers import (
        normalize_url as nav_normalize_url,
    )
    from app.agents.browser.ui_helpers import (
        accept_cookies_if_present as ui_accept_cookies_if_present,
    )
    from app.agents.browser.ui_helpers import (
        click_continue_shopping_if_present as ui_click_continue_shopping_if_present,
    )
    from app.agents.browser.ui_helpers import (
        kill_overlays as ui_kill_overlays,
    )
    from app.agents.browser.ui_helpers import (
        safe_wait_selector as ui_safe_wait_selector,
    )
except ImportError:
    ui_safe_wait_selector = None
    ui_accept_cookies_if_present = None
    ui_click_continue_shopping_if_present = None
    ui_kill_overlays = None
    nav_normalize_url = None
    nav_normalize_abs_url = None

# P1-1 Phase 2: URL検証 pure functions を url_rules から import
from app.agents.browser.url_rules import (
    check_origin_allowed,
    classify_candidate,
    extract_etld_plus_one,
    extract_origin,
    is_same_site,
    normalize_candidate_url,
    validate_candidate_url,
)

# P1-1 Phase 3: Moncler Mixin
from app.agents.browser.moncler_nav import MonclerNavMixin

# P1-1 Phase 4: Locale Mixin
from app.agents.browser.locale_manager import LocaleMixin

# P1-1 Phase 5: Fallback Mixin
from app.agents.browser.nav_fallbacks import FallbackMixin

logger = logging.getLogger(__name__)


# ヘルパー関数（モジュールレベル）
def is_same_origin(url: str, base: str) -> bool:
    """URL が同じオリジンかどうかを判定する"""
    try:
        from urllib.parse import urlparse

        u = urlparse(url)
        b = urlparse(base)
        return (u.scheme, u.netloc) == (b.scheme, b.netloc)
    except Exception:
        return False


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


class NavigationDriver(MonclerNavMixin, LocaleMixin, FallbackMixin):
    """
    Stage 3A-1: 骨組みのみ
    Stage 3A-3: trap 判定の観測フック追加

    PLP ナビゲーションロジックを BrowserUseAgent から分離するためのクラス。
    このステップでは、メソッドシグネチャのみを定義し、実際のロジックは Stage 3A-2 で移行します。
    Stage 3A-3 では、trap 判定の観測フックを追加します（挙動は変更しない）。
    """

    def __init__(
        self,
        page: Page,
        *,
        trap_checker: TrapCheckerFn | None = None,
        telemetry: TelemetryClient | None = None,
        strategy: Any = None,
    ) -> None:
        """
        NavigationDriver を初期化

        Args:
            page: SessionManager から渡される Page オブジェクト
            trap_checker: trap 判定関数（オプション、Stage 3A-3: 観測用）
            telemetry: TelemetryClient（オプション、Stage 3B で使用）
            strategy: StrategyPlugin（オプション、このステージでは使わなくてもよい）
        """
        self.page = page
        self.trap_checker = trap_checker
        self.telemetry = telemetry
        self.strategy = strategy

    async def run_plp_flow(self, ctx: NavigationContext) -> NavigationOutcome:
        """
        PLP フローを実行する

        Stage 3A-1: このステップでは、ctx をそのまま返すだけのスタブ実装。
        Stage 3A-2-2: ensure_plp_materialized を呼び出して materialize を実行。
        Stage 3A-2-3: trap 判定・復旧ロジックを実行。
        Stage 3A-3: trap 判定の観測フックを追加（挙動は変更しない）。

        CR-ATELIER-002 Step 4: 実ブラウザ検証と最終修正
        - URLバリデーションとロケール制御の実DOMベース調整
        - Telemetry/ログの実データに合わせた具体化
        - 成功基準（Acceptance Criteria）の充足確認ロジック

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            NavigationOutcome: ナビゲーション結果
        """
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)
        locale_correction_count = 0  # CR-ATELIER-002 Step 6: Locale補正の回数をカウント（初期化）

        # --- Stage 3A-2-3: trap 判定・復旧ロジック ---
        url = self.page.url or entry
        attempted_recover = False

        # trap_checker が提供されている場合はそれを使用、なければ内部メソッドを使用
        # Stage 3A-2-5: _looks_like_trap_or_legal の場合は site_config を渡す
        if self.trap_checker:
            trap_check_fn = self.trap_checker
        else:
            # 内部メソッドを使用する場合、site_config を渡すラッパーを作成
            def trap_check_with_config(url: str) -> bool:
                return self._looks_like_trap_or_legal(url, ctx.site_config)

            trap_check_fn = trap_check_with_config

        if trap_check_fn and url:
            try:
                if trap_check_fn(url):
                    outcome.trap_detected = True
                    outcome.trap_reason = f"initial_url={url}"
                    logger.warning("[NavigationDriver] trap-like url detected: %s", url)

                    # 復旧を試みる
                    attempted_recover = True
                    recovered = await self.recover_plp(ctx)
                    outcome.recovered = recovered

                    # CR-ATELIER-002 Step 2: Recovery 後のロケールチェック
                    # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
                    if recovered:
                        try:
                            await self._ensure_expected_locale(ctx)
                        except Exception as e:
                            logger.warning(f"[NavigationDriver] Locale Guard after recovery failed: {e}", exc_info=True)

                    if recovered:
                        # 回復後に再度 trap 判定
                        # Stage 3A-2-5: trap_check_fn が内部メソッドの場合は site_config を渡す
                        if self.trap_checker:
                            check_result = trap_check_fn(self.page.url)
                        else:
                            check_result = self._looks_like_trap_or_legal(self.page.url, ctx.site_config)
                        if check_result:
                            # 回復後も trap の場合は例外を投げる
                            raise ValueError(
                                f"Landing page looks like legal/trap (even after recovery attempt): {self.page.url}"
                            )
                        logger.info("[NavigationDriver] Recovery navigation seems successful.")
                    else:
                        # 回復に失敗した場合
                        raise ValueError(f"Landing page looks like legal/trap (recovery failed): {url}")
            except ValueError:
                # ValueError はそのまま再スロー
                raise
            except Exception as e:
                # その他のエラーはログに記録して続行
                logger.debug("[NavigationDriver] trap_checker/recover failed: %s", e)

        # --- CR-ATELIER-002 Step 2: Locale Guard - ロケール一貫性チェックと自動修正 ---
        # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
        # 初回PLPナビゲーション直後（home_urlからPLPへ飛んだ直後）にロケールチェック
        # CR-ATELIER-002 Step 6: Locale補正の回数をカウント（修正が行われた場合のみカウント）
        locale_correction_before = self.page.url or ""
        try:
            await self._ensure_expected_locale(ctx)
            locale_correction_after = self.page.url or ""
            # URL が変更された場合、locale_correction_count をインクリメント
            if locale_correction_before != locale_correction_after:
                locale_correction_count += 1
        except Exception as e:
            logger.warning(f"[NavigationDriver] Locale Guard failed: {e}", exc_info=True)
            # Locale Guard は失敗しても続行（Guard なので壊さない）

        # --- Stage 3A-2-2: ensure_plp_materialized を呼び出し ---
        materialized = False
        tiles_detected = False
        try:
            materialized = await self.ensure_plp_materialized(ctx)
            outcome.plp_materialized = materialized
            if not materialized:
                logger.warning("[NavigationDriver] PLP materialization failed")
        except Exception as e:
            logger.error(f"[NavigationDriver] ensure_plp_materialized failed: {e}")
            outcome.plp_materialized = False

        # materialized == False の場合（例外または False 返却）でも、タイルが検出されているか確認
        if not materialized:
            try:
                # ensure_plp_materialized 内で使用されているのと同じロジックで tile_selector_str を構築
                plp_cfg = (ctx.site_config.get("selectors", {}) or {}).get("plp", {}) or {}
                pdp_cfg = (ctx.site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
                tile_selectors = _dedupe_keep_order(
                    (plp_cfg.get("tile_selectors", []) or [])
                    + (plp_cfg.get("pdp_link_selectors", []) or [])
                    + (pdp_cfg.get("pdp_link_selectors", []) or [])
                    + [
                        # Moncler 用フォールバック（.html で終わるリンク）
                        "a[href$='.html']",
                        "[data-testid='product-card']",
                        "[data-test='product-card']",
                        "div[class*='product-card' i]",
                        "div[class*='product-tile' i]",
                    ]
                )
                if tile_selectors:
                    tile_selector_str = ", ".join(tile_selectors)
                    tile_count = await self.page.locator(tile_selector_str).count()
                    if tile_count > 0:
                        tiles_detected = True
                        logger.info(f"[NavigationDriver] Tiles detected ({tile_count}) despite materialization failure")
                        logger.info("[NavigationDriver] tiles_detected set to True, will attempt to record PLP state")
            except Exception as check_e:
                logger.warning(f"[NavigationDriver] Failed to check tile count: {check_e}")

        # PLP materialization が成功した場合、または例外で失敗したがタイルが検出された場合に DOM snapshot を保存
        condition_result = materialized or tiles_detected
        if condition_result and self.telemetry:
            try:
                # site_config からセレクタを取得
                pdp_cfg = (ctx.site_config.get("selectors") or {}).get("pdp", {}) or {}
                selectors = (pdp_cfg.get("pdp_link_selectors") or []) + (pdp_cfg.get("plp_container_selectors") or [])
                logger.info(
                    f"[NavigationDriver] Recording PLP state: materialized={materialized}, tiles_detected={tiles_detected}"
                )
                await self.telemetry.record_plp_state(
                    self.page,
                    name="plp_dom_initial_materialized",
                    selectors=selectors if selectors else None,
                    site_config=ctx.site_config,
                )
                logger.info("[NavigationDriver] Saved PLP DOM snapshot and selector counts")
            except Exception as e:
                logger.warning(f"[NavigationDriver] Failed to record PLP state: {e}", exc_info=True)
        elif not condition_result:
            logger.debug(
                f"[NavigationDriver] Skipping PLP state recording: materialized={materialized}, tiles_detected={tiles_detected}"
            )
        elif not self.telemetry:
            logger.warning("[NavigationDriver] Telemetry not available, skipping PLP state recording")

        # --- Stage 3A-2-3: materialize 後の trap 再チェック ---
        if outcome.plp_materialized and trap_check_fn:
            try:
                if trap_check_fn(self.page.url):
                    if not attempted_recover:
                        # まだ回復試行していない場合は試す
                        logger.warning("[NavigationDriver] trap-like url after materialize: %s", self.page.url)
                        recovered = await self.recover_plp(ctx)
                        outcome.recovered = recovered
                        # CR-ATELIER-002 Step 2: Recovery 後のロケールチェック
                        # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
                        # CR-ATELIER-002 Step 6: Locale補正の回数をカウント
                        if recovered:
                            locale_correction_before_recovery_post = self.page.url or ""
                            try:
                                await self._ensure_expected_locale(ctx)
                                locale_correction_after_recovery_post = self.page.url or ""
                                if locale_correction_before_recovery_post != locale_correction_after_recovery_post:
                                    locale_correction_count += 1
                            except Exception as e:
                                logger.warning(
                                    f"[NavigationDriver] Locale Guard after recovery (post-materialize) failed: {e}",
                                    exc_info=True,
                                )
                        if recovered:
                            if trap_check_fn(self.page.url):
                                raise ValueError(
                                    f"After materialize still on legal/trap page (even after recovery attempt): {self.page.url}"
                                )
                            logger.info("[NavigationDriver] Recovery navigation (post-materialize) seems successful.")
                        else:
                            raise ValueError(
                                f"After materialize still on legal/trap page (recovery failed): {self.page.url}"
                            )
                    else:
                        # 既に回復試行済みで、materialize したらまた trap に戻った場合
                        raise ValueError(f"After materialize, bounced back to legal/trap page: {self.page.url}")
            except ValueError:
                # ValueError はそのまま再スロー
                raise
            except Exception as e:
                logger.debug("[NavigationDriver] post-materialize trap check failed: %s", e)

        # --- CR-ATELIER-002 Step 1: Trap ページ検出（DOM ベース） ---
        trap_info = await self._detect_trap_page(ctx)
        if trap_info is not None:
            # Telemetry に現在の PLP 状態を記録
            if self.telemetry:
                try:
                    logger.warning(
                        f"[TrapDetector] Trap page detected: type={trap_info['type']}, "
                        f"reason={trap_info['reason']}, URL={self.page.url}"
                    )
                    await self.telemetry.record_plp_state(
                        self.page,
                        name="plp_trap_page",
                        selectors=None,
                        site_config=ctx.site_config,
                    )
                    logger.info("[TrapDetector] Saved trap page DOM snapshot to Telemetry")
                except Exception as e:
                    logger.warning(f"[TrapDetector] Failed to record trap-page PLP state: {e}", exc_info=True)
            else:
                logger.warning("[TrapDetector] Telemetry not available, skipping trap page recording")

            # 専用例外を投げる（上位の Self-Healing ロジックで扱えるようにする）
            raise TrapPageDetected(
                trap_type=trap_info["type"],
                reason=trap_info["reason"],
                url=self.page.url,
            )

        # --- Stage 3A-2-4: PDP リンク収集と fallback ロジック ---
        pdp_links: list[str] = []

        try:
            pdp_links = await self.collect_pdp_links(ctx)
            outcome.pdp_links = pdp_links

            # CR-ATELIER-002 Step 6: Moncler専用のoutcome情報を取得
            moncler_outcome_info = None
            if hasattr(ctx, "moncler_outcome"):
                moncler_outcome_info = ctx.moncler_outcome
            elif isinstance(ctx, dict) and "moncler_outcome" in ctx:
                moncler_outcome_info = ctx["moncler_outcome"]

            outcome.moncler_outcome = moncler_outcome_info

            logger.info(f"[NavigationDriver] Collected {len(pdp_links)} PDP links")
        except Exception as e:
            logger.error(f"[NavigationDriver] collect_pdp_links failed: {e}")
            outcome.pdp_links = []

        # PDP リンクが不足している場合の fallback
        if not pdp_links:
            log_file_path = None
            if ctx.run_context:
                log_file_path = ctx.run_context.get_path("system.log")
            logger.error(
                f"[NavigationDriver] No PDP links found (collected {len(pdp_links)} links), "
                "trying fallback strategies..."
            )
            if log_file_path and log_file_path.exists():
                logger.error(f"[NavigationDriver] Full error log available at: {log_file_path}")
            elif ctx.run_context:
                logger.error(f"[NavigationDriver] Error log will be saved to: {ctx.run_context.get_path('system.log')}")

            # Fallback 1: ヘッダ検索
            try:
                did_search = await self.header_search_fallback(ctx)
                if did_search:
                    outcome.fallback_used = "header_search"
                    logger.info("[NavigationDriver] Header search fallback succeeded, collecting PDP links again...")
                    # CR-ATELIER-002 Step 2: Header search fallback 後のロケールチェック
                    # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
                    # CR-ATELIER-002 Step 6: Locale補正の回数をカウント
                    locale_correction_before_header = self.page.url or ""
                    try:
                        await self._ensure_expected_locale(ctx)
                        locale_correction_after_header = self.page.url or ""
                        if locale_correction_before_header != locale_correction_after_header:
                            locale_correction_count += 1
                    except Exception as e:
                        logger.warning(
                            f"[NavigationDriver] Locale Guard after header search failed: {e}", exc_info=True
                        )
                    try:
                        pdp_links = await self.collect_pdp_links(ctx)
                        outcome.pdp_links = pdp_links
                        logger.info(f"[NavigationDriver] After header search, collected {len(pdp_links)} PDP links")
                    except Exception as e:
                        logger.warning(f"[NavigationDriver] collect_pdp_links after header search failed: {e}")
            except Exception as e:
                logger.warning(f"[NavigationDriver] header_search_fallback failed: {e}")

            # Fallback 2: カードクリック（PDP リンクがまだ不足している場合）
            if not pdp_links:
                try:
                    clicked_url = await self.click_first_card_or_link(ctx)
                    if clicked_url:
                        outcome.fallback_used = "click_first_card"
                        outcome.pdp_links = [clicked_url]
                        logger.info(f"[NavigationDriver] Click first card fallback succeeded, got URL: {clicked_url}")
                    else:
                        logger.warning("[NavigationDriver] Click first card fallback failed")
                except Exception as e:
                    logger.warning(f"[NavigationDriver] click_first_card_or_link failed: {e}")

        # CR-ATELIER-002 Step 4-4: 成功基準（Acceptance Criteria）の充足確認
        # 「down jacket」クエリで1 run実行したとき:
        # - nav_outcome.collected_pdp_links >= 1
        # - run.ok == True（TrapPageDetectedではなく正常終了）
        # - PLPのURLは /en-int/women/outerwear/all-down-jackets/ + shipToCountry=GB
        # - 抽出されたPDP URLは、すべて /en-int/.../products/... を指し、404/検索/ロケールゲートではない

        # 最終的に PDP リンクが 0 件の場合、例外を投げる（旧 _run_plp_flow と同じ条件）
        if not outcome.pdp_links:
            raise ValueError(
                f"No PDP links found after all phases (materialize, collect, fallbacks). URL={self.page.url}"
            )

        # CR-ATELIER-002 Step 4-4: 抽出されたPDP URLの検証
        # すべてのPDP URLが /en-int/.../products/... を指し、404/検索/ロケールゲートではないことを確認
        valid_pdp_count = 0
        invalid_pdp_reasons = []
        for pdp_url in outcome.pdp_links:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(pdp_url)
                path = parsed.path or ""

                # /en-int/ で始まるか
                if not path.startswith("/en-int/"):
                    invalid_pdp_reasons.append(f"{pdp_url}: no /en-int/ path")
                    continue

                # /products/ を含むか
                if "/products/" not in path and "/product/" not in path:
                    invalid_pdp_reasons.append(f"{pdp_url}: no /products/ path")
                    continue

                # trapページパターンを含まないか
                trap_patterns = ["/404", "/not-found", "/search", "/client-service"]
                if any(trap in path.lower() for trap in trap_patterns):
                    invalid_pdp_reasons.append(f"{pdp_url}: trap pattern detected")
                    continue

                # 二重ロケールパターンを含まないか
                double_locale_pattern = re.compile(r"/en-[a-z]{2}/en-int/", re.I)
                if double_locale_pattern.search(path):
                    invalid_pdp_reasons.append(f"{pdp_url}: double locale pattern")
                    continue

                valid_pdp_count += 1
            except Exception as e:
                invalid_pdp_reasons.append(f"{pdp_url}: validation error: {e}")

        if invalid_pdp_reasons:
            logger.warning(
                f"[PLP→PDP][Moncler] Found {len(invalid_pdp_reasons)} invalid PDP URLs: "
                f"{invalid_pdp_reasons[:5]}"  # 最初の5件のみ表示
            )

        # 成功基準のログ出力
        logger.info(
            f"[PLP→PDP][Moncler] Acceptance Criteria check: "
            f"collected_pdp_links={len(outcome.pdp_links)}, "
            f"valid_pdp_count={valid_pdp_count}, "
            f"trap_detected={outcome.trap_detected}, "
            f"plp_materialized={outcome.plp_materialized}"
        )

        # CR-ATELIER-002 Step 6-3: Telemetry に moncler_plp_pdp_outcome を保存
        site_code = ctx.site_config.get("site_code") or ctx.site_config.get("site") or ctx.site or ""
        if site_code == "MONCLER_OFFICIAL" and self.telemetry and outcome.moncler_outcome:
            try:
                # tiles_detected を取得（materialized または tiles_detected が True の場合）
                tiles_detected_count = 0
                if materialized or tiles_detected:
                    try:
                        plp_cfg = (ctx.site_config.get("selectors", {}) or {}).get("plp", {}) or {}
                        tile_selectors = plp_cfg.get("tile_selectors", []) or []
                        if tile_selectors:
                            tile_selector_str = ", ".join(tile_selectors)
                            tiles_detected_count = await self.page.locator(tile_selector_str).count()
                    except Exception:
                        pass

                # outcome dict を構築
                outcome_dict = {
                    "plp_materialized": outcome.plp_materialized or tiles_detected,
                    "tiles_detected": tiles_detected_count,
                    "pdp_links_raw": outcome.moncler_outcome.get("raw_count", 0),
                    "pdp_links_accepted": outcome.moncler_outcome.get("accepted_count", len(outcome.pdp_links)),
                    "selector_layers_used": outcome.moncler_outcome.get("layers_used", []),
                    "layer_stats": outcome.moncler_outcome.get("layer_stats", {}),
                    "locale_corrections": locale_correction_count,
                    "trap_detected": outcome.trap_detected,
                    "current_url": self.page.url or entry,
                    "run_id": getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                }

                # TelemetryClient に保存
                if hasattr(self.telemetry, "_service"):
                    await self.telemetry._service.record_moncler_plp_pdp_outcome(outcome_dict)
                else:
                    # TelemetryClient の場合、直接 save_json を使用
                    from app.agents.browser.telemetry import TelemetryContext

                    tctx = TelemetryContext(
                        site=site_code,
                        query=ctx.query,
                        run_id=getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                        stage="plp",
                    )
                    await self.telemetry.save_json("moncler_plp_pdp_outcome", outcome_dict, tctx)

                logger.info(
                    f"[Telemetry][Moncler] Saved PLP→PDP outcome: "
                    f"raw={outcome_dict['pdp_links_raw']}, "
                    f"accepted={outcome_dict['pdp_links_accepted']}, "
                    f"layers={outcome_dict['selector_layers_used']}"
                )

                # CR-ATELIER-002 Step 6-3: Self-Healing / Selector Discovery のトリガー判定
                should_trigger_self_healing = False
                failure_reason = None

                # トリガ条件をチェック
                if outcome_dict["pdp_links_raw"] == 0:
                    should_trigger_self_healing = True
                    failure_reason = "raw_zero"
                elif outcome_dict["pdp_links_accepted"] == 0:
                    should_trigger_self_healing = True
                    failure_reason = "rejected_all"
                elif (
                    "secondary" in outcome_dict["selector_layers_used"]
                    or "tertiary" in outcome_dict["selector_layers_used"]
                ):
                    should_trigger_self_healing = True
                    failure_reason = "secondary_or_tertiary_used"
                elif outcome_dict["trap_detected"]:
                    should_trigger_self_healing = True
                    failure_reason = "trap_detected"
                elif outcome_dict["locale_corrections"] >= 3:  # 閾値は設定可能にする
                    should_trigger_self_healing = True
                    failure_reason = "locale_corrections_exceeded"

                if should_trigger_self_healing:
                    logger.warning(
                        f"[SelfHealing][Moncler] Triggered because reason={failure_reason}, "
                        f"raw={outcome_dict['pdp_links_raw']}, "
                        f"accepted={outcome_dict['pdp_links_accepted']}, "
                        f"layers={outcome_dict['selector_layers_used']}"
                    )

                    # Self-Healing Agent と Selector Discovery Agent を呼び出す
                    try:
                        await self._trigger_moncler_self_healing(
                            ctx=ctx,
                            failure_reason=failure_reason,
                            outcome_dict=outcome_dict,
                        )
                    except Exception as e:
                        logger.warning(f"[SelfHealing][Moncler] Failed to trigger self-healing: {e}", exc_info=True)
                else:
                    logger.debug(
                        f"[SelfHealing][Moncler] No trigger conditions met: "
                        f"raw={outcome_dict['pdp_links_raw']}, "
                        f"accepted={outcome_dict['pdp_links_accepted']}, "
                        f"layers={outcome_dict['selector_layers_used']}"
                    )
            except Exception as e:
                logger.warning(
                    f"[Telemetry][Moncler] Failed to save PLP→PDP outcome or trigger self-healing: {e}", exc_info=True
                )

        outcome.locale_corrections = locale_correction_count

        logger.debug(
            f"[NavigationDriver] run_plp_flow: entry_url={entry}, trap_detected={outcome.trap_detected}, "
            f"plp_materialized={outcome.plp_materialized}, recovered={outcome.recovered}, "
            f"pdp_links={len(outcome.pdp_links)}, fallback_used={outcome.fallback_used}"
        )

        # CR-ATELIER-002 Step 4-5: テストと検証手順（人間が実行）
        #
        # pytest 実行:
        #   python -m pytest tests/test_moncler_pdp_url.py -q -v
        #
        # 実 run 検証:
        #   python -m app.scripts.run_site moncler --query "down jacket" --headful
        #
        # LATEST run を確認し、以下をチェック:
        #   - result.json 内の ok == true
        #   - nav_outcome.collected_pdp_links >= 1
        #   - 抽出された PDP URL が想定のパターンに一致している
        #     （すべて /en-int/.../products/... を指し、404/検索/ロケールゲートではない）

        return outcome

    async def collect_pdp_links(
        self,
        ctx: NavigationContext,
    ) -> list[str]:
        """
        Stage 3A-2-1:
        旧 BrowserUseAgent._collect_pdp_links のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持されている。

        Phase 1a: Global <a href> sweep + Regex Filter
        Phase 1b: Selector-based補完
        Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        Phase 3: Noise Filtering & Saving

        CR-ATELIER-002 Step 3: Moncler専用のPDP抽出ロジックを追加
        CR-E2E-003: 候補収集とreject理由の記録

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            List[str]: PDP リンクのリスト
        """
        page = self.page
        site_config = ctx.site_config
        run_context = ctx.run_context
        target_url = page.url
        found_links: set[str] = set()

        # CR-E2E-003: 候補を段階的に収集
        all_candidates: list[LinkCandidate] = []

        # CR-ATELIER-002 Step 4: Moncler専用のPDP抽出ロジック（実ブラウザ検証版）
        # site_config のキーまたは ctx.site から site_code を取得
        (site_config.get("site_code") or site_config.get("site") or ctx.site or "")
        # CR-ATELIER-003 Phase B: Moncler 専用処理は MonclerPdpHandler に移行
        # NavigationDriver はブランド非依存の低レイヤとして維持
        # Moncler の処理は MonclerPlpHandler 経由で MonclerPdpHandler に委譲される

        # Phase 1a: Global <a href> sweep + Regex Filter
        # CR-E2E-003: 候補を収集（origin判定は後で分類）
        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
        except Exception as e:
            logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}")
            raw_hrefs = []
        # CR-E2E-003B拡張: Moncler向けに.html形式のPDPも拾う
        pdp_rx = re.compile(r"/(products?|p)/", re.I)
        # .html形式のPDPパターン（/en-int/.../...html または /en-jp/en-int/.../...html）
        html_pdp_rx = re.compile(r"/(?:en-int|en-jp/en-int)/[^/]+/[^/]+/[^/]+\.html", re.I)
        phase1a_candidates: list[LinkCandidate] = []
        for href in raw_hrefs:
            # OneTrustドメインのhrefを除外（ノイズ除去）
            if "onetrust.com" in href.lower():
                continue
            # /products/ または .html形式のPDPパターンにマッチする場合
            if pdp_rx.search(href) or html_pdp_rx.search(href):
                # CR-E2E-003B: URL正規化を先に実行
                norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                candidate = LinkCandidate(
                    url=href,
                    phase="1a",
                    normalized_url=norm_url,
                    source_selector="global_sweep",  # CR-E2E-003A
                )
                candidate = classify_candidate(candidate, target_url, site_config)
                # CR-E2E-003B: 正規化情報をproduct_url_rulesに追加
                if candidate.product_url_rules:
                    candidate.product_url_rules["normalization_info"] = norm_info
                phase1a_candidates.append(candidate)
                all_candidates.append(candidate)
                if candidate.accepted:
                    found_links.add(norm_url)
        if found_links:
            logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")

        # Phase 1b: Selector-based補完
        # Stage 3A-2-5: site_config["selectors"]["plp"] から取得（pdp ではなく plp を使用）
        plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp_selectors = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

        # plp.pdp_link_selectors を優先、なければ pdp.pdp_link_selectors を使用
        pdp_link_selectors = _dedupe_keep_order(
            (plp_selectors.get("pdp_link_selectors", []) or []) + (pdp_selectors.get("pdp_link_selectors", []) or [])
        )

        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not pdp_link_selectors:
            pdp_link_selectors = [
                "a[href*='/products/']",
                "a[href*='/product/']",
                "a[href*='/p/']",
                "[data-component*='ProductCard'] a[href]",
                "[class*='product-card'] a[href]",
                "article [data-testid*='product']:is(a, * a)",
                "[data-testid*='card'] a[href]",
                "[data-testid*='product-card'] a[href]",
                "a[data-product-url]",
                "[data-qa='product-tile'] a[href]",
            ]

        PLP_PDP_LINK_SELECTORS = pdp_link_selectors
        phase1b_candidates: list[LinkCandidate] = []
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                rejected_count = 0
                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
                    if not href:
                        # CR-E2E-003A: 候補として記録（reject理由付き）
                        candidate = LinkCandidate(
                            url="",
                            phase="1b",
                            normalized_url="",
                            reject_reasons=[RejectReason.NO_HREF.value],
                            source_selector=sel,  # CR-E2E-003A
                        )
                        phase1b_candidates.append(candidate)
                        all_candidates.append(candidate)
                        rejected_count += 1
                        continue
                    # OneTrustドメインのhrefを除外（ノイズ除去）
                    if "onetrust.com" in href.lower():
                        continue
                    # CR-E2E-003B: URL正規化を先に実行
                    norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                    # CR-E2E-003A: 候補として収集（origin判定は後で分類）
                    candidate = LinkCandidate(
                        url=href,
                        phase="1b",
                        normalized_url=norm_url,
                        source_selector=sel,  # CR-E2E-003A
                    )
                    candidate = classify_candidate(candidate, target_url, site_config)
                    # CR-E2E-003B: 正規化情報をproduct_url_rulesに追加
                    if candidate.product_url_rules:
                        candidate.product_url_rules["normalization_info"] = norm_info
                    phase1b_candidates.append(candidate)
                    all_candidates.append(candidate)
                    if candidate.accepted:
                        found_links.add(norm_url)
                        matched_count += 1
                    else:
                        rejected_count += 1
                if matched_count > 0:
                    logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
                elif nodes and rejected_count > 0:
                    # 要素は見つかったが、リンク抽出に失敗した場合
                    logger.warning(
                        f"[PLP→PDP][1b] selector='{sel}' found {len(nodes)} elements, "
                        f"but {rejected_count} were rejected."
                    )
            except Exception as e:
                logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")

        # Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        phase2_candidates: list[LinkCandidate] = []
        if not found_links:
            logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
            try:
                deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)
                for href in deep_hrefs:
                    # CR-E2E-003B: URL正規化を先に実行
                    norm_url, norm_info = normalize_candidate_url(href, target_url, site_config)
                    # CR-E2E-003A: 候補として収集（origin判定は後で分類）
                    candidate = LinkCandidate(
                        url=href,
                        phase="2",
                        normalized_url=norm_url,
                        source_selector="deep_extraction",  # CR-E2E-003A
                    )
                    candidate = classify_candidate(candidate, target_url, site_config)
                    # CR-E2E-003B: 正規化情報をproduct_url_rulesに追加
                    if candidate.product_url_rules:
                        candidate.product_url_rules["normalization_info"] = norm_info
                    phase2_candidates.append(candidate)
                    all_candidates.append(candidate)
                    if candidate.accepted:
                        found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
            except Exception as e:
                logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")

        # CR-E2E-003B拡張: 「候補あり・全reject」の場合の再フィルタリング（根拠のある緩和）
        if not found_links and all_candidates:
            # site_configから再フィルタリング設定を取得
            refilter_config = (site_config or {}).get("pdp_link_refilter", {})
            if refilter_config.get("enabled", False):
                logger.info(
                    f"[PLP→PDP][Refilter] {len(all_candidates)} candidates found but all rejected. "
                    "Attempting evidence-based relaxed filtering..."
                )

                # CR-E2E-003B: 根拠のある緩和の順序
                # 1. forbidden_pathに当たってない候補を残す
                for candidate in all_candidates:
                    if not candidate.normalized_url:
                        continue
                    # forbidden_pathに当たっていない & domain_allowed & 商品カード由来など強い根拠があるものだけ試す
                    if (
                        candidate.product_url_rules
                        and not candidate.product_url_rules.get("forbidden_path_matched", True)
                        and candidate.product_url_rules.get("domain_allowed", False)
                        and candidate.source_selector
                        and any(
                            keyword in candidate.source_selector.lower()
                            for keyword in ["product", "card", "tile", "item"]
                        )
                    ):
                        found_links.add(candidate.normalized_url)
                        candidate.accepted = True
                        candidate.reject_reasons = [
                            r for r in candidate.reject_reasons if r not in (RejectReason.NO_PRODUCTS_PATH.value,)
                        ]
                        candidate.notes = (candidate.notes or "") + " [Refilter: product card source]"
                        logger.debug(
                            f"[PLP→PDP][Refilter] Accepted candidate (product card source): {candidate.normalized_url}"
                        )

                # 2. same-siteのみ許可（forbidden_pathに当たってない場合）
                if not found_links and refilter_config.get("allow_same_site_only", False):
                    for candidate in all_candidates:
                        if not candidate.normalized_url:
                            continue
                        # same-site判定: forbidden_pathに当たっていない場合のみ
                        if (
                            is_same_site(candidate.normalized_url, target_url)
                            and candidate.product_url_rules
                            and not candidate.product_url_rules.get("forbidden_path_matched", False)
                        ):
                            # same-siteの場合は、domain/subdomain rejectを無視
                            relaxed_reasons = [
                                r
                                for r in candidate.reject_reasons
                                if r
                                not in (
                                    RejectReason.DIFFERENT_DOMAIN.value,
                                    RejectReason.DIFFERENT_SUBDOMAIN.value,
                                    RejectReason.DIFFERENT_ORIGIN.value,
                                )
                            ]
                            # 他のreject理由がなければ採用
                            if not relaxed_reasons:
                                found_links.add(candidate.normalized_url)
                                candidate.accepted = True
                                candidate.reject_reasons = []
                                candidate.notes = (candidate.notes or "") + " [Refilter: same-site]"
                                logger.debug(
                                    f"[PLP→PDP][Refilter] Accepted same-site candidate: {candidate.normalized_url}"
                                )

                # 3. 特定のreject理由を無視（forbidden_pathに当たってない場合）
                if not found_links:
                    ignore_reasons = refilter_config.get("ignore_reject_reasons", [])
                    if ignore_reasons:
                        for candidate in all_candidates:
                            if not candidate.normalized_url:
                                continue
                            # forbidden_pathに当たっていない場合のみ
                            if candidate.product_url_rules and not candidate.product_url_rules.get(
                                "forbidden_path_matched", False
                            ):
                                # 無視するreject理由を除外
                                filtered_reasons = [r for r in candidate.reject_reasons if r not in ignore_reasons]
                                # 残りのreject理由がなければ採用
                                if not filtered_reasons:
                                    found_links.add(candidate.normalized_url)
                                    candidate.accepted = True
                                    candidate.reject_reasons = []
                                    candidate.notes = (candidate.notes or "") + f" [Refilter: ignored {ignore_reasons}]"
                                    logger.debug(
                                        f"[PLP→PDP][Refilter] Accepted candidate (ignored {ignore_reasons}): {candidate.normalized_url}"
                                    )

                if found_links:
                    logger.info(
                        f"[PLP→PDP][Refilter] Evidence-based relaxed filtering accepted {len(found_links)} links."
                    )
                else:
                    logger.warning(
                        "[PLP→PDP][Refilter] No candidates accepted after relaxed filtering. "
                        "Consider click fallback or selector adjustment."
                    )

        links = sorted(list(found_links))
        if not links:
            # 詳細な診断情報をログに出力
            log_file_path = None
            if ctx and ctx.run_context:
                log_file_path = ctx.run_context.get_path("system.log")
            logger.error(
                f"[PLP→PDP] No PDP hrefs found after all phases. "
                f"Found {len(found_links)} links total, {len(all_candidates)} candidates collected. "
                f"Target URL: {target_url}. "
                "This indicates a selector mismatch or URL validation failure."
            )
            if log_file_path and log_file_path.exists():
                logger.error(f"[PLP→PDP] Full error log available at: {log_file_path}")
            elif ctx and ctx.run_context:
                logger.error(f"[PLP→PDP] Error log will be saved to: {ctx.run_context.get_path('system.log')}")
            # Phase 1b で見つかった要素数を診断用に出力
            try:
                total_elements_found = 0
                for sel in PLP_PDP_LINK_SELECTORS:
                    try:
                        nodes = await page.query_selector_all(sel)
                        if nodes:
                            total_elements_found += len(nodes)
                            # 最初の要素の属性をサンプルとして取得
                            if total_elements_found == len(nodes):
                                sample_node = nodes[0]
                                sample_attrs = {
                                    "tag": await sample_node.evaluate("el => el.tagName"),
                                    "href": await sample_node.get_attribute("href"),
                                    "data-href": await sample_node.get_attribute("data-href"),
                                    "data-product-url": await sample_node.get_attribute("data-product-url"),
                                    "class": await sample_node.get_attribute("class"),
                                }
                                logger.debug(f"[PLP→PDP] Sample element from selector '{sel}': {sample_attrs}")
                    except Exception:
                        pass
                if total_elements_found > 0:
                    logger.warning(
                        f"[PLP→PDP] Found {total_elements_found} elements matching selectors, "
                        "but none passed URL validation or product URL check."
                    )
            except Exception as e:
                logger.debug(f"[PLP→PDP] Diagnostic info collection failed: {e}")

        # Phase 3: Noise Filtering & Saving
        cleaned: list[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u):
                cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")

        # CR-E2E-003: 候補収集の証跡を保存
        link_collection_summary = await self._save_link_collection_evidence(
            all_candidates=all_candidates,
            accepted_links=cleaned,
            run_context=run_context,
            ctx=ctx,
        )

        # CR-E2E-003: NavigationContextにサマリを保存（後でBrowserOrchestratorでevidenceに追加）
        ctx.link_collection_summary = link_collection_summary

        try:
            sample = cleaned[:20]
            logger.debug(f"[PLP→PDP] sample={sample}")
            # Stage 3B: TelemetryClient を使用して JSON を保存
            if self.telemetry and ctx.run_context:
                from app.agents.browser.telemetry import TelemetryContext

                tctx = TelemetryContext(
                    site=ctx.site, query=ctx.query, run_id=getattr(ctx.run_context, "run_id", None), stage="plp"
                )
                await self.telemetry.save_json("raw_pdp_links_v85.5", {"links": cleaned, "sample": sample}, tctx)
                await self.telemetry._service.save_raw_hrefs(cleaned, name="raw_hrefs_final_cleaned")
            elif ctx.run_context and hasattr(ctx.run_context, "save_json"):
                # フォールバック: 既存の run_context.save_json を使用
                ctx.run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
                # フォールバック: 既存のobservability.py関数を使用
                from app.utils.observability import save_raw_hrefs

                if callable(save_raw_hrefs):
                    res = save_raw_hrefs(ctx.run_context, cleaned, name="raw_hrefs_final_cleaned")
                    if asyncio.iscoroutine(res):
                        await res
        except Exception as e:
            logger.debug(f"[PLP→PDP] TelemetryClient.save_raw_hrefs failed: {e}")
        return cleaned

    async def _save_link_collection_evidence(
        self,
        all_candidates: list[LinkCandidate],
        accepted_links: list[str],
        run_context: Any | None,
        ctx: NavigationContext,
    ) -> dict[str, Any]:
        """
        CR-E2E-003A: リンク収集の証跡を保存し、サマリを返す

        Phase別のJSONファイルを保存し、検証レポートを生成する。

        Args:
            all_candidates: すべての候補リスト
            accepted_links: 受け入れられたリンクリスト
            run_context: RunContext（任意）
            ctx: NavigationContext

        Returns:
            Dict[str, Any]: サマリデータ（evidence.link_collectionに追加する用）
        """
        summary: dict[str, Any] = {
            "total_candidates": len(all_candidates),
            "total_valid": len(accepted_links),
            "top_reject_reasons": {},
            "sample_candidates": [],
        }

        if not run_context or not hasattr(run_context, "save_json"):
            return summary

        # CR-E2E-003B拡張: validation_reportを初期化（finallyブロックで確実に保存するため）
        validation_report: dict[str, Any] | None = None

        try:
            # CR-E2E-003A: 候補をphase別に分類（最大200件まで）
            phase1_candidates: list[dict[str, Any]] = []
            phase2_candidates: list[dict[str, Any]] = []
            phase3_candidates: list[dict[str, Any]] = []

            for candidate in all_candidates[:200]:  # 上限200件
                candidate_dict = {
                    "phase": candidate.phase,
                    "source_selector": candidate.source_selector or "",
                    "raw_href": candidate.url,
                    "resolved_url": candidate.normalized_url or "",
                    "origin": candidate.origin or "",
                    "passed": candidate.accepted,
                    "reject_reasons": candidate.reject_reasons if candidate.reject_reasons else [],  # 空配列でも必須
                    "notes": candidate.notes or "",
                    # CR-E2E-003B: product_url_rulesを追加
                    "product_url_rules": candidate.product_url_rules if candidate.product_url_rules else {},
                }

                if candidate.phase in ("1a", "1b"):
                    phase1_candidates.append(candidate_dict)
                elif candidate.phase == "2":
                    phase2_candidates.append(candidate_dict)
                elif candidate.phase == "3":
                    phase3_candidates.append(candidate_dict)

            # CR-E2E-003A: Phase別JSONファイルを保存
            run_context.save_json(
                "pdp_link_candidates_phase1.json",
                {
                    "phase": "1a/1b",
                    "candidates": phase1_candidates,
                    "total": len(phase1_candidates),
                },
            )

            run_context.save_json(
                "pdp_link_candidates_phase2.json",
                {
                    "phase": "2",
                    "candidates": phase2_candidates,
                    "total": len(phase2_candidates),
                },
            )

            # CR-E2E-003B拡張: Phase 3 (HAR/Network) のJSONファイルを保存
            if phase3_candidates:
                run_context.save_json(
                    "pdp_link_candidates_phase3.json",
                    {
                        "phase": "3",
                        "candidates": phase3_candidates,
                        "total": len(phase3_candidates),
                    },
                )

            # CR-E2E-003A: reject理由の集計
            reject_reason_counts: dict[str, int] = {}
            for candidate in all_candidates:
                for reason in candidate.reject_reasons:
                    reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

            # 上位のreject理由を取得（最大10件）
            top_reject_reasons = dict(sorted(reject_reason_counts.items(), key=lambda x: x[1], reverse=True)[:10])

            # CR-E2E-003B: domain/path別の内訳を集計
            domain_rejected_count = 0
            path_rejected_count = 0
            allow_path_matched_count = 0
            forbidden_path_matched_count = 0
            allow_path_pattern_counts: dict[str, int] = {}
            forbidden_path_pattern_counts: dict[str, int] = {}

            for candidate in all_candidates:
                if candidate.product_url_rules:
                    rules = candidate.product_url_rules
                    if not rules.get("domain_allowed", True):
                        domain_rejected_count += 1
                    if rules.get("forbidden_path_matched", False):
                        forbidden_path_matched_count += 1
                        pattern = rules.get("forbidden_path_pattern")
                        if pattern:
                            forbidden_path_pattern_counts[pattern] = forbidden_path_pattern_counts.get(pattern, 0) + 1
                    if rules.get("allow_path_matched", False):
                        allow_path_matched_count += 1
                        pattern = rules.get("allow_path_pattern")
                        if pattern:
                            allow_path_pattern_counts[pattern] = allow_path_pattern_counts.get(pattern, 0) + 1
                    if not rules.get("allow_path_matched", False) and not rules.get("forbidden_path_matched", False):
                        path_rejected_count += 1

            # CR-E2E-003A: 検証レポートを生成（CR-E2E-003B拡張: 候補URL全件のreject reasonを必ず出す）
            validation_report = {
                "total_candidates": len(all_candidates),
                "total_valid": len(accepted_links),
                "total_rejected": len(all_candidates) - len(accepted_links),
                "reject_reason_counts": reject_reason_counts,
                "top_reject_reasons": top_reject_reasons,
                # CR-E2E-003B: domain/path別の内訳
                "domain_rejected_count": domain_rejected_count,
                "path_rejected_count": path_rejected_count,
                "allow_path_matched_count": allow_path_matched_count,
                "forbidden_path_matched_count": forbidden_path_matched_count,
                "allow_path_pattern_counts": dict(
                    sorted(allow_path_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "forbidden_path_pattern_counts": dict(
                    sorted(forbidden_path_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                # CR-E2E-003B拡張: 候補URL全件のreject reason（最大200件）
                "all_rejected_candidates": [
                    {
                        "raw_href": c.url,
                        "resolved_url": c.normalized_url,
                        "origin": c.origin or "",
                        "reject_reasons": c.reject_reasons,
                        "phase": c.phase,
                        "source_selector": c.source_selector or "",
                        "notes": c.notes or "",
                        "product_url_rules": c.product_url_rules if c.product_url_rules else {},
                    }
                    for c in all_candidates
                    if not c.accepted
                ][:200],  # 最大200件
                "sample_rejected": [
                    {
                        "raw_href": c.url,
                        "resolved_url": c.normalized_url,
                        "origin": c.origin or "",
                        "reject_reasons": c.reject_reasons,
                        "phase": c.phase,
                        "source_selector": c.source_selector or "",
                        "notes": c.notes or "",
                        "product_url_rules": c.product_url_rules if c.product_url_rules else {},
                    }
                    for c in all_candidates
                    if not c.accepted
                ][:10],  # 最大10件（サンプル）
            }

            # CR-E2E-003B拡張: finallyでflush（ファイルが空にならないように）
            try:
                run_context.save_json("pdp_link_validation_report.json", validation_report)
            except Exception as e:
                logger.error(f"[PLP→PDP] Failed to save validation report: {e}", exc_info=True)
                # フォールバック: 直接ファイルに書き込む
                try:
                    report_path = run_context.get_path("pdp_link_validation_report.json")
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(validation_report, f, indent=2, ensure_ascii=False)
                except Exception as e2:
                    logger.error(f"[PLP→PDP] Failed to write validation report directly: {e2}", exc_info=True)

            # CR-E2E-003A: サマリを更新（evidence.link_collection用）
            summary["total_candidates"] = len(all_candidates)
            summary["total_valid"] = len(accepted_links)
            summary["top_reject_reasons"] = top_reject_reasons
            summary["sample_candidates"] = [
                {
                    "raw_href": c.url,
                    "resolved_url": c.normalized_url,
                    "origin": c.origin or "",
                    "reject_reasons": c.reject_reasons,
                    "phase": c.phase,
                    "source_selector": c.source_selector or "",
                    "passed": c.accepted,
                }
                for c in all_candidates[:10]  # 最大10件
            ]

            logger.info(
                f"[PLP→PDP][CR-E2E-003A] Collected {len(all_candidates)} candidates, "
                f"accepted {len(accepted_links)}, rejected {len(all_candidates) - len(accepted_links)}"
            )
        except Exception as e:
            logger.warning(f"[PLP→PDP][CR-E2E-003A] Failed to save link collection evidence: {e}", exc_info=True)
            # エラー時でも最小限のサマリを返す
            summary["error"] = str(e)
        finally:
            # CR-E2E-003B拡張: finallyでvalidation_reportを確実に保存（120sタイムアウト時でも残る）
            if run_context and validation_report is not None:
                try:
                    report_path = run_context.get_path("pdp_link_validation_report.json")
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(validation_report, f, indent=2, ensure_ascii=False)
                    logger.info(f"[PLP→PDP] Saved validation report (finally) to: {report_path}")
                except Exception as e:
                    logger.error(f"[PLP→PDP] Failed to save validation report in finally: {e}", exc_info=True)

        return summary

    async def recover_plp(self, ctx: NavigationContext) -> bool:
        """
        Stage 3A-2-3:
        旧 BrowserUseAgent._force_plp_recover のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持すること。

        PLP を回復する（強制的に PLP URL にナビゲート）。
        成功したら True / 失敗したら False を返す。
        """
        page = self.page
        site_config = ctx.site_config
        target_url = ctx.entry_url

        try:
            await self._force_plp_recover(page, site_config, target_url)
            # 回復後に trap 判定を再チェック
            # Stage 3A-2-5: site_config を渡す
            if self._looks_like_trap_or_legal(page.url, site_config):
                logger.warning(f"[recover_plp] Still trap-like after recovery: {page.url}")
                return False
            return True
        except Exception as e:
            logger.debug(f"[recover_plp] Recovery failed: {e}")
        return False

    async def _detect_trap_page(self, ctx: NavigationContext) -> dict[str, str] | None:
        """
        CR-ATELIER-002 Step 1: Trap ページ検出（DOM ベース）

        PLP ではない状態（404 / location gate / 想定外ロケール＋検索ページ）を検出する。

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            trap を検出した場合は dict（type, reason を含む）、検出されなければ None
        """
        page = self.page
        site_config = ctx.site_config
        current_url = page.url or ""

        try:
            # 1. 404 ページの検出
            # h1 要素に "It's not here" が含まれるかチェック
            try:
                h1_text = await page.locator("h1").first.inner_text(timeout=2000)
                if h1_text and "It's not here" in h1_text:
                    return {
                        "type": "404",
                        "reason": f"h1 contains 'It's not here': {h1_text[:50]}",
                    }
            except Exception:
                pass

            # URL パターンに `/404` や `not-found` が含まれるかチェック
            url_lower = current_url.lower()
            if "/404" in url_lower or "not-found" in url_lower:
                return {
                    "type": "404",
                    "reason": f"URL contains /404 or not-found: {current_url}",
                }

            # 2. Location gate の検出
            # 本文に "Select your location" が含まれるかチェック
            try:
                body_text = await page.locator("body").first.inner_text(timeout=2000)
                if body_text and "Select your location" in body_text:
                    # product リスト（商品カード）が存在しないかチェック
                    # Moncler の product card セレクタを確認
                    plp_cfg = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
                    pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

                    # 商品カードのセレクタ候補
                    product_selectors = [
                        "[data-component='ProductCard']",
                        "[data-testid*='product']",
                        "article[data-product-id]",
                        ".product-card",
                        ".c-product-card",
                        "a[href*='/products/']",
                    ]
                    # site_config からも取得
                    product_selectors.extend(
                        (plp_cfg.get("tile_selectors", []) or []) + (pdp_cfg.get("pdp_link_selectors", []) or [])
                    )

                    product_found = False
                    for sel in product_selectors[:5]:  # 最初の5つだけチェック（パフォーマンス）
                        try:
                            count = await page.locator(sel).count()
                            if count > 0:
                                product_found = True
                                break
                        except Exception:
                            continue

                    if not product_found:
                        return {
                            "type": "location_gate",
                            "reason": "Contains 'Select your location' but no product cards found",
                        }
            except Exception:
                pass

            # 3. 想定外ロケール＋検索ページの検出
            # URL パスに `/en-lt/` や `/en-de/` など、`/en-int/` 以外のロケールが含まれるかチェック
            # かつ URL パスに `/search` が含まれるかチェック
            locale_cfg = site_config.get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "en-int")
            target_locale_path = f"/{prefer_locale}/"

            # 想定外ロケールパターン（/en-int/ 以外のロケールセグメント）
            unexpected_locale_patterns = [
                "/en-lt/",
                "/en-de/",
                "/en-fr/",
                "/en-jp/",
                "/en-us/",
            ]

            # URL に想定外ロケールが含まれ、かつ /search が含まれる場合
            if "/search" in url_lower:
                for pattern in unexpected_locale_patterns:
                    if pattern in url_lower and target_locale_path not in url_lower:
                        return {
                            "type": "unexpected_locale_search",
                            "reason": f"URL contains unexpected locale '{pattern}' and '/search': {current_url}",
                        }

            # 二重ロケールパターン（例: /en-lt/en-int/search）
            if target_locale_path in url_lower and "/search" in url_lower:
                # 二重ロケールが含まれているかチェック
                for pattern in unexpected_locale_patterns:
                    if pattern in url_lower:
                        return {
                            "type": "unexpected_locale_search",
                            "reason": f"URL contains double locale pattern '{pattern}' and '/search': {current_url}",
                        }

            return None

        except Exception as e:
            logger.debug(f"[TrapDetector] Error during trap page detection: {e}", exc_info=True)
            return None

    async def _click_first_card(
        self,
        page: Page,
        site_config: dict[str, Any],
    ) -> Page | None:
        """
        最初のカードをクリックする（骨組みのみ）

        Stage 3A-1: このステップでは、メソッドシグネチャのみ定義。

        Args:
            page: Playwright Page オブジェクト
            site_config: サイト設定

        Returns:
            Optional[Page]: クリック後の新しい Page（存在する場合）、または None
        """
        # Stage 3A-1: スタブ実装
        logger.debug("[NavigationDriver] _click_first_card (stub): returning None")
        return None

    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        """セレクタが出現するまで安全に待機する。ui_helpers に委譲。"""
        if ui_safe_wait_selector is not None:
            return await ui_safe_wait_selector(page, selector, timeout_ms=timeout_ms, state=state)
        if not page or page.is_closed():
            return False
        try:
            await asyncio.wait_for(
                page.wait_for_selector(selector, state=state, timeout=timeout_ms), timeout=(timeout_ms / 1000.0) + 0.5
            )
            return True
        except asyncio.CancelledError:
            logger.debug(f"[safe_wait_selector] Cancelled for '{selector}'")
            raise
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[safe_wait_selector] Timeout/Error for '{selector}': {e}")
            return False

    # Stage 3A-2-2: ヘルパーメソッド（BrowserUseAgent から移植）
    def _time_left_ms(self, start_t: float, budget_ms: int) -> int:
        """残り時間をミリ秒で返す"""
        used = int((time.monotonic() - start_t) * 1000)
        return max(0, budget_ms - used)

    async def _accept_cookies_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        """Cookie 同意バナーがあればクリックする。ui_helpers に委譲。"""
        if ui_accept_cookies_if_present is not None:
            return await ui_accept_cookies_if_present(page, site_config)
        return False

    async def _kill_overlays(self, page: Page, site_config: dict[str, Any] | None = None) -> None:
        """オーバーレイを削除する。ui_helpers に委譲。"""
        if ui_kill_overlays is not None:
            await ui_kill_overlays(page)
            return
        with contextlib.suppress(Exception):
            await page.evaluate("""
              (() => {
                const sels = ['.overlay','.backdrop','.modal-backdrop','#onetrust-banner-sdk','.cookie-banner','[aria-modal="true"]','.cmp-ui-overlay','.cmp-modal','.drawer--open'];
                document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                const b = document.body; if (b) { b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }
                const html=document.documentElement; if (html) { html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }
              })();
            """)

    async def ensure_plp_materialized(self, ctx: NavigationContext) -> bool:
        """
        Stage 3A-2-2:
        旧 BrowserUseAgent._ensure_plp_materialized のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持すること。

        PLP をスクロールしてタイルが十分に出るまで待つ処理。
        """
        page = self.page
        site_config = ctx.site_config
        settings = ctx.settings
        start_t = ctx.start_t
        budget_ms = ctx.budget_ms
        target_url = ctx.entry_url

        # Stage 3A-2-5: site_config から tile_selectors を取得（優先順位: plp.tile_selectors > pdp.pdp_link_selectors）
        plp_cfg = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

        tile_selectors = _dedupe_keep_order(
            (plp_cfg.get("tile_selectors", []) or [])  # 新規: plp.tile_selectors を優先
            + (plp_cfg.get("pdp_link_selectors", []) or [])  # plp.pdp_link_selectors も使用
            + (pdp_cfg.get("pdp_link_selectors", []) or [])
            + (pdp_cfg.get("plp_container_selectors", []) or [])
            + [
                "a[data-product-url]",
                "[data-product-url]",
                "[data-qa='product-tile']",
                ".product-card",
                ".c-product-card",
                ".c-product-tile",
                "[data-testid*='product' i]",
            ]
        )
        tile_selector_str = ", ".join(tile_selectors)
        target_min_tiles = 8
        max_scroll_attempts = int(max(settings.get("plp_scroll_rounds", 10), 10))
        run_ctx = ctx.run_context

        locale_recover_attempts = 0
        locale_recover_max = int(settings.get("locale_recover_max", 5))

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                logger.warning("[Materialize] Timed out.")
                return False

            # v88.6.x: Attemptごとに遅延表示ゲート/バナーを掃除する
            # 最初の試行では特に Cookie バナーを確実に閉じる
            if attempt == 0:
                try:
                    cookie_closed = await self._accept_cookies_if_present(page, site_config)
                    if cookie_closed:
                        logger.info("[Materialize] Cookie banner closed, waiting for content to load...")
                        # Cookie バナーを閉じた後、コンテンツが読み込まれるまで待機
                        await page.wait_for_timeout(1000)
                except Exception as e:
                    logger.debug(f"[Materialize] Cookie banner handling failed: {e}")
            else:
                with contextlib.suppress(Exception):
                    await self._accept_cookies_if_present(page, site_config)

            try:
                # Stage 3A-2-5: site_config を渡す
                # タイムアウトを避けるため、asyncio.wait_for でラップ
                import asyncio

                try:
                    await asyncio.wait_for(self._dismiss_geo_modal(page, site_config), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.debug("[Materialize] Geo modal dismissal timed out (non-fatal), continuing")
                except Exception as geo_e:
                    logger.debug(f"[Materialize] Geo modal dismissal failed (non-fatal): {geo_e}")
            except Exception as geo_e:
                logger.debug(f"[Materialize] Geo modal dismissal failed (non-fatal): {geo_e}")
                pass
            with contextlib.suppress(Exception):
                # Stage 3A-2-5: site_config を渡す
                await self._kill_overlays(page, site_config)

            # Stage 4: ロケールリダイレクトの検出（汎用化）
            current_url = (page.url or "").lower()
            locale_cfg = site_config.get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "")
            allowed_domain = site_config.get("allowed_domain", "")

            # ターゲットロケール以外のロケールにリダイレクトされた場合を検出
            if prefer_locale and allowed_domain:
                # 現在のURLがターゲットロケールを含まず、別ロケールセグメントが含まれている場合
                target_locale_path = f"/{prefer_locale}/"
                if (
                    allowed_domain.lower() in current_url
                    and target_locale_path not in current_url
                    and _LOCALE_SEG_RE.search(current_url)
                ):
                    logger.warning(
                        f"[Materialize] Detected locale redirect away from {prefer_locale} mid-attempt: {current_url}"
                    )
                    if locale_recover_attempts >= locale_recover_max:
                        logger.error("[Materialize] Locale recovery exceeded max attempts. Aborting.")
                        return False
                    locale_recover_attempts += 1
                    if target_url:
                        await self._force_plp_recover(page, site_config, target_url)
                        await page.wait_for_timeout(800)
                        # CR-ATELIER-002 Step 2: Locale redirect recovery 後のロケールチェック
                        # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
                        try:
                            await self._ensure_expected_locale(ctx)
                        except Exception as locale_e:
                            logger.warning(
                                f"[Materialize] Locale Guard after locale redirect recovery failed: {locale_e}",
                                exc_info=True,
                            )
                        continue

            if run_ctx is not None and hasattr(run_ctx, "take_screenshot") and attempt < 3:
                try:
                    await run_ctx.take_screenshot(page, f"30_plp_materialize_attempt_{attempt + 1:02d}")
                except Exception as ss_e:
                    logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=800)
            except Exception as e:
                logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}")
                break

            # Stage 4: ロケールゲートが途中で出た場合に備えて閉じておく（汎用化）
            try:
                modal_title = page.locator("text=Select your location").first
                if await modal_title.count() > 0:
                    logger.info("[GeoModal] Locale gate header detected during PLP materialization.")
                    close_btn = page.locator(
                        "button[aria-label*='close' i], "
                        "button:has-text('Close'), "
                        "button:has-text('×'), "
                        ".modal__close, .c-modal__close"
                    ).first
                    if await close_btn.count() > 0:
                        await close_btn.click(timeout=3000)
                        await page.wait_for_timeout(500)
                        logger.info("[GeoModal] Locale gate closed.")
            except Exception as e:
                logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

            try:
                count = await page.locator(tile_selector_str).count()
                logger.info(f"[Materialize] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")

                # Cookie バナーがまだ表示されている場合は、閉じる処理を再試行
                if count == 0 and attempt < 2:
                    try:
                        banner_container = page.locator("#onetrust-banner-sdk, #onetrust-pc-sdk")
                        if await banner_container.count() > 0:
                            is_visible = await banner_container.first.is_visible()
                            if is_visible:
                                logger.warning(
                                    "[Materialize] Cookie banner still visible, attempting to close again..."
                                )
                                await self._accept_cookies_if_present(page, site_config)
                                await page.wait_for_timeout(1500)  # バナーを閉じた後、コンテンツが読み込まれるまで待機
                                # タイル数を再カウント
                                count = await page.locator(tile_selector_str).count()
                                logger.info(f"[Materialize] After closing banner, found {count} tiles.")
                    except Exception as e:
                        logger.debug(f"[Materialize] Cookie banner check failed: {e}")

                if count >= target_min_tiles:
                    logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
                if count < 4 and attempt >= 1:
                    logger.warning(
                        f"[Materialize] Low tiles ({count}) after {attempt + 1} attempts, forcing recovery hop."
                    )
                    if target_url:
                        try:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(500)
                            # CR-ATELIER-002 Step 2: Recovery hop 後のロケールチェック
                            # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
                            try:
                                await self._ensure_expected_locale(ctx)
                            except Exception as locale_e:
                                logger.warning(
                                    f"[Materialize] Locale Guard after recovery hop failed: {locale_e}", exc_info=True
                                )
                            rec_count = await page.locator(tile_selector_str).count()
                            logger.info(f"[Materialize] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return True
                        except Exception as rec_e:
                            logger.warning(f"[Materialize] Recovery hop failed: {rec_e}")
                    return False
            except asyncio.CancelledError:
                logger.warning("[Materialize] Cancelled during tile count.")
                return False
            except Exception as e:
                logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
        if final_count > 0:
            logger.warning(
                f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty."
            )
            return True
        logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
        return False

    async def _click_continue_shopping_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        """CONTINUE SHOPPING ボタンがあればクリックする。ui_helpers に委譲。"""
        if ui_click_continue_shopping_if_present is not None:
            return await ui_click_continue_shopping_if_present(page, site_config)
        return False


# P1-1 Phase 2: 後方互換 re-export
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
    "classify_candidate",
]
