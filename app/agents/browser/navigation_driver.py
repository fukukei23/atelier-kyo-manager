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

if TYPE_CHECKING:
    from app.agents.browser.telemetry import TelemetryClient

# Stage 3A-2-1: extractor モジュールから looks_like_product_url を import
try:
    from app.agents.browser.extractor import (
        VISIBLE_PRICE_SELECTORS,
        extract_moncler_pdp_links,
        looks_like_product_url,
    )
except ImportError:
    # フォールバック: モジュールが見つからない場合の処理
    def looks_like_product_url(url: str) -> bool:
        """フォールバック実装"""
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

logger = logging.getLogger(__name__)

# Stage 3A-2-2: ロケールセグメント判定用の正規表現
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)


# CR-E2E-003: reject理由の列挙型
class RejectReason(str, Enum):
    """PDPリンク候補のreject理由"""

    NO_HREF = "no_href"
    DIFFERENT_ORIGIN = "different_origin"
    DIFFERENT_DOMAIN = "different_domain"  # CR-E2E-003A拡張: eTLD+1が異なる
    DIFFERENT_SUBDOMAIN = "different_subdomain"  # CR-E2E-003A拡張: サブドメインが異なる
    BLOCKED_ORIGIN = "blocked_origin"
    NOT_PRODUCT_URL = "not_product_url"
    NO_PRODUCTS_PATH = "no_products_path"
    BLOCKED_DOMAIN = "blocked_domain"
    TRAP_PATTERN = "trap_pattern"
    NOISE_PATTERN = "noise_pattern"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass
class LinkCandidate:
    """PDPリンク候補（CR-E2E-003/003A/003B）"""

    url: str  # raw_href
    phase: str  # "1a", "1b", "2", "moncler"
    normalized_url: str  # resolved_url
    reject_reasons: list[str] = field(default_factory=list)
    accepted: bool = False
    source_selector: str | None = None  # CR-E2E-003A: セレクタ情報
    origin: str | None = None  # CR-E2E-003A: origin情報
    notes: str | None = None  # CR-E2E-003A: 補足情報
    product_url_rules: dict[str, Any] | None = None  # CR-E2E-003B: 判定根拠


# CR-ATELIER-002 Step 1: Trap ページ検出用の例外クラス
class TrapPageDetected(Exception):
    """PLP ではなく trap ページ（404 / location gate / 想定外ロケール）が検出されたことを示す例外"""

    def __init__(self, trap_type: str, reason: str, url: str):
        self.trap_type = trap_type  # "404", "location_gate", "unexpected_locale_search"
        self.reason = reason
        self.url = url
        message = f"Trap page detected: {trap_type} - {reason} (URL: {url})"
        super().__init__(message)


# Stage 3A-2-1: ヘルパー関数（BrowserUseAgent から移植）
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


# Stage 3A-3: trap 判定関数の型定義
TrapCheckerFn = Callable[[str], bool]  # URL を受けて trap かどうか判定


@dataclass
class NavigationContext:
    """ナビゲーション実行時のコンテキスト情報"""

    site: str
    query: str
    site_config: dict[str, Any]
    settings: dict[str, Any]
    run_context: Any  # RunContext を直接 import しない（循環回避）
    start_t: float
    budget_ms: int
    entry_url: str | None = None
    context: Any = None  # Stage 3A-2-4: BrowserContext（fallback で必要、click_first_card_or_link で使用）
    link_collection_summary: dict[str, Any] | None = None  # CR-E2E-003: リンク収集のサマリ
    link_collection_summary: dict[str, Any] | None = None  # CR-E2E-003: リンク収集のサマリ


@dataclass
class NavigationOutcome:
    """ナビゲーション実行結果"""

    entry_url: str
    plp_materialized: bool = False
    trap_detected: bool = False
    trap_reason: str | None = None
    recovered: bool = False
    pdp_links: list[str] = None  # Stage 3A-2-4: PDP リンクのリスト（BrowserUseAgent で使用）
    fallback_used: str | None = (
        None  # Stage 3A-2-4: 使用した fallback の種類（"header_search" または "click_first_card"、BrowserUseAgent で使用）
    )
    locale_corrections: int = 0  # CR-ATELIER-002 Step 6: Locale補正の回数
    moncler_outcome: dict[str, Any] | None = None  # CR-ATELIER-002 Step 6: Moncler抽出結果の詳細情報

    def __post_init__(self):
        """pdp_links のデフォルト値を設定"""
        if self.pdp_links is None:
            self.pdp_links = []


class NavigationDriver:
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

    def _extract_etld_plus_one(self, hostname: str) -> str | None:
        """
        CR-E2E-003A拡張: eTLD+1を抽出（例: www.moncler.com -> moncler.com）

        Args:
            hostname: ホスト名

        Returns:
            Optional[str]: eTLD+1またはNone
        """
        if not hostname:
            return None
        hostname = hostname.lower().strip()
        # 簡易実装: 最後の2つのドメイン部分を取得
        # より正確にはpublicsuffixlistを使うが、ここでは簡易版
        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    def _is_same_site(self, url1: str, url2: str) -> bool:
        """
        CR-E2E-003A拡張: 2つのURLが同じサイト（eTLD+1）か判定

        Args:
            url1: URL1
            url2: URL2

        Returns:
            bool: 同じサイトの場合True
        """
        try:
            parsed1 = urlparse(url1)
            parsed2 = urlparse(url2)
            etld1 = self._extract_etld_plus_one(parsed1.netloc)
            etld2 = self._extract_etld_plus_one(parsed2.netloc)
            return etld1 and etld2 and etld1 == etld2
        except Exception:
            return False

    def _check_origin_allowed(
        self,
        url: str,
        base_url: str,
        site_config: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """
        CR-E2E-003/003A拡張: originが許可されているかチェック（eTLD+1判定対応）

        Args:
            url: 検証対象URL
            base_url: ベースURL
            site_config: サイト設定（任意）

        Returns:
            (is_allowed, reject_reason): 許可されている場合True、拒否理由（拒否時）
        """
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_url)
            url_origin = (parsed.scheme, parsed.netloc)
            base_origin = (base_parsed.scheme, base_parsed.netloc)

            # 同一originの場合は許可
            if url_origin == base_origin:
                return True, None

            # CR-E2E-003A拡張: eTLD+1判定（same-site）
            if self._is_same_site(url, base_url):
                # サブドメインが異なる場合は警告付きで許可
                if parsed.netloc.lower() != base_parsed.netloc.lower():
                    return True, None  # 許可（reject理由は記録しない）
                return True, None

            # site_configから設定を取得（Missing-safe）
            if site_config:
                allowed_origins = site_config.get("allowed_origins", [])
                blocked_origins = site_config.get("blocked_origins", [])
                allowed_host_suffixes = site_config.get("allowed_host_suffixes", [])
                # CR-E2E-003A拡張: allowed_domains（配列）とallowed_domain（文字列）の両方に対応
                allowed_domains = site_config.get("allowed_domains", [])
                if not allowed_domains:
                    allowed_domain = site_config.get("allowed_domain")
                    if allowed_domain:
                        allowed_domains = [allowed_domain]
            else:
                allowed_origins = []
                blocked_origins = []
                allowed_host_suffixes = []
                allowed_domains = []

            # blocked_originsチェック
            if blocked_origins:
                for blocked in blocked_origins:
                    if blocked in parsed.netloc.lower():
                        return False, RejectReason.BLOCKED_ORIGIN.value

            # CR-E2E-003A拡張: allowed_domainsチェック（eTLD+1判定）
            if allowed_domains:
                url_etld = self._extract_etld_plus_one(parsed.netloc)
                for allowed_domain in allowed_domains:
                    allowed_etld = self._extract_etld_plus_one(allowed_domain)
                    if url_etld and allowed_etld and url_etld == allowed_etld:
                        return True, None

            # allowed_originsチェック
            if allowed_origins:
                for allowed in allowed_origins:
                    if allowed in parsed.netloc.lower():
                        return True, None

            # allowed_host_suffixesチェック
            if allowed_host_suffixes:
                for suffix in allowed_host_suffixes:
                    if parsed.netloc.lower().endswith(suffix):
                        return True, None

            # CR-E2E-003A拡張: reject理由を分解
            # eTLD+1が異なる場合はdifferent_domain、サブドメインのみ異なる場合はdifferent_subdomain
            url_etld = self._extract_etld_plus_one(parsed.netloc)
            base_etld = self._extract_etld_plus_one(base_parsed.netloc)
            if url_etld and base_etld and url_etld != base_etld:
                return False, RejectReason.DIFFERENT_DOMAIN.value
            elif parsed.netloc.lower() != base_parsed.netloc.lower():
                return False, RejectReason.DIFFERENT_SUBDOMAIN.value

            # デフォルト: 外部originは拒否（ただし候補として残す）
            return False, RejectReason.DIFFERENT_ORIGIN.value
        except Exception:
            return False, RejectReason.VALIDATION_ERROR.value

    def _extract_origin(self, url: str) -> str | None:
        """
        CR-E2E-003A: URLからoriginを抽出

        Args:
            url: URL

        Returns:
            Optional[str]: origin（scheme://netloc）またはNone
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            pass
        return None

    def _normalize_candidate_url(
        self,
        url: str,
        base_url: str,
        site_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        CR-E2E-003B: 候補URLを正規化（相対URL、クエリ、フラグメント、末尾スラッシュ、localeパラメータ等を統一）

        Args:
            url: 正規化対象URL
            base_url: ベースURL（相対URL解決用）
            site_config: サイト設定（任意）

        Returns:
            (normalized_url, normalization_info): 正規化されたURLと正規化情報
        """
        normalization_info = {
            "original": url,
            "was_relative": False,
            "removed_query_params": [],
            "removed_fragment": False,
            "normalized": url,
        }

        try:
            # プロトコル相対URL（//example.com/path）の処理
            if url.startswith("//"):
                base_parsed = urlparse(base_url)
                normalized = f"{base_parsed.scheme}:{url}"
                normalization_info["was_relative"] = True
            # 相対URLを絶対URLに変換
            elif url.startswith("/") or not url.startswith("http"):
                normalized = urljoin(base_url, url)
                normalization_info["was_relative"] = True
            else:
                normalized = url

            parsed = urlparse(normalized)

            # クエリパラメータの処理
            query_params = {}
            if parsed.query:
                from urllib.parse import parse_qs

                query_dict = parse_qs(parsed.query)

                # site_configから削除すべきクエリパラメータを取得
                remove_params = []
                if site_config:
                    url_rules = site_config.get("url_rules", {})
                    if isinstance(url_rules, dict):
                        normalize_rules = url_rules.get("normalize_rules", {})
                        if isinstance(normalize_rules, dict):
                            remove_params = normalize_rules.get("remove_query_params", [])

                # 削除対象以外のパラメータを保持
                for key, values in query_dict.items():
                    if key not in remove_params:
                        query_params[key] = values
                    else:
                        normalization_info["removed_query_params"].append(key)

            # フラグメントの処理
            fragment = ""
            if site_config:
                url_rules = site_config.get("url_rules", {})
                if isinstance(url_rules, dict):
                    normalize_rules = url_rules.get("normalize_rules", {})
                    if isinstance(normalize_rules, dict) and not normalize_rules.get("remove_fragment", True):
                        fragment = parsed.fragment
                    else:
                        normalization_info["removed_fragment"] = True

            # URLを再構築
            from urllib.parse import urlencode, urlunparse

            normalized_query = urlencode(query_params, doseq=True) if query_params else ""
            normalized = urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.params, normalized_query, fragment)
            )

            normalization_info["normalized"] = normalized
            return normalized, normalization_info
        except Exception as e:
            logger.debug(f"[URL Normalize] Failed to normalize URL: {e}")
            normalization_info["error"] = str(e)
            return url, normalization_info

    def _validate_candidate_url(
        self,
        url: str,
        normalized_url: str,
        base_url: str,
        site_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        CR-E2E-003B: 候補URLを検証し、判定根拠を返す

        Args:
            url: 元のURL（raw_href）
            normalized_url: 正規化済みURL
            base_url: ベースURL
            site_config: サイト設定（任意）

        Returns:
            Dict[str, Any]: 検証結果と判定根拠
            {
                "domain_allowed": bool,
                "allow_path_matched": bool,
                "allow_path_pattern": Optional[str],  # マッチしたパターン
                "forbidden_path_matched": bool,
                "forbidden_path_pattern": Optional[str],  # マッチしたパターン
                "locale_ok": bool,
                "final_decision": "accept" | "reject",
                "reject_reasons": List[str],
                "product_url_rules": Dict[str, Any],
            }
        """
        result = {
            "domain_allowed": False,
            "allow_path_matched": False,
            "allow_path_pattern": None,
            "forbidden_path_matched": False,
            "forbidden_path_pattern": None,
            "locale_ok": False,
            "final_decision": "reject",
            "reject_reasons": [],
            "product_url_rules": {},
        }

        try:
            parsed = urlparse(normalized_url)
            path = parsed.path or ""
            host = parsed.netloc.lower() if parsed.netloc else ""

            # 1. ドメイン許可チェック
            if site_config:
                allowed_domains = site_config.get("allowed_domains", [])
                if not allowed_domains:
                    allowed_domain = site_config.get("allowed_domain")
                    if allowed_domain:
                        allowed_domains = [allowed_domain]

                domain_allowed = False
                if allowed_domains:
                    url_etld = self._extract_etld_plus_one(host)
                    for allowed_domain in allowed_domains:
                        allowed_etld = self._extract_etld_plus_one(allowed_domain)
                        if url_etld and allowed_etld and url_etld == allowed_etld:
                            domain_allowed = True
                            break

                result["domain_allowed"] = domain_allowed

                if not domain_allowed:
                    result["reject_reasons"].append(RejectReason.DIFFERENT_DOMAIN.value)
                    result["final_decision"] = "reject"
                    return result
            else:
                # site_configがない場合は、base_urlと同じドメインかチェック
                base_parsed = urlparse(base_url)
                base_host = base_parsed.netloc.lower() if base_parsed.netloc else ""
                if host and base_host:
                    url_etld = self._extract_etld_plus_one(host)
                    base_etld = self._extract_etld_plus_one(base_host)
                    result["domain_allowed"] = url_etld and base_etld and url_etld == base_etld
                    if not result["domain_allowed"]:
                        result["reject_reasons"].append(RejectReason.DIFFERENT_DOMAIN.value)
                        result["final_decision"] = "reject"
                        return result
                else:
                    result["domain_allowed"] = True  # デフォルトで許可

            # 2. forbidden_path_patternsチェック（優先）
            if site_config:
                url_rules = site_config.get("url_rules", {})
                if isinstance(url_rules, dict):
                    forbidden_path_patterns = url_rules.get("forbidden_path_patterns", [])
                    for pattern in forbidden_path_patterns:
                        if re.search(pattern, path, re.I):
                            result["forbidden_path_matched"] = True
                            result["forbidden_path_pattern"] = pattern
                            result["reject_reasons"].append(RejectReason.NOISE_PATTERN.value)
                            result["final_decision"] = "reject"
                            return result

                    # 3. allow_path_patternsチェック
                    allow_path_patterns = url_rules.get("allow_path_patterns", [])
                    if allow_path_patterns:
                        for pattern in allow_path_patterns:
                            if re.search(pattern, path, re.I):
                                result["allow_path_matched"] = True
                                result["allow_path_pattern"] = pattern
                                break

                    # allow_path_patternsが設定されていてマッチしない場合
                    if allow_path_patterns and not result["allow_path_matched"]:
                        result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                        result["final_decision"] = "reject"
                        return result
                    elif not allow_path_patterns:
                        # 従来の/products?/チェック
                        if not re.search(r"/products?/", path, re.I):
                            result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                            result["final_decision"] = "reject"
                            return result
                else:
                    # 従来の/products?/チェック
                    if not re.search(r"/products?/", path, re.I):
                        result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                        result["final_decision"] = "reject"
                        return result
            else:
                # site_configがない場合も従来の/products?/チェック
                if not re.search(r"/products?/", path, re.I):
                    result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                    result["final_decision"] = "reject"
                    return result
                else:
                    result["allow_path_matched"] = True

            # 4. locale_okチェック（LocaleGuardの結果を参照）
            # 簡易チェック: /en-int/で始まるか
            result["locale_ok"] = path.lower().startswith("/en-int/")

            # 5. looks_like_product_urlチェック（allow_path_matchedがTrueの場合はスキップ）
            if not result["allow_path_matched"]:
                looks_like_product = looks_like_product_url(normalized_url)
                if not looks_like_product:
                    result["reject_reasons"].append(RejectReason.NOT_PRODUCT_URL.value)
                    result["final_decision"] = "reject"
                    return result

            # 6. すべてのチェックをパスした場合
            if not result["reject_reasons"]:
                result["final_decision"] = "accept"

            # product_url_rulesを構築
            result["product_url_rules"] = {
                "domain_allowed": result["domain_allowed"],
                "allow_path_matched": result["allow_path_matched"],
                "allow_path_pattern": result["allow_path_pattern"],
                "forbidden_path_matched": result["forbidden_path_matched"],
                "forbidden_path_pattern": result["forbidden_path_pattern"],
                "locale_ok": result["locale_ok"],
                "final_decision": result["final_decision"],
                "reject_reasons": result["reject_reasons"],
            }

            return result
        except Exception as e:
            logger.warning(f"[Validate Candidate URL] Failed to validate URL: {e}", exc_info=True)
            result["reject_reasons"].append(RejectReason.VALIDATION_ERROR.value)
            result["final_decision"] = "reject"
            result["error"] = str(e)
            return result

    def _classify_candidate(
        self,
        candidate: LinkCandidate,
        base_url: str,
        site_config: dict[str, Any] | None = None,
    ) -> LinkCandidate:
        """
        CR-E2E-003/003A/003B: 候補を分類し、reject理由を記録

        CR-E2E-003B拡張: _validate_candidate_urlを使用して体系的な検証を行う

        Args:
            candidate: リンク候補
            base_url: ベースURL
            site_config: サイト設定（任意）

        Returns:
            LinkCandidate: reject理由が記録された候補
        """
        # CR-E2E-003A: origin情報を記録
        if candidate.normalized_url:
            candidate.origin = self._extract_origin(candidate.normalized_url)

        # CR-E2E-003A拡張: スキーム除外チェック（javascript:, mailto:, tel:など）
        parsed_url = urlparse(candidate.normalized_url)
        if parsed_url.scheme and parsed_url.scheme.lower() not in ("http", "https", ""):
            candidate.reject_reasons.append(RejectReason.NO_HREF.value)
            candidate.notes = f"invalid scheme: {parsed_url.scheme}"
            candidate.product_url_rules = {
                "domain_allowed": False,
                "allow_path_matched": False,
                "forbidden_path_matched": False,
                "locale_ok": False,
                "final_decision": "reject",
                "reject_reasons": [RejectReason.NO_HREF.value],
                "error": "invalid scheme",
            }
            return candidate

        # OneTrustドメイン除外（早期チェック）
        if "onetrust.com" in parsed_url.netloc.lower():
            candidate.reject_reasons.append(RejectReason.BLOCKED_DOMAIN.value)
            candidate.notes = "blocked domain: onetrust.com"
            candidate.product_url_rules = {
                "domain_allowed": False,
                "allow_path_matched": False,
                "forbidden_path_matched": False,
                "locale_ok": False,
                "final_decision": "reject",
                "reject_reasons": [RejectReason.BLOCKED_DOMAIN.value],
            }
            return candidate

        # CR-E2E-003B: _validate_candidate_urlを使用して検証
        validation_result = self._validate_candidate_url(
            url=candidate.url,
            normalized_url=candidate.normalized_url,
            base_url=base_url,
            site_config=site_config,
        )

        # 検証結果を候補に反映
        candidate.reject_reasons.extend(validation_result["reject_reasons"])
        candidate.product_url_rules = validation_result.get("product_url_rules", {})

        # notesに詳細を記録
        if validation_result["forbidden_path_matched"]:
            candidate.notes = f"forbidden path pattern: {validation_result.get('forbidden_path_pattern', 'unknown')}"
        elif not validation_result["domain_allowed"]:
            candidate.notes = f"different domain (eTLD+1): {candidate.origin}"
        elif not validation_result["allow_path_matched"]:
            candidate.notes = "path did not match allow_path_patterns"
        elif not validation_result["locale_ok"]:
            candidate.notes = "locale mismatch"

        # すべてのチェックをパスした場合のみaccept
        if validation_result["final_decision"] == "accept":
            candidate.accepted = True

        return candidate

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
                norm_url, norm_info = self._normalize_candidate_url(href, target_url, site_config)
                candidate = LinkCandidate(
                    url=href,
                    phase="1a",
                    normalized_url=norm_url,
                    source_selector="global_sweep",  # CR-E2E-003A
                )
                candidate = self._classify_candidate(candidate, target_url, site_config)
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
                    norm_url, norm_info = self._normalize_candidate_url(href, target_url, site_config)
                    # CR-E2E-003A: 候補として収集（origin判定は後で分類）
                    candidate = LinkCandidate(
                        url=href,
                        phase="1b",
                        normalized_url=norm_url,
                        source_selector=sel,  # CR-E2E-003A
                    )
                    candidate = self._classify_candidate(candidate, target_url, site_config)
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
                    norm_url, norm_info = self._normalize_candidate_url(href, target_url, site_config)
                    # CR-E2E-003A: 候補として収集（origin判定は後で分類）
                    candidate = LinkCandidate(
                        url=href,
                        phase="2",
                        normalized_url=norm_url,
                        source_selector="deep_extraction",  # CR-E2E-003A
                    )
                    candidate = self._classify_candidate(candidate, target_url, site_config)
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
                    if candidate.product_url_rules:
                        # forbidden_path_matchedがFalseの場合
                        if not candidate.product_url_rules.get("forbidden_path_matched", True):
                            # domain_allowedがTrueで、allow_path_matchedがFalseでも、商品カード由来など強い根拠があるものだけ試す
                            if candidate.product_url_rules.get("domain_allowed", False):
                                # source_selectorが商品カード関連の場合は採用を試す
                                if candidate.source_selector and any(
                                    keyword in candidate.source_selector.lower()
                                    for keyword in ["product", "card", "tile", "item"]
                                ):
                                    found_links.add(candidate.normalized_url)
                                    candidate.accepted = True
                                    candidate.reject_reasons = [
                                        r
                                        for r in candidate.reject_reasons
                                        if r not in (RejectReason.NO_PRODUCTS_PATH.value,)
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
                        # same-site判定
                        if self._is_same_site(candidate.normalized_url, target_url):
                            # forbidden_pathに当たっていない場合のみ
                            if candidate.product_url_rules and not candidate.product_url_rules.get(
                                "forbidden_path_matched", False
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

    async def _collect_moncler_pdp_links(self, ctx: NavigationContext) -> set[str]:
        """
        CR-ATELIER-002 Step 3: Moncler専用のPDP抽出ロジック

        Monclerの実際のDOM構造に合わせたセレクタを使用してPDPリンクを抽出する。
        - /products/ パターンを含むURLを優先
        - 外部ドメイン（onetrust.com等）を明示的に除外
        - Moncler本体のドメイン（moncler.com）のみを対象とする

        Args:
            ctx: ナビゲーションコンテキスト

        Returns:
            Set[str]: 抽出されたPDPリンクのセット
        """
        page = self.page
        target_url = page.url
        found_links: set[str] = set()

        # Moncler専用のセレクタ（/products/ パターンを含む）
        moncler_selectors = [
            # ProductCardコンポーネント内のリンク
            "article[data-component*='ProductCard'] a[href*='/products/']",
            "article[data-component*='ProductCard'] a[href*='/product/']",
            "[data-component*='ProductCard'] a[href*='/products/']",
            "[data-component*='ProductCard'] a[href*='/product/']",
            # data-testidベースのセレクタ
            "[data-testid*='product-card'] a[href*='/products/']",
            "[data-testid*='product-tile'] a[href*='/products/']",
            "[data-test*='product-card'] a[href*='/products/']",
            # classベースのセレクタ
            ".product-card a[href*='/products/']",
            ".c-product-card a[href*='/products/']",
            ".product-tile a[href*='/products/']",
            ".c-product-tile a[href*='/products/']",
            # 汎用セレクタ（/products/ を含む）
            "a[href*='/en-int/products/']",
            "a[href*='/products/']",
            # data-qaベース
            "[data-qa='product-tile'] a[href*='/products/']",
            "[data-qa*='product'] a[href*='/products/']",
        ]

        logger.info(f"[PLP→PDP][Moncler] Starting Moncler-specific PDP extraction from URL: {target_url}")

        # Phase 1: Moncler専用セレクタで抽出
        for sel in moncler_selectors:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                rejected_count = 0
                rejection_reasons = []

                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
                    if not href:
                        rejected_count += 1
                        if rejected_count == 1:
                            rejection_reasons.append("no href attribute")
                        continue

                    # URLを正規化
                    norm_url = self._normalize_abs_url(target_url, href)

                    # Moncler専用のURLバリデーション
                    if not self._is_valid_moncler_pdp_url(norm_url, target_url):
                        rejected_count += 1
                        if rejected_count == 1:
                            reason = self._get_moncler_rejection_reason(norm_url, target_url)
                            rejection_reasons.append(f"{reason}: {norm_url}")
                        continue

                    found_links.add(norm_url)
                    matched_count += 1

                if matched_count > 0:
                    logger.info(f"[PLP→PDP][Moncler] selector='{sel}' added {matched_count} links")
                elif nodes and rejected_count > 0:
                    logger.debug(
                        f"[PLP→PDP][Moncler] selector='{sel}' found {len(nodes)} elements, "
                        f"but {rejected_count} were rejected. "
                        f"Reasons: {', '.join(rejection_reasons[:2])}"
                    )
            except Exception as e:
                logger.debug(f"[PLP→PDP][Moncler] selector='{sel}' failed: {e}")

        # Phase 2: グローバルsweep（Moncler専用フィルタリング）
        if not found_links:
            logger.debug("[PLP→PDP][Moncler] Selector-based extraction found no links, trying global sweep...")
            try:
                raw_hrefs: list[str] = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
                )
                pdp_rx = re.compile(r"/products?/", re.I)
                for href in raw_hrefs:
                    # OneTrustドメインのhrefを除外（ノイズ除去）
                    if "onetrust.com" in href.lower():
                        continue
                    if pdp_rx.search(href):
                        norm_url = self._normalize_abs_url(target_url, href)
                        if self._is_valid_moncler_pdp_url(norm_url, target_url):
                            found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][Moncler] Global sweep found {len(found_links)} links")
            except Exception as e:
                logger.warning(f"[PLP→PDP][Moncler] Global sweep failed: {e}")

        # デバッグ用ログ
        if found_links:
            sample_urls = list(found_links)[:5]
            logger.debug(f"[PLP→PDP][Moncler] Sample PDP URLs: {sample_urls}")
        else:
            logger.warning("[PLP→PDP][Moncler] No valid PDP links found after Moncler-specific extraction")

        return found_links

    def _is_valid_moncler_pdp_url(self, url: str, base_url: str) -> bool:
        """
        CR-ATELIER-002 Step 3: Moncler専用のURLバリデーション

        MonclerのPDP URLとして有効かどうかを判定する。

        Args:
            url: 検証対象のURL
            base_url: ベースURL（同一オリジン判定用）

        Returns:
            bool: 有効なMoncler PDP URLの場合True
        """
        try:
            parsed = urlparse(url)

            # スキームチェック
            if parsed.scheme not in ("http", "https"):
                return False

            # ホストチェック（Moncler本体のドメインのみ）
            host = parsed.netloc.lower()
            if not host.endswith("moncler.com"):
                return False

            # 外部ドメインの明示的な除外（onetrust.com等）
            blocked_domains = [
                "onetrust.com",
                "monclergroup.com",
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "pinterest.com",
            ]
            for blocked in blocked_domains:
                if blocked in host:
                    return False

            # パスチェック（/products/ または /product/ を含む）
            path = parsed.path or ""
            if not re.search(r"/products?/", path, re.I):
                return False

            # 404やtrapページのパターンを除外
            trap_patterns = [
                r"/404",
                r"/not-found",
                r"/search\?",
                r"/legal/",
                r"/client-service",
            ]
            for pattern in trap_patterns:
                if re.search(pattern, path, re.I):
                    return False

            # 同一オリジン判定
            if not is_same_origin(url, base_url):
                return False

            # looks_like_product_url のチェックも実行
            return looks_like_product_url(url)
        except Exception as e:
            logger.debug(f"[PLP→PDP][Moncler] URL validation failed for {url}: {e}")
            return False

    def _get_moncler_rejection_reason(self, url: str, base_url: str) -> str:
        """
        Moncler URLバリデーションでrejectされた理由を取得

        Args:
            url: 検証対象のURL
            base_url: ベースURL

        Returns:
            str: reject理由
        """
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path or ""

            # ホストチェック
            if not host.endswith("moncler.com"):
                return "external_domain"

            # ブロックドメイン
            if "onetrust.com" in host:
                return "blocked_domain_onetrust"
            if "monclergroup.com" in host:
                return "blocked_domain_monclergroup"

            # パスチェック
            if not re.search(r"/products?/", path, re.I):
                return "no_products_path"

            # 同一オリジン
            if not is_same_origin(url, base_url):
                return "different_origin"

            # looks_like_product_url
            if not looks_like_product_url(url):
                return "not_product_url_pattern"

            return "unknown"
        except Exception:
            return "validation_error"

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

    # ==============================================================================
    # CR-ATELIER-002 Step 3: Moncler専用 PDP 抽出ロジック設計案（実装済み）
    # ==============================================================================
    #
    # 【実装済みのMoncler専用ロジック】
    #
    # NavigationDriver.collect_pdp_links() 内で、MONCLER_OFFICIAL の場合に
    # _collect_moncler_pdp_links() を優先的に呼び出す分岐が実装済み。
    #
    # 分岐案（実装済み）:
    # ```python
    # site_code = site_config.get("site_code") or site_config.get("site") or ""
    # if site_code == "MONCLER_OFFICIAL":
    #     moncler_links = await self._collect_moncler_pdp_links(ctx)
    #     if moncler_links:
    #         # Moncler専用ロジックで見つかった場合は、それを返す
    #         return cleaned_moncler_links
    # # フォールバック: 汎用ロジック
    # ```
    #
    # 【Moncler専用抽出ロジックの実装済み機能】
    # - _collect_moncler_pdp_links(): Moncler専用セレクタで抽出
    # - _is_valid_moncler_pdp_url(): Moncler専用URLバリデーション
    # - _get_moncler_rejection_reason(): reject理由の取得（デバッグ用）
    #
    # 【改善案: ログ・Telemetryの強化】
    # 以下の情報をログ／Telemetryに残す案:
    #
    # 1. Moncler専用抽出を試したかどうか
    #    logger.info("[PLP→PDP][Moncler] Using moncler-specific extractor...")
    #
    # 2. 抽出候補として拾ったhrefの一覧（上限10件程度）
    #    logger.debug("[PLP→PDP][Moncler] Raw hrefs (first 10): %s", raw_hrefs[:10])
    #
    # 3. 何件が「origin mismatch」でrejectされ、何件が「locale mismatch」でrejectされたか
    #    logger.warning(
    #        "[PLP→PDP][Moncler] Extraction summary: raw=%d, origin_rejected=%d, "
    #        "locale_rejected=%d, path_rejected=%d, accepted=%d",
    #        len(raw_hrefs), origin_rejected, locale_rejected, path_rejected, len(accepted_hrefs),
    #    )
    #
    # 4. accepted==0の場合、raw_hrefsをTelemetry JSONにダンプ
    #    # TODO (CR-ATELIER-002): When accepted==0, dump raw_hrefs to Telemetry JSON
    #    # for offline analysis.
    #    if not accepted_hrefs and self.telemetry and ctx.run_context:
    #        from app.agents.browser.telemetry import TelemetryContext
    #        tctx = TelemetryContext(
    #            site=ctx.site,
    #            query=ctx.query,
    #            run_id=getattr(ctx.run_context, "run_id", None),
    #            stage="plp"
    #        )
    #        await self.telemetry.save_json(
    #            "moncler_pdp_extraction_debug",
    #            {
    #                "raw_hrefs": raw_hrefs[:50],
    #                "rejection_stats": {
    #                    "origin_mismatch": origin_rejected,
    #                    "locale_mismatch": locale_rejected,
    #                    "path_mismatch": path_rejected,
    #                },
    #                "url": page.url,
    #            },
    #            tctx,
    #        )
    #
    # ==============================================================================

    async def run_deep_extraction(
        self,
        page: Page,
        site_config: dict[str, Any],
    ) -> list[str]:
        """
        深い抽出を実行する（骨組みのみ）

        Stage 3A-1: このステップでは、メソッドシグネチャのみ定義。
        実際のロジック移動は Stage 3A-2 で行います。

        Args:
            page: Playwright Page オブジェクト
            site_config: サイト設定

        Returns:
            List[str]: 抽出されたリンクのリスト（現時点では空リストを返す）
        """
        # Stage 3A-1: スタブ実装
        logger.debug("[NavigationDriver] run_deep_extraction (stub): returning empty list")
        return []

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

    def is_expected_locale_path(self, path: str, expected_locale: str) -> bool:
        """
        CR-E2E-003B拡張: Locale判定の共通化

        パスが期待ロケールで始まるか判定する。
        - ^/{expected_locale}(/|$) をTrueとする
        - ^/en-[a-z]{2}/{expected_locale}(/|$) をTrueとする（国コードprefix付き）

        Args:
            path: 検証対象パス
            expected_locale: 期待ロケール（例: "en-int"）

        Returns:
            bool: 期待ロケールで始まる場合True
        """
        if not path:
            return False
        path_lower = path.lower()
        expected_lower = expected_locale.lower()

        # パターン1: ^/{expected_locale}(/|$)
        if path_lower.startswith(f"/{expected_lower}/") or path_lower == f"/{expected_lower}":
            return True

        # パターン2: ^/en-[a-z]{2}/{expected_locale}(/|$)
        locale_prefix_pattern = re.compile(rf"^/en-[a-z]{{2}}/{expected_lower}(/|$)", re.I)
        return bool(locale_prefix_pattern.match(path_lower))

    def _is_locale_stable(
        self,
        url: str,
        site_config: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        CR-E2E-003B拡張: ロケールが安定しているか判定

        Args:
            url: 検証対象URL
            site_config: サイト設定（任意）

        Returns:
            (is_stable, diagnostics): 安定している場合True、診断情報
        """
        diagnostics = {
            "url": url,
            "path_ok": False,
            "country_ok": False,
            "reject_path_matched": False,
            "trap_pattern_matched": False,
            "stable": False,
        }

        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            path = parsed.path or ""
            query_params = parse_qs(parsed.query)

            # locale_policyから設定を取得
            locale_policy = {}
            if site_config:
                nav_cfg = site_config.get("navigation", {}) or {}
                locale_policy = nav_cfg.get("locale_policy", {}) or {}

            target_locale = locale_policy.get("target_locale", "en-int")
            target_country = locale_policy.get("target_country", "GB")
            reject_path_prefixes = locale_policy.get("reject_path_prefixes", [])
            trap_url_patterns = (nav_cfg.get("trap_url_patterns", []) if site_config else []) or []

            # パスチェック（CR-E2E-003B拡張: 共通関数を使用）
            path_ok = self.is_expected_locale_path(path, target_locale)
            diagnostics["path_ok"] = path_ok

            # 国チェック（CR-E2E-003B拡張: パスが/en-jp/で始まる場合はcountry_okを緩和）
            ship_to_country = query_params.get("shipToCountry", [])
            # パスが/en-jp/で始まる場合は、shipToCountryがJPでも許容
            path_starts_with_jp = path.lower().startswith("/en-jp/")
            if path_starts_with_jp:
                # /en-jp/の場合は、shipToCountryがJPまたはGBのどちらでもOK
                country_ok = (target_country in ship_to_country or "JP" in ship_to_country) if ship_to_country else True
            else:
                country_ok = target_country in ship_to_country if ship_to_country else False
            diagnostics["country_ok"] = country_ok

            # reject_path_prefixesチェック
            reject_path_matched = False
            for prefix in reject_path_prefixes:
                if path.lower().startswith(prefix.lower()):
                    reject_path_matched = True
                    break
            diagnostics["reject_path_matched"] = reject_path_matched

            # trap_url_patternsチェック（CR-E2E-003B拡張: /en-xx/{expected_locale}/ を例外化）
            trap_pattern_matched = False
            for pattern in trap_url_patterns:
                if pattern in path.lower():
                    # CR-E2E-003B拡張: /en-int/search は PLP として許容
                    if pattern == "/search" and path.lower().startswith(f"/{target_locale}/search"):
                        continue
                    # CR-E2E-003B拡張: /en-xx/{expected_locale}/ パターンをtrap判定から除外
                    if self.is_expected_locale_path(path, target_locale):
                        continue
                    trap_pattern_matched = True
                    break
            diagnostics["trap_pattern_matched"] = trap_pattern_matched

            # 安定判定
            stable = path_ok and country_ok and not reject_path_matched and not trap_pattern_matched
            diagnostics["stable"] = stable

            return stable, diagnostics
        except Exception as e:
            logger.warning(f"[LocaleGuard] _is_locale_stable failed: {e}", exc_info=True)
            diagnostics["error"] = str(e)
            return False, diagnostics

    async def _ensure_expected_locale(self, ctx: NavigationContext) -> None:
        """
        CR-ATELIER-002 Step 2: Locale Guard - ロケール一貫性チェックと自動修正
        CR-ATELIER-002 Step 5-3: Redirect / Locale 挙動の扱い整理
        CR-E2E-003B拡張: モーダル検出→国/言語選択→再矯正→再判定（最大3回）

        【責務】:
        - Pre-condition: Moncler の PLP/検索 URL
        - Post-condition:
          - page.url が /en-int/... で始まる
          - 「明らかな Trap（検索トップ / ロケールゲート / 404）」でないこと
          - 二重ロケールパターン（/en-lt/en-int/...）を検出して修正

        【Search ページの扱い】:
        - /en-int/search であっても、DOM 上に product tile が並んでいるなら PLP 同等として扱う
        - ただし、明らかな検索トップページ（検索ボックスのみ）は Trap として扱う

        【役割分担】:
        - LocaleGuard: 現在のページ自体を /en-int/...&shipToCountry=GB に揃える
        - TrapDetector: 明らかな Trap ページ（404、ロケールゲート、検索トップ）を検出
        - URL バリデーション: PDP 候補リンクをフィルタする

        Args:
            ctx: ナビゲーションコンテキスト
        """
        page = self.page
        site_config = ctx.site_config
        run_context = ctx.run_context

        # 診断情報を収集
        diagnostics = {
            "attempts": [],
            "http_errors": [],
            "final_url": None,
            "final_stable": False,
        }

        # HTTPレスポンスエラーを記録
        http_errors = []

        async def on_response(response):
            if response.status in [404, 410]:
                http_errors.append(
                    {
                        "url": response.url,
                        "status": response.status,
                        "timestamp": time.time(),
                    }
                )

        page.on("response", on_response)

        current_url = page.url or ""

        # MONCLER_OFFICIAL のみを対象とする
        # site_config のキーまたは ctx.site から site_code を取得
        site_code = site_config.get("site_code") or site_config.get("site") or ctx.site or ""
        if site_code != "MONCLER_OFFICIAL":
            logger.debug(f"[LocaleGuard] Skipping locale check for site: {site_code}")
            return

        # locale_policyから設定を取得
        nav_cfg = (site_config.get("navigation", {}) or {}) if site_config else {}
        locale_policy = nav_cfg.get("locale_policy", {}) or {}
        location_modal_cfg = nav_cfg.get("location_modal", {}) or {}

        target_locale = locale_policy.get("target_locale", "en-int")
        target_country = locale_policy.get("target_country", "GB")
        max_attempts = locale_policy.get("max_correction_attempts", 3)
        stability_check_delay_ms = locale_policy.get("stability_check_delay_ms", 2000)
        require_stable = locale_policy.get("require_stable_before_proceed", True)

        # スクリーンショット保存用のヘルパー
        async def save_screenshot(stage: str) -> str | None:
            """段階スクショを保存"""
            if not run_context:
                return None
            try:
                timestamp = int(time.time() * 1000)
                filename = f"locale_{stage}_{timestamp}.png"
                # RunContextはscreenshots_path属性を持つ
                path = run_context.screenshots_path / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(path))
                return str(path.relative_to(run_context.run_path))
            except Exception as e:
                logger.debug(f"[LocaleGuard] Failed to save screenshot {stage}: {e}")
                return None

        try:
            from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

            parsed = urlparse(current_url)
            path = parsed.path or ""
            query_params = parse_qs(parsed.query)

            # 期待値のチェック（CR-E2E-003B拡張: 共通関数を使用）
            expected_country = target_country

            # パスチェック（共通関数を使用）
            path_ok = self.is_expected_locale_path(path, target_locale)

            # クエリに `shipToCountry=GB` が含まれているかチェック（/en-jp/の場合は緩和）
            ship_to_country = query_params.get("shipToCountry", [])
            path_starts_with_jp = path.lower().startswith("/en-jp/")
            if path_starts_with_jp:
                # /en-jp/の場合は、shipToCountryがJPまたはGBのどちらでもOK
                country_ok = (
                    (expected_country in ship_to_country or "JP" in ship_to_country) if ship_to_country else True
                )
            else:
                country_ok = expected_country in ship_to_country if ship_to_country else False

            # CR-ATELIER-002 Step2: LocaleGuard - ensure Moncler stays on /en-int + shipToCountry=GB
            # 両方満たされている場合は何もしないが、INFOログを必ず出す
            if path_ok and country_ok:
                logger.info(f"[LocaleGuard] Checked locale, no change: {current_url}")
                # CR-E2E-003B拡張: 早期終了時にも診断情報を保存
                final_stable, final_diag = self._is_locale_stable(current_url, site_config)
                diagnostics["final_url"] = current_url
                diagnostics["final_stable"] = final_stable
                diagnostics["http_errors"] = http_errors
                diagnostics["attempts"] = [
                    {
                        "attempt": 0,
                        "url_before": current_url,
                        "url_after": current_url,
                        "stable_after": final_stable,
                        "stability_diagnostics": final_diag,
                        "note": "Locale already stable, no correction needed",
                    }
                ]
                if run_context:
                    try:
                        # RunContextはrun_path属性を持つ
                        diagnostics_path = run_context.run_path / "locale_diagnostics.json"
                        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(diagnostics_path, "w", encoding="utf-8") as f:
                            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
                        logger.info(f"[LocaleGuard] Saved locale diagnostics (stable case) to: {diagnostics_path}")
                    except Exception as e:
                        logger.warning(f"[LocaleGuard] Failed to save locale diagnostics: {e}", exc_info=True)
                else:
                    logger.warning("[LocaleGuard] run_context is None, cannot save locale diagnostics")
                return

            # ロケールがずれている場合、修正を試みる
            logger.warning(
                f"[LocaleGuard] Locale mismatch detected: path_ok={path_ok}, country_ok={country_ok}, URL={current_url}"
            )

            # 正しいPLP URLを取得
            # MonclerPLPStrategy の _HARD_PLP_URL を優先、なければ site_config から取得
            corrected_url = None

            # StrategyPlugin から取得を試みる
            if self.strategy and hasattr(self.strategy, "_HARD_PLP_URL"):
                corrected_url = self.strategy._HARD_PLP_URL
                logger.debug(f"[LocaleGuard] Using strategy _HARD_PLP_URL: {corrected_url}")

            # site_config から取得
            if not corrected_url:
                # plp_recovery の plp_hard_nav を確認
                nav_cfg = site_config.get("navigation", {}) or {}
                plp_recovery_cfg = nav_cfg.get("plp_recovery", {}) or {}
                corrected_url = (
                    plp_recovery_cfg.get("plp_hard_nav")
                    or site_config.get("seed_plp_url")
                    or site_config.get("fallback_url")
                    or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                    or site_config.get("home_url")
                )

            # それでも取得できない場合は、現在のURLを修正
            # CR-E2E-003B拡張: path_ok==Trueの場合は矯正URL生成をスキップ
            if not corrected_url and not path_ok:
                # CR-ATELIER-002 Step 4-2: 現在のURLを修正して `/en-int/` と `shipToCountry=GB` を付与
                # パスを正規化（冪等性を確保：既に期待ロケールが含まれる場合は二重挿入しない）
                if not self.is_expected_locale_path(path, target_locale):
                    # 既存のロケールセグメントを削除して `/en-int/` を追加
                    path_parts = [p for p in path.split("/") if p]
                    # ロケールセグメントをスキップ（二重ロケールも処理）
                    while path_parts and _LOCALE_SEG_RE.match(path_parts[0] or ""):
                        path_parts.pop(0)
                    # `/en-int/` を先頭に追加（既に含まれていない場合のみ）
                    if not any(target_locale.lower() in p.lower() for p in path_parts):
                        normalized_path = f"/{target_locale}/" + "/".join(path_parts)
                    else:
                        # 既に期待ロケールが含まれている場合は、先頭に追加しない
                        normalized_path = "/" + "/".join(path_parts)
                    if not normalized_path.endswith("/") and normalized_path != "/":
                        normalized_path += "/"
                else:
                    # 既に期待ロケールで始まっている場合は変更しない
                    normalized_path = path

                # クエリパラメータを修正（冪等性を確保：既に設定されている場合は追加しない）
                if "forceLocale" not in query_params or target_locale not in query_params.get("forceLocale", []):
                    query_params["forceLocale"] = [target_locale]
                if "shipToCountry" not in query_params or expected_country not in query_params.get("shipToCountry", []):
                    query_params["shipToCountry"] = [expected_country]
                # 既存のクエリパラメータも保持
                normalized_query = urlencode(query_params, doseq=True)

                corrected_url = urlunparse(
                    (parsed.scheme, parsed.netloc, normalized_path, parsed.params, normalized_query, parsed.fragment)
                )
                logger.debug(f"[LocaleGuard] Constructed corrected URL from current URL: {corrected_url}")

            # CR-E2E-003B拡張: モーダル検出→国/言語選択→再矯正→再判定（最大3回）
            attempt_count = 0
            while attempt_count < max_attempts:
                attempt_count += 1
                attempt_info = {
                    "attempt": attempt_count,
                    "url_before": page.url or "",
                    "modal_detected": False,
                    "modal_handled": False,
                    "navigation_performed": False,
                    "url_after": None,
                    "stable_after": False,
                }

                # beforeスクショを保存
                screenshot_before = await save_screenshot(f"before_attempt_{attempt_count}")
                if screenshot_before:
                    attempt_info["screenshot_before"] = screenshot_before

                # モーダル検出
                if location_modal_cfg.get("enabled", True):
                    detection_selectors = location_modal_cfg.get("detection_selectors", [])
                    for sel in detection_selectors:
                        try:
                            locator = page.locator(sel).first
                            # CR-E2E-003B拡張: 例外安全化（TimeoutErrorとCancelledErrorを捕捉）
                            try:
                                is_visible = await locator.is_visible(timeout=1000)
                            except (asyncio.CancelledError, Exception) as e:
                                # TimeoutError, CancelledError, その他の例外を捕捉
                                if isinstance(e, asyncio.CancelledError):
                                    logger.debug(f"[LocaleGuard] Modal detection cancelled for selector '{sel}'")
                                else:
                                    logger.debug(f"[LocaleGuard] Modal detection selector '{sel}' failed: {e}")
                                continue
                                if is_visible:
                                    attempt_info["modal_detected"] = True
                                    logger.info(f"[LocaleGuard] Location modal detected (attempt {attempt_count})")

                                    # modalスクショを保存
                                    screenshot_modal = await save_screenshot(f"modal_attempt_{attempt_count}")
                                    if screenshot_modal:
                                        attempt_info["screenshot_modal"] = screenshot_modal

                                    # 国/言語選択
                                    country_selectors = location_modal_cfg.get("country_selectors", [])
                                    location_selected = False
                                    for country_sel in country_selectors:
                                        try:
                                            country_locator = page.locator(country_sel).first
                                            # CR-E2E-003B拡張: 例外安全化
                                            try:
                                                country_visible = await country_locator.is_visible(timeout=1000)
                                            except (asyncio.CancelledError, Exception) as e:
                                                if isinstance(e, asyncio.CancelledError):
                                                    logger.debug(
                                                        f"[LocaleGuard] Country selection cancelled for selector '{country_sel}'"
                                                    )
                                                continue
                                                if country_visible:
                                                    await country_locator.click(timeout=3000)
                                                    wait_after = location_modal_cfg.get("wait_after_selection_ms", 2000)
                                                    await page.wait_for_timeout(wait_after)
                                                    location_selected = True
                                                    attempt_info["modal_handled"] = True
                                                    logger.info(
                                                        f"[LocaleGuard] Selected location using selector: {country_sel}"
                                                    )
                                                    break
                                        except Exception:
                                            continue

                                    if not location_selected:
                                        # モーダルを閉じる試行
                                        close_selectors = location_modal_cfg.get("close_selectors", [])
                                        for close_sel in close_selectors:
                                            try:
                                                close_locator = page.locator(close_sel).first
                                                # CR-E2E-003B拡張: 例外安全化
                                                try:
                                                    close_visible = await close_locator.is_visible(timeout=1000)
                                                except (asyncio.CancelledError, Exception) as e:
                                                    if isinstance(e, asyncio.CancelledError):
                                                        logger.debug(
                                                            f"[LocaleGuard] Close modal cancelled for selector '{close_sel}'"
                                                        )
                                                    continue
                                                    if close_visible:
                                                        await close_locator.click(timeout=2000)
                                                        await page.wait_for_timeout(1000)
                                                        break
                                            except Exception:
                                                continue
                                    break
                            except Exception as e:  # noqa: B025 — inner except catches Exception broadly, outer provides fallback logging
                                # is_visible()が失敗した場合は次のセレクタを試す
                                logger.debug(f"[LocaleGuard] Modal detection selector '{sel}' failed: {e}")
                                continue
                        except Exception as e:
                            logger.debug(f"[LocaleGuard] Modal detection failed for selector '{sel}': {e}")
                            continue
                # ロケール安定性チェック
                current_url_check = page.url or ""
                is_stable, stability_diag = self._is_locale_stable(current_url_check, site_config)
                attempt_info["url_after"] = current_url_check
                attempt_info["stable_after"] = is_stable
                attempt_info["stability_diagnostics"] = stability_diag

                if is_stable and require_stable:
                    logger.info(f"[LocaleGuard] Locale is stable after attempt {attempt_count}: {current_url_check}")
                    # afterスクショを保存
                    screenshot_after = await save_screenshot(f"after_attempt_{attempt_count}")
                    if screenshot_after:
                        attempt_info["screenshot_after"] = screenshot_after
                    diagnostics["attempts"].append(attempt_info)
                    diagnostics["final_url"] = current_url_check
                    diagnostics["final_stable"] = True
                    break

                # まだ安定していない場合、再矯正を試みる（CR-E2E-003B拡張: path_ok==Trueの場合はスキップ）
                if attempt_count < max_attempts:
                    # 現在のURLのpath_okをチェック
                    parsed_current_check = urlparse(current_url_check)
                    path_current_check = parsed_current_check.path or ""
                    path_ok_current = self.is_expected_locale_path(path_current_check, target_locale)

                    # path_ok==Trueの場合は矯正URL生成・navigateを実行しない
                    if path_ok_current:
                        logger.info(
                            f"[LocaleGuard] Attempt {attempt_count}: path_ok=True, skipping correction: {current_url_check}"
                        )
                        diagnostics["attempts"].append(attempt_info)
                        continue

                    # 再矯正URLを構築
                    parsed_current = urlparse(current_url_check)
                    path_current = parsed_current.path or ""
                    query_current = parse_qs(parsed_current.query)

                    # パスを正規化（冪等性を確保：既に期待ロケールが含まれる場合は二重挿入しない）
                    if not self.is_expected_locale_path(path_current, target_locale):
                        # ロケールが異なる場合のみ補正
                        path_parts = [p for p in path_current.split("/") if p]
                        while path_parts and _LOCALE_SEG_RE.match(path_parts[0] or ""):
                            path_parts.pop(0)
                        # 既に期待ロケールが含まれていない場合のみ追加
                        if not any(target_locale.lower() in p.lower() for p in path_parts):
                            normalized_path = f"/{target_locale}/" + "/".join(path_parts)
                        else:
                            normalized_path = "/" + "/".join(path_parts)
                        if not normalized_path.endswith("/") and normalized_path != "/":
                            normalized_path += "/"
                    else:
                        # 既に期待ロケールの場合は変更しない
                        normalized_path = path_current

                    # クエリパラメータを修正（冪等性を確保：既に設定されている場合は追加しない）
                    if "forceLocale" not in query_current or target_locale not in query_current.get("forceLocale", []):
                        query_current["forceLocale"] = [target_locale]
                    if "shipToCountry" not in query_current or target_country not in query_current.get(
                        "shipToCountry", []
                    ):
                        query_current["shipToCountry"] = [target_country]
                    normalized_query = urlencode(query_current, doseq=True)

                    corrected_url = urlunparse(
                        (
                            parsed_current.scheme,
                            parsed_current.netloc,
                            normalized_path,
                            parsed_current.params,
                            normalized_query,
                            parsed_current.fragment,
                        )
                    )

                    if corrected_url != current_url_check:
                        logger.warning(
                            f"[LocaleGuard] Attempt {attempt_count}: Navigating to corrected URL: {corrected_url}"
                        )
                        try:
                            await page.goto(corrected_url, wait_until="domcontentloaded", timeout=30000)
                            await page.wait_for_timeout(stability_check_delay_ms)
                            attempt_info["navigation_performed"] = True
                        except Exception as e:
                            logger.warning(f"[LocaleGuard] Navigation failed on attempt {attempt_count}: {e}")
                            attempt_info["navigation_error"] = str(e)

                diagnostics["attempts"].append(attempt_info)

            # 最終的な安定性チェック
            final_url = page.url or ""
            final_stable, final_diag = self._is_locale_stable(final_url, site_config)
            diagnostics["final_url"] = final_url
            diagnostics["final_stable"] = final_stable
            diagnostics["http_errors"] = http_errors

            # locale_diagnostics.jsonを保存
            if run_context:
                try:
                    # RunContextはrun_path属性を持つ
                    diagnostics_path = run_context.run_path / "locale_diagnostics.json"
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(diagnostics_path, "w", encoding="utf-8") as f:
                        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
                    logger.info(f"[LocaleGuard] Saved locale diagnostics to: {diagnostics_path}")
                except Exception as e:
                    logger.warning(f"[LocaleGuard] Failed to save locale diagnostics: {e}", exc_info=True)

            if not final_stable and require_stable:
                logger.warning(f"[LocaleGuard] Locale is not stable after {max_attempts} attempts: {final_url}")

            # 旧コードの残り部分は削除（上記のwhileループで実装済み）

        except (asyncio.CancelledError, Exception) as e:
            # CR-E2E-003B拡張: 例外安全化（CancelledErrorとTimeoutErrorを捕捉）
            if isinstance(e, asyncio.CancelledError):
                logger.warning(f"[LocaleGuard] Locale correction cancelled: {e}")
            else:
                logger.warning(f"[LocaleGuard] Failed to normalize locale: {e}", exc_info=True)
            # 例外発生時でも診断情報を保存
            diagnostics["final_url"] = page.url if hasattr(self, "page") and self.page else current_url
            diagnostics["final_stable"] = False
            diagnostics["error"] = str(e)
        finally:
            # CR-E2E-003B拡張: finallyで診断ファイルを確実に保存（120sタイムアウト時でも残る）
            if run_context:
                try:
                    diagnostics_path = run_context.run_path / "locale_diagnostics.json"
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(diagnostics_path, "w", encoding="utf-8") as f:
                        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
                    logger.info(f"[LocaleGuard] Saved locale diagnostics (finally) to: {diagnostics_path}")
                except Exception as e:
                    logger.error(f"[LocaleGuard] Failed to save locale diagnostics in finally: {e}", exc_info=True)

    # --- 内部用の private メソッド（型だけ定義） ---
    # Stage 3A-1: これらのメソッドは、Stage 3A-2 で実装を移行します

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

    def _looks_like_trap_or_legal(self, url: str, site_config: dict[str, Any] | None = None) -> bool:
        """
        Stage 3A-2-3:
        旧 BrowserUseAgent._looks_like_trap_or_legal のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持すること。

        明らかに商品一覧ではなく、法務/クッキー/ヘルプ系に飛ばされてると判断したら True。
        こういうページに張り付いてもPDPは取れないので、早期abortさせる。

        ★ V88.5.9: 先に軽量正規化を行ってから判定する。
        - /en-jp/en-int/ を /en-int/ に置換
        - #product-information-panel 等のハッシュを除去

        Stage 3A-2-5: site_config から trap_url_patterns と legal_url_patterns を取得
        """
        try:
            # Stage 4: 二重ロケールの早期修正（site_configに基づく）
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            normalize_double_locale = locale_cfg.get("normalize_double_locale", False)

            if normalize_double_locale:
                sp = urlsplit(url)
                path = sp.path or ""
                double_locale_patterns = locale_cfg.get("double_locale_patterns", [])
                for pattern in double_locale_patterns:
                    from_pattern = pattern.get("from", "")
                    to_pattern = pattern.get("to", "")
                    if from_pattern and to_pattern:
                        path = path.replace(from_pattern, to_pattern)
                # "PDPアンカー"などのハッシュは評価前に捨てる
                sp = sp._replace(path=path, fragment="")
                url = urlunsplit(sp)
        except Exception:
            pass  # 正規化に失敗しても、元のURLで判定を続行

        try:
            u = urlparse(url)
            full_lower = url.lower()
            path_lower = (u.path or "").lower()
            host = (u.netloc or "").lower()

            # Stage 3A-2-5: site_config から trap パターンを取得
            nav_cfg = (site_config or {}).get("navigation", {}) or {}
            trap_patterns = nav_cfg.get("trap_url_patterns", [])
            legal_patterns = nav_cfg.get("legal_url_patterns", [])

            # Stage 4: site_config から trap パターンをチェック（Moncler固有ロジックを削除）
            if trap_patterns and any(pattern.lower() in full_lower for pattern in trap_patterns):
                logger.warning(f"[_looks_like_trap] Detected trap pattern: {url}")
                return True

            if legal_patterns and any(pattern.lower() in path_lower for pattern in legal_patterns):
                logger.warning(f"[_looks_like_trap] Detected legal pattern: {url}")
                return True

            # trap_domains をチェック
            trap_domains = nav_cfg.get("trap_domains", [])
            if trap_domains and any(domain.lower() in host for domain in trap_domains):
                logger.warning(f"[_looks_like_trap] Detected trap domain: {url}")
                return True

            # locale_gate_detection をチェック
            locale_gate_cfg = nav_cfg.get("locale_gate_detection", {}) or {}
            if locale_gate_cfg.get("enabled", False):
                target_locale = locale_gate_cfg.get("target_locale", "")
                gate_paths = locale_gate_cfg.get("gate_paths", [])
                if target_locale and gate_paths:
                    # ホストがallowed_domainに一致し、パスがgate_pathsに一致する場合
                    allowed_domain = site_config.get("allowed_domain", "")
                    if allowed_domain and allowed_domain.lower() in host:
                        if path_lower in [p.lower() for p in gate_paths]:
                            logger.warning(f"[_looks_like_trap] Detected locale gate: {url}")
                            return True

            # デフォルトのリーガルキーワード（site_configに定義がない場合のフォールバック）
            # ただし、これは最小限に抑える
            default_legal_keywords = ["/cookie-policy", "/privacy", "/legal", "/help", "/account", "/login"]
            if not legal_patterns:  # site_configにlegal_patternsが定義されていない場合のみ
                if any(kw in path_lower for kw in default_legal_keywords):
                    logger.warning(f"[_looks_like_trap] Detected default legal keyword: {url}")
                    return True

            return False

        except Exception:
            return False

    # Stage 3A-2-1: ヘルパーメソッド（BrowserUseAgent から移植）
    def _normalize_abs_url(self, base_url: str, href: str) -> str:
        """
        CR-E2E-003A拡張: URLを絶対URLに正規化。navigation_helpers に委譲可能なら委譲。
        プロトコル相対・スキーム除外は自前で処理。
        """
        if not href:
            return ""
        if nav_normalize_abs_url is not None:
            try:
                out = nav_normalize_abs_url(base_url, href)
                if out:
                    parsed = urlparse(out)
                    if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
                        pass  # スキーム除外は下の自前処理へ
                    else:
                        return out
            except Exception:
                pass
        try:
            if href.startswith("//"):
                base_parsed = urlparse(base_url)
                href = f"{base_parsed.scheme}:{href}"
            absu = urljoin(base_url, href)
            parsed = urlparse(absu)
            if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
                return href
            parts = list(urlsplit(absu))
            if parts[2].endswith("/"):
                parts[2] = parts[2].rstrip("/")
            parts[3] = ""
            parts[4] = ""
            return urlunsplit(parts)
        except Exception:
            return href

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

    async def _run_deep_extraction_phase2(self, page: Page, site_config: dict[str, Any]) -> list[str]:
        """
        Stage 3A-2-1:
        旧 BrowserUseAgent._run_deep_extraction_phase2 のロジックをここに移す。
        挙動・ログ・例外の流れはそのまま維持すること。

        Deep Extraction Phase 2: JSON-LD, onclick, data-* 属性からリンクを抽出する
        """
        logger.debug("[Phase 2] Running deep extraction (JSON-LD, onclick, data-*, ...)")
        # ★ 88.6.2: (BugFix) 括弧が過剰だった SyntaxError を修正
        container_sels: list[str] = ((site_config.get("selectors") or {}).get("pdp") or {}).get(
            "plp_container_selectors", []
        ) or []
        for cont in container_sels or []:
            await self.safe_wait_selector(page, cont, timeout_ms=1000, state="visible")
        try:
            for _ in range(2):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(200)
        except Exception:
            pass

        # V86.0: Strict mode violation prevention + V88.2: Get ElementHandle
        # Stage 4: タイムアウト処理改善（CancelledErrorを適切に処理）
        scope = page.locator("main, [role='main'], #main, #app")
        handle: ElementHandle | None = None
        try:
            # タイムアウトを短くして、CancelledErrorを避ける
            # ただし、asyncio.wait_forでラップすると、タイムアウト時にCancelledErrorが発生する可能性がある
            # そのため、直接wait_forを呼び出し、タイムアウトエラーをキャッチする
            try:
                await scope.first.wait_for(state="attached", timeout=4000)
                handle = await scope.first.element_handle(timeout=4000)
            except asyncio.CancelledError:
                logger.debug("[Phase 2] Cancelled while waiting for scope")
                raise
            except Exception as e_handle:
                # (V88.6.2: ログレベルは warning のまま)
                logger.warning(
                    f"[Phase 2] Could not get element handle for scope: {e_handle}. Falling back to page evaluate."
                )
                handle = None  # Ensure handle is None if getting it failed
        except asyncio.CancelledError:
            logger.debug("[Phase 2] Cancelled in outer try block")
            raise
        except Exception as e_outer:
            logger.warning(f"[Phase 2] Outer exception: {e_outer}. Falling back to page evaluate.")
            handle = None

        # JS Script that takes an optional node context
        _js_script = """
          (node) => {
            const area = node || document;
            const out = [];
            const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
            area.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
              const a = el.closest("a") || el;
              const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
              if (cand) push(cand);
            });
            area.querySelectorAll("[onclick]").forEach(el => {
              const oc = el.getAttribute("onclick") || "";
              const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
              if (m1 && m1[1]) push(m1[1]);
              const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
              if (m2 && m2[1]) push(m2[1]);
            });
            area.querySelectorAll("script[type='application/ld+json']").forEach(s => {
              try {
                const data = JSON.parse(s.textContent || "null");
                const arr = Array.isArray(data) ? data : [data];
                const pushAny = (v) => { if (v && typeof v === "string") push(v); };
                arr.forEach(d => {
                  if (!d || typeof d !== "object") return;
                  pushAny(d.url || d['@id']);
                  if (Array.isArray(d.offers)) {
                    d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                  } else if (d.offers && typeof d.offers === "object") {
                    pushAny(d.offers.url || d.offers['@id']);
                  }
                  if (d.itemListElement && Array.isArray(d.itemListElement)) {
                    d.itemListElement.forEach(it => {
                      if (it && it.item && (it.item.url || it.item['@id'])) {
                        pushAny(it.item.url || it.item['@id']);
                      }
                    });
                  }
                });
              } catch(e) {}
            });
            return out.filter(Boolean);
          }
        """

        hrefs: list[str] = []
        try:
            if handle:
                # Execute JS within the specific element context
                hrefs = await handle.evaluate(_js_script)
                logger.debug("[Phase 2] Deep extraction performed using element handle.")
            else:
                # --- V88.3.0: Safer Fallback ---
                # ドキュメント全体を対象にした無引数版を直接渡す
                hrefs = await page.evaluate("""
                  () => {
                    const out = [];
                    const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
                    document.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
                      const a = el.closest("a") || el;
                      const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
                      if (cand) push(cand);
                    });
                    document.querySelectorAll("[onclick]").forEach(el => {
                      const oc = el.getAttribute("onclick") || "";
                      const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
                      if (m1 && m1[1]) push(m1[1]);
                      const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
                      if (m2 && m2[1]) push(m2[1]);
                    });
                    document.querySelectorAll("script[type='application/ld+json']").forEach(s => {
                      try {
                        const data = JSON.parse(s.textContent || "null");
                        const arr = Array.isArray(data) ? data : [data];
                        const pushAny = (v) => { if (v && typeof v === "string") push(v); };
                        arr.forEach(d => {
                          if (!d || typeof d !== "object") return;
                          pushAny(d.url || d['@id']);
                          if (Array.isArray(d.offers)) {
                            d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                          } else if (d.offers && typeof d.offers === "object") {
                            pushAny(d.offers.url || d.offers['@id']);
                          }
                          if (d.itemListElement && Array.isArray(d.itemListElement)) {
                            d.itemListElement.forEach(it => {
                              if (it && it.item && (it.item.url || it.item['@id'])) {
                                pushAny(it.item.url || it.item['@id']);
                            }
                            });
                          }
                        });
                      } catch(e) {}
                    });
                    return out.filter(Boolean);
                  }
                """)
                logger.debug("[Phase 2] Deep extraction performed using page evaluate (fallback).")
                # --- V88.3.0 修正ここまで ---
        except Exception as e:
            logger.warning(f"[Phase 2] Deep extraction evaluate failed: {e}")
            hrefs = []  # Ensure hrefs is a list even on error

        return _dedupe_keep_order(hrefs)

    # Stage 3A-2-2: ヘルパーメソッド（BrowserUseAgent から移植）
    def _time_left_ms(self, start_t: float, budget_ms: int) -> int:
        """残り時間をミリ秒で返す"""
        used = int((time.monotonic() - start_t) * 1000)
        return max(0, budget_ms - used)

    def _normalize_url(self, url: str, site_config: dict[str, Any]) -> str:
        """
        Stage 4: URLを汎用的に正規化する

        site_config["locale"]["normalize_rules"] と force_query_params を使用して
        ロケール正規化とクエリパラメータの追加を行う。
        /en-int/ などのハードコードを排除。
        """
        u = urlparse(url)
        path = (u.path or "/").replace("//", "/")

        # site_configからロケール設定を取得（既存設定との互換性を確保）
        locale_cfg = site_config.get("locale", {}) or {}

        # normalize_rules: locale.normalize_rules を優先、なければルートレベルの normalize_rules を参照
        normalize_rules = locale_cfg.get("normalize_rules", [])
        if not normalize_rules:
            normalize_rules = site_config.get("normalize_rules", [])

        # replace_rules も normalize_rules として扱う（既存設定との互換性）
        if not normalize_rules:
            replace_rules = locale_cfg.get("replace_rules", [])
            if replace_rules:
                normalize_rules = [{"from": r.get("from", ""), "to": r.get("to", "")} for r in replace_rules]

        prefer_locale = locale_cfg.get("prefer", None)

        # normalize_double_locale: フラグがない場合は replace_rules の存在で判断
        normalize_double_locale = locale_cfg.get("normalize_double_locale", False)
        if not normalize_double_locale and locale_cfg.get("replace_rules"):
            normalize_double_locale = True

        # 二重ロケールの正規化（例: /en-jp/en-int/ → /en-int/）
        if normalize_double_locale:
            double_locale_patterns = locale_cfg.get("double_locale_patterns", [])
            # replace_rules を double_locale_patterns として扱う（既存設定との互換性）
            if not double_locale_patterns:
                replace_rules = locale_cfg.get("replace_rules", [])
                if replace_rules:
                    double_locale_patterns = [{"from": r.get("from", ""), "to": r.get("to", "")} for r in replace_rules]

            for pattern in double_locale_patterns:
                from_pattern = pattern.get("from", "")
                to_pattern = pattern.get("to", "")
                if from_pattern and to_pattern:
                    path = path.replace(from_pattern, to_pattern)

        # normalize_rules を適用
        for rule in normalize_rules:
            # 既存の normalize_rules 形式（if_url_contains/replace）にも対応
            if "if_url_contains" in rule and "replace" in rule:
                if_url_contains = rule.get("if_url_contains", "")
                replace_dict = rule.get("replace", {})
                if if_url_contains in path:
                    for from_pattern, to_pattern in replace_dict.items():
                        path = path.replace(from_pattern, to_pattern)
            else:
                # 標準形式（from/to）
                from_pattern = rule.get("from", "")
                to_pattern = rule.get("to", "")
                if from_pattern and to_pattern:
                    path = path.replace(from_pattern, to_pattern)

        # ロケールセグメントの処理
        if prefer_locale:
            seg = [s for s in path.split("/") if s]
            i = 0
            # 先頭のロケールセグメントをスキップ
            while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""):
                i += 1
            # 既存のprefer_localeを削除してから追加
            seg = [s for s in seg[i:] if s.lower() != prefer_locale.lower()]
            norm = f"/{prefer_locale}/" + "/".join(seg)
        else:
            norm = path

        if not norm.endswith("/") and norm != "/":
            norm += "/"

        # クエリパラメータの処理
        q = dict(parse_qsl(u.query))

        # force_query_params を追加（既存設定との互換性を確保）
        force_params = locale_cfg.get("force_query_params", {})
        # discovery_settings.force_query_params も参照（既存設定との互換性）
        if not force_params:
            ds = site_config.get("discovery_settings", {}) or {}
            force_params = ds.get("force_query_params", {})
        if force_params:
            q.update(force_params)

        # ensure_params の処理（normalize_rules内のensure_params）
        for rule in normalize_rules:
            ensure_params = rule.get("ensure_params", {})
            if ensure_params:
                q.update(ensure_params)

        # URLを再構築
        if q:
            from urllib.parse import urlencode

            norm += "?" + urlencode(q)

        return f"{u.scheme}://{u.netloc}{norm}"

    async def _accept_cookies_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        """Cookie 同意バナーがあればクリックする。ui_helpers に委譲。"""
        if ui_accept_cookies_if_present is not None:
            return await ui_accept_cookies_if_present(page, site_config)
        return False

    async def _dismiss_geo_modal(self, page: Page, site_config: dict[str, Any] | None = None) -> None:
        """ジオ / ロケール関係のモーダルを潰す"""
        # Stage 3A-2-5: site_config["navigation"]["overlays"]["geo_modal_selectors"] から取得
        geo_selectors = []
        if site_config:
            nav_cfg = site_config.get("navigation", {}) or {}
            overlays_cfg = nav_cfg.get("overlays", {}) or {}
            geo_selectors = overlays_cfg.get("geo_modal_selectors", [])

        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not geo_selectors:
            geo_selectors = [
                "text=STAY HERE",
                "text=REMAIN HERE",
                "text=REMAIN IN ENGLISH",
                "text=CONTINUE SHOPPING",
                "text=ショッピングを続ける",
            ]

        for sel in geo_selectors:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.click(timeout=3000)
                    break
            except Exception:
                continue

        async def _click_first(loc: Locator, desc: str) -> bool:
            try:
                if await loc.count() == 0:
                    return False
                target = loc.first
                try:
                    if not await target.is_visible(timeout=2000):
                        await target.scroll_into_view_if_needed(timeout=3000)
                except Exception as scroll_e:
                    logger.debug(f"[GeoModal] scroll_into_view failed ({desc}): {scroll_e}, skipping")
                try:
                    await target.click(timeout=5000)
                    logger.info(f"[GeoModal] Clicked {desc}")
                    await page.wait_for_timeout(300)
                    return True
                except Exception as click_e:
                    logger.debug(f"[GeoModal] Click failed ({desc}): {click_e}")
                    return False
            except Exception as e:
                logger.debug(f"[GeoModal] Click failed ({desc}): {e}")
                return False

        async def _wait_for_target_locale(timeout_ms: int = 4000) -> bool:
            """Stage 4: ターゲットロケールへの遷移を待つ（汎用化）"""
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "")

            if not prefer_locale:
                # ロケール設定がない場合は常にTrueを返す
                return True

            try:
                # ターゲットロケールがURLに含まれているかチェック
                locale_path = f"/{prefer_locale}/"
                await page.wait_for_function(
                    f"() => location.href.includes('{locale_path}')",
                    timeout=timeout_ms,
                )
                return True
            except Exception:
                return locale_path in (page.url or "").lower()

        try:
            # Stage 4: 汎用的なロケールゲートヘッダーの検出
            header = page.locator("text=Select your location").first
            header_visible = await header.count() > 0
            if header_visible:
                logger.info("[GeoModal] Locale gate header detected.")

            # Stage 4: site_configから優先ロケールを取得（汎用化）
            geo_modal_preferred_locale = overlays_cfg.get("geo_modal_preferred_locale", "")
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", geo_modal_preferred_locale)

            # 優先ロケールに基づく候補セレクタ（デフォルトはen-gb）
            if prefer_locale and "gb" in prefer_locale.lower():
                # United Kingdom / English の候補
                preferred_candidates = [
                    page.get_by_text(re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("link", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"United\s+Kingdom.*English", re.I)),
                    page.get_by_role("link", name=re.compile(r"United\s+Kingdom.*English", re.I)),
                    page.locator("[data-testid*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("[data-component*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("button:has-text('United Kingdom EN')"),
                    page.locator("text=/United\\s+Kingdom\\s*\\|\\s*English/i"),
                ]
            else:
                # その他のロケールの場合は、geo_modal_selectorsを使用
                preferred_candidates = []
                for sel in geo_selectors:
                    try:
                        preferred_candidates.append(page.locator(sel).first)
                    except Exception:
                        continue

            for loc in preferred_candidates:
                if await _click_first(loc, f"Preferred locale selector ({prefer_locale})"):
                    if await _wait_for_target_locale():
                        return
                    break

            close_candidates = [
                page.locator("button[aria-label*='close' i]"),
                page.locator("button:has-text('Close')"),
                page.locator(".modal__close, .c-modal__close"),
                page.locator("[data-testid*='close' i]"),
                page.locator("div[data-editorial-component='ticker-top-banner'] button[aria-label*='close' i]"),
            ]
            for loc in close_candidates:
                if await _click_first(loc, "locale gate close button"):
                    if await _wait_for_target_locale():
                        return
                    break
        except Exception as e:
            logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

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

    async def _force_plp_recover(self, page: Page, site_config: dict[str, Any], target_url: str | None) -> None:
        """
        Stage 4: PLP 回復（汎用化）
        site_config["navigation"]["plp_recovery"] と discovery_settings.fallback_url を使用
        """
        try:
            # site_configからPLP回復設定を取得
            nav_cfg = site_config.get("navigation", {}) or {}
            plp_recovery_cfg = nav_cfg.get("plp_recovery", {}) or {}
            recovery_enabled = plp_recovery_cfg.get("enabled", True)

            if not recovery_enabled:
                logger.debug("[recover] PLP recovery is disabled in site_config")
                return

            # PLP URL候補の取得（優先順位: target_url > plp_hard_nav > seed_plp_url > fallback_url > discovery_settings.fallback_url > home_url）
            plp = (
                target_url
                or site_config.get("plp_hard_nav")
                or site_config.get("seed_plp_url")
                or site_config.get("fallback_url")
                or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                or plp_recovery_cfg.get("fallback_url")
                or site_config.get("home_url")
            )

            if not plp:
                logger.debug("[recover] no PLP candidate found; skip")
                return

            # ロケール正規化（site_configに基づく）
            normalize_locale = plp_recovery_cfg.get("normalize_locale", True)
            if normalize_locale:
                plp = self._normalize_url(plp, site_config)

            logger.info("[recover] Forcing PLP navigation: %s", plp)
            await page.goto(url=plp, wait_until="domcontentloaded")
        except Exception as e:
            logger.debug("[recover] force PLP failed: %r", e)

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
                # 現在のURLがターゲットロケールを含まない場合
                target_locale_path = f"/{prefer_locale}/"
                if allowed_domain.lower() in current_url and target_locale_path not in current_url:
                    # ロケールセグメントが含まれているかチェック
                    if _LOCALE_SEG_RE.search(current_url):
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

    async def _click_and_capture_navigation(
        self,
        click_coro: Callable,
        page: Page,
        context: BrowserContext,
        *,
        url_regex: re.Pattern | None = None,
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Page | None:
        """
        Stage 3A-2-4:
        クリック操作を実行し、ナビゲーション（ポップアップ、SPA遷移など）をキャプチャする。
        """
        if url_regex is None:
            url_regex = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

        popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
        same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))
        spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms)) if url_regex else None
        sel_spa = (
            ", ".join(VISIBLE_PRICE_SELECTORS)
            if VISIBLE_PRICE_SELECTORS
            else "[itemprop=price],[class*=price],[data-testid*=price]"
        )
        spa_price_task = asyncio.create_task(page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms))

        try:
            await click_coro()
        except Exception:
            for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
                if t and not t.done():
                    t.cancel()
            return None

        tasks = {popup_task, same_tab_nav_task, spa_price_task}
        if spa_url_task:
            tasks.add(spa_url_task)

        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout_ms / 1000, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if not done:
                return None
            winner = next(iter(done))
            new_page = winner.result() if winner is popup_task else page
            log_msg = (
                "popup"
                if winner is popup_task
                else "framenav"
                if winner is same_tab_nav_task
                else "SPA URL"
                if winner is spa_url_task
                else "SPA Price"
            )
            logger.debug(f"[_click_and_capture] Winner: {log_msg}")
            try:
                if new_page.url == "about:blank":
                    await new_page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Exception as e_blank:
                logger.debug(f"[_click_and_capture] Wait for about:blank failed: {e_blank}")
            with contextlib.suppress(Exception):
                await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10))
            if url_regex:
                try:
                    await new_page.wait_for_url(url_regex, timeout=max(1000, timeout_ms // 4))
                except Exception as e_url_final:
                    logger.debug(f"[_click_and_capture] Final wait_for_url failed: {e_url_final}")
            return new_page
        except Exception as e_wait:
            logger.warning(f"[_click_and_capture] Nav race failed: {e_wait}")
            return None
        finally:
            for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
                if t and not t.done():
                    t.cancel()

    async def header_search_fallback(self, ctx: NavigationContext) -> bool:
        """
        Stage 3A-2-4:
        旧 BrowserUseAgent._plp_header_search_fallback のロジックをここに移行。
        PLP の検索UIを使って query を再投入し、PLP を再構成する fallback。
        成功したら True, 変化なしなら False。
        """
        page = self.page
        site_config = ctx.site_config
        query = ctx.query
        start_t = ctx.start_t
        budget_ms = ctx.budget_ms

        # Stage 3A-2-5: site_config["navigation"]["header_search"] から取得
        nav_cfg = site_config.get("navigation", {}) or {}
        hs_cfg = nav_cfg.get("header_search", {}) or {}

        # フォールバック: 既存の ui 構造もサポート
        ui = (site_config.get("selectors", {}) or {}).get("ui", {}) or {}

        # Stage 4: 文字列とリストの両方に対応（site_configで文字列が設定されている場合がある）
        def _ensure_list(value):
            """文字列をリストに変換、リストはそのまま、None/空は空リスト"""
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, list):
                return value
            return []

        sel_open = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("search_open_selector"))
            + _ensure_list(ui.get("search_open"))
            + [
                "button[aria-label='Search']",
                "[aria-label*='Search' i]",
            ]
        )
        sel_input = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("search_input_selector"))
            + _ensure_list(ui.get("search_input"))
            + [
                "form[role='search'] input",
                "input[type='search']",
                "input[name='q']",
                "[data-testid*='search' i] input",
                "[role='search'] input",
                "dialog input[type='search']",
            ]
        )
        sel_submit = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("submit_selector"))
            + _ensure_list(ui.get("search_submit"))
            + ["form[role='search'] button[type='submit']"]
        )
        clear_before_type = hs_cfg.get("clear_before_type", True)

        try:
            opened = False
            for s in sel_open:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0:
                    await el.click(timeout=3000)
                    opened = True
                    await asyncio.sleep(0.2)
                    await self.safe_wait_selector(
                        page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible"
                    )
                    logger.debug(f"[Fallback] opened search with '{s}'")
                    break
            if not opened:
                await page.keyboard.press("/")
                await self.safe_wait_selector(page, "input[type='search']", timeout_ms=5000, state="visible")
            found_input = False
            for s in sel_input:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_visible():
                    # Stage 3A-2-5: clear_before_type が True の場合は先にクリア
                    if clear_before_type:
                        await el.clear(timeout=2000)
                    await el.fill(query, timeout=8000)
                    found_input = True
                    logger.debug(f"[Fallback] filled '{query}' into '{s}'")
                    break
            if not found_input:
                raise ValueError("Input field not found")
            submitted = False
            for s in sel_submit:
                if self._time_left_ms(start_t, budget_ms) <= 0:
                    break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_enabled():
                    await el.click(timeout=5000)
                    submitted = True
                    logger.debug(f"[Fallback] submitted with '{s}'")
                    break
            if not submitted:
                await page.keyboard.press("Enter")
                logger.debug("[Fallback] submitted with Enter key.")
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    logger.debug("[Fallback] Optional main wait timed out.")
            return True
        except Exception:
            logger.warning("[Fallback] UI search failed. Trying direct search URL.")
            try:
                # Stage 4: site_configから検索URLテンプレートを取得（既存設定との互換性を確保）
                url_template = hs_cfg.get("url_template", "")
                base_url_key = hs_cfg.get("base_url", "home_url")

                # url_templateがリストの場合は最初の要素を使用
                if isinstance(url_template, list):
                    url_template = url_template[0] if url_template else ""

                if not url_template:
                    # フォールバック: discovery_settingsから取得（既存設定との互換性）
                    ds = site_config.get("discovery_settings", {}) or {}
                    url_templates = ds.get("url_templates", {}) or {}
                    url_template = url_templates.get("search", "")
                    # リストの場合は最初の要素を使用
                    if isinstance(url_template, list):
                        url_template = url_template[0] if url_template else ""

                if not url_template or not isinstance(url_template, str):
                    logger.warning("[Fallback] No valid search URL template found in site_config")
                    return False

                # ベースURLを取得
                if base_url_key == "home_url":
                    base_url = site_config.get("home_url", "")
                else:
                    base_url = site_config.get(base_url_key, site_config.get("home_url", ""))

                # base_urlがリストの場合は最初の要素を使用
                if isinstance(base_url, list):
                    base_url = base_url[0] if base_url else ""

                if not base_url or not isinstance(base_url, str):
                    logger.warning("[Fallback] No valid base URL found in site_config")
                    return False

                # URLテンプレートのプレースホルダを置換
                # {query} を置換
                search_url = url_template.replace("{query}", quote_plus(query))

                # {locale} を置換（locale.preferを使用）
                locale_cfg = site_config.get("locale", {}) or {}
                prefer_locale = locale_cfg.get("prefer", "")
                if "{locale}" in search_url and prefer_locale:
                    search_url = search_url.replace("{locale}", prefer_locale)

                # 相対URLの場合はbase_urlと結合
                if not search_url.startswith("http"):
                    from urllib.parse import urljoin

                    search_url = urljoin(base_url, search_url)

                logger.info(f"[Fallback] Using search URL from site_config: {search_url}")
                await page.goto(url=search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    logger.debug("[Fallback] Optional main wait (URL) timed out.")
                return True
            except Exception as final_e:
                logger.error(f"[Fallback] Direct search URL failed: {final_e}")
                return False

    async def click_first_card_or_link(self, ctx: NavigationContext) -> str | None:
        """
        Stage 3A-2-4:
        旧 BrowserUseAgent._click_first_card_or_link のロジックをここに移行。
        PLP 上の最初のカード/リンクをクリックして PDP へ遷移し、遷移先 URL を返す。ダメなら None。
        """
        page = self.page
        site_config = ctx.site_config
        context = ctx.context

        if context is None:
            logger.warning("[Fallback:click-card] BrowserContext is not available")
            return None

        # Stage 3A-2-5: site_config["navigation"]["fallback"]["click_first_card"] から取得
        nav_cfg = site_config.get("navigation", {}) or {}
        fb_cfg = nav_cfg.get("fallback", {}) or {}
        click_cfg = fb_cfg.get("click_first_card", {}) or {}

        # フォールバック: 既存の pdp 構造もサポート
        pdp = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}

        # enabled が False の場合はスキップ
        if not click_cfg.get("enabled", True):
            logger.debug("[Fallback:click-card] click_first_card is disabled in site_config")
            return None

        # card_selectors を取得（優先順位: navigation.fallback.click_first_card.card_selectors > selectors.plp.card_selectors > selectors.pdp.pdp_link_selectors）
        link_sel = _dedupe_keep_order(
            (click_cfg.get("card_selectors", []) or [])
            + (plp_selectors.get("card_selectors", []) or [])
            + (pdp.get("pdp_link_selectors", []) or [])
        )

        plp_boxes = _dedupe_keep_order(
            (plp_selectors.get("container_selectors", []) or [])
            + (pdp.get("plp_container_selectors", []) or [])
            + (["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"])
        )
        block_ng = set(
            (click_cfg.get("blocklist_href_substrings", []) or [])
            + (pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
        )
        url_pat = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

        if link_sel:
            for s in link_sel:
                try:
                    loc = page.locator(s)
                    count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i)
                        href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed()
                            newp = await self._click_and_capture_navigation(
                                lambda: el.click(timeout=5000), page, context, url_regex=url_pat
                            )
                            if newp:
                                return newp.url
                except Exception:
                    continue

        # CR-E2E-003B拡張: MonclerPLPStrategyが検出できているセレクタを優先
        # site_configからtile_selectorsを取得（Moncler向けに最適化）
        tile_selectors_from_config = plp_selectors.get("tile_selectors", []) or []
        tile_selectors = tile_selectors_from_config + [
            "[data-qa='product-tile']",
            "[data-testid='product-tile']",
            "div[data-testid='product-tile']",
            "div[class*='product-tile' i]",
            "div.product-tile",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]",
        ]
        # 重複を除去
        tile_selectors = _dedupe_keep_order(tile_selectors)

        # まず、box スコープなしで直接 tile_selectors を試す
        for tile_sel in tile_selectors:
            try:
                card = page.locator(tile_sel).first
                # 要素が存在するか確認（CR-E2E-003B拡張: タイムアウトを5000msに延長）
                try:
                    await card.wait_for(state="attached", timeout=5000)
                except Exception as e:
                    logger.debug(f"[Fallback:click-card] wait_for failed for '{tile_sel}': {e}")
                    continue

                count = await card.count()
                if count > 0:
                    await card.scroll_into_view_if_needed(timeout=3000)
                    newp = await self._click_and_capture_navigation(
                        lambda: card.click(timeout=5000), page, context, url_regex=url_pat
                    )
                    if newp:
                        return newp.url
            except Exception as e:
                logger.debug(f"[Fallback:click-card] Selector '{tile_sel}' failed: {e}")
                continue

        # box スコープ付きで試す（フォールバック）
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    # セレクタを組み立て
                    selector = f"{box} {tile_sel}".strip()
                    card = page.locator(selector).first

                    # 要素が存在するか確認（タイムアウトを短く設定）
                    # Stage 4: タイムアウトエラーを適切に処理（CancelledErrorを避ける）
                    try:
                        count = await card.count()
                        if count == 0:
                            continue
                        # 短いタイムアウトで確認（要素が存在する場合のみ）
                        await asyncio.wait_for(card.wait_for(state="attached", timeout=2000), timeout=2.5)
                    except asyncio.CancelledError:
                        logger.debug(f"[Fallback:click-card] Cancelled for '{tile_sel}'")
                        raise
                    except asyncio.TimeoutError:
                        logger.debug(f"[Fallback:click-card] Timeout for '{tile_sel}'")
                        continue
                    except Exception as e:
                        logger.debug(f"[Fallback:click-card] wait_for failed for '{tile_sel}': {e}")
                        continue

                    count = await card.count()
                    if count > 0:
                        await card.scroll_into_view_if_needed(timeout=3000)
                        newp = await self._click_and_capture_navigation(
                            lambda: card.click(timeout=5000), page, context, url_regex=url_pat
                        )
                        if newp:
                            return newp.url
                except Exception as e:
                    logger.debug(f"[Fallback:click-card] Selector '{selector}' failed: {e}")
                    continue

        logger.warning("[Fallback:click-card] Could not find any clickable link or card.")
        return None

    async def _trigger_moncler_self_healing(
        self,
        ctx: NavigationContext,
        failure_reason: str,
        outcome_dict: dict[str, Any],
    ) -> None:
        """
        CR-ATELIER-002 Step 6-3: Moncler 専用の Self-Healing / Selector Discovery をトリガー
        CR-ATELIER-002 Step 7-4: パッチ生成モジュールを呼び出してファイル保存

        Args:
            ctx: ナビゲーションコンテキスト
            failure_reason: 失敗理由（"raw_zero", "rejected_all", "secondary_or_tertiary_used", "trap_detected", "locale_corrections_exceeded"）
            outcome_dict: Telemetry に保存した outcome 情報
        """
        try:
            # Self-Healing Agent と Selector Discovery Agent をインポート
            from app.agents.moncler_patch_builder import process_moncler_self_healing_results
            from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
            from app.agents.self_healing_agent import SelfHealingAgent

            # DOM スナップショットのパスを取得
            dom_snapshot_path = None
            if ctx.run_context:
                with contextlib.suppress(Exception):
                    dom_snapshot_path = str(ctx.run_context.get_path("failure_dom.html"))

            # failure_payload を構築
            failure_payload = {
                "site": "MONCLER_OFFICIAL",
                "url": self.page.url or ctx.entry_url or "",
                "failure_reason": failure_reason,
                "dom_snapshot_path": dom_snapshot_path,
                "layer_stats": outcome_dict.get("layer_stats", {}),
                "rejection_stats": {},  # outcome_dict から取得可能な場合は追加
                "selectors_current": ctx.site_config.get("selectors", {}),
                "run_id": getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                "timestamp": outcome_dict.get("timestamp"),
            }

            # Self-Healing Agent を呼び出す
            self_healing_result = None
            try:
                self_healing_agent = SelfHealingAgent()
                # handle_moncler_failure が実装されている場合は呼び出す
                if hasattr(self_healing_agent, "handle_moncler_failure"):
                    self_healing_result = await self_healing_agent.handle_moncler_failure(failure_payload)
                    logger.info(
                        f"[SelfHealing][Moncler] Self-healing result: {self_healing_result.get('analysis', 'N/A')}"
                    )
                else:
                    logger.debug("[SelfHealing][Moncler] handle_moncler_failure not implemented, skipping")
            except Exception as e:
                logger.warning(f"[SelfHealing][Moncler] Failed to call self-healing agent: {e}", exc_info=True)

            # Selector Discovery Agent を呼び出す
            selector_discovery_result = None
            try:
                selector_discovery_agent = SelectorDiscoveryAgent()
                # propose_moncler_selectors が実装されている場合は呼び出す
                if hasattr(selector_discovery_agent, "propose_moncler_selectors"):
                    discovery_payload = {
                        "dom_snapshot_path": dom_snapshot_path,
                        "selectors_current": ctx.site_config.get("selectors", {}),
                        "layer_stats": outcome_dict.get("layer_stats", {}),
                        "rejection_stats": {},  # outcome_dict から取得可能な場合は追加
                        "run_id": getattr(ctx.run_context, "run_id", None) if ctx.run_context else None,
                    }
                    selector_discovery_result = await selector_discovery_agent.propose_moncler_selectors(
                        discovery_payload
                    )
                    logger.info(
                        f"[SelectorDiscovery][Moncler] Proposed {len(selector_discovery_result.get('candidate_selectors', []))} selectors"
                    )
                else:
                    logger.debug("[SelectorDiscovery][Moncler] propose_moncler_selectors not implemented, skipping")
            except Exception as e:
                logger.warning(
                    f"[SelectorDiscovery][Moncler] Failed to call selector discovery agent: {e}", exc_info=True
                )

            # CR-ATELIER-002 Step 7-4: パッチ生成モジュールを呼び出してファイル保存
            if ctx.run_context and (self_healing_result or selector_discovery_result):
                try:
                    run_id = getattr(ctx.run_context, "run_id", None) or "unknown"
                    site = ctx.site_config.get("site_code") or ctx.site_config.get("site") or "MONCLER_OFFICIAL"
                    current_url = self.page.url or ctx.entry_url or ""

                    # moncler_outcome を構築（outcome_dict から）
                    moncler_outcome = {
                        "plp_materialized": outcome_dict.get("plp_materialized", False),
                        "tiles_detected": outcome_dict.get("tiles_detected", 0),
                        "pdp_links_raw": outcome_dict.get("pdp_links_raw", 0),
                        "pdp_links_accepted": outcome_dict.get("pdp_links_accepted", 0),
                        "layer_stats": outcome_dict.get("layer_stats", {}),
                        "rejection_stats": {},  # 必要に応じて追加
                        "locale_corrections": outcome_dict.get("locale_corrections", 0),
                        "trap_detected": outcome_dict.get("trap_detected", False),
                    }

                    # パッチ生成モジュールを呼び出す
                    saved_paths = await process_moncler_self_healing_results(
                        run_context=ctx.run_context,
                        run_id=run_id,
                        site=site,
                        current_url=current_url,
                        moncler_outcome=moncler_outcome,
                        self_healing_result=self_healing_result,
                        selector_discovery_result=selector_discovery_result,
                        current_site_config=ctx.site_config,
                        generate_markdown=True,  # Markdown レポートも生成
                    )

                    if saved_paths:
                        logger.info(
                            f"[PatchBuilder][Moncler] Generated patch files: "
                            f"analysis={saved_paths.get('analysis')}, "
                            f"patch={saved_paths.get('patch_candidate')}"
                        )
                except Exception as e:
                    logger.warning(f"[PatchBuilder][Moncler] Failed to generate patch files: {e}", exc_info=True)
        except Exception as e:
            logger.warning(
                f"[SelfHealing][Moncler] Failed to trigger self-healing/selector discovery: {e}", exc_info=True
            )
