# -*- coding: utf-8 -*-
# ==============================================================================
# File: app/agents/browser/plp_driver.py
# Version: 2.0.0 (Stage 4: 汎用 PLP Driver 拡張版)
# Purpose: PLP → PDP ナビゲーションロジックの汎用ドライバ
# ==============================================================================
"""
Stage 4: 汎用 PLP Driver 拡張版

PLP → PDP のナビゲーションロジックを完全に site_config ベースで動作する
汎用的なドライバに拡張。ブランド固有ロジックはすべて設定ファイルに移行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, TYPE_CHECKING
from urllib.parse import urlparse, urljoin

from playwright.async_api import Page, BrowserContext, Locator

if TYPE_CHECKING:
    from app.core.run_context import RunContext
    from app.agents.browser.telemetry import TelemetryClient

logger = logging.getLogger(__name__)

# Stage 3A-2-1: ロケールセグメント判定用の正規表現
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))


@dataclass
class PlpNavigationResult:
    """
    PLP → PDP ナビゲーションの結果（拡張版）
    
    Stage 4: 拡張フィールドを追加
    - recovery_successful: リカバリが成功したか
    - overlays_handled: 処理したオーバーレイの種類
    - navigation_method: ナビゲーション方法（"new_tab", "same_tab", "spa"）
    - errors: 発生したエラーメッセージのリスト
    """
    pdp_url: str
    pdp_opened_in_new_tab: bool
    plp_url: str
    tiles_seen: int
    trap_detected: bool
    trap_reason: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False  # Stage 4: 追加
    overlays_handled: List[str] = field(default_factory=list)  # Stage 4: 追加
    navigation_method: Optional[str] = None  # Stage 4: 追加 ("new_tab", "same_tab", "spa")
    errors: List[str] = field(default_factory=list)  # Stage 4: 追加


class PlpDriver:
    """
    汎用 PLP Driver - site_configベースで動作
    
    Stage 4: 完全汎用化
    - ブランド固有ロジックをすべて site_config に移行
    - コードは汎用的なナビゲーションロジックのみ
    - NavigationDriver への依存を最小化（将来DI可能に）
    """
    
    def __init__(
        self,
        page: Page,
        context: BrowserContext,
        *,
        site_config: Dict[str, Any],
        run_context: "RunContext",
        logger: Optional[logging.Logger] = None,
        telemetry: Optional["TelemetryClient"] = None,
    ) -> None:
        self.page = page
        self.context = context
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        self.telemetry = telemetry
        
        # Stage 4: NavigationDriver への依存を将来DIできるように（オプショナル）
        self._navigation_driver: Optional[Any] = None
    
    async def navigate_to_pdp(
        self,
        *,
        target_url: Optional[str] = None,
        timeout_ms: int = 60000,
        # Stage 4: 後方互換性のため、既存のシグネチャもサポート
        start_t: Optional[float] = None,
        budget_ms: Optional[int] = None,
    ) -> PlpNavigationResult:
        """
        PLP → PDP ナビゲーションを実行する（メインエントリーポイント）
        
        Stage 4: 新しいシグネチャに対応しつつ、後方互換性を維持
        
        Args:
            target_url: ターゲットPLP URL（リカバリ用）
            timeout_ms: タイムアウト（ミリ秒）
            start_t: 開始時刻（後方互換性用、指定時は使用）
            budget_ms: 予算時間（後方互換性用、指定時は使用）
            
        Returns:
            PlpNavigationResult: ナビゲーション結果
        """
        # 後方互換性: 既存のシグネチャ（start_t, budget_ms）が指定されている場合
        if start_t is not None and budget_ms is not None:
            actual_start_t = start_t
            actual_budget_ms = budget_ms
        else:
            actual_start_t = time.time()
            actual_budget_ms = timeout_ms
        
        plp_url = self.page.url
        errors: List[str] = []
        overlays_handled: List[str] = []
        
        try:
            # 1. Overlay処理（Cookie、Geo、その他）
            await self._handle_overlays(overlays_handled)
            
            # 2. PLPタイルのマテリアライズ
            plp_config = self._get_plp_config()
            tiles_seen = await self._materialize_tiles(
                plp_config=plp_config,
                start_t=actual_start_t,
                budget_ms=actual_budget_ms,
                target_url=target_url,
            )
            
            if tiles_seen == 0:
                error_msg = f"PLP did not materialize (no product tiles). URL={plp_url}"
                errors.append(error_msg)
                raise ValueError(error_msg)
            
            # 3. Trap検出と回復
            trap_config = self._get_trap_config()
            trap_detected = False
            trap_reason = None
            recovery_attempted = False
            recovery_successful = False
            
            if self._is_trap_page(self.page.url, trap_config):
                trap_detected = True
                trap_reason = f"Trap/legal page detected: {self.page.url}"
                self.logger.warning(f"[PlpDriver] {trap_reason}")
                
                if target_url:
                    recovery_attempted = True
                    recovery_successful = await self._recover_from_trap(
                        target_url=target_url,
                        trap_config=trap_config,
                    )
                    
                    if recovery_successful:
                        # 回復後に再度trap判定
                        if self._is_trap_page(self.page.url, trap_config):
                            error_msg = f"Still on trap/legal page after recovery: {self.page.url}"
                            errors.append(error_msg)
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"Trap recovery failed. URL={self.page.url}"
                        errors.append(error_msg)
                        raise ValueError(error_msg)
            
            # 4. タイルクリック → PDP遷移
            pdp_page = await self._click_tile_and_navigate(
                plp_config=plp_config,
                timeout_ms=min(10000, actual_budget_ms // 6),
            )
            
            if pdp_page is None:
                error_msg = "Failed to navigate to PDP from PLP tile"
                errors.append(error_msg)
                raise ValueError(error_msg)
            
            pdp_url = pdp_page.url
            pdp_opened_in_new_tab = (pdp_page is not self.page)
            navigation_method = (
                "new_tab" if pdp_opened_in_new_tab else
                "same_tab" if pdp_url != plp_url else
                "spa"
            )
            
            # 新タブが開かれた場合、self.page を更新
            if pdp_opened_in_new_tab:
                self.page = pdp_page
            
            return PlpNavigationResult(
                pdp_url=pdp_url,
                pdp_opened_in_new_tab=pdp_opened_in_new_tab,
                plp_url=plp_url,
                tiles_seen=tiles_seen,
                trap_detected=trap_detected,
                trap_reason=trap_reason,
                recovery_attempted=recovery_attempted,
                recovery_successful=recovery_successful,
                overlays_handled=overlays_handled,
                navigation_method=navigation_method,
                errors=errors,
            )
            
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            self.logger.error(f"[PlpDriver] Navigation failed: {e}", exc_info=True)
            raise
    
    # === Private Methods: 設定取得 ===
    
    def _get_plp_config(self) -> Dict[str, Any]:
        """
        site_configからPLP設定を取得（後方互換性維持）
        
        Stage 4: 新しい selectors.plp.* スキーマを優先しつつ、
        既存の selectors.pdp.* や discovery_settings からもフォールバック
        """
        selectors = (self.site_config.get("selectors") or {})
        plp_cfg = selectors.get("plp") or {}
        pdp_cfg = selectors.get("pdp") or {}
        discovery = self.site_config.get("discovery_settings", {}) or {}
        plp_discovery = discovery.get("plp") or {}
        
        # 後方互換性: pdp.* から plp.* にフォールバック
        return {
            "product_tiles": (
                plp_cfg.get("product_tiles") or
                pdp_cfg.get("plp_container_selectors") or
                []
            ),
            "product_link": (
                plp_cfg.get("product_link") or
                pdp_cfg.get("pdp_link_selectors") or
                []
            ),
            "container": (
                plp_cfg.get("container") or
                pdp_cfg.get("plp_container_selectors") or
                ["main", "[role='main']"]
            ),
            "click_strategy": plp_cfg.get("click_strategy", "link"),
            "wait_for_navigation": plp_cfg.get("wait_for_navigation", True),
            "min_tiles": (
                plp_cfg.get("min_tiles") or
                plp_discovery.get("min_tiles") or
                8
            ),
            "max_scroll_rounds": (
                plp_cfg.get("max_scroll_rounds") or
                plp_discovery.get("scroll_rounds") or
                discovery.get("plp_scroll_rounds") or
                10
            ),
            "scroll_pause_ms": (
                plp_cfg.get("scroll_pause_ms") or
                plp_discovery.get("scroll_pause_ms") or
                160
            ),
            "target_load_state": (
                plp_cfg.get("target_load_state") or
                plp_discovery.get("wait_until") or
                "networkidle"
            ),
            "wait_for_selectors": (
                plp_cfg.get("wait_for_selectors") or
                plp_discovery.get("wait_for_selectors") or
                []
            ),
        }
    
    def _get_overlay_config(self) -> Dict[str, Any]:
        """
        site_configからOverlay設定を取得
        
        Stage 4: navigation.overlays.* と selectors.ui.* を統合
        """
        nav_cfg = (self.site_config.get("navigation") or {})
        overlays_cfg = nav_cfg.get("overlays") or {}
        selectors = (self.site_config.get("selectors") or {})
        ui_cfg = selectors.get("ui") or {}
        
        return {
            "cookie": {
                "selectors": (
                    overlays_cfg.get("cookie_banner", {}).get("selectors") or
                    ui_cfg.get("cookie_accept") or
                    []
                ),
                "wait_after_click_ms": (
                    overlays_cfg.get("cookie_banner", {}).get("wait_after_click_ms") or
                    500
                ),
            },
            "geo": {
                "selectors": (
                    overlays_cfg.get("geo_popup", {}).get("selectors") or
                    overlays_cfg.get("geo_modal_selectors") or
                    []
                ),
                "wait_after_click_ms": (
                    overlays_cfg.get("geo_popup", {}).get("wait_after_click_ms") or
                    500
                ),
            },
            "other": overlays_cfg.get("other_overlays") or {},
        }
    
    def _get_trap_config(self) -> Dict[str, Any]:
        """
        site_configからTrap設定を取得
        
        Stage 4: navigation.trap.* スキーマを優先しつつ、
        既存の navigation.trap_url_patterns からもフォールバック
        """
        nav_cfg = (self.site_config.get("navigation") or {})
        trap_cfg = nav_cfg.get("trap") or {}
        
        # 後方互換性: navigation.trap_url_patterns も読み込む
        legacy_patterns = nav_cfg.get("trap_url_patterns") or []
        
        return {
            "detect_by_url": {
                "patterns": (
                    trap_cfg.get("detect_by_url", {}).get("patterns") or
                    legacy_patterns or
                    []
                ),
                "exact_matches": (
                    trap_cfg.get("detect_by_url", {}).get("exact_matches") or
                    []
                ),
            },
            "detect_by_selector": (
                trap_cfg.get("detect_by_selector") or
                []
            ),
            "recovery_actions": (
                trap_cfg.get("recovery_actions") or
                [
                    {"action": "go_back", "max_attempts": 1},
                    {"action": "goto_target", "target_url_key": "seed_plp_url", "max_attempts": 1},
                ]
            ),
        }
    
    # === Private Methods: タイルマテリアライズ ===
    
    async def _materialize_tiles(
        self,
        *,
        plp_config: Dict[str, Any],
        start_t: float,
        budget_ms: int,
        target_url: Optional[str] = None,
    ) -> int:
        """
        PLPタイルをマテリアライズ（Stage 4: site_configベース）
        
        既存の _materialize_plp_tiles() を site_config ベースにリファクタリング
        """
        # 既存の実装を呼び出し（後方互換性）
        return await self._materialize_plp_tiles(
            start_t=start_t,
            budget_ms=budget_ms,
            target_url=target_url,
        )
    
    async def _materialize_plp_tiles(
        self,
        *,
        start_t: float,
        budget_ms: int,
        target_url: Optional[str] = None,
    ) -> int:
        """
        PLP タイルをマテリアライズ（スクロールしながら一定数ロード）する。
        
        後方互換性のため保持。新しいコードは _materialize_tiles() を使用。
        
        Returns:
            見つかったタイル数
        """
        plp_config = self._get_plp_config()
        
        # site_config ベースでタイルセレクタを取得
        product_tiles = plp_config.get("product_tiles", [])
        product_link = plp_config.get("product_link", [])
        container = plp_config.get("container", [])
        
        # デフォルトセレクタと組み合わせ
        tile_selectors = _dedupe_keep_order(
            product_link +
            product_tiles +
            container +
            [
                "a[data-product-url]",
                "[data-product-url]",
                "[data-qa='product-tile']",
                ".product-card",
                ".c-product-card",
                ".c-product-tile",
                "[data-testid*='product' i]"
            ]
        )
        tile_selector_str = ", ".join(tile_selectors) if tile_selectors else "a[href*='/product'], a[href*='/p/']"
        
        target_min_tiles = plp_config.get("min_tiles", 8)
        max_scroll_attempts = int(plp_config.get("max_scroll_rounds", 10))
        scroll_pause_ms = plp_config.get("scroll_pause_ms", 160)
        target_load_state = plp_config.get("target_load_state", "networkidle")
        wait_for_selectors = plp_config.get("wait_for_selectors", [])
        
        locale_recover_attempts = 0
        locale_recover_max = 5
        
        # wait_for_selectors を待機
        for sel in wait_for_selectors:
            try:
                await self.page.wait_for_selector(sel, state="visible", timeout=2000)
            except Exception:
                pass
        
        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                self.logger.warning("[PlpDriver] Materialize timed out.")
                break
            
            # オーバーレイ処理
            overlays_handled: List[str] = []
            await self._handle_overlays(overlays_handled)
            
            # ロケールリダイレクト検出
            if target_url and self._detected_locale_redirect():
                if locale_recover_attempts >= locale_recover_max:
                    self.logger.error("[PlpDriver] Locale recovery exceeded max attempts.")
                    break
                locale_recover_attempts += 1
                trap_config = self._get_trap_config()
                await self._recover_from_trap(target_url=target_url, trap_config=trap_config)
                await self.page.wait_for_timeout(800)
                continue
            
            # スクロール
            try:
                scroll_steps = 6
                for _ in range(scroll_steps):
                    await self.page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await self.page.wait_for_timeout(scroll_pause_ms)
                try:
                    await self.page.wait_for_load_state(target_load_state, timeout=800)
                except Exception:
                    pass
            except Exception as e:
                self.logger.warning(f"[PlpDriver] Scroll failed on attempt {attempt + 1}: {e}")
                break
            
            # タイル数カウント
            try:
                count = await self.page.locator(tile_selector_str).count()
                self.logger.info(f"[PlpDriver] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")
                if count >= target_min_tiles:
                    self.logger.info(f"[PlpDriver] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return count
                if count < 4 and attempt >= 1:
                    self.logger.warning(f"[PlpDriver] Low tiles ({count}) after {attempt+1} attempts.")
                    if target_url:
                        try:
                            trap_config = self._get_trap_config()
                            await self._recover_from_trap(target_url=target_url, trap_config=trap_config)
                            await self.page.wait_for_timeout(500)
                            rec_count = await self.page.locator(tile_selector_str).count()
                            self.logger.info(f"[PlpDriver] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return rec_count
                        except Exception as rec_e:
                            self.logger.warning(f"[PlpDriver] Recovery hop failed: {rec_e}")
                    return count if count > 0 else 0
            except asyncio.CancelledError:
                self.logger.warning("[PlpDriver] Cancelled during tile count.")
                break
            except Exception as e:
                self.logger.warning(f"[PlpDriver] Could not count tiles on attempt {attempt + 1}: {e}")
        
        # 最終カウント
        try:
            final_count = await self.page.locator(tile_selector_str).count()
            if final_count > 0:
                self.logger.warning(f"[PlpDriver] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding.")
                return final_count
        except Exception:
            pass
        
        self.logger.error("[PlpDriver] Failed: No product tiles found after all scroll attempts.")
        return 0
    
    # === Private Methods: ナビゲーション ===
    
    async def _click_tile_and_navigate(
        self,
        *,
        plp_config: Dict[str, Any],
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """
        タイルをクリックしてPDPに遷移（Stage 4: site_configベース）
        
        既存の _click_tile_and_navigate_to_pdp() を site_config ベースにラップ
        """
        # 既存の実装を呼び出し
        return await self._click_tile_and_navigate_to_pdp(
            url_regex=None,
            timeout_ms=timeout_ms,
        )
    
    async def _click_tile_and_navigate_to_pdp(
        self,
        *,
        url_regex: Optional[Pattern[str]] = None,
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """
        タイルをクリックして PDP に遷移する。
        新タブ／同タブ／SPA の URL 変化を待つレース処理を行う。
        
        後方互換性のため保持。新しいコードは _click_tile_and_navigate() を使用。
        
        Args:
            url_regex: PDP URL のパターン（デフォルト: /product[s]?/|/p/|/pp/）
            timeout_ms: タイムアウト（ミリ秒）
            
        Returns:
            遷移後の Page オブジェクト（新タブの場合は新しいPage、同タブの場合は self.page）
        """
        if url_regex is None:
            url_regex = re.compile(r"/product[s]?/|/p/|/pp/", re.I)
        
        plp_config = self._get_plp_config()
        pdp_cfg = (self.site_config.get("selectors", {}) or {}).get("pdp") or {}
        
        # site_config から設定を取得
        product_link = plp_config.get("product_link", [])
        container = plp_config.get("container", [])
        click_strategy = plp_config.get("click_strategy", "link")
        
        link_sel = product_link or pdp_cfg.get("pdp_link_selectors", [])
        plp_boxes = container or pdp_cfg.get("plp_container_selectors", [
            "main",
            "section[role='main']",
            "#main",
            "[id*='product' i]",
            "[class*='product' i]"
        ])
        block_ng = set(pdp_cfg.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
        
        # リンクセレクタから試行
        if link_sel and click_strategy in ("link", "both"):
            for s in link_sel:
                try:
                    loc = self.page.locator(s)
                    count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i)
                        href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed()
                            new_page = await self._click_and_wait_for_navigation(
                                lambda: el.click(timeout=5000),
                                url_regex=url_regex,
                                timeout_ms=timeout_ms,
                            )
                            if new_page:
                                return new_page
                except Exception:
                    continue
        
        # タイルセレクタから試行
        if click_strategy in ("tile", "both"):
            product_tiles = plp_config.get("product_tiles", [])
            tile_selectors = product_tiles or [
                "[data-qa='product-tile']",
                ".c-product-tile",
                ".product-card",
                "[data-testid*='product-card']",
                "article[data-product-id]"
            ]
            for box in plp_boxes:
                for tile_sel in tile_selectors:
                    try:
                        card = self.page.locator(f"{box} {tile_sel}").first
                        await card.scroll_into_view_if_needed()
                        if await card.count() > 0:
                            new_page = await self._click_and_wait_for_navigation(
                                lambda: card.click(timeout=5000),
                                url_regex=url_regex,
                                timeout_ms=timeout_ms,
                            )
                            if new_page:
                                return new_page
                    except Exception:
                        continue
        
        self.logger.warning("[PlpDriver] Could not find any clickable link or card.")
        return None
    
    async def _click_and_wait_for_navigation(
        self,
        click_coro,
        *,
        url_regex: Optional[Pattern[str]],
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """
        クリック後のナビゲーションを待機（新タブ/同タブ/SPA）
        
        Stage 4: 既存の _click_and_capture_navigation() をラップ（名前変更）
        """
        return await self._click_and_capture_navigation(
            click_coro=click_coro,
            url_regex=url_regex,
            timeout_ms=timeout_ms,
        )
    
    async def _click_and_capture_navigation(
        self,
        click_coro,
        *,
        url_regex: Optional[Pattern[str]],
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """
        タイルクリック後のナビゲーションを検出する（新タブ／同タブ／SPA のレース処理）。
        
        後方互換性のため保持。新しいコードは _click_and_wait_for_navigation() を使用。
        
        Args:
            click_coro: クリック処理のコルーチン
            url_regex: PDP URL のパターン
            wait_state: 待機するロード状態
            timeout_ms: タイムアウト（ミリ秒）
            
        Returns:
            遷移後の Page オブジェクト
        """
        from app.agents.browser.extractor import VISIBLE_PRICE_SELECTORS
        
        popup_task = asyncio.create_task(
            self.context.wait_for_event("page", timeout=timeout_ms)
        )
        same_tab_nav_task = asyncio.create_task(
            self.page.wait_for_event("framenavigated", timeout=timeout_ms)
        )
        spa_url_task = (
            asyncio.create_task(self.page.wait_for_url(url_regex, timeout=timeout_ms))
            if url_regex else None
        )
        sel_spa = ", ".join(VISIBLE_PRICE_SELECTORS) or "[itemprop=price],[class*=price],[data-testid*=price]"
        spa_price_task = asyncio.create_task(
            self.page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms)
        )
        
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
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            
            if not done:
                return None
            
            winner = next(iter(done))
            new_page = winner.result() if winner is popup_task else self.page
            
            log_msg = (
                "popup" if winner is popup_task else
                "framenav" if winner is same_tab_nav_task else
                "SPA URL" if winner is spa_url_task else
                "SPA Price"
            )
            self.logger.debug(f"[PlpDriver] Navigation winner: {log_msg}")
            
            try:
                if new_page.url == "about:blank":
                    await new_page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Exception as e_blank:
                self.logger.debug(f"[PlpDriver] Wait for about:blank failed: {e_blank}")
            
            try:
                await new_page.wait_for_load_state(
                    wait_state,
                    timeout=max(500, timeout_ms // 10)
                )
            except Exception:
                pass
            
            if url_regex:
                try:
                    await new_page.wait_for_url(
                        url_regex,
                        timeout=max(1000, timeout_ms // 4)
                    )
                except Exception as e_url_final:
                    self.logger.debug(f"[PlpDriver] Final wait_for_url failed: {e_url_final}")
            
            return new_page
        except Exception as e_wait:
            self.logger.warning(f"[PlpDriver] Nav race failed: {e_wait}")
            return None
        finally:
            for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task):
                if t and not t.done():
                    t.cancel()
    
    # === Private Methods: Trap検出と回復 ===
    
    def _is_trap_page(self, url: str, trap_config: Dict[str, Any]) -> bool:
        """
        Trapページかどうかを判定（Stage 4: site_configベース）
        
        NavigationDriver のロジックを再利用するが、将来DIできるように実装
        """
        # Stage 4: NavigationDriver への依存を最小化（将来DI可能に）
        if self._navigation_driver:
            # 将来: DIされたNavigationDriverを使用
            return self._navigation_driver._looks_like_trap_or_legal(url, self.site_config)
        else:
            # 現在: 直接NavigationDriverをインスタンス化
            from app.agents.browser.navigation_driver import NavigationDriver
            
            driver = NavigationDriver(page=None)  # type: ignore
            return driver._looks_like_trap_or_legal(url, self.site_config)
    
    def _looks_like_trap_or_legal(self, url: str) -> bool:
        """
        Trap/Legal ページかどうかを判定する（後方互換性のため保持）。
        
        新しいコードは _is_trap_page() を使用。
        """
        trap_config = self._get_trap_config()
        return self._is_trap_page(url, trap_config)
    
    async def _recover_from_trap(
        self,
        *,
        target_url: str,
        trap_config: Dict[str, Any],
    ) -> bool:
        """
        Trapページから回復する（Stage 4: site_configベースのリカバリアクション）
        
        Returns:
            bool: リカバリが成功したかどうか
        """
        recovery_actions = trap_config.get("recovery_actions", [])
        
        for action_cfg in recovery_actions:
            action = action_cfg.get("action")
            max_attempts = action_cfg.get("max_attempts", 1)
            
            for attempt in range(max_attempts):
                try:
                    if action == "go_back":
                        await self.page.go_back(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            self.logger.info(f"[PlpDriver] Recovery successful via 'go_back'")
                            return True
                    
                    elif action == "goto_target":
                        target = (
                            target_url or
                            self.site_config.get(action_cfg.get("target_url_key", "seed_plp_url")) or
                            ""
                        )
                        if target:
                            await self.page.goto(url=target, wait_until="domcontentloaded")
                            await self.page.wait_for_timeout(1000)
                            if not self._is_trap_page(self.page.url, trap_config):
                                self.logger.info(f"[PlpDriver] Recovery successful via 'goto_target': {target}")
                                return True
                    
                    elif action == "reload":
                        await self.page.reload(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            self.logger.info(f"[PlpDriver] Recovery successful via 'reload'")
                            return True
                    
                except Exception as e:
                    self.logger.warning(f"[PlpDriver] Recovery action '{action}' (attempt {attempt + 1}/{max_attempts}) failed: {e}")
                    continue
        
        # フォールバック: NavigationDriver の _force_plp_recover を使用
        try:
            from app.agents.browser.navigation_driver import NavigationDriver
            
            driver = NavigationDriver(page=self.page)
            await driver._force_plp_recover(self.page, self.site_config, target_url)
            if not self._is_trap_page(self.page.url, trap_config):
                self.logger.info(f"[PlpDriver] Recovery successful via NavigationDriver._force_plp_recover")
                return True
        except Exception as e:
            self.logger.warning(f"[PlpDriver] NavigationDriver recovery fallback failed: {e}")
        
        return False
    
    # === Private Methods: Overlay処理 ===
    
    async def _handle_overlays(self, overlays_handled: List[str]) -> None:
        """
        Overlay処理（Cookie、Geo、その他）（Stage 4: site_configベース）
        
        Args:
            overlays_handled: 処理したオーバーレイの種類を追加するリスト
        """
        overlay_config = self._get_overlay_config()
        
        # Cookie処理
        if await self._handle_cookie_banner(overlay_config["cookie"]):
            overlays_handled.append("cookie")
        
        # Geo処理
        if await self._handle_geo_modal(overlay_config["geo"]):
            overlays_handled.append("geo")
        
        # その他のオーバーレイ削除
        await self._remove_generic_overlays(overlay_config["other"])
    
    async def _handle_cookie_banner(self, config: Dict[str, Any]) -> bool:
        """
        Cookieバナーを処理（Stage 4: site_configベース）
        
        Returns:
            bool: Cookieバナーが処理されたかどうか
        """
        selectors = config.get("selectors", [])
        wait_ms = config.get("wait_after_click_ms", 500)
        
        # デフォルトセレクタも追加
        if not selectors:
            selectors = [
                "#onetrust-accept-btn-handler",
                "button:has-text('ACCEPT ALL')",
                "button:has-text('CONTINUE WITHOUT ACCEPTING')",
                "button[aria-label*='Accept' i]"
            ]
        
        for sel in selectors:
            try:
                loc = self.page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=3000)
                    await self.page.wait_for_timeout(wait_ms)
                    self.logger.info(f"[PlpDriver] Cookie banner accepted via: {sel}")
                    return True
            except Exception:
                continue
        return False
    
    async def _handle_geo_modal(self, config: Dict[str, Any]) -> bool:
        """
        Geo Modalを処理（Stage 4: site_configベース）
        
        Returns:
            bool: Geo Modalが処理されたかどうか
        """
        selectors = config.get("selectors", [])
        wait_ms = config.get("wait_after_click_ms", 500)
        
        # デフォルトセレクタも追加
        if not selectors:
            selectors = [
                "button:has-text('Select your location')",
                "button[aria-label*='close' i]",
                "button:has-text('Close')",
                ".modal__close",
                ".c-modal__close"
            ]
        
        for sel in selectors:
            try:
                loc = self.page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=3000)
                    await self.page.wait_for_timeout(wait_ms)
                    self.logger.info(f"[PlpDriver] Geo modal dismissed via: {sel}")
                    return True
            except Exception:
                continue
        return False
    
    async def _remove_generic_overlays(self, config: Dict[str, Any]) -> None:
        """
        その他のオーバーレイを削除（Stage 4: site_configベース）
        """
        remove_selectors = config.get("remove_selectors", [])
        remove_body_classes = config.get("remove_body_classes", [])
        
        # デフォルトセレクタも追加
        if not remove_selectors:
            remove_selectors = [
                '.overlay', '.backdrop', '.modal-backdrop',
                '#onetrust-banner-sdk', '.cookie-banner',
                '[aria-modal="true"]', '.cmp-ui-overlay',
                '.cmp-modal', '.drawer--open'
            ]
        
        if not remove_selectors and not remove_body_classes:
            # デフォルト処理を実行
            try:
                await self.page.evaluate("""
                    (() => {
                        const sels = [
                            '.overlay', '.backdrop', '.modal-backdrop',
                            '#onetrust-banner-sdk', '.cookie-banner',
                            '[aria-modal="true"]', '.cmp-ui-overlay',
                            '.cmp-modal', '.drawer--open'
                        ];
                        document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                    })()
                """)
            except Exception as e:
                self.logger.debug(f"[PlpDriver] Default overlay removal failed: {e}")
            return
        
        js_code = f"""
        (() => {{
            const sels = {json.dumps(remove_selectors)};
            document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
            
            const bodyClasses = {json.dumps(remove_body_classes)};
            const body = document.body;
            if (body) {{
                bodyClasses.forEach(cls => body.classList.remove(cls));
                body.style.overflow = '';
            }}
            
            const html = document.documentElement;
            if (html) {{
                html.style.overflow = '';
                bodyClasses.forEach(cls => html.classList.remove(cls));
            }}
        }})();
        """
        
        try:
            await self.page.evaluate(js_code)
        except Exception as e:
            self.logger.debug(f"[PlpDriver] Overlay removal failed: {e}")
    
    # === 後方互換性のためのメソッド（既存コードとの互換性維持） ===
    
    async def _accept_cookies_if_present(self) -> bool:
        """
        Cookie バナーを処理する（後方互換性のため保持）。
        
        新しいコードは _handle_cookie_banner() を使用。
        """
        overlay_config = self._get_overlay_config()
        return await self._handle_cookie_banner(overlay_config["cookie"])
    
    async def _dismiss_geo_modal(self) -> None:
        """
        Geo モーダルを処理する（後方互換性のため保持）。
        
        新しいコードは _handle_geo_modal() を使用。
        """
        overlay_config = self._get_overlay_config()
        await self._handle_geo_modal(overlay_config["geo"])
    
    async def _kill_overlays(self) -> None:
        """
        オーバーレイを削除する（後方互換性のため保持）。
        
        新しいコードは _remove_generic_overlays() を使用。
        """
        overlay_config = self._get_overlay_config()
        await self._remove_generic_overlays(overlay_config["other"])
    
    # === Private Methods: ヘルパー ===
    
    def _detected_locale_redirect(self) -> bool:
        """ロケールリダイレクトが検出されたかどうか。"""
        current_url = (self.page.url or "").lower()
        locale_cfg = (self.site_config.get("locale", {}) or {})
        prefer_locale = locale_cfg.get("prefer", "")
        allowed_domain = self.site_config.get("allowed_domain", "")
        
        if prefer_locale and allowed_domain:
            target_locale_path = f"/{prefer_locale}/"
            if allowed_domain.lower() in current_url and target_locale_path not in current_url:
                if _LOCALE_SEG_RE.search(current_url):
                    return True
        return False
    
    def _time_left_ms(self, start_t: float, budget_ms: int) -> float:
        """残り時間を計算する。"""
        elapsed_ms = (time.time() - start_t) * 1000
        return max(0, budget_ms - elapsed_ms)
