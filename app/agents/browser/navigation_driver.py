# -*- coding: utf-8 -*-
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
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit, urlparse, parse_qsl, quote_plus

from playwright.async_api import ElementHandle, Locator, Page, BrowserContext

if TYPE_CHECKING:
    from app.agents.browser.telemetry import TelemetryService, TelemetryClient, TelemetryContext

# Stage 3A-2-1: extractor モジュールから looks_like_product_url を import
try:
    from app.agents.browser.extractor import looks_like_product_url, VISIBLE_PRICE_SELECTORS
except ImportError:
    # フォールバック: モジュールが見つからない場合の処理
    def looks_like_product_url(url: str) -> bool:
        """フォールバック実装"""
        return True
    VISIBLE_PRICE_SELECTORS = ["[itemprop=price]", "[class*=price]", "[data-testid*=price]"]

logger = logging.getLogger(__name__)

# Stage 3A-2-2: ロケールセグメント判定用の正規表現
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

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

def _dedupe_keep_order(items: List[str]) -> List[str]:
    """重複を削除しつつ順序を保持する"""
    return list(dict.fromkeys([i for i in (items or []) if i]))

# Stage 3A-3: trap 判定関数の型定義
TrapCheckerFn = Callable[[str], bool]  # URL を受けて trap かどうか判定


@dataclass
class NavigationContext:
    """ナビゲーション実行時のコンテキスト情報"""
    site: str
    query: str
    site_config: Dict[str, Any]
    settings: Dict[str, Any]
    run_context: Any  # RunContext を直接 import しない（循環回避）
    start_t: float
    budget_ms: int
    entry_url: Optional[str] = None
    context: Any = None  # Stage 3A-2-4: BrowserContext（fallback で必要、click_first_card_or_link で使用）


@dataclass
class NavigationOutcome:
    """ナビゲーション実行結果"""
    entry_url: str
    plp_materialized: bool = False
    trap_detected: bool = False
    trap_reason: Optional[str] = None
    recovered: bool = False
    pdp_links: List[str] = None  # Stage 3A-2-4: PDP リンクのリスト（BrowserUseAgent で使用）
    fallback_used: Optional[str] = None  # Stage 3A-2-4: 使用した fallback の種類（"header_search" または "click_first_card"、BrowserUseAgent で使用）
    
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
        trap_checker: Optional[TrapCheckerFn] = None,
        telemetry: Optional["TelemetryClient"] = None,
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
        
        Args:
            ctx: ナビゲーションコンテキスト
            
        Returns:
            NavigationOutcome: ナビゲーション結果
        """
        entry = ctx.entry_url or self.page.url
        outcome = NavigationOutcome(entry_url=entry)
        
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
                        raise ValueError(
                            f"Landing page looks like legal/trap (recovery failed): {url}"
                        )
            except ValueError:
                # ValueError はそのまま再スロー
                raise
            except Exception as e:
                # その他のエラーはログに記録して続行
                logger.debug("[NavigationDriver] trap_checker/recover failed: %s", e)
        
        # --- Stage 3A-2-2: ensure_plp_materialized を呼び出し ---
        try:
            materialized = await self.ensure_plp_materialized(ctx)
            outcome.plp_materialized = materialized
            if not materialized:
                logger.warning("[NavigationDriver] PLP materialization failed")
        except Exception as e:
            logger.error(f"[NavigationDriver] ensure_plp_materialized failed: {e}")
            outcome.plp_materialized = False
        
        # --- Stage 3A-2-3: materialize 後の trap 再チェック ---
        if outcome.plp_materialized and trap_check_fn:
            try:
                if trap_check_fn(self.page.url):
                    if not attempted_recover:
                        # まだ回復試行していない場合は試す
                        logger.warning("[NavigationDriver] trap-like url after materialize: %s", self.page.url)
                        recovered = await self.recover_plp(ctx)
                        outcome.recovered = recovered
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
                        raise ValueError(
                            f"After materialize, bounced back to legal/trap page: {self.page.url}"
                        )
            except ValueError:
                # ValueError はそのまま再スロー
                raise
            except Exception as e:
                logger.debug("[NavigationDriver] post-materialize trap check failed: %s", e)
        
        # --- Stage 3A-2-4: PDP リンク収集と fallback ロジック ---
        pdp_links: List[str] = []
        try:
            pdp_links = await self.collect_pdp_links(ctx)
            outcome.pdp_links = pdp_links
            logger.info(f"[NavigationDriver] Collected {len(pdp_links)} PDP links")
        except Exception as e:
            logger.error(f"[NavigationDriver] collect_pdp_links failed: {e}")
            outcome.pdp_links = []

        # PDP リンクが不足している場合の fallback
        if not pdp_links:
            logger.warning("[NavigationDriver] No PDP links found, trying fallback strategies...")
            
            # Fallback 1: ヘッダ検索
            try:
                did_search = await self.header_search_fallback(ctx)
                if did_search:
                    outcome.fallback_used = "header_search"
                    logger.info("[NavigationDriver] Header search fallback succeeded, collecting PDP links again...")
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

        # 最終的に PDP リンクが 0 件の場合、例外を投げる（旧 _run_plp_flow と同じ条件）
        if not outcome.pdp_links:
            raise ValueError(
                f"No PDP links found after all phases (materialize, collect, fallbacks). URL={self.page.url}"
            )

        logger.debug(
            f"[NavigationDriver] run_plp_flow: entry_url={entry}, trap_detected={outcome.trap_detected}, "
            f"plp_materialized={outcome.plp_materialized}, recovered={outcome.recovered}, "
            f"pdp_links={len(outcome.pdp_links)}, fallback_used={outcome.fallback_used}"
        )
        return outcome

    async def collect_pdp_links(
        self,
        ctx: NavigationContext,
    ) -> List[str]:
        """
        Stage 3A-2-1:
        旧 BrowserUseAgent._collect_pdp_links のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持されている。
        
        Phase 1a: Global <a href> sweep + Regex Filter
        Phase 1b: Selector-based補完
        Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        Phase 3: Noise Filtering & Saving
        
        Args:
            ctx: ナビゲーションコンテキスト
            
        Returns:
            List[str]: PDP リンクのリスト
        """
        page = self.page
        site_config = ctx.site_config
        settings = ctx.settings
        run_context = ctx.run_context
        target_url = page.url
        found_links: Set[str] = set()

        # Phase 1a: Global <a href> sweep + Regex Filter
        try:
            raw_hrefs: List[str] = await page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)")
        except Exception as e:
            logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}")
            raw_hrefs = []
        pdp_rx = re.compile(r"/(products?|p)/", re.I)
        for href in raw_hrefs:
            if pdp_rx.search(href):
                norm_url = self._normalize_abs_url(target_url, href)
                if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                    found_links.add(norm_url)
        if found_links:
            logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")

        # Phase 1b: Selector-based補完
        # Stage 3A-2-5: site_config["selectors"]["plp"] から取得（pdp ではなく plp を使用）
        plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
        pdp_selectors = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        
        # plp.pdp_link_selectors を優先、なければ pdp.pdp_link_selectors を使用
        pdp_link_selectors = _dedupe_keep_order(
            (plp_selectors.get("pdp_link_selectors", []) or []) +
            (pdp_selectors.get("pdp_link_selectors", []) or [])
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
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                for n in nodes:
                    href = await n.get_attribute("href") or await n.get_attribute("data-href") or await n.get_attribute("data-product-url") or await n.get_attribute("data-url")
                    if not href:
                        continue
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                        found_links.add(norm_url)
                        matched_count += 1
                if matched_count > 0:
                    logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
            except Exception as e:
                logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")

        # Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        if not found_links:
            logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
            try:
                deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)
                for href in deep_hrefs:
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                        found_links.add(norm_url)
                if found_links:
                    logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
            except Exception as e:
                logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")

        links = sorted(list(found_links))
        if not links:
            logger.warning("[PLP→PDP] No PDP hrefs found after all phases.")
        return []

        # Phase 3: Noise Filtering & Saving
        cleaned: List[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u):
                cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")
        try:
            sample = cleaned[:20]
            logger.debug(f"[PLP→PDP] sample={sample}")
            # Stage 3B: TelemetryClient を使用して JSON を保存
            if self.telemetry and ctx.run_context:
                from app.agents.browser.telemetry import TelemetryContext
                tctx = TelemetryContext(
                    site=ctx.site,
                    query=ctx.query,
                    run_id=getattr(ctx.run_context, "run_id", None),
                    stage="plp"
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
            logger.debug(f"[PLP→PDP] TelemetryService.save_raw_hrefs failed: {e}")
        except Exception:
            pass
        return cleaned

    async def run_deep_extraction(
        self,
        page: Page,
        site_config: Dict[str, Any],
    ) -> List[str]:
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

    # --- 内部用の private メソッド（型だけ定義） ---
    # Stage 3A-1: これらのメソッドは、Stage 3A-2 で実装を移行します

    async def _click_first_card(
        self,
        page: Page,
        site_config: Dict[str, Any],
    ) -> Optional[Page]:
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

    def _looks_like_trap_or_legal(self, url: str, site_config: Optional[Dict[str, Any]] = None) -> bool:
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
            if trap_patterns:
                if any(pattern.lower() in full_lower for pattern in trap_patterns):
                    logger.warning(f"[_looks_like_trap] Detected trap pattern: {url}")
                    return True
            
            if legal_patterns:
                if any(pattern.lower() in path_lower for pattern in legal_patterns):
                    logger.warning(f"[_looks_like_trap] Detected legal pattern: {url}")
                    return True
            
            # trap_domains をチェック
            trap_domains = nav_cfg.get("trap_domains", [])
            if trap_domains:
                if any(domain.lower() in host for domain in trap_domains):
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
        """URL を絶対URLに正規化する（query と fragment を除去）"""
        try:
            absu = urljoin(base_url, href)
            parts = list(urlsplit(absu))
            if parts[2].endswith('/'):
                parts[2] = parts[2].rstrip('/')
            parts[3] = ""  # query
            parts[4] = ""  # fragment
            return urlunsplit(parts)
        except Exception:
            return href

    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        """セレクタが出現するまで安全に待機する（Stage 4: タイムアウト処理改善）"""
        if not page or page.is_closed():
            return False
        try:
            await asyncio.wait_for(
                page.wait_for_selector(selector, state=state, timeout=timeout_ms),
                timeout=(timeout_ms / 1000.0) + 0.5
            )
            return True
        except asyncio.CancelledError:
            logger.debug(f"[safe_wait_selector] Cancelled for '{selector}'")
            raise
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[safe_wait_selector] Timeout/Error for '{selector}': {e}")
            return False

    async def _run_deep_extraction_phase2(self, page: Page, site_config: Dict[str, Any]) -> List[str]:
        """
        Stage 3A-2-1:
        旧 BrowserUseAgent._run_deep_extraction_phase2 のロジックをここに移す。
        挙動・ログ・例外の流れはそのまま維持すること。
        
        Deep Extraction Phase 2: JSON-LD, onclick, data-* 属性からリンクを抽出する
        """
        logger.debug("[Phase 2] Running deep extraction (JSON-LD, onclick, data-*, ...)")
        # ★ 88.6.2: (BugFix) 括弧が過剰だった SyntaxError を修正
        container_sels: List[str] = (
            ((site_config.get("selectors") or {}).get("pdp") or {}).get("plp_container_selectors", []) or []
        )
        for cont in (container_sels or []):
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
        handle: Optional[ElementHandle] = None
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
                logger.warning(f"[Phase 2] Could not get element handle for scope: {e_handle}. Falling back to page evaluate.")
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

        hrefs: List[str] = []
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

    def _normalize_url(self, url: str, site_config: Dict[str, Any]) -> str:
        """
        Stage 4: URLを汎用的に正規化する
        
        site_config["locale"]["normalize_rules"] と force_query_params を使用して
        ロケール正規化とクエリパラメータの追加を行う。
        /en-int/ などのハードコードを排除。
        """
        u = urlparse(url)
        path = (u.path or "/").replace("//", "/")
        
        # site_configからロケール設定を取得（既存設定との互換性を確保）
        locale_cfg = (site_config.get("locale", {}) or {})
        
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
            ds = (site_config.get("discovery_settings", {}) or {})
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

    async def _accept_cookies_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
        """Cookie 同意バナーがあればクリックする"""
        # Stage 3A-2-5: site_config["navigation"]["overlays"]["cookie_banner_selectors"] から取得
        nav_cfg = (site_config.get("navigation", {}) or {})
        overlays_cfg = nav_cfg.get("overlays", {}) or {}
        cookie_selectors = overlays_cfg.get("cookie_banner_selectors", [])
        
        # フォールバック: 既存の ui 構造もサポート
        ui = (site_config.get("selectors", {}) or {}).get("ui", {}) or {}
        
        candidates = _dedupe_keep_order(
            (cookie_selectors or []) +
            (ui.get("cookie_accept", []) or []) + [
                "#onetrust-accept-btn-handler",
                "button:has-text('ACCEPT ALL')",
                "button:has-text('CONTINUE WITHOUT ACCEPTING')",
                "button[aria-label*='Accept' i]",
            ]
        )
        for sel in candidates:
            try:
                node = page.locator(sel).first
                if await node.count() > 0 and await node.is_visible():
                    await node.click(timeout=3000)
                    await asyncio.sleep(0.2)
                    return True
            except Exception:
                continue
        return False

    async def _dismiss_geo_modal(self, page: Page, site_config: Optional[Dict[str, Any]] = None) -> None:
        """ジオ / ロケール関係のモーダルを潰す"""
        # Stage 3A-2-5: site_config["navigation"]["overlays"]["geo_modal_selectors"] から取得
        geo_selectors = []
        if site_config:
            nav_cfg = (site_config.get("navigation", {}) or {})
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
                if not await target.is_visible():
                    await target.scroll_into_view_if_needed()
                await target.click(timeout=5000)
                logger.info(f"[GeoModal] Clicked {desc}")
                await page.wait_for_timeout(300)
                return True
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

    async def _kill_overlays(self, page: Page, site_config: Optional[Dict[str, Any]] = None) -> None:
        """オーバーレイを削除する"""
        # Stage 3A-2-5: site_config["navigation"]["overlays"]["generic_close_buttons"] から取得
        overlay_selectors = []
        if site_config:
            nav_cfg = (site_config.get("navigation", {}) or {})
            overlays_cfg = nav_cfg.get("overlays", {}) or {}
            overlay_selectors = overlays_cfg.get("generic_close_buttons", [])
        
        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not overlay_selectors:
            overlay_selectors = [
                '.overlay', '.backdrop', '.modal-backdrop', '#onetrust-banner-sdk',
                '.cookie-banner', '[aria-modal="true"]', '.cmp-ui-overlay', '.cmp-modal', '.drawer--open'
            ]
        
        try:
            # JavaScript の配列として渡す
            sels_str = ','.join([f"'{s}'" for s in overlay_selectors])
            await page.evaluate(f"""
              (() => {{
                const sels = [{sels_str}];
                document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                const b = document.body; if (b) {{ b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }}
                const html=document.documentElement; if (html) {{ html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }}
              }})();
            """)
        except Exception:
            pass

    async def _force_plp_recover(self, page: Page, site_config: Dict[str, Any], target_url: Optional[str]) -> None:
        """
        Stage 4: PLP 回復（汎用化）
        site_config["navigation"]["plp_recovery"] と discovery_settings.fallback_url を使用
        """
        try:
            # site_configからPLP回復設定を取得
            nav_cfg = (site_config.get("navigation", {}) or {})
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
            (plp_cfg.get("tile_selectors", []) or []) +  # 新規: plp.tile_selectors を優先
            (plp_cfg.get("pdp_link_selectors", []) or []) +  # plp.pdp_link_selectors も使用
            (pdp_cfg.get("pdp_link_selectors", []) or [])
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
            try:
                await self._accept_cookies_if_present(page, site_config)
            except Exception:
                pass
            try:
                # Stage 3A-2-5: site_config を渡す
                await self._dismiss_geo_modal(page, site_config)
            except Exception:
                pass
            try:
                # Stage 3A-2-5: site_config を渡す
                await self._kill_overlays(page, site_config)
            except Exception:
                pass

            # Stage 4: ロケールリダイレクトの検出（汎用化）
            current_url = (page.url or "").lower()
            locale_cfg = (site_config.get("locale", {}) or {})
            prefer_locale = locale_cfg.get("prefer", "")
            allowed_domain = site_config.get("allowed_domain", "")
            
            # ターゲットロケール以外のロケールにリダイレクトされた場合を検出
            if prefer_locale and allowed_domain:
                # 現在のURLがターゲットロケールを含まない場合
                target_locale_path = f"/{prefer_locale}/"
                if allowed_domain.lower() in current_url and target_locale_path not in current_url:
                    # ロケールセグメントが含まれているかチェック
                    if _LOCALE_SEG_RE.search(current_url):
                        logger.warning(f"[Materialize] Detected locale redirect away from {prefer_locale} mid-attempt: {current_url}")
                        if locale_recover_attempts >= locale_recover_max:
                            logger.error("[Materialize] Locale recovery exceeded max attempts. Aborting.")
                            return False
                        locale_recover_attempts += 1
                        if target_url:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(800)
                            continue

            if run_ctx is not None and hasattr(run_ctx, "take_screenshot") and attempt < 3:
                try:
                    await run_ctx.take_screenshot(
                        page,
                        f"30_plp_materialize_attempt_{attempt + 1:02d}"
                    )
                except Exception as ss_e:
                    logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                try:
                    await page.wait_for_load_state("networkidle", timeout=800)
                except Exception:
                    pass
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
                if count >= target_min_tiles:
                    logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
                if count < 4 and attempt >= 1:
                    logger.warning(f"[Materialize] Low tiles ({count}) after {attempt+1} attempts, forcing recovery hop.")
                    if target_url:
                        try:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(500)
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
            logger.warning(f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty.")
            return True
        logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
        return False

    async def _click_continue_shopping_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
        """CONTINUE SHOPPING ボタンがあればクリックする"""
        ui = (site_config.get("selectors", {}) or {}).get("ui", {}) or {}
        candidates = _dedupe_keep_order(
            (ui.get("continue_shopping", []) or []) + [
                "a:has-text('CONTINUE SHOPPING')",
                "button:has-text('CONTINUE SHOPPING')",
                "[role='button']:has-text('CONTINUE SHOPPING')",
                "text=/\\bCONTINUE\\s+SHOPPING\\b/i",
            ]
        )
        for _ in range(3):
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            except Exception:
                pass
            for sel in candidates:
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0 and await el.is_visible():
                        await el.click(timeout=3000)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            pass
                        return True
                except Exception:
                    continue
        return False

    async def _click_and_capture_navigation(
        self,
        click_coro: Callable,
        page: Page,
        context: BrowserContext,
        *,
        url_regex: Optional[re.Pattern] = None,
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """
        Stage 3A-2-4:
        クリック操作を実行し、ナビゲーション（ポップアップ、SPA遷移など）をキャプチャする。
        """
        if url_regex is None:
            url_regex = re.compile(r"/product[s]?/|/p/|/pp/", re.I)

        popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
        same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))
        spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms)) if url_regex else None
        sel_spa = ", ".join(VISIBLE_PRICE_SELECTORS) if VISIBLE_PRICE_SELECTORS else "[itemprop=price],[class*=price],[data-testid*=price]"
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
            try:
                await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10))
            except Exception:
                pass
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
        settings = ctx.settings
        query = ctx.query
        start_t = ctx.start_t
        budget_ms = ctx.budget_ms

        # Stage 3A-2-5: site_config["navigation"]["header_search"] から取得
        nav_cfg = (site_config.get("navigation", {}) or {})
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
            _ensure_list(hs_cfg.get("search_open_selector")) +
            _ensure_list(ui.get("search_open")) + [
                "button[aria-label='Search']",
                "[aria-label*='Search' i]",
            ]
        )
        sel_input = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("search_input_selector")) +
            _ensure_list(ui.get("search_input")) + [
                "form[role='search'] input",
                "input[type='search']",
                "input[name='q']",
                "[data-testid*='search' i] input",
                "[role='search'] input",
                "dialog input[type='search']",
            ]
        )
        sel_submit = _dedupe_keep_order(
            _ensure_list(hs_cfg.get("submit_selector")) +
            _ensure_list(ui.get("search_submit")) + 
            ["form[role='search'] button[type='submit']"]
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
                    await self.safe_wait_selector(page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible")
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
                    ds = (site_config.get("discovery_settings", {}) or {})
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
                locale_cfg = (site_config.get("locale", {}) or {})
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

    async def click_first_card_or_link(self, ctx: NavigationContext) -> Optional[str]:
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
        nav_cfg = (site_config.get("navigation", {}) or {})
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
            (click_cfg.get("card_selectors", []) or []) +
            (plp_selectors.get("card_selectors", []) or []) +
            (pdp.get("pdp_link_selectors", []) or [])
        )
        
        plp_boxes = _dedupe_keep_order(
            (plp_selectors.get("container_selectors", []) or []) +
            (pdp.get("plp_container_selectors", []) or []) +
            (["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"])
        )
        block_ng = set(
            (click_cfg.get("blocklist_href_substrings", []) or []) +
            (pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
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

        tile_selectors = [
            "[data-qa='product-tile']",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]",
        ]
        # まず、box スコープなしで直接 tile_selectors を試す
        for tile_sel in tile_selectors:
            try:
                card = page.locator(tile_sel).first
                # 要素が存在するか確認（タイムアウトを短く設定）
                try:
                    await card.wait_for(state="attached", timeout=2000)
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
