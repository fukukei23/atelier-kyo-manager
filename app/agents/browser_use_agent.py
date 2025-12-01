# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : app/agents/browser_use_agent.py
# Version   : 88.6.2J (Hotfix SyntaxError and minor refactors)
# Date (JST): 2025年11月7日 07:54
# Usage     : drop-in replacement for previous BrowserUseAgent
# ------------------------------------------------------------------------------
# 変更要旨
#  - ★ 88.6.2: (BugFix) _run_deep_extraction_phase2 内の
#    `container_sels` 代入行にあった致命的な SyntaxError (括弧過剰) を修正。
#  - ★ 88.6.2: (Refactor) _click_first_card_or_link 内の
#    `tile_selectors` の定義を読みやすさのため整形 (動作変更なし)。
#  - ★ 88.6.1: (Refactor) _force_plp_recover 内の正規化メソッド呼び出しを
#    `_normalize_en_int_url` から `_normalize_to_en_int_url` に統一。
#  - ★ 88.6.1: (Refactor) 不要になった `_normalize_en_int_url` を削除。
#  - ★ 88.6.1: (BugFix) _force_plp_recover 内の `goto` に `url=` を明記。
#  - ★ 88.6.1: (BugFix) _handle_run_failure が参照するDOMファイル名を
#    `fail_dom.html` から `failure_dom.html` に修正 (可観測性)。
#  - ★ 88.6.0: (BugFix) _force_plp_recover メソッドの定義欠落を修正済。
# ==============================================================================

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qsl, quote_plus, urlsplit, urlunsplit

# --- Playwright imports (robust) ---
from playwright.async_api import Page, BrowserContext, Locator, ElementHandle
try:
    from playwright.async_api import Error as PlaywrightError
except Exception:
    try:
        from playwright.sync_api import Error as PlaywrightError
    except Exception:
        try:
            from playwright._impl._errors import Error as PlaywrightError
        except Exception:
            class PlaywrightError(Exception):
                pass
# === InteractiveRepairSession (Atlas-style guided loop) tentative import ===
try:
    from app.agents.interactive_repair_session import InteractiveRepairSession
except Exception:
    InteractiveRepairSession = None  # will be injected later
# --- End of robust imports ---

# --- Session manager (Stage 1 extraction) ---
from app.agents.browser.session_manager import SessionManager, EXTERNAL_BLOCKLIST_HOSTS
from app.agents.browser.navigation_driver import NavigationContext, NavigationDriver
from app.agents.browser.plp_driver import PlpDriver, PlpNavigationResult
from app.agents.browser.extractor import (
    BrowserExtractionService,
    PDPSizeSelectPolicy,
    VISIBLE_PRICE_SELECTORS,
    DEFAULT_PDP_PARALLEL_LIMIT,
)

# PRICE_SELECTORS のエイリアス（互換性のため）
PRICE_SELECTORS = VISIBLE_PRICE_SELECTORS

# --- 専用パッチの動的インポート ---
try:
    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
except Exception:
    moncler_plp_recovery = None
# ---

# --- Strategy Plugins ---
try:
    from app.agents.plugins.base import StrategyPlugin
    from app.agents.plugins.moncler_plp_v1 import MonclerPLPStrategy
except Exception:
    StrategyPlugin = None  # type: ignore
    MonclerPLPStrategy = None  # type: ignore
# ---

# 堅牢なインポート試行
try:
    from app.core.run_context import RunContext
    from app.utils.visual_regression import compare_and_maybe_update
    from app.models.result_models import DiscoveryResult
    from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
    from app.agents.browser.telemetry import TelemetryService, TelemetryClient, TelemetryContext
    # Stage 3B: observability.py の関数は TelemetryService に移行
    # 後方互換性のため、observability.py も残す（段階的移行）
    from app.utils.observability import (
        save_dom, count_selectors, save_raw_hrefs, write_fail_snapshot
    )
    # V88.5.1: (Patch) AiLlmController は run_with_repair 内部でのみ import
    # from app.utils.ai_llm_controller import AiLlmController
    # MONCLER専用 DrissionPage ハンドラのインポート（未導入時は None）
    try:
        from app.specialized.moncler_handler import MonclerDrissionHandler
    except ImportError:
        MonclerDrissionHandler = None  # type: ignore
except ImportError:
    # 実行環境によってはパスが通っていない可能性を考慮
    logging.warning("Failed to import modules from standard paths. Trying relative imports...")
    try:
        from ..core.run_context import RunContext
        from ..utils.visual_regression import compare_and_maybe_update
        from ..models.result_models import DiscoveryResult
        from .selector_discovery_agent import SelectorDiscoveryAgent
        from .browser.telemetry import TelemetryService, TelemetryClient, TelemetryContext
        from ..utils.observability import (
            save_dom, count_selectors, save_raw_hrefs, write_fail_snapshot
        )
        # MONCLER専用 DrissionPage ハンドラのインポート（未導入時は None）
        try:
            from ..specialized.moncler_handler import MonclerDrissionHandler
        except ImportError:
            MonclerDrissionHandler = None  # type: ignore
    except ImportError as e:
         logging.critical(f"Relative import also failed: {e}. Some functionalities might be broken.")
         # 最低限動作するためのモックやプレースホルダを定義する (必要に応じて)
         class RunContext: pass # type: ignore
         @dataclass
         class DiscoveryResult: # type: ignore
             ok: bool = False
             site: str = ""
             query: str = ""
             proposal: Dict[str, Any] = field(default_factory=dict)
             evidence: Dict[str, Any] = field(default_factory=dict)
             ai_analysis: Optional[Any] = None
             message: Optional[str] = None
             screenshot: Optional[str] = None
             file_path: Optional[str] = None
         def compare_and_maybe_update(*args, **kwargs): pass
         def extract_title_price(*args, **kwargs): return {}
         class SelectorDiscoveryAgent:
             def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None): pass
         class TelemetryService: # type: ignore
             def __init__(self, run_context=None, logger=None, **kwargs): pass
             async def save_dom(self, page, name): pass
             async def save_json(self, name, payload): pass
             async def save_screenshot(self, page, name): pass
             async def write_fail_snapshot(self, page, reason, tctx, extra=None): pass
         # MonclerDrissionHandler も定義（インポート失敗時は None）
         MonclerDrissionHandler = None  # type: ignore
         @dataclass
         class TelemetryContext: # type: ignore
             site: str = ""
             query: str = ""
             run_id: Optional[str] = None
             stage: Optional[str] = None
         class TelemetryClient: # type: ignore
             def __init__(self, run_context=None, base_dir: str = ""): pass
             async def save_dom(self, page, name, tctx): pass
             async def save_json(self, name, payload, tctx): pass
             async def save_screenshot(self, page, name, tctx): pass
             async def write_fail_snapshot(self, page, reason, tctx, extra=None): pass
         def save_dom(*args, **kwargs): pass
         def count_selectors(*args, **kwargs): pass
         def save_raw_hrefs(*args, **kwargs): pass
         def write_fail_snapshot(*args, **kwargs): pass


logger = logging.getLogger(__name__)
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000  # 120s watchdog

PLUGIN_REGISTRY: Dict[str, StrategyPlugin] = {}

# ==============================================================================
# Helper Functions
# (V88.6.0: _looks_like_trap_or_legal はクラスメソッドに移動)
# ==============================================================================

def is_same_origin(url: str, base: str) -> bool:
    try:
        u = urlparse(url); b = urlparse(base)
        return (u.scheme, u.netloc) == (b.scheme, b.netloc)
    except Exception: return False

def _unpack_vrt(res, *, threshold: Optional[float] = None):
    if isinstance(res, dict):
        d = float(res.get("diff_percent", 0.0))
        ok = bool(res.get("is_ok", (threshold is None or d <= float(threshold or 0.0))))
        return {"diff_percent": d, "is_ok": ok}
    if isinstance(res, (tuple, list)):
        if not res: return None
        return _unpack_vrt(res[0], threshold=threshold)
    if isinstance(res, (int, float)):
        d = float(res)
        ok = (threshold is None) or (d <= float(threshold or 0.0))
        return {"diff_percent": d, "is_ok": ok}
    return None

def _dedupe_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys([i for i in (items or []) if i]))


# ==============================================================================
# BrowserUseAgent Class
# ==============================================================================

class BrowserUseAgent:
    """
    Playwright を駆動して PLP/PDP を探索するメインエージェント。

    TODO: LocaleGateHandler などの共通クラスにロケーションゲート処理を抽象化予定。
    """
    # ★ V88.6.0: インスタンスメソッドに変更 (旧 v88.5.9J のグローバル関数)
    # Stage 4: _looks_like_trap_or_legal() を削除し、NavigationDriver._looks_like_trap_or_legal() を使用
    # このメソッドは NavigationDriver 経由で呼び出す
    def _looks_like_trap_or_legal(self, url: str, site_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Stage 4: Trap判定のヘルパー（NavigationDriver経由）
        NavigationDriver._looks_like_trap_or_legal() を呼び出すためのラッパー
        
        Args:
            url: チェックするURL
            site_config: サイト設定（Noneの場合は警告を出してFalseを返す）
        """
        if site_config is None:
            # site_configがない場合は警告を出してFalseを返す
            # ただし、既存コードとの互換性のため、このメソッドは残す
            logger.warning("[_looks_like_trap_or_legal] site_config is None, returning False (Stage 4 migration)")
            return False
        
        from app.agents.browser.navigation_driver import NavigationDriver
        # NavigationDriverインスタンスを作成（pageは不要なのでNone）
        driver = NavigationDriver(page=None)  # type: ignore
        return driver._looks_like_trap_or_legal(url, site_config)

    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.logger = logger
        self.run_context: Optional[RunContext] = None # Temporarily attach RunContext during run()
        
        # Stage 3B: TelemetryService インスタンス（run_context が設定された後に初期化）
        self._telemetry: Optional[TelemetryService] = None

        # --- V88.6.x: Session handles are managed by SessionManager ---
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._session_manager: Optional[SessionManager] = None
        self.extraction_service = BrowserExtractionService(self.logger, self.runtime_kwargs)
    
    def _ensure_telemetry(self) -> TelemetryService:
        """
        TelemetryService インスタンスを取得（遅延初期化）
        
        Returns:
            TelemetryService インスタンス
        """
        if self._telemetry is None:
            if self.run_context is None:
                raise ValueError("run_context must be set before using TelemetryService")
            self._telemetry = TelemetryService(run_context=self.run_context, logger=self.logger)
        return self._telemetry

    def _attach_session(self, session: SessionManager) -> None:
        self._session_manager = session
        self._context = session.context
        self._page = session.page

    def _detach_session(self) -> None:
        self._session_manager = None
        self._context = None
        self._page = None

    async def _bootstrap_session_page(
        self,
        *,
        page: Page,
        site: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
        settings: Dict[str, Any],
        target_url: str,
        likely_plp: bool = False,
    ) -> Page:
        """
        SessionManager で初期化済みの page をターゲットURLに遷移させ、最初の整形を行う。
        旧 _open_session の後半（初回ナビゲーション/locale回復等）をこちらへ移設する。
        """
        if not page or page.is_closed():
            raise ValueError("BrowserUseAgent: Session page is not available or already closed.")
        if not target_url:
            raise ValueError("BrowserUseAgent: target_url が指定されていません。起点URLが必要です。")

        # --- 初期ナビゲーション ---
        await page.goto(url=target_url, wait_until="domcontentloaded")
        await self._accept_cookies_if_present(page, site_config)
        await self._dismiss_geo_modal(page, site_config)
        await self._kill_overlays(page)
        await self._click_continue_shopping_if_present(page, site_config)

        if settings.get("enable_human_like"):
            try:
                await self._human_like_mouse_move(page)
                await self._human_like_scroll(page)
            except Exception as e:
                self.logger.debug(f"[HumanLike] skipped: {e}")

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=800)
        except Exception:
            pass
        await page.wait_for_timeout(120)

        # Stage 4: ロケールゲート検出と回復（汎用化）
        nav_cfg = (site_config.get("navigation", {}) or {})
        locale_gate_cfg = nav_cfg.get("locale_gate_detection", {}) or {}
        trap_domains = nav_cfg.get("trap_domains", [])
        allowed_domain = site_config.get("allowed_domain", "")
        
        # ロケールゲート検出が有効な場合
        if locale_gate_cfg.get("enabled", False) and not likely_plp:
            try:
                gate_links_count = await page.evaluate(
                    "() => document.querySelectorAll(\"a[href*='/en-'], a[href*='/it-'], a[href*='/fr-'], a[href*='/de-']\").length"
                )
                if gate_links_count and gate_links_count >= 10:
                    self.logger.warning(
                        f"[LocaleGate] Locale gate detected ({gate_links_count} links). Forcing navigation to PLP (PDP run only)."
                    )
                    # site_configからfallback URLを取得
                    fixed_url = (
                        site_config.get("seed_plp_url")
                        or site_config.get("fallback_url")
                        or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                        or site_config.get("home_url")
                    )
                    if fixed_url:
                        # URL正規化を適用
                        fixed_url = self._normalize_url(fixed_url, site_config)
                        await page.goto(url=fixed_url, wait_until="domcontentloaded")
                        await self._click_continue_shopping_if_present(page, site_config)
                        try:
                            await page.wait_for_load_state("networkidle", timeout=2000)
                        except Exception:
                            pass
                        await self._accept_cookies_if_present(page, site_config)
            except Exception as gate_e:
                self.logger.warning(f"[LocaleGate] Gate detection failed: {gate_e}")

            if settings.get("enable_locale_escape"):
                await self._force_en_int(page, site_config)
                await self._click_continue_shopping_if_present(page, site_config)
                await run_context.take_screenshot(page, "12_after_locale_escape")

            # trap_domains へのリダイレクトを検出
            try:
                current_url_lower = (page.url or "").lower()
                if trap_domains:
                    for trap_domain in trap_domains:
                        if trap_domain.lower() in current_url_lower:
                            # site_configからfallback URLを取得
                            fixed_url = (
                                site_config.get("seed_plp_url")
                                or site_config.get("fallback_url")
                                or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                                or site_config.get("home_url")
                            )
                            if fixed_url:
                                fixed_url = self._normalize_url(fixed_url, site_config)
                                logger.warning(f"[TrapDomain] Bounced to trap domain ({trap_domain}). Forcing back to PLP: {fixed_url}")
                                await page.goto(url=fixed_url, wait_until="domcontentloaded")
                                await self._accept_cookies_if_present(page, site_config)
                                break
            except Exception as trap_e:
                logger.warning(f"[TrapDomain] Trap domain detection failed: {trap_e}")

        return page

    def _build_pdp_prepare_hook(
        self,
        *,
        site_config: Dict[str, Any],
        settings: Dict[str, Any],
        run_context: RunContext,
    ):
        async def _prepare(page: Page):
            await self._kill_overlays(page)
            await self._click_continue_shopping_if_present(page, site_config)
            await self._dismiss_geo_modal(page, site_config)
            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "pdp", settings)

        return _prepare

    async def _open_session(
        self,
        *,
        site: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
        settings: Dict[str, Any],
        target_url: str,
        timeout_ms: int,
        likely_plp: bool = False,
    ) -> Page:
        """
        SessionManager を直接利用できない既存フロー（run_with_repair等）のための
        互換ラッパー。Playwright 依存ロジックは SessionManager 側に集約済み。
        """
        if self._session_manager and self._page and not self._page.is_closed():
            self.logger.warning("[SessionManager] Existing session detected. Reusing current page.")
            return self._page

        session = SessionManager(
            site=site,
            site_config=site_config,
            run_context=run_context,
            settings=settings,
            target_url=target_url,
            timeout_ms=timeout_ms,
            likely_plp=likely_plp,
            runtime_kwargs=self.runtime_kwargs,
            logger=self.logger,
            # Stage 4: url_normalizerを汎用化（site_configを渡すラムダ関数）
            url_normalizer=lambda url: self._normalize_url(url, site_config) if hasattr(self, '_normalize_url') else url,
        )
        await session.open()
        self._attach_session(session)

        page = session.page
        if page is None:
            raise ValueError("SessionManager failed to provision a Playwright page.")

        return await self._bootstrap_session_page(
            page=page,
            site=site,
            site_config=site_config,
            run_context=run_context,
            settings=settings,
            target_url=target_url,
            likely_plp=likely_plp,
        )

    async def _close_session(
        self,
        run_context: Optional[RunContext],
        settings: Dict[str, Any],
    ) -> None:
        """
        SessionManager ベースのセッションを明示的にクローズする薄いラッパー。
        旧コードとの互換性を保つため signature を維持する。
        """
        if not self._session_manager:
            self._detach_session()
            return
        try:
            await self._session_manager.close()
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to close browser session: {e}")
        finally:
            self._detach_session()


    # --- V88.5.0: `run_with_repair` (Full Implementation) ---
    # --- V88.5.1: (User Patch) Applied robustness patches ---
    # --- V88.5.2: (User Patch) Applied governance (timeout, result eval) ---
    async def run_with_repair(
        self,
        *,
        site: str,
        query: str,
        site_config: Dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
        max_steps: int = 5,
        repair_budget_ms: int = 60000  # V88.5.2: 修復予算の時間（ミリ秒）
    ) -> "DiscoveryResult":
        """
        Atlas型の「自己修復つきスクレイピング」フロー（実ブラウザ継続）。
        1. Playwrightセッションを開く
        2. 通常のPLP/PDP抽出を試す (VRT, Patch含む)
        3. ダメなら LLM駆動のインタラクティブ修復ループ (時間・ステップ監視付き)
        4. 成果セレクタを overrides.local.json 等へセーブ
        5. セッションを閉じる
        """
        settings = self._resolve_run_settings(site_config)
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        self.run_context = run_context
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower() # 'learn' は run() で処理

        # Merge learned selectors (V88.5.0: Moved from run() to here and run())
        try:
            instance_dir = Path(run_context.run_path).parent.parent
            learned_path = instance_dir / "sites" / site.upper() / "learned_selectors.json"
            if learned_path.exists():
                learned = json.loads(learned_path.read_text(encoding="utf-8"))
                sc_sel = site_config.setdefault("selectors", {}).setdefault("pdp", {})
                for k in ("price_selectors","title_selectors","pdp_link_selectors"):
                    v = learned.get(k)
                    if v: sc_sel[k] = list(dict.fromkeys(list(v) + list(sc_sel.get(k, []))))
                logger.info(f"[LEARN] loaded and merged selectors: {learned_path}")
        except Exception as e: logger.warning(f"[LEARN] merge skipped: {e}")


        page: Optional[Page] = None
        base_result: Optional[DiscoveryResult] = None
        healed_result: Optional[DiscoveryResult] = None # V88.5.2: 修復結果を格納

        ### PHASE 1: 通常フローを試す（ブラウザは開くがまだ閉じない）
        try:
            page = await self._open_session(
                site=site,
                site_config=site_config,
                run_context=run_context,
                settings=settings,
                target_url=target_url,
                timeout_ms=timeout_ms,
                likely_plp=likely_plp,
            )

            # 直前フック (VRT / Moncler回復パッチ 等)
            await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
            if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "plp", settings)

            # Moncler legacy patch is PDP-only; PLP flows rely on plugin/target URL normalization.
            if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery and not likely_plp:
                try:
                    await moncler_plp_recovery(page, site_config, {"query": query, "shipTo": "GB"})
                except Exception as _e:
                    self.logger.warning(f"[MonclerPatch] skipped: {_e}")

            context = self._context
            if context is None:
                raise ValueError("BrowserContext was not initialized by _open_session")

            start_t, budget_ms = self._start_watchdog(
                settings.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)
            )

            # mode='learn' は run() 側専用。明示的に弾く。
            if mode == "learn":
                raise ValueError("mode='learn' is not supported in run_with_repair. Use run() instead.")

            if likely_plp:
                # ★ V88.5.9: target_url を _run_plp_flow に渡す
                base_result = await self._run_plp_flow(
                    page, context, site, query,
                    site_config, settings, run_context,
                    target_url=target_url, # ★ 回復試行のため
                    start_t=start_t, budget_ms=budget_ms
                )
            else:
                base_result = await self._run_pdp_flow(
                    page, site, query, settings, run_context, site_config
                )

        except Exception as e:
            # run本体が途中で死んだ場合でも base_result を必ず構築
            # V88.5.1: (Patch) _open_session() 自体が失敗した場合、 self._page は None の可能性がある
            # _handle_run_failure は None を受け取れる
            base_result = await self._handle_run_failure(
                e, site, query, site_config, run_context, self._page
            )

        ### PHASE 2: 成功してたらここで終了。修復は不要。
        if getattr(base_result, "ok", False):
            try:
                await self._close_session(run_context, settings)
            finally:
                if hasattr(self, "run_context"):
                    del self.run_context
            return base_result

        # 5. 失敗→ インタラクティブ修復
        self.logger.warning(f"[run_with_repair] Initial run failed. Entering guided repair loop (Atlas-style). Reason: {base_result.message}")

        ### PHASE 3: 失敗なのでインタラクティブ修復に入る
        if InteractiveRepairSession is None:
            self.logger.error("[run_with_repair] InteractiveRepairSession not available, cannot self-heal.")
            try:
                await self._close_session(run_context, settings)
            finally:
                if hasattr(self, "run_context"):
                    del self.run_context
            return base_result

        # LLMコントローラを用意
        try:
            from app.utils.ai_llm_controller import AiLlmController
            llm_ctrl = AiLlmController(mode="Chat/Default")
        except Exception as e:
            self.logger.error(f"[run_with_repair] Failed to instantiate AiLlmController: {e}. Aborting repair.")
            try:
                await self._close_session(run_context, settings)
            finally:
                if hasattr(self, "run_context"):
                    del self.run_context
            return base_result

        # failure_contextをまとめて InteractiveRepairSession に渡すために整形
        failure_ev = base_result.evidence or {}
        failure_ctx_from_run = failure_ev.get("failure_context", {})
        failure_ctx = {
            "final_url": failure_ev.get("final_url"),
            "page_html_path": failure_ctx_from_run.get("dom_snapshot_path"),
            "screenshot_path": (failure_ctx_from_run.get("screenshots") or [None])[0],
            "exception_message": (failure_ctx_from_run.get("errors") or [""])[0],
            "selectors_used": {
                "pdp": (site_config.get("selectors", {}) or {}).get("pdp", {}),
            },
            "intent_description": failure_ctx_from_run.get("intent_description",
                "Goal: Extract PLP items or PDP price. Initial attempt failed."),
        }

        self.logger.info(f"[run_with_repair] Starting InteractiveRepairSession with failure context (Error: {failure_ctx['exception_message']})")

        # --- V88.5.2: ここで Atlasループ開始 (タイムアウト監視付き) ---
        repair_out = None
        repair_status = "unknown_error" # デフォルト

        try:
            # NOTE: InteractiveRepairSession はファイル先頭でimport済み
            repair_session = InteractiveRepairSession(
                ai_controller=llm_ctrl,
                run_context=run_context,
                max_steps=max_steps, # V88.5.2: ステップ上限を渡す
            )

            # run_repair_loop が sync の場合/async の場合どっちも耐えるラッパー
            maybe_coro = repair_session.run_repair_loop(
                page=self._page,  # ★ セッション中のpageを引き継ぐ
                site_key=site,
                intent="Collect PLP items and PDP prices",
                initial_failure=failure_ctx, # V88.5.2: v88.5.1 のリッチなコンテキストを維持
            )

            # V88.5.2: ユーザー要求の repair_budget_ms でタイムアウト監視
            repair_budget_sec = max(5.0, float(repair_budget_ms) / 1000.0)

            if asyncio.iscoroutine(maybe_coro):
                self.logger.info(f"[run_with_repair] Waiting for async repair loop (budget: {repair_budget_sec}s)")
                repair_out = await asyncio.wait_for(maybe_coro, timeout=repair_budget_sec)
            else:
                # 同期版のタイムアウト監視は難しいが、v88.5.1J のフォールバックを維持
                self.logger.warning("[run_with_repair] Running sync repair loop (cannot enforce budget)")
                repair_out = maybe_coro # 同期実行

            repair_status = "completed" # タイムアウト/例外なく完了

        except asyncio.TimeoutError:
            self.logger.error(f"[run_with_repair] InteractiveRepairSession timed out after {repair_budget_sec}s.")
            repair_status = "timeout_exceeded" # V88.5.2: タイムアウト
            # V88.5.2: タイムアウトした場合、失敗として healed_result を設定
            healed_result = DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=f"Repair loop timed out after {repair_budget_sec}s",
                evidence={"status": "timeout_exceeded", "initial_failure": base_result.evidence}
            )

        except Exception as repair_e:
            self.logger.error(
                f"[run_with_repair] InteractiveRepairSession failed catastrophically: {repair_e}",
                exc_info=True
            )
            repair_status = "catastrophic_failure"
            # V88.5.2: 修復が例外で死んだ場合、失敗として healed_result を設定
            healed_result = DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=f"Repair loop failed: {repair_e}",
                evidence={"status": "catastrophic_failure", "initial_failure": base_result.evidence}
            )

        # --- V88.5.2: 修復結果の厳格な評価 ---
        if repair_status == "completed" and repair_out:
            # 6. 修復結果の処理
            selectors_update = repair_out.get("selectors_update")
            code_patch = repair_out.get("code_patch", "")

            if selectors_update:
                # overrides_store.update_site_selectors(...) を使ってマージ
                try:
                    from app.utils.overrides_store import update_site_selectors
                    site_block = selectors_update.get(site.upper()) or selectors_update.get(site) or {}
                    new_sels = (site_block.get("selectors") or {})
                    if new_sels:
                        overrides_path = "app/config/sites/overrides.local.json" # TODO: パスを動的にすべきかも
                        updated, diff_txt = update_site_selectors(
                            site=site.upper(),
                            new_selectors=new_sels,
                            overrides_path=overrides_path,
                       )
                        if updated and diff_txt:
                            self.logger.info(f"[run_with_repair] Merged selectors_update to {overrides_path}")
                            run_context.save_text("repair_selector_diff.patch", diff_txt)
                        else:
                            self.logger.info("[run_with_repair] Selectors update did not result in changes.")
                    else:
                        self.logger.info("[run_with_repair] Selectors update was empty.")
                except ImportError:
                     self.logger.error("[run_with_repair] `app.utils.overrides_store` not found. Cannot merge selectors.")
                except Exception as merge_e:
                    self.logger.warning(f"[run_with_repair] could not merge selectors_update: {merge_e}")

            if code_patch:
                self.logger.info("[run_with_repair] Received code patch suggestion. Saving to artifacts.")
                run_context.save_text("browser_use_agent.patch", code_patch)

            # V88.5.2: 成果物（セレクタ or パッチ）があるか？
            if selectors_update or code_patch:
                self.logger.info("[run_with_repair] Repair loop completed and produced artifacts.")
                # 7. 成功した扱いのDiscoveryResult を組み立てる
                healed_result = DiscoveryResult(
                    ok=True, # 成果物があるので成功とみなす
                    site=site,
                    query=query,
                    message="Recovered via InteractiveRepairSession",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "selectors_update": selectors_update,
                        "code_patch": code_patch,
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []),
                        "status": "recovered",
                    },
                )
            else:
                # V88.5.2: ユーザー要求。修復が完了したが成果物がない場合 (max_steps 超過や stalled など)
                self.logger.warning("[run_with_repair] Repair loop completed but produced no artifacts (likely exhausted or stalled).")
                status_from_repair = repair_out.get("status", "exhausted_steps") # 'stalled' や 'exhausted_steps' を期待
                healed_result = DiscoveryResult(
                    ok=False, # 成果物がないので失敗
                    site=site,
                    query=query,
                    message=f"Repair loop finished without artifacts (Status: {status_from_repair})",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []),
                        "status": status_from_repair,
                        "initial_failure": base_result.evidence
                    }
                )

        elif not healed_result: # タイムアウトでも例外でもないが、repair_out が None (or status != completed)
             self.logger.error(f"[run_with_repair] Repair loop finished abnormally (Status: {repair_status}, repair_out: {repair_out})")
             healed_result = DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=f"Repair loop failed with unknown status: {repair_status}",
                evidence={"status": repair_status, "initial_failure": base_result.evidence}
            )

        # 8. セッションを閉じて、修復結果（成功または失敗）を返す
        try:
            await self._close_session(run_context, settings)
        finally:
            if hasattr(self, "run_context"):
                del self.run_context

        # V88.5.2: 修復フェーズに入った場合、必ず healed_result (修復成功/修復失敗/タイムアウト) が返る
        return healed_result

    # --- Settings Resolution ---
    def _resolve_run_settings(self, site_config: Dict[str, Any]) -> Dict[str, Any]:
        ds = site_config.get("discovery_settings", {}) or {}
        site_key_guess = (
            site_config.get("site_key")
            or site_config.get("id")
            or self.runtime_kwargs.get("site")
            or ""
        )
        site_key_guess = str(site_key_guess or "").upper()
        vrt = ds.get("vrt", {}) or {}
        self.logger.info(f"[Debug] runtime enable_video flag: {self.runtime_kwargs.get('enable_video')}")
        enable_har = self.runtime_kwargs.get("enable_har", ds.get("enable_har", True))
        enable_trace = self.runtime_kwargs.get("enable_trace", ds.get("enable_trace", True))

        # --- enable_video resolution (CLI > site config > env > default) ---
        cli_enable_video = self.runtime_kwargs.get("enable_video")
        cfg_enable_video = ds.get("enable_video")
        env_raw = os.getenv("ATK_ENABLE_VIDEO") or os.getenv("ENABLE_VIDEO")
        env_enable_video: Optional[bool] = None
        if env_raw is not None:
            env_enable_video = str(env_raw).strip().lower() in ("1", "true", "yes", "y", "on")

        if cli_enable_video is not None:
            enable_video = bool(cli_enable_video)
        elif cfg_enable_video is not None:
            enable_video = bool(cfg_enable_video)
        elif env_enable_video is not None:
            enable_video = env_enable_video
        else:
            enable_video = True if site_key_guess == "MONCLER_OFFICIAL" else False

        default_accept_language = "en-GB,en;q=0.8"
        if site_key_guess == "MONCLER_OFFICIAL":
            default_accept_language = "en-US,en;q=0.8"

        settings = {
            "timeout_sec": self.runtime_kwargs.get("timeout_sec") or ds.get("timeout_sec", 60),
            "headless": self.runtime_kwargs.get("headless", True),
            "slow_mo": self.runtime_kwargs.get("slow_mo", 0),
            "viewport": ds.get("viewport"), "user_agent": ds.get("user_agent"),
            "extra_http_headers": ds.get("extra_http_headers"),
            "accept_language": ds.get("accept_language", default_accept_language),
            "enable_har": enable_har, "enable_trace": enable_trace,
            "enable_video": enable_video,
            "enable_locale_escape": bool(ds.get("enable_locale_escape", True)),
            "overall_plp_budget_ms": int(ds.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)),
            "pdp_parallel_limit": int(ds.get("pdp_parallel_limit", DEFAULT_PDP_PARALLEL_LIMIT)),
            "pdp_retry_once": bool(ds.get("pdp_retry_once", True)),
            "enable_visual_regression_check": bool(ds.get("enable_visual_regression_check", False)),
            "vrt_scope": (vrt.get("scope") or "none").lower(),
            "vrt_threshold": float(vrt.get("threshold", 0.02)),
            "vrt_hard_fail_threshold": float(vrt.get("hard_fail_threshold", 0.05)),
            "vrt_fail_on_hard_threshold": bool(vrt.get("fail_on_hard_threshold", True)),
            "vrt_baseline_dir": vrt.get("baseline_dir"),
            "vrt_plp_selector": vrt.get("plp_selector") or "full_page",
            "vrt_pdp_selector": vrt.get("pdp_selector") or "full_page",
            "vrt_auto_update_baseline": bool(vrt.get("auto_update_baseline", False)),
            "vrt_save_failed_diff_only": bool(vrt.get("save_failed_diff_only", True)),
            "wait_for_selectors": _dedupe_keep_order(ds.get("wait_for_selectors") or []),
            "wait_until": ds.get("wait_until") or "domcontentloaded",
            "plp_scroll_rounds": int(ds.get("plp_scroll_rounds", 10)),
            "extra_block_routes": _dedupe_keep_order(ds.get("extra_block_routes") or []),
            "pdp_price_wait_ms": int(ds.get("pdp_price_wait_ms", 4000)),
            "locale_recover_max": int(ds.get("locale_recover_max", 5)),
            "enable_human_like": bool(self.runtime_kwargs.get("enable_human_like", ds.get("enable_human_like", False))),
            "enable_ua_rotation": bool(self.runtime_kwargs.get("enable_ua_rotation", ds.get("enable_ua_rotation", False))),
            "enable_viewport_rotation": bool(self.runtime_kwargs.get("enable_viewport_rotation", ds.get("enable_viewport_rotation", False))),
        }
        try:
            pdp_policy_cfg = ds.get("pdp_size_select_policy", {})
            settings["pdp_size_select_policy"] = PDPSizeSelectPolicy(
                mode=pdp_policy_cfg.get("mode", "off"),
                prefer_labels=pdp_policy_cfg.get("prefer_labels", [])
            )
        except Exception as e:
            logger.warning(f"Could not parse PDPSizeSelectPolicy: {e}. Defaulting to 'off'.")
            settings["pdp_size_select_policy"] = PDPSizeSelectPolicy()
        return settings

    # --- Time Budget Helpers ---
    @staticmethod
    def _start_watchdog(budget_ms: int) -> Tuple[float, int]:
        return time.monotonic(), int(budget_ms)

    @staticmethod
    def _time_left_ms(start_t: float, budget_ms: int) -> int:
        used = int((time.monotonic() - start_t) * 1000)
        return max(0, budget_ms - used)

    @staticmethod
    def _slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
        return max(500, min(left_ms, cap_ms))

    # --- Safe Wait ---
    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        if not page or page.is_closed(): return False
        try:
            await page.wait_for_selector(selector, state=state, timeout=timeout_ms)
            return True
        except Exception: return False

    # --- UI Helpers ---
    async def _kill_overlays(self, page: Page) -> None:
        try:
            await page.evaluate("""
              (() => {
                const sels = ['.overlay','.backdrop','.modal-backdrop','#onetrust-banner-sdk','.cookie-banner','[aria-modal="true"]','.cmp-ui-overlay','.cmp-modal','.drawer--open'];
                document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
                const b = document.body; if (b) { b.classList.remove('modal-open','locked','no-scroll','overflow-hidden'); b.style.overflow=''; }
                const html=document.documentElement; if (html) { html.style.overflow=''; html.classList.remove('no-scroll','overflow-hidden'); }
              })();
            """)
        except Exception: pass

    async def _click_continue_shopping_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        candidates = _dedupe_keep_order((ui.get("continue_shopping") or []) + ["a:has-text('CONTINUE SHOPPING')", "button:has-text('CONTINUE SHOPPING')", "[role='button']:has-text('CONTINUE SHOPPING')", "text=/\\bCONTINUE\\s+SHOPPING\\b/i"])
        for _ in range(3):
            try: await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            except Exception: pass
            for sel in candidates:
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0 and await el.is_visible():
                        await el.click(timeout=3000)
                        try: await page.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception: pass
                        return True
                except Exception: continue
            await page.wait_for_timeout(1200)
        return False

    async def _pause_for_operator(self, page: Optional[Page], run_context: Optional[RunContext], label: str) -> None:
        """Headful 実行中に人間が介入して操作できるよう一時停止する。"""
        if not self.runtime_kwargs.get("interactive_pause"):
            return
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.lower())
        if page and run_context:
            try:
                await run_context.take_screenshot(page, f"50_operator_{slug}")
            except Exception as e:
                self.logger.debug(f"[OperatorPause] screenshot failed: {e}")
        prompt = (
            f"\n[OperatorPause] '{label}' で一時停止中です。"
            f" Playwright ウィンドウを操作したら Enter を押して再開してください..."
        )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: input(prompt))
        except (EOFError, RuntimeError):
            self.logger.warning("[OperatorPause] input() が使えません。即座に続行します。")
    async def _accept_cookies_if_present(self, page: Page, site_config: Dict[str, Any]) -> bool:
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        candidates = _dedupe_keep_order((ui.get("cookie_accept") or []) + ["#onetrust-accept-btn-handler", "button:has-text('ACCEPT ALL')", "button:has-text('CONTINUE WITHOUT ACCEPTING')", "button[aria-label*='Accept' i]"])
        for sel in candidates:
            try:
                node = page.locator(sel).first
                if await node.count() > 0 and await node.is_visible():
                    await node.click(timeout=3000); await asyncio.sleep(0.2); return True
            except Exception: continue
        return False

    async def _dismiss_geo_modal(self, page: Page, site_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Stage 4: ジオ / ロケール関係のモーダルを潰す（汎用化）
        
        site_config["navigation"]["overlays"]["geo_modal_selectors"] を使用
        """
        # Stage 4: site_configからgeo_modal_selectorsを取得
        nav_cfg = (site_config.get("navigation", {}) or {}) if site_config else {}
        overlays_cfg = nav_cfg.get("overlays", {}) or {}
        geo_selectors = overlays_cfg.get("geo_modal_selectors", [])
        
        # デフォルトのセレクタ（site_configに定義がない場合）
        default_selectors = [
            "text=STAY HERE",
            "text=REMAIN HERE",
            "text=REMAIN IN ENGLISH",
            "text=CONTINUE SHOPPING",
            "text=ショッピングを続ける",
        ]
        
        # site_configのセレクタとデフォルトをマージ
        all_selectors = _dedupe_keep_order(geo_selectors + default_selectors)
        
        for sel in all_selectors:
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
                self.logger.info(f"[GeoModal] Clicked {desc}")
                await page.wait_for_timeout(300)
                return True
            except Exception as e:
                self.logger.debug(f"[GeoModal] Click failed ({desc}): {e}")
                return False

        # Stage 4: ターゲットロケールへの遷移を待つ（汎用化）
        async def _wait_for_target_locale(timeout_ms: int = 4000) -> bool:
            if not site_config:
                return True
            locale_cfg = (site_config.get("locale", {}) or {})
            prefer_locale = locale_cfg.get("prefer", "")
            
            if not prefer_locale:
                return True
            
            try:
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
                self.logger.info("[GeoModal] Locale gate header detected.")

            # Stage 4: site_configから優先ロケールを取得（汎用化）
            if not site_config:
                # site_configがない場合はデフォルトの処理のみ
                return
            
            nav_cfg = (site_config.get("navigation", {}) or {})
            overlays_cfg = nav_cfg.get("overlays", {}) or {}
            geo_modal_preferred_locale = overlays_cfg.get("geo_modal_preferred_locale", "")
            locale_cfg = (site_config.get("locale", {}) or {})
            prefer_locale = locale_cfg.get("prefer", geo_modal_preferred_locale)
            
            # 優先ロケールに基づく候補セレクタ（デフォルトはen-gb）
            if prefer_locale and "gb" in prefer_locale.lower():
                # United Kingdom / English の候補
                preferred_candidates = [
                    page.get_by_text(re.compile(r"UNITED\\s+KINGDOM\\s*\\|\\s*ENGLISH", re.I)),
                    page.get_by_role("link", name=re.compile(r"UNITED\\s+KINGDOM\\s*\\|\\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"UNITED\\s+KINGDOM\\s*\\|\\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"United\\s+Kingdom.*English", re.I)),
                    page.get_by_role("link", name=re.compile(r"United\\s+Kingdom.*English", re.I)),
                    page.locator("[data-testid*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("[data-component*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("button:has-text('United Kingdom EN')"),
                    page.locator("text=/United\\s+Kingdom\\s*\\|\\s*English/i"),
                ]
            else:
                # その他のロケールの場合は、geo_modal_selectorsを使用
                geo_selectors = overlays_cfg.get("geo_modal_selectors", [])
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

            try:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(200)
            except Exception:
                pass

            try:
                await page.evaluate(
                    """
                    () => {
                      const headers = Array.from(
                        document.querySelectorAll('*')
                      ).filter(el => el.textContent && el.textContent.includes('Select your location'));
                      const roots = headers.map(h => h.closest('div[role="dialog"], [data-testid*="modal"], .modal, .c-modal')).filter(Boolean);
                      roots.forEach(root => root.remove());
                    }
                    """
                )
            except Exception:
                pass
        except Exception:
            return

    # --- Locale Helpers (Moncler specific but could be generalized) ---
    # Stage 4: _normalize_to_en_int_url() を削除し、NavigationDriver._normalize_url() を使用
    def _normalize_url(self, url: str, site_config: Dict[str, Any]) -> str:
        """
        Stage 4: URL正規化のヘルパー（NavigationDriver経由）
        NavigationDriver._normalize_url() を呼び出すためのラッパー
        
        Args:
            url: 正規化するURL
            site_config: サイト設定
            
        Returns:
            str: 正規化されたURL
        """
        from app.agents.browser.navigation_driver import NavigationDriver
        # NavigationDriverインスタンスを作成（pageは不要なのでNone）
        # ただし、_normalize_urlはインスタンスメソッドなので、page=Noneでも動作するようにする
        try:
            driver = NavigationDriver(page=None)  # type: ignore
            return driver._normalize_url(url, site_config)
        except Exception as e:
            # フォールバック: NavigationDriverが使えない場合は元のURLを返す
            logger.warning(f"[_normalize_url] NavigationDriver failed, returning original URL: {e}")
            return url

    # Stage 4: _force_en_int() は Moncler 専用パッチに移行すべき機能のため、このメソッドは削除または無効化
    # 必要に応じて browser_use_moncler_patch.py の moncler_plp_recovery() を使用
    async def _force_en_int(self, page: Page, site_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Stage 4: このメソッドは Moncler 専用の機能のため、site_config の enable_moncler_patch フラグで制御
        """
        if site_config is None:
            logger.warning("[_force_en_int] site_config is None, skipping (Stage 4 migration)")
            return
        
        # Moncler専用パッチが有効な場合のみ実行
        ds = (site_config.get("discovery_settings", {}) or {})
        if not ds.get("enable_moncler_patch", False):
            logger.debug("[_force_en_int] Moncler patch is disabled in site_config")
            return
        
        try:
            # Cookie注入は site_config から取得（Moncler専用パッチに移行すべき）
            if page.context:
                # この部分は browser_use_moncler_patch.py に移行すべき
                logger.debug("[_force_en_int] Cookie injection should be handled by moncler_plp_recovery()")
        except Exception: pass
        try:
            fixed = self._normalize_url(page.url, site_config)
            if fixed != page.url:
                await page.goto(url=fixed, wait_until="domcontentloaded")
                try: await page.wait_for_load_state("networkidle", timeout=1500)
                except Exception: pass
        except Exception: pass

    # --- PLP Materialize ---
    # Stage 3A-2-2: 実体は NavigationDriver.ensure_plp_materialized に移行済み
    # このメソッドは NavigationDriver が使われていない場合のフォールバックとして残している
    # TODO: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能
    async def _ensure_plp_materialized(self, page: Page, site_config: Dict[str, Any], settings: Dict[str, Any], *, start_t: float, budget_ms: int, target_url: Optional[str] = None) -> bool:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        tile_selectors = _dedupe_keep_order((pdp_cfg.get("pdp_link_selectors") or []) + (pdp_cfg.get("plp_container_selectors") or []) + ["a[data-product-url]", "[data-product-url]", "[data-qa='product-tile']", ".product-card", ".c-product-card", ".c-product-tile", "[data-testid*='product' i]"])
        tile_selector_str = ", ".join(tile_selectors)
        target_min_tiles = 8
        max_scroll_attempts = int(max(settings.get("plp_scroll_rounds", 10), 10))
        run_ctx = getattr(self, "run_context", None)

        locale_recover_attempts = 0
        locale_recover_max = int(settings.get("locale_recover_max", 5))

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0: self.logger.warning("[Materialize] Timed out."); return False

            # v88.6.x: Attemptごとに遅延表示ゲート/バナーを掃除する
            try:
                await self._accept_cookies_if_present(page, site_config)
            except Exception:
                pass
            try:
                await self._dismiss_geo_modal(page, site_config)
            except Exception:
                pass
            try:
                await self._kill_overlays(page)
            except Exception:
                pass

            # Stage 4: ロケールリダイレクトの検出（汎用化）
            current_url = (page.url or "").lower()
            locale_cfg = (site_config.get("locale", {}) or {})
            prefer_locale = locale_cfg.get("prefer", "")
            allowed_domain = site_config.get("allowed_domain", "")
            
            # ターゲットロケール以外のロケールにリダイレクトされた場合を検出
            if prefer_locale and allowed_domain:
                target_locale_path = f"/{prefer_locale}/"
                if allowed_domain.lower() in current_url and target_locale_path not in current_url:
                    # ロケールセグメントが含まれているかチェック
                    if _LOCALE_SEG_RE.search(current_url):
                        self.logger.warning(f"[Materialize] Detected locale redirect away from {prefer_locale} mid-attempt: {current_url}")
                        if locale_recover_attempts >= locale_recover_max:
                            self.logger.error("[Materialize] Locale recovery exceeded max attempts. Aborting.")
                            return False
                        locale_recover_attempts += 1
                        if target_url:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(800)
                            continue

            if run_ctx is not None and attempt < 3:
                try:
                    await run_ctx.take_screenshot(
                        page,
                        f"30_plp_materialize_attempt_{attempt + 1:02d}"
                    )
                except Exception as ss_e:
                    self.logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6): await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))"); await page.wait_for_timeout(160)
                try: await page.wait_for_load_state("networkidle", timeout=800)
                except Exception: pass
            except Exception as e: self.logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}"); break

            # Moncler locale gate が途中で出た場合に備えて閉じておく
            try:
                modal_title = page.locator("text=Select your location").first
                if await modal_title.count() > 0:
                    self.logger.info("[GeoModal] Locale gate header detected during PLP materialization.")
                    close_btn = page.locator(
                        "button[aria-label*='close' i], "
                        "button:has-text('Close'), "
                        "button:has-text('×'), "
                        ".modal__close, .c-modal__close"
                    ).first
                    if await close_btn.count() > 0:
                        await close_btn.click(timeout=3000)
                        await page.wait_for_timeout(500)
                        self.logger.info("[GeoModal] Locale gate closed.")
            except Exception as e:
                self.logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

            try:
                count = await page.locator(tile_selector_str).count()
                self.logger.info(f"[Materialize] Attempt {attempt + 1}/{max_scroll_attempts}, found {count} tiles.")
                if count >= target_min_tiles:
                    self.logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles})."); return True
                if count < 4 and attempt >= 1:
                    self.logger.warning(f"[Materialize] Low tiles ({count}) after {attempt+1} attempts, forcing recovery hop.")
                    if target_url:
                        try:
                            await self._force_plp_recover(page, site_config, target_url)
                            await page.wait_for_timeout(500)
                            rec_count = await page.locator(tile_selector_str).count()
                            self.logger.info(f"[Materialize] After recovery hop, tiles={rec_count}")
                            if rec_count >= target_min_tiles:
                                return True
                        except Exception as rec_e:
                            self.logger.warning(f"[Materialize] Recovery hop failed: {rec_e}")
                    return False
            except asyncio.CancelledError:
                self.logger.warning("[Materialize] Cancelled during tile count.")
                return False
            except Exception as e: self.logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
        if final_count > 0: self.logger.warning(f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty."); return True
        self.logger.error("[Materialize] Failed: No product tiles found after all scroll attempts."); return False

    # --- Price / Size Selection ---
    async def _read_price_or_none(self, page: Page) -> Optional[str]:
        try:
            for sel in PRICE_SELECTORS:
                loc = page.locator(sel)
                count = await loc.count()
                if count == 0: continue
                for i in range(count):
                    el = loc.nth(i)
                    try:
                        tag = (await el.evaluate("e => e && e.tagName") or "").lower()
                        if not tag: continue
                        content = await (el.get_attribute("content") if tag == "meta" else el.inner_text())
                    except Exception: continue
                    if content:
                        s = content.strip()
                        if s and re.search(r'\d', s): self.logger.debug(f"Price found via selector '{sel}' (nth={i}): {s}"); return s
        except Exception as e: self.logger.warning(f"Error: Exception during _read_price_or_none (outer loop): {e}")
        self.logger.debug("Price string not found (_read_price_or_none)."); return None

    # --- PDP Extraction ---
    # --- PLP -> PDP Link Collection (Robust v85.5) ---
    def _normalize_abs_url(self, base_url: str, href: str) -> str:
        try:
            absu = urljoin(base_url, href)
            parts = list(urlsplit(absu))
            if parts[2].endswith('/'): parts[2] = parts[2].rstrip('/')
            parts[3] = ""; parts[4] = "" # query & fragment
            return urlunsplit(parts)
        except Exception: return href

    # Stage 3A-2-1: 実体は NavigationDriver.collect_pdp_links に移行済み
    # このメソッドは薄いラッパーとして残している（互換性維持）
    # 注意: _run_plp_flow 内では NavigationDriver.run_plp_flow の結果（nav_outcome.pdp_links）を優先的に使用
    # TODO: Stage 3A-2 完了後、すべての呼び出しが NavigationDriver 経由になったら削除可能
    async def _collect_pdp_links(self, page: Page, site_config: Dict, settings: Dict, run_context: RunContext) -> List[str]:
        """
        Stage 3A-2-1:
        PDPリンク収集の本体は NavigationDriver.collect_pdp_links に移譲。
        互換性維持のため、既存シグネチャは残しておく。
        
        注意: このメソッドは _run_plp_flow 内から呼ばれるため、
        将来的には _run_plp_flow から直接 NavigationDriver を呼ぶ形に変更する。
        """
        # Stage 3A-2-1: NavigationContext を構築して NavigationDriver を呼ぶ
        from app.agents.browser.navigation_driver import NavigationContext, NavigationDriver
        import time
        
        # NavigationContext を構築（site, query は runtime_kwargs から取得）
        nav_ctx = NavigationContext(
            site=self.runtime_kwargs.get("site", "UNKNOWN"),
            query=self.runtime_kwargs.get("query", ""),
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            start_t=time.time(),
            budget_ms=int(settings.get("timeout_sec", 60)) * 1000,
        )
        
        # Stage 3B: TelemetryClient を初期化
        telemetry = TelemetryClient(run_context)
        navigation_driver = NavigationDriver(
            page=page,
            telemetry=telemetry,  # Stage 3B: TelemetryClient を渡す
            trap_checker=None,
            strategy=None,
        )
        return await navigation_driver.collect_pdp_links(nav_ctx)

    # --- V88.3.0J: _run_deep_extraction_phase2 Safer Fallback Evaluate ---
    # --- V88.6.2J: (BugFix) SyntaxError on container_sels ---
    async def _run_deep_extraction_phase2(self, page: Page, site_config: Dict) -> List[str]:
        logger.debug("[Phase 2] Running deep extraction (JSON-LD, onclick, data-*, ...)")
        # ★ 88.6.2: (BugFix) 括弧が過剰だった SyntaxError を修正
        container_sels: List[str] = (
            ((site_config.get("selectors") or {}).get("pdp") or {}).get("plp_container_selectors", []) or []
        )
        for cont in (container_sels or []): await self.safe_wait_selector(page, cont, timeout_ms=1000, state="visible")
        try:
            for _ in range(2): await page.evaluate("window.scrollBy(0, document.body.scrollHeight)"); await page.wait_for_timeout(200)
        except Exception: pass

        # V86.0: Strict mode violation prevention + V88.2: Get ElementHandle
        scope = page.locator("main, [role='main'], #main, #app")
        handle: Optional[ElementHandle] = None
        try:
            await scope.first.wait_for(state="attached", timeout=4000)
            handle = await scope.first.element_handle(timeout=4000) # Get handle
        except Exception as e_handle:
             # (V88.6.2: ログレベルは warning のまま)
             logger.warning(f"[Phase 2] Could not get element handle for scope: {e_handle}. Falling back to page evaluate.")
             handle = None # Ensure handle is None if getting it failed

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
            if (handle):
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
            hrefs = [] # Ensure hrefs is a list even on error

        return _dedupe_keep_order(hrefs)
    # --- V88.2.0-V88.3.0 修正ここまで ---

    # --- VRT ---
    async def _perform_vrt(self, page: Page, scope: str, settings: Dict[str, Any]):
        try:
            from pathlib import Path as _P
            site_name = self.runtime_kwargs.get("site") or "GENERIC"
            baseline_dir = _P(settings.get("vrt_baseline_dir") or f"app/visual_baselines/{site_name}")
            baseline_dir.mkdir(parents=True, exist_ok=True)
            sel = settings.get("vrt_plp_selector") if scope=="plp" else settings.get("vrt_pdp_selector")
            res = await compare_and_maybe_update(page=page, baseline_path=baseline_dir / f"{scope}.png", selector=sel, threshold=settings.get("vrt_threshold", 0.02), hard_fail_threshold=settings.get("vrt_hard_fail_threshold", 0.05), auto_update_baseline=settings.get("auto_update_baseline", False), save_failed_diff_only=settings.get("save_failed_diff_only", True))
            vrt_result = _unpack_vrt(res, threshold=settings.get("vrt_threshold", 0.02))
            if vrt_result and not vrt_result.get("is_ok", True):
                diff = float(vrt_result.get("diff_percent", 0.0))
                hard = diff > float(settings.get("vrt_hard_fail_threshold", 0.05))
                msg = f"[VRT][{scope.upper()}] diff={diff:.4f} (thr={settings.get('vrt_threshold', 0.02)})"
                if hard and settings.get("vrt_fail_on_hard_threshold", True) and scope == "pdp": raise PlaywrightError(msg + f" HARD>{settings.get('vrt_hard_fail_threshold', 0.05)}")
                logger.warning(msg + (" HARD" if hard else ""))
        except Exception as vrt_e: logger.warning(f"[VRT][{scope.upper()}] skipped: {vrt_e}")

    # --- Human-like interaction helpers ---
    async def _human_like_pause(self, page: Page, *, min_ms: int = 400, max_ms: int = 900):
        import random
        await page.wait_for_timeout(random.randint(min_ms, max_ms))

    async def _human_like_mouse_move(self, page: Page):
        import random
        try:
            box = await page.evaluate("""() => ({ w: window.innerWidth, h: window.innerHeight })""")
            w, h = int(box.get("w", 1280)), int(box.get("h", 720))
        except Exception:
            w, h = 1280, 720
        moves = random.randint(3, 6)
        for _ in range(moves):
            x = random.randint(int(w * 0.1), int(w * 0.9))
            y = random.randint(int(h * 0.1), int(h * 0.9))
            await page.mouse.move(x, y, steps=random.randint(5, 12))
            await self._human_like_pause(page, min_ms=120, max_ms=280)

    async def _human_like_scroll(self, page: Page):
        import random
        try:
            total_height = await page.evaluate("() => document.body ? document.body.scrollHeight : 0")
        except Exception:
            total_height = 0
        if not total_height:
            await page.mouse.wheel(0, random.randint(200, 600))
            await self._human_like_pause(page, min_ms=200, max_ms=400)
            return
        viewport = await page.evaluate("() => ({h: window.innerHeight || 800})")
        vh = int(viewport.get("h", 800))
        steps = random.randint(2, 4)
        for _ in range(steps):
            delta = random.randint(int(vh * 0.3), int(vh * 0.6))
            await page.mouse.wheel(0, delta)
            await self._human_like_pause(page, min_ms=200, max_ms=500)

    # --- Main Run Logic (V88.5.0: Refactored for session management) ---
    async def run(self, *, site: str, query: str, site_config: Dict[str, Any], run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:

        # --- MONCLER 専用 DrissionPage ルート ---
        if site.upper() == "MONCLER_OFFICIAL" and MonclerDrissionHandler is not None:
            self.logger.info("[BrowserUseAgent] MONCLER検知: DrissionPageエンジンへ切り替えます")
            
            try:
                handler = MonclerDrissionHandler(runtime_kwargs=self.runtime_kwargs)
                
                # DrissionPage は同期ライブラリなので、スレッドオフロードで呼び出す
                discovery_result = await asyncio.to_thread(
                    handler.run,
                    query=query,
                    site_config=site_config,
                    run_context=run_context,
                    target_url=target_url,
                )
                
                # 結果が成功した場合はそのまま返す
                if discovery_result.ok:
                    self.logger.info("[BrowserUseAgent] DrissionPageルートで成功しました")
                    return discovery_result
                else:
                    # 失敗した場合は警告を出して Playwright ルートにフォールバック
                    self.logger.warning(
                        "[BrowserUseAgent] DrissionPageルートで失敗。Playwrightルートにフォールバックします: %s",
                        discovery_result.message,
                    )
                    # フォールバックのため、そのまま下の Playwright ロジックに続行
                    
            except Exception as e:
                self.logger.warning(
                    "[BrowserUseAgent] DrissionPageルートでエラー発生。Playwrightルートにフォールバックします: %s",
                    e,
                )
                # エラー情報を run_context に記録
                try:
                    run_context.save_json("drission_fallback_error.json", {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })
                except Exception:
                    pass
                # フォールバックのため、そのまま下の Playwright ロジックに続行

        # --- ここから下は既存の Playwright ロジック（変更しない） ---
        settings = self._resolve_run_settings(site_config)
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        self.run_context = run_context  # Attach for downstream helpers
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()
        
        # Stage 3B: TelemetryClient を初期化
        telemetry = TelemetryClient(run_context)
        tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id)

        # Stage 3C: Plugin API (Facade化) - 直接 PLUGIN_REGISTRY を import しない
        from app.agents.browser.plugin_api import PluginAPI

        plugin_api = PluginAPI(self.logger)
        plugin = plugin_api.get_plugin(site)
        plugin_ctx = {"query": query, "site_config": site_config}

        nav_url = plugin_api.before_navigate(
            plugin,
            site=site,
            target_url=target_url,
            ctx=plugin_ctx,
        )

        try:
            instance_dir = Path(run_context.run_path).parent.parent
            learned_path = instance_dir / "sites" / site.upper() / "learned_selectors.json"
            if learned_path.exists():
                learned = json.loads(learned_path.read_text(encoding="utf-8"))
                sc_sel = site_config.setdefault("selectors", {}).setdefault("pdp", {})
                for k in ("price_selectors", "title_selectors", "pdp_link_selectors"):
                    v = learned.get(k)
                    if v:
                        sc_sel[k] = list(dict.fromkeys(list(v) + list(sc_sel.get(k, []))))
                logger.info(f"[LEARN] loaded and merged selectors: {learned_path}")
        except Exception as e:
            logger.warning(f"[LEARN] merge skipped: {e}")

        page: Optional[Page] = None

        try:
            async with SessionManager(
                site=site,
                site_config=site_config,
                run_context=run_context,
                settings=settings,
                target_url=nav_url,
                timeout_ms=timeout_ms,
                likely_plp=likely_plp,
                runtime_kwargs=self.runtime_kwargs,
                logger=self.logger,
                # Stage 4: url_normalizerを汎用化（site_configを渡すラムダ関数）
            url_normalizer=lambda url: self._normalize_url(url, site_config) if hasattr(self, '_normalize_url') else url,
            ) as session:
                self._attach_session(session)
                page = session.page
                if page is None:
                    raise ValueError("SessionManager failed to provision a Playwright page.")

                page = await self._bootstrap_session_page(
                    page=page,
                    site=site,
                    site_config=site_config,
                    run_context=run_context,
                    settings=settings,
                    target_url=nav_url,
                    likely_plp=likely_plp,
                )

                await plugin_api.after_navigate(
                    plugin,
                    site=site,
                    page=page,
                    ctx=plugin_ctx,
                )

                await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
                if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                    await self._perform_vrt(page, "plp", settings)

                if (
                    site.upper() == "MONCLER_OFFICIAL"
                    and moncler_plp_recovery is not None
                    and plugin is None
                ):
                    try:
                        await moncler_plp_recovery(page, site_config, query)
                    except Exception as _e:
                        self.logger.warning(f"[MonclerPatch] skipped: {_e}")

                start_t, budget_ms = self._start_watchdog(
                    settings.get("overall_plp_budget_ms", OVERALL_PLP_BUDGET_MS_DEFAULT)
                )

                context = session.context
                if context is None:
                    raise ValueError("SessionManager did not provide a BrowserContext.")

                if mode == "learn":
                    return await self._run_learning_flow(
                        page, context, site, site_config, settings, run_context, start_t=start_t, budget_ms=budget_ms
                    )

                if likely_plp:
                    if (
                        site.upper() == "MONCLER_OFFICIAL"
                        and moncler_plp_recovery is not None
                        and plugin is None
                    ):
                        try:
                            await moncler_plp_recovery(page, site_config, query)
                        except Exception as _e:
                            self.logger.warning(f"[MonclerPatch] skipped: {_e}")
                    if plugin:
                        plp_ctx = {"plp_min_cards": 24, "query": query}

                        materialized = await plugin_api.materialize_plp(
                            plugin,
                            site=site,
                            page=page,
                            ctx=plp_ctx,
                        )
                        # materialized が False / None の場合も、現状と同じく
                        # 「デフォルト PLP フロー継続」で特別な制御は追加しない

                        asserted = await plugin_api.assert_plp(
                            plugin,
                            site=site,
                            page=page,
                            ctx=plp_ctx,
                        )
                        # asserted が False / None でも挙動は変えない（ログだけ）

                    nav_ctx = NavigationContext(
                        site=site,
                        query=query,
                        site_config=site_config,
                        settings=settings,
                        run_context=run_context,
                        start_t=start_t,
                        budget_ms=budget_ms,
                        entry_url=nav_url,
                        context=context,  # Stage 3A-2-4: BrowserContext を追加
                    )
                    # Stage 3B: TelemetryService を NavigationDriver に渡す
                    telemetry = self._ensure_telemetry()
                    # Stage 3A-3: trap_checker を NavigationDriver に渡す（観測用）
                    navigation_driver = NavigationDriver(
                        page=page,
                        # Stage 4: site_configを渡す
                        trap_checker=lambda url: self._looks_like_trap_or_legal(url, site_config),
                        telemetry=telemetry,  # Stage 3B: TelemetryClient を渡す
                        strategy=plugin,
                    )
                    try:
                        nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
                    except Exception as nav_e:
                        self.logger.debug(
                            f"[NavigationDriver] Stage3A-3 failed (fallback to legacy): {nav_e}"
                        )
                        nav_outcome = None

                    # Stage 3A-3: trap 観測結果をログに出す（挙動は変更しない）
                    if nav_outcome and getattr(nav_outcome, "trap_detected", False):
                        self.logger.warning(
                            "[NavigationDriver] trap-like url observed (legacy PLP flow will still run): %s",
                            nav_outcome.trap_reason,
                        )

                    skip_materialize = bool(getattr(nav_outcome, "plp_materialized", False))

                    return await self._run_plp_flow(
                        page,
                        context,
                        site,
                        query,
                        site_config,
                        settings,
                        run_context,
                        target_url=nav_url,
                        start_t=start_t,
                        budget_ms=budget_ms,
                        skip_materialize=skip_materialize,
                        nav_outcome=nav_outcome,  # Stage 3A-2: NavigationDriver の結果を渡す
                        plugin=plugin,  # Stage 3A-2-2: plugin を渡す
                    )

                return await self._run_pdp_flow(page, site, query, settings, run_context, site_config)

        except Exception as e:
            return await self._handle_run_failure(e, site, query, site_config, run_context, page or self._page)
        finally:
            self._detach_session()
            if hasattr(self, "run_context"):
                del self.run_context

    # --- Flow Logic: PLP ---
    # ★ V88.5.9: 回復ロジックを組み込み (v88.5.7 の即時失敗を置き換え)
    async def _run_plp_flow(self, page: Page, context: BrowserContext, site: str, query: str,
                            site_config: Dict, settings: Dict, run_context: RunContext,
                            target_url: str,
                            *, start_t: float, budget_ms: int, skip_materialize: bool = False,
                            nav_outcome: Optional[Any] = None, plugin: Optional[Any] = None) -> DiscoveryResult:
        """
        Stage 3A-1: NavigationDriver への最小限の委譲を追加。
        
        このステップでは、NavigationDriver を初期化して run_plp_flow を呼び出す配線だけを作成し、
        実際のナビゲーションロジックは現時点ではまだ旧実装を使います。
        """

        # --- Stage 3A-1: NavigationDriver への最小限の委譲 ---
        # NavigationContext を組み立て
        nav_ctx = NavigationContext(
            site=site,
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            start_t=start_t,
            budget_ms=budget_ms,
            entry_url=target_url,
            context=context,  # Stage 3A-2-4: BrowserContext を追加
        )
        
        # NavigationDriver を初期化
        # Stage 3B: TelemetryService を NavigationDriver に渡す
        telemetry = self._ensure_telemetry()
        # plugin が渡されていない場合は取得
        if plugin is None:
            from app.agents.browser.plugin_api import PluginAPI
            plugin_api = PluginAPI(logger=self.logger)
            plugin = plugin_api.get_plugin(site)
        # Stage 3A-3: trap_checker を NavigationDriver に渡す（観測用）
        navigation_driver = NavigationDriver(
            page=page,
            trap_checker=lambda url: self._looks_like_trap_or_legal(url),
            telemetry=telemetry,  # Stage 3B: TelemetryService を渡す
            strategy=plugin,
        )
        
        # run_plp_flow を呼び出す（現時点ではスタブ実装なので、ctx をそのまま返すだけ）
        try:
            nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
            self.logger.debug(f"[_run_plp_flow] NavigationDriver.run_plp_flow called (stub): entry_url={nav_outcome.entry_url}")
        except Exception as nav_e:
            self.logger.debug(f"[_run_plp_flow] NavigationDriver.run_plp_flow failed (fallback to legacy): {nav_e}")
            nav_outcome = None

        # --- Stage 3A-2-3: NavigationDriver が trap 判定・復旧を実行済みかチェック ---
        attempted_recover = False
        if nav_outcome and nav_outcome.trap_detected:
            # NavigationDriver が既に trap を検出し、復旧も試行済み
            if nav_outcome.recovered:
                self.logger.debug("[_run_plp_flow] NavigationDriver already handled trap detection and recovery")
                attempted_recover = True
            else:
                # NavigationDriver が trap を検出したが復旧に失敗した場合、例外を投げる
                raise ValueError(
                    f"Landing page looks like legal/trap (NavigationDriver recovery failed): {nav_outcome.trap_reason}"
                )
        elif nav_outcome is None or not nav_outcome.trap_detected:
            # NavigationDriver が trap を検出していない場合、従来の処理を実行
            # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
            # ★ V88.5.9: 1. 入口で trap でも、まず 1 回だけ回復ナビを試みる
            # Stage 4: site_configを渡す
            if self._looks_like_trap_or_legal(page.url, site_config):
                self.logger.warning("[_looks_like_trap] initial trap-like url: %s", page.url)
                attempted_recover = True
                await self._force_plp_recover(page, site_config, target_url)
                # 正規化後URLでもう一度だけ判定
                if self._looks_like_trap_or_legal(page.url, site_config):
                    raise ValueError(f"Landing page looks like legal/trap (even after recovery attempt): {page.url}")
                self.logger.info("[_looks_like_trap] Recovery navigation seems successful.")

        await self._pause_for_operator(page, run_context, "before_plp_materialize")

        # --- Stage 3A-2-2: NavigationDriver が materialize 済みならスキップ ---
        if skip_materialize:
            ok_materialized = True
        elif nav_outcome and nav_outcome.plp_materialized:
            # NavigationDriver が既に materialize を実行済み
            ok_materialized = True
            self.logger.debug("[_run_plp_flow] NavigationDriver already materialized PLP, skipping _ensure_plp_materialized")
        else:
            # NavigationDriver が materialize を実行していない場合、従来の処理を実行
            ok_materialized = await self._ensure_plp_materialized(
                page, site_config, settings,
                start_t=start_t, budget_ms=budget_ms, target_url=target_url
            )

        # ★ V88.5.7: (Fail Fast) まともなPLP(タイル0枚)が出なかったらすぐ諦める
        if not ok_materialized:
            raise ValueError(
                f"PLP did not materialize (no product tiles). URL={page.url}"
            )

        # --- Stage 3A-2-3: NavigationDriver が materialize 後の trap 再チェックも処理済みの場合 ---
        if nav_outcome and nav_outcome.plp_materialized:
            # NavigationDriver が materialize 後の trap 再チェックも完了している
            # NavigationDriver の run_plp_flow 内で既に trap 判定・復旧が完了しているため、
            # ここでの trap 再チェックは不要（NavigationDriver が例外を投げているはず）
            self.logger.debug("[_run_plp_flow] Using NavigationDriver result (post-materialize trap check already done)")
        else:
            # Stage 3A-2-3: NavigationDriver が使われていない場合の従来処理
            # Stage 4: site_configを渡す
            # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
            # ★ V88.5.9: スクロール後のURL再チェック (v88.5.6ロジック)
            if self._looks_like_trap_or_legal(page.url, site_config):
                # V88.5.9: まだ回復トライしてなければ、ここで試す
                if not attempted_recover:
                    self.logger.warning("[_looks_like_trap] trap-like url after materialize: %s", page.url)
                    await self._force_plp_recover(page, site_config, target_url)
                    if self._looks_like_trap_or_legal(page.url, site_config):
                         raise ValueError(f"After materialize still on legal/trap page (even after recovery attempt): {page.url}")
                    self.logger.info("[_looks_like_trap] Recovery navigation (post-materialize) seems successful.")
                else:
                    # 既に回復試行済みで、スクロールしたらまたトラップに戻った場合
                    raise ValueError(f"After materialize, bounced back to legal/trap page: {page.url}")

        try:
            # Stage 3B: TelemetryService を使用
            telemetry = self._ensure_telemetry()
            pdp_cfg_a = (site_config.get('selectors') or {}).get('pdp', {}) or {}
            selectors = (
                (pdp_cfg_a.get('pdp_link_selectors') or []) +
                (pdp_cfg_a.get('plp_container_selectors') or [])
            )
            await telemetry.record_plp_state(
                page,
                name="plp_dom_initial_materialized",
                selectors=selectors,
                site_config=site_config,
            )
        except Exception as e:
            logger.warning(f"[Hook A1] TelemetryService failed, falling back to observability: {e}")
            # フォールバック: 既存のobservability.py関数を使用
            try:
                # Stage 3B: TelemetryClient 経由で保存（フォールバック時も TelemetryClient を使用）
                fallback_telemetry = TelemetryClient(run_context)
                fallback_tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp")
                await fallback_telemetry.save_dom(page, "plp_dom_initial_materialized", fallback_tctx)
                pdp_cfg_a = (site_config.get('selectors') or {}).get('pdp', {}) or {}
                await count_selectors(
                    run_context,
                    page,
                    (pdp_cfg_a.get('pdp_link_selectors') or []) +
                    (pdp_cfg_a.get('plp_container_selectors') or []),
                    name="selector_counts_plp_initial"
                )
            except Exception as e2:
                logger.warning(f"[Hook A1] Fallback also failed: {e2}")

        # Stage 3A-2-4: NavigationDriver が既に PDP リンクを収集している場合はそれを使用
        # NavigationDriver.run_plp_flow 内で collect_pdp_links と fallback が実行されている
        if nav_outcome and nav_outcome.pdp_links:
            # NavigationDriver が既に PDP リンクを収集済み
            pdp_links = nav_outcome.pdp_links
            self.logger.debug(f"[_run_plp_flow] Using NavigationDriver result: {len(pdp_links)} PDP links")
            if nav_outcome.fallback_used:
                self.logger.info(f"[_run_plp_flow] NavigationDriver used fallback: {nav_outcome.fallback_used}")
        else:
            # NavigationDriver が PDP リンクを収集していない場合、従来の処理を実行
            # Stage 3A-2-1: NavigationDriver を直接使用（既に構築済みの nav_ctx を再利用）
            navigation_driver = NavigationDriver(
                page=page,
                telemetry=self._ensure_telemetry(),
                trap_checker=None,
                strategy=None,
            )
            pdp_links = await navigation_driver.collect_pdp_links(nav_ctx)

        # Fallback logic (header search, click first card)
        # V88.5.7: このブロックは ok_materialized=True (タイル1枚以上) だが
        # pdp_links=[] だった場合にのみ実行される
        # Stage 3A-2-4: NavigationDriver が既に fallback を実行している場合はスキップ
        if not pdp_links and (not nav_outcome or not nav_outcome.fallback_used):

            # Stage 4: site_configを渡す
            # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
            # ★ 3. ここでも trap判定をはさむ (v88.5.6ロジック)
            # (V88.5.9: 回復ロジックは既に試したので、ここでは検知のみ)
            if not pdp_links and self._looks_like_trap_or_legal(page.url, site_config):
                raise ValueError(
                    f"No PDP links and URL looks like trap/legal page: {page.url}"
                )

            try:
                self.logger.debug("[Fallback] Trying header search UI...")
                did_search = await self._plp_header_search_fallback(page, query, site_config, settings, run_context, context, start_t=start_t, budget_ms=budget_ms)
                if did_search:
                    await self._click_continue_shopping_if_present(page, site_config)
                    try: anchors = await page.locator("a[href*='/p/'], a[href*='/product/']").count()
                    except Exception: anchors = 0
                    if anchors < 6:
                        self.logger.debug(f"[Fallback] Materializing after search (anchors={anchors}<6)")
                        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms, target_url=target_url)
                    try:
                        # Stage 3B: TelemetryService を使用
                        telemetry = self._ensure_telemetry()
                        pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                        selectors = (
                            (pdp_cfg_a2.get("pdp_link_selectors") or []) +
                            (pdp_cfg_a2.get("plp_container_selectors") or [])
                        )
                        await telemetry.record_plp_state(
                            page,
                            name="plp_dom_search_fallback",
                            selectors=selectors,
                            site_config=site_config,
                        )
                    except Exception as e:
                        logger.warning(f"[Hook A3] TelemetryService failed, falling back to observability: {e}")
                        # フォールバック: 既存のobservability.py関数を使用
                        try:
                            # Stage 3B: TelemetryClient 経由で保存（フォールバック時も TelemetryClient を使用）
                            fallback_telemetry = TelemetryClient(run_context)
                            fallback_tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="plp")
                            await fallback_telemetry.save_dom(page, "plp_dom_search_fallback", fallback_tctx)
                            pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                            await count_selectors(run_context, page, (pdp_cfg_a2.get("pdp_link_selectors") or []) + (pdp_cfg_a2.get("plp_container_selectors") or []), name="selector_counts_after_search_fallback")
                        except Exception as e2:
                            logger.warning(f"[Hook A3] Fallback also failed: {e2}")
                    # Stage 3A-2-1: NavigationDriver を直接使用（nav_ctx を再利用）
                    # 注意: NavigationDriver.run_plp_flow が既に fallback を実行している場合は不要
                    if not nav_outcome or not nav_outcome.fallback_used:
                        pdp_links = await navigation_driver.collect_pdp_links(nav_ctx)
                    else:
                        pdp_links = nav_outcome.pdp_links if nav_outcome else []

                    # --- V88.5.5: 早期失敗ロジック ---
                    # Task C: PlpDriver を使用してタイルクリック → PDP遷移
                    # Stage 4: 拡張版 PlpDriver に対応
                    if not pdp_links:
                        self.logger.warning("[Fallback] No hrefs after search. Clicking first card using PlpDriver...")
                        try:
                            plp_driver = PlpDriver(
                                page=page,
                                context=context,
                                site_config=site_config,
                                run_context=run_context,
                                logger=self.logger,
                                telemetry=self._ensure_telemetry(),
                            )
                            # Stage 4: 新しいシグネチャを使用（target_url, timeout_ms を優先）
                            # Task 1: budget_ms = None 対応で安全化
                            default_timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
                            if budget_ms is not None:
                                timeout_ms = min(budget_ms, default_timeout_ms)
                            else:
                                timeout_ms = default_timeout_ms
                            nav_result = await plp_driver.navigate_to_pdp(
                                target_url=target_url,
                                timeout_ms=timeout_ms,
                                # 後方互換性のため、旧パラメータも渡す（内部で fallback として使用）
                                start_t=start_t,
                                budget_ms=budget_ms,
                            )
                            
                            # Stage 4: 新しいフィールドを RunContext に保存
                            run_context.save_json("plp_navigation_result.json", {
                                "pdp_url": nav_result.pdp_url,
                                "plp_url": nav_result.plp_url,
                                "tiles_seen": nav_result.tiles_seen,
                                "trap_detected": nav_result.trap_detected,
                                "trap_reason": nav_result.trap_reason,
                                "recovery_attempted": nav_result.recovery_attempted,
                                "recovery_successful": nav_result.recovery_successful,
                                "overlays_handled": nav_result.overlays_handled,
                                "navigation_method": nav_result.navigation_method,
                                "errors": nav_result.errors,
                                "pdp_opened_in_new_tab": nav_result.pdp_opened_in_new_tab,
                            })
                            
                            # Stage 4: 詳細なログ出力
                            if nav_result.trap_detected:
                                if nav_result.recovery_successful:
                                    self.logger.info(
                                        f"[PlpDriver] Trap detected and recovered: {nav_result.trap_reason} "
                                        f"(overlays: {nav_result.overlays_handled}, method: {nav_result.navigation_method})"
                                    )
                                else:
                                    self.logger.warning(
                                        f"[PlpDriver] Trap detected but recovery failed: {nav_result.trap_reason} "
                                        f"(errors: {nav_result.errors})"
                                    )
                            else:
                                self.logger.info(
                                    f"[PlpDriver] Successfully navigated to PDP: {nav_result.pdp_url} "
                                    f"(tiles_seen: {nav_result.tiles_seen}, "
                                    f"overlays: {nav_result.overlays_handled}, "
                                    f"method: {nav_result.navigation_method})"
                                )
                            
                            # エラーがある場合は警告
                            if nav_result.errors:
                                for error in nav_result.errors:
                                    self.logger.warning(f"[PlpDriver] Error: {error}")
                            
                            # PlpDriver が PDP に遷移した場合、そのページで PDP 抽出を実行
                            # PlpDriver 内で既にページ遷移が完了しているので、plp_driver.page を使用
                            pdp_page = plp_driver.page
                            return await self._run_pdp_flow(pdp_page, site, query, settings, run_context, site_config)
                        except Exception as plp_e:
                            self.logger.warning(f"[Fallback] PlpDriver failed: {plp_e}", exc_info=True)
                            # フォールバック: 既存の _click_first_card_or_link を使用
                            new_page = await self._click_first_card_or_link(page, site_config, settings, context)
                            if new_page:
                                return await self._run_pdp_flow(new_page or page, site, query, settings, run_context, site_config)
                            # new_page も取れなかった → ここで即ギブアップ (V88.5.5)
                            raise ValueError("No PDP links and click fallback failed (gave up early for speed).")
                    # --- V88.5.5: 修正ここまで ---

            except Exception as _e:
                # _plp_header_search_fallback や _click_first_card_or_link 自体が例外を投げた場合
                # (V88.5.5: 早期ギブアップの ValueError もここに含まれる)
                self.logger.warning(f"[Fallback:header-search] failed or gave up early: {_e}", exc_info=True)
                # 最終手段として、この例外をそのまま投げるか、
                # あるいは、pdp_links が空のまま次の if pdp_links: に進める
                # ここでは後者を選択し、最終的に if pdp_links: の外側でエラーになるようにする
                pass

        if pdp_links:
            prepare_hook = self._build_pdp_prepare_hook(site_config=site_config, settings=settings, run_context=run_context)
            return await self.extraction_service.extract_from_pdp_list(
                page=page,
                context=context,
                site=site,
                query=query,
                pdp_links=pdp_links,
                site_config=site_config,
                settings=settings,
                run_context=run_context,
                start_t=start_t,
                budget_ms=budget_ms,
                prepare_page=prepare_hook,
            )
        raise ValueError("All PDP attempts failed after all recovery attempts.")

    # Stage 3A-2-4: 実体は NavigationDriver.header_search_fallback に移行済み
    # このメソッドは NavigationDriver が使われていない場合のフォールバックとして残している
    # TODO: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能
    # Stage 4: _plp_header_search_fallback() を削除し、NavigationDriver.header_search_fallback() を使用
    async def _plp_header_search_fallback(self, page, query: str, site_config, settings, run_context, context: BrowserContext, *, start_t: float, budget_ms: int) -> bool:
        """
        Stage 4: ヘッダ検索フォールバックのヘルパー（NavigationDriver経由）
        NavigationDriver.header_search_fallback() を呼び出すためのラッパー
        """
        from app.agents.browser.navigation_driver import NavigationDriver, NavigationContext
        # NavigationContextを作成
        nav_ctx = NavigationContext(
            site=getattr(run_context, "site", ""),
            query=query,
            site_config=site_config,
            settings=settings,
            run_context=run_context,
            start_t=start_t,
            budget_ms=budget_ms,
            entry_url=page.url,
            context=context,
        )
        # NavigationDriverインスタンスを作成
        driver = NavigationDriver(page=page)
        return await driver.header_search_fallback(nav_ctx)

    # ★ V88.5.5: タイムアウトを 15000ms -> 5000ms に短縮
    async def _click_and_capture_navigation(self, click_coro, page: Page, context: BrowserContext, *, url_regex: Optional[re.Pattern] = re.compile(r"/product[s]?/|/p/|/pp/", re.I), wait_state: str = "domcontentloaded", timeout_ms: int = 5000) -> Optional[Page]:
        popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
        same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))
        spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms)) if url_regex else None
        sel_spa = ", ".join(VISIBLE_PRICE_SELECTORS) or "[itemprop=price],[class*=price],[data-testid*=price]"
        spa_price_task = asyncio.create_task(page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms))
        try: await click_coro()
        except Exception: [t.cancel() for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task) if t and not t.done()]; return None
        tasks = {popup_task, same_tab_nav_task, spa_price_task}; tasks.add(spa_url_task) if spa_url_task else None
        try:
            # V88.5.5: timeout_ms が 5000 になったため、 asyncio.wait のタイムアウトも 5.0 秒になる
            done, pending = await asyncio.wait(tasks, timeout=timeout_ms/1000, return_when=asyncio.FIRST_COMPLETED)
            [t.cancel() for t in pending]
            if not done: return None
            winner = next(iter(done))
            new_page = winner.result() if winner is popup_task else page
            log_msg = f"Winner: {'popup' if winner is popup_task else 'framenav' if winner is same_tab_nav_task else 'SPA URL' if winner is spa_url_task else 'SPA Price'}"
            self.logger.debug(f"[_click_and_capture] {log_msg}")
            try:
                if new_page.url == "about:blank": await new_page.wait_for_load_state('domcontentloaded', timeout=1500)
            except Exception as e_blank: self.logger.debug(f"[_click_and_capture] Wait for about:blank failed: {e_blank}")
            try: await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10)) # 500ms
            except Exception: pass
            if url_regex:
                try: await new_page.wait_for_url(url_regex, timeout=max(1000, timeout_ms // 4)) # 1250ms
                except Exception as e_url_final: self.logger.debug(f"[_click_and_capture] Final wait_for_url failed: {e_url_final}")
            return new_page
        except Exception as e_wait: self.logger.warning(f"[_click_and_capture] Nav race failed: {e_wait}"); return None
        finally: [t.cancel() for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task) if t and not t.done()]

    # Stage 3A-2-4: 実体は NavigationDriver.click_first_card_or_link に移行済み
    # このメソッドは NavigationDriver が使われていない場合のフォールバックとして残している
    # TODO: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能
    async def _click_first_card_or_link(self, page: Page, site_config: Dict, settings: Dict, context: BrowserContext) -> Optional[Page]:
        pdp = (site_config.get("selectors", {}).get("pdp") or {})
        link_sel = pdp.get("pdp_link_selectors", [])
        plp_boxes = pdp.get("plp_container_selectors", ["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"])
        block_ng = set(pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
        url_pat = re.compile(r"/product[s]?/|/p/|/pp/", re.I)
        if link_sel:
            for s in link_sel:
                try:
                    loc = page.locator(s); count = await loc.count()
                    for i in range(count):
                        el = loc.nth(i); href = (await el.get_attribute("href")) or (await el.get_attribute("data-href")) or ""
                        if href and not any(bad in href for bad in block_ng):
                            await el.scroll_into_view_if_needed(); newp = await self._click_and_capture_navigation(lambda: el.click(timeout=5000), page, context, url_regex=url_pat)
                            if newp: return newp
                except Exception: continue
        # ★ 88.6.2: (Refactor) 可読性のため整形
        tile_selectors = [
            "[data-qa='product-tile']",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]"
        ]
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    card = page.locator(f"{box} {tile_sel}").first; await card.scroll_into_view_if_needed()
                    if await card.count() > 0:
                        newp = await self._click_and_capture_navigation(lambda: card.click(timeout=5000), page, context, url_regex=url_pat)
                        if newp: return newp
                except Exception: continue
        self.logger.warning("[Fallback:click-card] Could not find any clickable link or card."); return None

    # --- Flow Logic: PDP ---
    async def _run_pdp_flow(
        self,
        page: Page,
        site: str,
        query: str,
        settings: Dict,
        run_context: RunContext,
        site_config: Dict[str, Any],
    ) -> DiscoveryResult:
        logger.info("[Mode] PDP (detail)")
        prepare_hook = self._build_pdp_prepare_hook(site_config=site_config, settings=settings, run_context=run_context)
        return await self.extraction_service.extract_single_pdp(
            page=page,
            context=self._context,
            site=site,
            query=query,
            settings=settings,
            run_context=run_context,
            site_config=site_config,
            target_url=page.url,
            prepare_page=prepare_hook,
        )

    # ★ V88.6.0: (Fix) ご提示いただいた差分パッチ L.59-L.90 (v88.5.9J で欠落していた) をここに追加
    # ★NEW: 既定の“強制PLP復帰”保険（本命）
    # ★ V88.6.1: (Refactor) 呼び出しを _normalize_to_en_int_url に修正
    # ★ V88.6.1: (BugFix) goto に url= を明記
    # Stage 3A-2-3: 実体は NavigationDriver.recover_plp に移行済み
    # このメソッドは NavigationDriver が使われていない場合のフォールバックとして残している
    # TODO: Stage 3A-2 完了後、NavigationDriver が常に使われるようになったら削除可能
    # Stage 4: _force_plp_recover() を削除し、NavigationDriver._force_plp_recover() を使用
    async def _force_plp_recover(self, page, site_config: dict, target_url: str | None) -> None:
        """
        Stage 4: PLP回復のヘルパー（NavigationDriver経由）
        NavigationDriver._force_plp_recover() を呼び出すためのラッパー
        """
        from app.agents.browser.navigation_driver import NavigationDriver
        # NavigationDriverインスタンスを作成
        driver = NavigationDriver(page=page)
        await driver._force_plp_recover(page, site_config, target_url)

    # ★NEW: ガード用の簡易版（_force_plp_recover が見つからない場合の代替）
    async def _inline_force_plp_recover(self, page, site_config: dict, target_url: str | None) -> None:
        await self._force_plp_recover(page, site_config, target_url)

    # ★ V88.6.1: (Refactor) 不要になった重複メソッドを削除
    # def _normalize_en_int_url(self, url: str, site_config: dict) -> str: ...

    # --- V88.1.0: Refined Failure Handling ---
    # --- V88.4.0: Add intent context ---
    async def _handle_run_failure(self, e: Exception, site: str, query: str, site_config: Dict,
                                  run_context: RunContext, page: Optional[Page]) -> DiscoveryResult:
        logger.error(f"Browser task failed (RunID: {run_context.run_id}): {e}", exc_info=True)
        final_url_on_fail = None

        # V88.5.0: page は self._page (インスタンス変数) または引数で渡されたもの
        active_page = page or self._page

        try:
            await self._pause_for_operator(active_page, run_context, "failure_inspection")
            if active_page and not active_page.is_closed():
                final_url_on_fail = active_page.url
        except Exception:
             pass

        # Stage 3B: TelemetryClient を使用して失敗スナップショットを保存
        dom_path_str = None
        try:
            # write_fail_snapshot は active_page (開いている可能性のあるページ) を使う
            telemetry = TelemetryClient(run_context)
            tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="fail_plp")
            await telemetry.write_fail_snapshot(
                active_page,
                reason=str(e),
                tctx=tctx,
                extra={"site_config": site_config, "final_url": final_url_on_fail}
            )
            # Infer standard path used by write_fail_snapshot
            # ★ V88.6.1: (BugFix) ファイル名を failure_dom.html に修正
            dom_guess = run_context.get_path("failure_dom.html")
            if dom_guess and Path(dom_guess).exists():
                dom_path_str = str(dom_guess)
        except Exception as hook_e:
            logger.warning(f"[Hook C2] TelemetryService.write_fail_snapshot failed, falling back: {hook_e}")
            # フォールバック: 既存のobservability.py関数を使用
            try:
                # Stage 3B: TelemetryClient 経由で保存
                await telemetry.write_fail_snapshot(
                    active_page,
                    reason=str(e),
                    tctx=tctx,
                    extra={"site_config": site_config, "final_url": final_url_on_fail}
                )
                dom_guess = run_context.get_path("failure_dom.html")
                if dom_guess and Path(dom_guess).exists():
                    dom_path_str = str(dom_guess)
            except Exception as hook_e2:
                logger.warning(f"[Hook C2] Fallback also failed: {hook_e2}")

        # Build failure_context for Orchestrator V16+
        # Dynamically get screenshots if RunContext supports it
        screenshots_list = []
        if hasattr(run_context, "screenshots") and isinstance(run_context.screenshots, list):
            screenshots_list = run_context.screenshots
        elif hasattr(run_context, "get_path"):
            try:
                # Fallback: find recent PNGs in run_dir
                run_dir = Path(run_context.run_path)
                recent_pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                screenshots_list = [str(p) for p in recent_pngs[:6]] # Take last 6
            except Exception:
                pass # Ignore if path finding fails


        failure_context = {
            "final_url": final_url_on_fail,
            "dom_snapshot_path": dom_path_str, # Use path obtained from write_fail_snapshot result
            "errors": [str(e)],
            "screenshots": screenshots_list, # Use dynamically obtained list
            # high-level intent / expectation so LLM can reason:
            "intent_description": (
                "Goal: reach a product listing page (PLP) and extract product cards "
                "and individual PDP links, then read price/title from PDP. "
                "We expected to see product tiles and extract price. "
                "Instead we hit an unexpected layout / modal / redirect."
            ),
            # future: we could add what selectors we attempted, etc.
            "selectors_tried_hint": "See site_config['selectors']['pdp'] and wait_for_selectors in settings.",
        }

        # Task B: Build upload bundle for diagnostics
        try:
            bundle_path = run_context.build_upload_bundle()
            if bundle_path:
                logger.info(f"[Failure Handler] Upload bundle created: {bundle_path}")
        except Exception as bundle_e:
            logger.warning(f"[Failure Handler] Failed to build upload bundle: {bundle_e}")

        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message=str(e),
            # ai_analysis removed (Orchestrator's responsibility now)
            evidence={
                "final_url": final_url_on_fail,
                "failure_context": failure_context, # Pass context to Orchestrator
            }
        )

    # --- Flow Logic: Learning ---
    async def _run_learning_flow(self, page: Page, context: BrowserContext, site: str, site_config: Dict, settings: Dict, run_context: RunContext, *, start_t: float, budget_ms: int) -> DiscoveryResult:
        self.logger.info(f"[LEARN] Starting learning flow for site: {site}")
        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)
        # Stage 3B: TelemetryService を使用
        try:
            telemetry = self._ensure_telemetry()
            await telemetry.save_dom(page, "learn_plp_dom_for_discovery")
        except Exception as e:
            logger.warning(f"[LEARN] TelemetryService.save_dom failed, falling back: {e}")
            # フォールバック: 既存のobservability.py関数を使用
            await save_dom(run_context, page, "learn_plp_dom_for_discovery")
        try:
            discovered_selectors = await self.discovery_agent.discover(page=page, context=context, run_context=run_context)
            if not discovered_selectors: raise ValueError("SelectorDiscoveryAgent returned no selectors.")
            self.logger.info(f"[LEARN] Discovered selectors: {json.dumps(discovered_selectors, indent=2)}")
        except Exception as e:
            self.logger.error(f"[LEARN] Selector discovery failed: {e}", exc_info=True)
            return await self._handle_run_failure(e, site, "(learning)", site_config, run_context, page) # Use failure handler
        try:
            await self._save_learned_selectors(site, discovered_selectors, run_context)
        except Exception as e:
            self.logger.error(f"[LEARN] Failed to save learned selectors: {e}", exc_info=True)
            # Saving failed, but discovery succeeded, so still return ok=False but with learned selectors
            return DiscoveryResult(ok=False, site=site, query="(learning)", message=f"Failed to save: {e}", evidence={"learned_selectors": discovered_selectors})
        return DiscoveryResult(ok=True, site=site, query="(learning)", message="Successfully learned and saved.", evidence={"learned_selectors": discovered_selectors})

    async def _save_learned_selectors(self, site: str, new_selectors: Dict[str, List[str]], run_context: RunContext) -> None:
        try:
            instance_dir = Path(run_context.run_path).parent.parent
            site_dir = instance_dir / "sites" / site.upper(); site_dir.mkdir(parents=True, exist_ok=True)
            learned_path = site_dir / "learned_selectors.json"
        except Exception:
            learned_path = Path(f"instance/sites/{site.upper()}/learned_selectors.json")
        learned_path.parent.mkdir(parents=True, exist_ok=True)
        existing_selectors = {}
        if learned_path.exists():
            try:
                existing_selectors = json.loads(learned_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        merged_selectors = {}
        all_keys = set(existing_selectors.keys()) | set(new_selectors.keys())
        for key in all_keys:
            merged_list = _dedupe_keep_order(new_selectors.get(key, []) + existing_selectors.get(key, []))
            if merged_list: merged_selectors[key] = merged_list
        
        # Task B: Use RunContext to save learned selectors
        # Save to run_context for this run, and also save to instance/sites/ for persistence
        try:
            # Save to run_context for this run
            run_context.save_json("learned_selectors.json", merged_selectors)
            self.logger.info(f"[LEARN] Saved learned selectors to run_context: {run_context.get_path('learned_selectors.json')}")
        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to save to run_context: {e}")
        
        # Also save to instance/sites/ for persistence across runs
        try:
            learned_path.write_text(json.dumps(merged_selectors, indent=2, ensure_ascii=False), encoding="utf-8")
            self.logger.info(f"[LEARN] Saved merged selectors to: {learned_path}")
        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to save to instance/sites/: {e}")
