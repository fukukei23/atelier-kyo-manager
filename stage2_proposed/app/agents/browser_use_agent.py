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
        await self._dismiss_geo_modal(page)
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

        # --- Moncler 固有の回復 ---
        if site.upper() == "MONCLER_OFFICIAL" and not likely_plp:
            try:
                gate_links_count = await page.evaluate(
                    "() => document.querySelectorAll(\"a[href*='/en-']\").length"
                )
                if gate_links_count and gate_links_count >= 10:
                    self.logger.warning(
                        f"[Moncler] Locale gate detected ({gate_links_count} links). Forcing navigation to PLP (PDP run only)."
                    )
                    fixed_url = (
                        "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
                        "?forceLocale=en-int&shipToCountry=GB"
                    )
                    await page.goto(url=fixed_url, wait_until="domcontentloaded")
                    await self._click_continue_shopping_if_present(page, site_config)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=2000)
                    except Exception:
                        pass
                    await self._accept_cookies_if_present(page, site_config)
            except Exception as gate_e:
                self.logger.warning(f"[Moncler] Gate detection failed: {gate_e}")

            if settings.get("enable_locale_escape"):
                await self._force_en_int(page)
                await self._click_continue_shopping_if_present(page, site_config)
                await run_context.take_screenshot(page, "12_after_locale_escape")

            try:
                if "monclergroup.com" in (page.url or "").lower():
                    fixed_url = (
                        "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
                        "?forceLocale=en-int&shipToCountry=GB"
                    )
                    logger.warning(f"[Moncler] Bounced to corporate. Forcing back to PLP: {fixed_url}")
                    await page.goto(url=fixed_url, wait_until="domcontentloaded")
                    await self._accept_cookies_if_present(page, site_config)
                    await self._click_continue_shopping_if_present(page, site_config)
            except Exception:
                pass

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
            await self._dismiss_geo_modal(page)
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
            url_normalizer=self._normalize_to_en_int_url,
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
from app.agents.browser.extractor import (
    BrowserExtractionService,
    PDPSizeSelectPolicy,
    VISIBLE_PRICE_SELECTORS,
    DEFAULT_PDP_PARALLEL_LIMIT,
    looks_like_product_url,
)

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
    from app.utils.observability import (
        save_dom, count_selectors, save_raw_hrefs, write_fail_snapshot
    )
    # V88.5.1: (Patch) AiLlmController は run_with_repair 内部でのみ import
    # from app.utils.ai_llm_controller import AiLlmController
except ImportError:
    # 実行環境によってはパスが通っていない可能性を考慮
    logging.warning("Failed to import modules from standard paths. Trying relative imports...")
    try:
        from ..core.run_context import RunContext
        from ..utils.visual_regression import compare_and_maybe_update
        from ..models.result_models import DiscoveryResult
        from .selector_discovery_agent import SelectorDiscoveryAgent
        from ..utils.observability import (
            save_dom, count_selectors, save_raw_hrefs, write_fail_snapshot
        )
    except ImportError as e:
         logging.critical(f"Relative import also failed: {e}. Some functionalities might be broken.")
         # 最低限動作するためのモックやプレースホルダを定義する (必要に応じて)
         class RunContext: pass # type: ignore
         class DiscoveryResult: pass # type: ignore
         def compare_and_maybe_update(*args, **kwargs): pass
         def extract_title_price(*args, **kwargs): return {}
         class SelectorDiscoveryAgent: pass
         def save_dom(*args, **kwargs): pass
         def count_selectors(*args, **kwargs): pass
         def save_raw_hrefs(*args, **kwargs): pass
         def write_fail_snapshot(*args, **kwargs): pass


logger = logging.getLogger(__name__)
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000  # 120s watchdog

PLUGIN_REGISTRY: Dict[str, StrategyPlugin] = {}
if StrategyPlugin and MonclerPLPStrategy:
    PLUGIN_REGISTRY = {
        "MONCLER_OFFICIAL": MonclerPLPStrategy(),
    }
# ここに他サイトの Strategy を追加予定

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
    def _looks_like_trap_or_legal(self, url: str) -> bool:
        """
        明らかに商品一覧ではなく、法務/クッキー/ヘルプ系に飛ばされてると判断したら True。
        こういうページに張り付いてもPDPは取れないので、早期abortさせる。

        ★ V88.5.9: 先に軽量正規化を行ってから判定する。
        - /en-jp/en-int/ を /en-int/ に置換
        - #product-information-panel 等のハッシュを除去
        """
        try:
            # V88.5.9: ローカルインポート (urllib.parse はグローバルで import 済)
            sp = urlsplit(url)
            path = sp.path or ""
            # 二重ロケールの早期修正
            path = path.replace("/en-jp/en-int/", "/en-int/").replace("/en-jp/", "/en-int/")
            # “PDPアンカー”などのハッシュは評価前に捨てる
            sp = sp._replace(path=path, fragment="")
            url = urlunsplit(sp)
        except Exception:
            pass # 正規化に失敗しても、元のURLで判定を続行

        try:
            u = urlparse(url)
            full_lower = url.lower()
            path_lower = (u.path or "").lower()
            host = (u.netloc or "").lower()

            # V88.5.7: /en-jp (日本向け)
            jp_locale = "moncler.com" in full_lower and "/en-jp" in path_lower

            # V88.5.8: コーポレートサイト
            corporate = "monclergroup.com" in host or "/brands/moncler" in path_lower

            # V88.6.3: Moncler のロケーションゲート/トップページを trap とみなす
            moncler_locale_gate = (
                "moncler.com" in host
                and path_lower in ("/en-int", "/en-int/", "/en-gb", "/en-gb/", "/en-us", "/en-us/")
            )

            # V88.5.6 以前のリーガルキーワード
            legal_kw = any(k in path_lower for k in (
                "/cookie-policy", "/cookies", "/privacy", "/legal", "/help",
                "/customer-service", "/customer_service", "/support",
                "/account", "/login", "/accessibility-statement", "/client-service/"
            ))

            # V88.5.9: 簡潔な return 形式に統合
            if jp_locale:
                logger.warning(f"[_looks_like_trap] Detected /en-jp locale trap: {url}")
            if corporate:
                logger.warning(f"[_looks_like_trap] Detected corporate site redirect/path: {url}")
            if legal_kw:
                logger.warning(f"[_looks_like_trap] Detected legal/help keyword trap: {url}")
            if moncler_locale_gate:
                logger.warning(f"[_looks_like_trap] Detected Moncler locale gate/home: {url}")

            return jp_locale or corporate or legal_kw or moncler_locale_gate

        except Exception:
            return False

    def __init__(self, runtime_kwargs: Optional[Dict[str, Any]] = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.logger = logger
        self.run_context: Optional[RunContext] = None # Temporarily attach RunContext during run()

        # --- V88.6.x: Session handles are managed by SessionManager ---
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._session_manager: Optional[SessionManager] = None
        self.extraction_service = BrowserExtractionService(self.logger, self.runtime_kwargs)

    def _attach_session(self, session: SessionManager) -> None:
        self._session_manager = session
        self._context = session.context
        self._page = session.page

    def _detach_session(self) -> None:
        self._session_manager = None
        self._context = None
        self._page = None


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

    async def _dismiss_geo_modal(self, page: Page) -> None:
        """
        ジオ / ロケール関係のモーダルを潰す。

        1. 汎用の「STAY HERE」系バナー
        2. Moncler の「Select your location」ロケーションゲート
           - 「UNITED KINGDOM | ENGLISH」を優先的に踏みに行く
        """

        for sel in [
            "text=STAY HERE",
            "text=REMAIN HERE",
            "text=REMAIN IN ENGLISH",
            "text=CONTINUE SHOPPING",
            "text=ショッピングを続ける",
        ]:
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

        async def _wait_for_en_int(timeout_ms: int = 4000) -> bool:
            try:
                await page.wait_for_function(
                    "() => location.href.includes('/en-int/') && !location.href.includes('/en-gb/')",
                    timeout=timeout_ms,
                )
                return True
            except Exception:
                return "/en-int/" in (page.url or "").lower()

        try:
            header = page.locator("text=Select your location").first
            header_visible = await header.count() > 0
            if header_visible:
                self.logger.info("[GeoModal] Moncler locale gate header detected.")

            uk_candidates = [
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
            for loc in uk_candidates:
                if await _click_first(loc, "United Kingdom / English selector"):
                    if await _wait_for_en_int():
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
                    if await _wait_for_en_int():
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
    def _normalize_to_en_int_url(self, url: str) -> str:
        u = urlparse(url)
        path = (u.path or "/").replace("//","/")
        path = path.replace("/en-gb/", "/en-int/")
        seg = [s for s in path.split("/") if s]
        i = 0
        while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""): i += 1
        seg = [s for s in seg[i:] if s.lower() != "en-int"]
        norm = "/en-int/" + "/".join(seg)
        if not norm.endswith("/"): norm += "/"
        q = dict(parse_qsl(u.query))
        q["forceLocale"] = "en-int"; q.setdefault("shipToCountry","GB")
        return urlunparse((u.scheme, u.netloc, norm, u.params, urlencode(q), u.fragment))

    async def _force_en_int(self, page: Page) -> None:
        try:
            if page.context:
                await page.context.add_cookies([{"name":"moncler-shipping-country","value":"GB","domain":".moncler.com","path":"/"}, {"name":"moncler-shipping-language","value":"en","domain":".moncler.com","path":"/"}])
        except Exception: pass
        try:
            fixed = self._normalize_to_en_int_url(page.url)
            if fixed != page.url:
                # V88.5.3: (BugFix) `url=` キーワード引数を明示的に指定
                await page.goto(url=fixed, wait_until="domcontentloaded")
                try: await page.wait_for_load_state("networkidle", timeout=1500)
                except Exception: pass
        except Exception: pass

    # --- PLP Materialize ---
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
                await self._dismiss_geo_modal(page)
            except Exception:
                pass
            try:
                await self._kill_overlays(page)
            except Exception:
                pass

            current_url = (page.url or "").lower()
            if "moncler.com/en-gb" in current_url:
                self.logger.warning("[Materialize] Detected EN-GB redirect mid-attempt.")
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

    async def _collect_pdp_links(self, page: Page, site_config: Dict, settings: Dict, run_context: RunContext) -> List[str]:
        target_url = page.url
        found_links: Set[str] = set()

        # Phase 1a: Global <a href> sweep + Regex Filter
        try:
            raw_hrefs: List[str] = await page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)")
        except Exception as e: logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}"); raw_hrefs = []
        pdp_rx = re.compile(r"/(products?|p)/", re.I)
        for href in raw_hrefs:
            if pdp_rx.search(href):
                norm_url = self._normalize_abs_url(target_url, href)
                if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url): found_links.add(norm_url)
        if found_links: logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")

        # Phase 1b: Selector-based補完
        selectors_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        PLP_PDP_LINK_SELECTORS = _dedupe_keep_order((selectors_cfg.get("pdp_link_selectors", []) or []) + ["a[href*='/products/']", "a[href*='/product/']", "a[href*='/p/']", "[data-component*='ProductCard'] a[href]", "[class*='product-card'] a[href]", "article [data-testid*='product']:is(a, * a)", "[data-testid*='card'] a[href]", "[data-testid*='product-card'] a[href]", "a[data-product-url]", "[data-qa='product-tile'] a[href]"])
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes: continue
                matched_count = 0
                for n in nodes:
                    href = await n.get_attribute("href") or await n.get_attribute("data-href") or await n.get_attribute("data-product-url") or await n.get_attribute("data-url")
                    if not href: continue
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                        found_links.add(norm_url); matched_count += 1
                if matched_count > 0: logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
            except Exception as e: logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")

        # Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
        if not found_links:
            logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
            try:
                deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)
                for href in deep_hrefs:
                    norm_url = self._normalize_abs_url(target_url, href)
                    if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url): found_links.add(norm_url)
                if found_links: logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
            except Exception as e: logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")

        links = sorted(list(found_links))
        if not links: logger.warning("[PLP→PDP] No PDP hrefs found after all phases."); return []

        # Phase 3: Noise Filtering & Saving
        cleaned: List[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u): cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")
        try:
            sample = cleaned[:20]; logger.debug(f"[PLP→PDP] sample={sample}")
            if self.run_context: self.run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
            # V87.0: Robust save_raw_hrefs call
            try:
                if callable(save_raw_hrefs) and run_context:
                    res = save_raw_hrefs(run_context, cleaned, name="raw_hrefs_final_cleaned")
                    if asyncio.iscoroutine(res): await res
            except Exception: pass
        except Exception: pass
        return cleaned

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

    # --- Browser Context Setup ---
    def _build_context_options(
        self,
        settings: Dict[str, Any],
        run_context: RunContext,
    ) -> Dict[str, Any]:
        """
        Playwright BrowserContext 用のオプションを一括構築する。
        ビデオ/HAR/トレースの保存先もここで定義しておく。
        """
        ctx_opts: Dict[str, Any] = {}

        import random
        viewport = settings.get("viewport")
        if settings.get("enable_viewport_rotation"):
            viewport = random.choice(VIEWPORT_POOL)
        if viewport:
            ctx_opts["viewport"] = viewport

        ctx_opts["locale"] = "en-GB"
        ctx_opts["timezone_id"] = "UTC"

        headers = (settings.get("extra_http_headers") or {}).copy()
        headers["Accept-Language"] = settings.get("accept_language") or "en-GB,en;q=0.8"
        ctx_opts["extra_http_headers"] = headers

        user_agent = settings.get("user_agent")
        if settings.get("enable_ua_rotation") and USER_AGENT_POOL:
            user_agent = random.choice(USER_AGENT_POOL)
        if user_agent:
            ctx_opts["user_agent"] = user_agent

        if settings.get("enable_har"):
            ctx_opts["record_har_path"] = str(run_context.get_path("network.har"))
        if settings.get("enable_trace"):
            trace_dir = run_context.get_path("trace")
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            ctx_opts["record_trace_dir"] = str(trace_dir)
            self.logger.debug(f"[Playwright] record_trace_dir={trace_dir}")
        if settings.get("enable_video"):
            videos_dir = run_context.get_path("videos")
            Path(videos_dir).mkdir(parents=True, exist_ok=True)
            ctx_opts["record_video_dir"] = str(videos_dir)
            ctx_opts["record_video_size"] = {"width": 1280, "height": 720}

        return ctx_opts

    async def _setup_routes(self, context: BrowserContext, settings: Dict[str, Any]):
        base_routes = ["**/*onetrust.com/**","**/*cookielaw.org/**","**/*cookiepro.com/**"]
        extra = settings.get("extra_block_routes") or []
        extra_hosts, extra_globs = [], []
        for x in extra:
            x = (x or "").strip()
            if not x: continue
            if "*" in x or "/" in x: extra_globs.append(x)
            else: extra_hosts.append(x.lstrip("."))
        block_hosts = tuple(h.lower().lstrip(".") for h in EXTERNAL_BLOCKLIST_HOSTS) + tuple(h.lower() for h in extra_hosts)
        async def _abort(route: Route): await route.abort()
        async def _locale_rewrite(route: Route):
            try:
                req = route.request; url = req.url
                if req.resource_type == "document" and "moncler.com" in url:
                    pu = urlparse(url); path_lower = (pu.path or "/").lower()
                    skip_paths = ("/search", "/account", "/customer", "/help", "/privacy", "/legal")
                    is_skippable = path_lower == "/" or any(path_lower.startswith(s) or path_lower.startswith(f"{s}/") for s in skip_paths)
                    if not is_skippable:
                        fixed = self._normalize_to_en_int_url(url)
                        if fixed != url: self.logger.info(f"[Route] URL normalized: {url} -> {fixed}"); await route.continue_(url=fixed); return
                    else: self.logger.debug(f"[Route] Skipping locale normalization for path: {path_lower}")
                host = urlparse(url).netloc.lower().strip(".")
                if any(host == b or host.endswith("." + b) for b in block_hosts): await route.abort(); return
                try: await route.fallback()
                except Exception: await route.continue_()
            except Exception as e:
                self.logger.warning(f"[Route] handler error: {e}")
                try: await route.continue_()
                except Exception: pass
        await context.route("**/*", _locale_rewrite)
        for pat in base_routes + extra_globs: await context.route(pat, _abort)

    def _get_session_file(self, site: str, site_config: Dict[str, Any]) -> Path:
        # site_config.discovery_settings.session_file があれば優先、なければデフォルトパス
        ds = site_config.get("discovery_settings") or {}
        sess_path = ds.get("session_file")
        if sess_path:
            return Path(sess_path)
        safe_site = "".join(c for c in site.lower() if c.isalnum() or c in ("_", "-")) or "default"
        return SESSION_DIR / f"{safe_site}.json"

    async def _apply_saved_session(self, context: BrowserContext, page: Page, site: str, site_config: Dict[str, Any]) -> None:
        """
        手動突破で保存した Cookie/LocalStorage を復元する。
        ファイル形式例:
        {
          "cookies": [ {name, value, domain, path, expires, httpOnly, secure, sameSite}, ... ],
          "localStorage": { "key": "value", ... }
        }
        """
        sess_file = self._get_session_file(site, site_config)
        if not sess_file.exists():
            return
        try:
            data = json.loads(sess_file.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"[Session] Failed to read session file {sess_file}: {e}")
            return

        # Cookie
        cookies = data.get("cookies") or []
        if cookies:
            try:
                await context.add_cookies(cookies)
                self.logger.info(f"[Session] Restored {len(cookies)} cookies from {sess_file}")
            except Exception as e:
                self.logger.warning(f"[Session] add_cookies failed: {e}")

        # LocalStorage
        ls = data.get("localStorage") or {}
        if ls:
            try:
                # set localStorage before any navigation
                # Playwright python add_init_script does not take args; embed as JSON
                import json as _json
                payload = _json.dumps(ls)
                script = f"""
                  (() => {{
                    try {{
                      const items = {payload};
                      for (const [k,v] of Object.entries(items || {{}})) {{
                        localStorage.setItem(k, v);
                      }}
                    }} catch (e) {{}}
                  }})();
                """
                await context.add_init_script(script)
                self.logger.info(f"[Session] Restored localStorage keys={list(ls.keys())[:5]} from {sess_file}")
            except Exception as e:
                self.logger.warning(f"[Session] localStorage restore failed: {e}")

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

    async def _setup_init_scripts(self, context: BrowserContext):
        # --- Baseline init script ---
        await context.add_init_script("""try { localStorage.setItem('a11y-contrast','off'); localStorage.setItem('high-contrast','off'); } catch(e){} Object.defineProperty(navigator, 'language', {get: () => 'en-GB'}); Object.defineProperty(navigator, 'languages', {get: () => ['en-GB','en']}); (function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions; Intl.DateTimeFormat.prototype.resolvedOptions=function(){const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();""")

        # --- Stealthish patches to reduce headless fingerprinting ---
        await context.add_init_script(r"""
          (() => {
            try {
              const rand = (min, max) => Math.random() * (max - min) + min;
              const jitter = (base, span) => base + rand(-span, span);

              // navigator.* tweaks
              const nav = navigator;
              if (nav) {
                const lang = (nav.language || 'en-GB');
                const langs = Array.isArray(nav.languages) && nav.languages.length ? nav.languages : ['en-GB','en'];
                Object.defineProperty(nav, 'webdriver', { get: () => undefined });
                Object.defineProperty(nav, 'hardwareConcurrency', { get: () => 8 });
                Object.defineProperty(nav, 'deviceMemory', { get: () => 8 });
                Object.defineProperty(nav, 'language', { get: () => lang });
                Object.defineProperty(nav, 'languages', { get: () => langs });
                Object.defineProperty(nav, 'maxTouchPoints', { get: () => 0 });
                Object.defineProperty(nav, 'platform', { get: () => 'Win32' });
              }

              // Canvas noise
              const patchCanvas = (proto) => {
                if (!proto) return;
                const toDataURL = proto.toDataURL;
                proto.toDataURL = function(...args) {
                  const ctx = this.getContext && this.getContext('2d');
                  if (ctx) {
                    const shift = () => (Math.random() - 0.5) * 2;
                    ctx.fillStyle = `rgba(${128+shift()},${128+shift()},${128+shift()},0.01)`;
                    ctx.fillRect(0, 0, 2, 2);
                  }
                  return toDataURL.apply(this, args);
                };
              };
              if (typeof HTMLCanvasElement !== 'undefined' && HTMLCanvasElement.prototype) {
                patchCanvas(HTMLCanvasElement.prototype);
              }
              if (typeof OffscreenCanvas !== 'undefined' && OffscreenCanvas.prototype) {
                patchCanvas(OffscreenCanvas.prototype);
              }

              // WebGL noise
              const patchWebGL = (proto) => {
                if (!proto) return;
                const getParameter = proto.getParameter;
                proto.getParameter = function(param) {
                  // Vendor/renderer slightly jittered
                  const VENDOR = 0x1F00, RENDERER = 0x1F01;
                  if (param === VENDOR) {
                    const v = getParameter.call(this, param);
                    return typeof v === 'string' ? v.replace(/Google Inc\./, 'Google LLC') : v;
                  }
                  if (param === RENDERER) {
                    const r = getParameter.call(this, param);
                    return typeof r === 'string' ? r.replace(/ANGLE \(|\)/g, '') : r;
                  }
                  return getParameter.call(this, param);
                };
              };
              if (typeof WebGLRenderingContext !== 'undefined' && WebGLRenderingContext.prototype) {
                patchWebGL(WebGLRenderingContext.prototype);
              }
              if (typeof WebGL2RenderingContext !== 'undefined' && WebGL2RenderingContext.prototype) {
                patchWebGL(WebGL2RenderingContext.prototype);
              }

              // AudioContext fingerprint jitter
              const patchAudio = (Cls) => {
                if (!Cls || !Cls.prototype) return;
                const getFloatFrequencyData = Cls.prototype.getFloatFrequencyData;
                if (getFloatFrequencyData) {
                  Cls.prototype.getFloatFrequencyData = function(arr) {
                    const res = getFloatFrequencyData.call(this, arr);
                    for (let i = 0; i < arr.length; i += Math.floor(arr.length / 8) || 1) {
                      arr[i] = arr[i] * (0.99 + Math.random() * 0.02);
                    }
                    return res;
                  };
                }
              };
              if (typeof AnalyserNode !== 'undefined') {
                patchAudio(AnalyserNode);
              }

              // Fonts enumeration shield
              if (typeof Navigator !== 'undefined' && Navigator.prototype) {
                const origFonts = Navigator.prototype.fonts;
                if (origFonts) {
                  Navigator.prototype.fonts = function() {
                    const it = origFonts.apply(this, arguments);
                    if (it && typeof it.status === 'string') return it;
                    return {
                      status: 'loaded',
                      check: () => true,
                      load: () => Promise.resolve(),
                      values: () => [].values()
                    };
                  };
                }
              }

            } catch (e) { /* swallow */ }
          })();
        """)

    # --- Main Run Logic (V88.5.0: Refactored for session management) ---
    async def run(self, *, site: str, query: str, site_config: Dict[str, Any], run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:

        settings = self._resolve_run_settings(site_config)
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        self.run_context = run_context  # Attach for downstream helpers
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        plugin = PLUGIN_REGISTRY.get(site.upper()) if PLUGIN_REGISTRY else None
        plugin_ctx = {"query": query, "site_config": site_config}
        nav_url = target_url
        if plugin:
            try:
                nav_url = plugin.before_navigate(target_url, plugin_ctx) or target_url
                self.logger.info(f"[Plugin:{site}] before_navigate => {nav_url}")
            except Exception as plugin_e:
                nav_url = target_url
                self.logger.warning(f"[Plugin:{site}] before_navigate failed: {plugin_e}")

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
                url_normalizer=self._normalize_to_en_int_url,
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

                if plugin:
                    try:
                        await plugin.after_navigate(page, plugin_ctx)
                    except Exception as plugin_e:
                        self.logger.warning(f"[Plugin:{site}] after_navigate failed: {plugin_e}")

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
                    if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None:
                        try:
                            await moncler_plp_recovery(page, site_config, query)
                        except Exception as _e:
                            self.logger.warning(f"[MonclerPatch] skipped (plp preflow): {_e}")
                    if plugin:
                        plp_ctx = {"plp_min_cards": 24, "query": query}
                        try:
                            materialized = await plugin.materialize(page, plp_ctx)
                            if materialized:
                                self.logger.info(f"[Plugin:{site}] PLP materialized successfully.")
                            else:
                                self.logger.warning(
                                    f"[Plugin:{site}] materialize failed (fallback to default PLP flow)."
                                )
                        except Exception as plugin_e:
                            self.logger.warning(
                                f"[Plugin:{site}] materialize raised {plugin_e!r} (ignored, continue default PLP flow)."
                            )
                        try:
                            asserted = await plugin.assert_plp(page, plp_ctx)
                        except Exception as plugin_e:
                            self.logger.warning(
                                f"[Plugin:{site}] assert_plp raised {plugin_e!r} (ignored, continue default PLP flow)."
                            )
                        else:
                            if not asserted:
                                self.logger.warning(
                                    f"[Plugin:{site}] assert_plp=False (continue with default PLP flow)."
                                )

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
                            target_url: str, # ★ V88.5.9: 回復で使うため追加
                            *, start_t: float, budget_ms: int) -> DiscoveryResult:

        # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
        # ★ V88.5.9: 1. 入口で trap でも、まず 1 回だけ回復ナビを試みる
        attempted_recover = False
        if self._looks_like_trap_or_legal(page.url):
            self.logger.warning("[_looks_like_trap] initial trap-like url: %s", page.url)
            attempted_recover = True
            await self._force_plp_recover(page, site_config, target_url)
            # 正規化後URLでもう一度だけ判定
            if self._looks_like_trap_or_legal(page.url):
                raise ValueError(f"Landing page looks like legal/trap (even after recovery attempt): {page.url}")
            self.logger.info("[_looks_like_trap] Recovery navigation seems successful.")

        await self._pause_for_operator(page, run_context, "before_plp_materialize")

        # ★ V88.5.7: _ensure_plp_materialized の戻り値を受け取る
        ok_materialized = await self._ensure_plp_materialized(
            page, site_config, settings,
            start_t=start_t, budget_ms=budget_ms, target_url=target_url
        )

        # ★ V88.5.7: (Fail Fast) まともなPLP(タイル0枚)が出なかったらすぐ諦める
        if not ok_materialized:
            raise ValueError(
                f"PLP did not materialize (no product tiles). URL={page.url}"
            )

        # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
        # ★ V88.5.9: スクロール後のURL再チェック (v88.5.6ロジック)
        if self._looks_like_trap_or_legal(page.url):
            # V88.5.9: まだ回復トライしてなければ、ここで試す
            if not attempted_recover:
                self.logger.warning("[_looks_like_trap] trap-like url after materialize: %s", page.url)
                await self._force_plp_recover(page, site_config, target_url)
                if self._looks_like_trap_or_legal(page.url):
                     raise ValueError(f"After materialize still on legal/trap page (even after recovery attempt): {page.url}")
                self.logger.info("[_looks_like_trap] Recovery navigation (post-materialize) seems successful.")
            else:
                # 既に回復試行済みで、スクロールしたらまたトラップに戻った場合
                raise ValueError(f"After materialize, bounced back to legal/trap page: {page.url}")

        try:
            await save_dom(run_context, page, "plp_dom_initial_materialized")
            pdp_cfg_a = (site_config.get('selectors') or {}).get('pdp', {}) or {}
            await count_selectors(
                run_context,
                page,
                (pdp_cfg_a.get('pdp_link_selectors') or []) +
                (pdp_cfg_a.get('plp_container_selectors') or []),
                name="selector_counts_plp_initial"
            )
        except Exception as e:
            logger.warning(f"[Hook A1] Failed: {e}")

        pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)

        # Fallback logic (header search, click first card)
        # V88.5.7: このブロックは ok_materialized=True (タイル1枚以上) だが
        # pdp_links=[] だった場合にのみ実行される
        if not pdp_links:

            # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
            # ★ 3. ここでも trap判定をはさむ (v88.5.6ロジック)
            # (V88.5.9: 回復ロジックは既に試したので、ここでは検知のみ)
            if not pdp_links and self._looks_like_trap_or_legal(page.url):
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
                        await save_dom(run_context, page, "plp_dom_search_fallback")
                        pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                        await count_selectors(run_context, page, (pdp_cfg_a2.get("pdp_link_selectors") or []) + (pdp_cfg_a2.get("plp_container_selectors") or []), name="selector_counts_after_search_fallback")
                    except Exception as e: logger.warning(f"[Hook A3] Failed: {e}")
                    pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)

                    # --- V88.5.5: 早期失敗ロジック ---
                    if not pdp_links:
                        self.logger.warning("[Fallback] No hrefs after search. Clicking first card...")
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

    async def _plp_header_search_fallback(self, page, query: str, site_config, settings, run_context, context: BrowserContext, *, start_t: float, budget_ms: int) -> bool:
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        sel_open = _dedupe_keep_order(ui.get("search_open", []) + ["button[aria-label='Search']", "[aria-label*='Search' i]"])
        sel_input = _dedupe_keep_order(ui.get("search_input", []) + ["form[role='search'] input", "input[type='search']", "input[name='q']", "[data-testid*='search' i] input", "[role='search'] input", "dialog input[type='search']"])
        sel_submit = _dedupe_keep_order(ui.get("search_submit", []) + ["form[role='search'] button[type='submit']"])
        try:
            opened = False
            for s in sel_open:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0: await el.click(timeout=3000); opened = True; await asyncio.sleep(0.2); await self.safe_wait_selector(page, "[role='search'], [data-overlay], dialog[open]", timeout_ms=5000, state="visible"); self.logger.debug(f"[Fallback] opened search with '{s}'"); break
            if not opened: await page.keyboard.press("/"); await self.safe_wait_selector(page, "input[type='search']", timeout_ms=5000, state="visible")
            found_input = False
            for s in sel_input:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_visible(): await el.fill(query, timeout=8000); found_input = True; self.logger.debug(f"[Fallback] filled '{query}' into '{s}'"); break
            if not found_input: raise ValueError("Input field not found")
            submitted = False
            for s in sel_submit:
                if self._time_left_ms(start_t, budget_ms) <= 0: break
                el = page.locator(s).first
                if await el.count() > 0 and await el.is_enabled(): await el.click(timeout=5000); submitted = True; self.logger.debug(f"[Fallback] submitted with '{s}'"); break
            if not submitted: await page.keyboard.press("Enter"); self.logger.debug("[Fallback] submitted with Enter key.")
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                try: await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception: self.logger.debug("[Fallback] Optional main wait timed out.")
            return True
        except Exception:
            self.logger.warning("[Fallback] UI search failed. Trying direct search URL.")
            try:
                search_url = f"https://www.moncler.com/en-int/search?q={quote_plus(query)}&forceLocale=en-int"
                # V88.5.3: (BugFix) `url=` キーワード引数を明示的に指定
                await page.goto(url=search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)
                try: await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception: self.logger.debug("[Fallback] Optional main wait (URL) timed out.")
                return True
            except Exception as final_e: self.logger.error(f"[Fallback] Direct search URL failed: {final_e}"); return False

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
    async def _force_plp_recover(self, page, site_config: dict, target_url: str | None) -> None:
        try:
            plp = (
                target_url
                or site_config.get("plp_hard_nav")
                or site_config.get("seed_plp_url")
                or site_config.get("fallback_url")
                or site_config.get("home_url")
            )
            if not plp:
                logger.debug("[recover] no PLP candidate found; skip")
                return
            # ロケール強制
            plp = self._normalize_to_en_int_url(plp) # V88.6.1: 修正
            logger.info("[recover] Forcing PLP navigation: %s", plp)
            await page.goto(url=plp, wait_until="domcontentloaded") # V88.6.1: 修正
        except Exception as e:
            logger.debug("[recover] force PLP failed: %r", e)

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

        # Call write_fail_snapshot to save DOM and screenshot
        dom_path_str = None
        try:
            # write_fail_snapshot は active_page (開いている可能性のあるページ) を使う
            await write_fail_snapshot(run_context, active_page, final_url_on_fail, e, site_config)
            # Infer standard path used by write_fail_snapshot
            # ★ V88.6.1: (BugFix) ファイル名を failure_dom.html に修正
            dom_guess = run_context.get_path("failure_dom.html")
            if dom_guess and Path(dom_guess).exists():
                dom_path_str = str(dom_guess)
        except Exception as hook_e:
            logger.warning(f"[Hook C2] write_fail_snapshot failed: {hook_e}")

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
        learned_path.write_text(json.dumps(merged_selectors, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info(f"[LEARN] Saved merged selectors to: {learned_path}")
