from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

from playwright.async_api import BrowserContext, Page

from app.agents.browser.extractor import (
    VISIBLE_PRICE_SELECTORS,
    BrowserExtractionService,
    looks_like_product_url,
)
from app.agents.browser.navigation_driver import (
    NavigationContext,
    NavigationDriver,
    _dedupe_keep_order,
    is_same_origin,
)
from app.agents.browser.session_manager import SessionManager
from app.agents.browser.ui_helpers import (
    accept_cookies_if_present as ui_accept_cookies_if_present,
    click_continue_shopping_if_present as ui_click_continue_shopping_if_present,
    dismiss_geo_modal as ui_dismiss_geo_modal,
    human_like_mouse_move as ui_human_like_mouse_move,
    human_like_pause as ui_human_like_pause,
    human_like_scroll as ui_human_like_scroll,
    kill_overlays as ui_kill_overlays,
    pause_for_operator as ui_pause_for_operator,
    safe_wait_selector as ui_safe_wait_selector,
)
from app.agents.browser.settings import (
    resolve_run_settings as settings_resolve_run_settings,
    time_left_ms as settings_time_left_ms,
)
from app.agents.selector_discovery_agent import SelectorDiscoveryAgent
from app.core.run_context import RunContext
from app.models.result_models import DiscoveryResult
from app.utils.observability import count_selectors, save_dom, save_raw_hrefs, write_fail_snapshot

try:
    from app.agents.interactive_repair_session import InteractiveRepairSession
except Exception:
    InteractiveRepairSession = None

try:
    from app.agents.browser_use_moncler_patch import moncler_plp_recovery
except Exception:
    moncler_plp_recovery = None


logger = logging.getLogger(__name__)
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

OVERALL_PLP_BUDGET_MS_DEFAULT = 120000  # 120s watchdog


# ==============================================================================
# Helper Functions
# (V88.6.0: _looks_like_trap_or_legal はクラスメソッドに移動)
# ==============================================================================


# ==============================================================================
# Module-level helpers
# ==============================================================================


def _merge_learned_selectors(site: str, site_config: dict[str, Any], run_context: RunContext) -> None:
    """学習済みセレクタを site_config にマージ。"""
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


# ==============================================================================
# BrowserUseAgent Class
# ==============================================================================


class BrowserUseAgent:
    """
    Playwright を駆動して PLP/PDP を探索するメインエージェント。

    TODO: LocaleGateHandler などの共通クラスにロケーションゲート処理を抽象化予定。
    """

    # Phase 2: LocaleMixin._looks_like_trap_or_legal に委譲
    def _looks_like_trap_or_legal(self, url: str) -> bool:
        """LocaleMixin._looks_like_trap_or_legal delegation (pure URL check, no self.page dependency)"""
        from app.agents.browser.locale_manager import LocaleMixin

        return LocaleMixin._looks_like_trap_or_legal(self, url)

    def __init__(self, runtime_kwargs: dict[str, Any] | None = None):
        self.runtime_kwargs = runtime_kwargs or {}
        self.discovery_agent = SelectorDiscoveryAgent(runtime_kwargs=self.runtime_kwargs)
        self.logger = logger
        self.run_context: RunContext | None = None  # Temporarily attach RunContext during run()

        # --- V88.6.x: Session handles are managed by SessionManager ---
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._session_manager: SessionManager | None = None
        self.extraction_service = BrowserExtractionService(self.logger, self.runtime_kwargs)

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
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
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

        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=800)
        await page.wait_for_timeout(120)

        # --- Moncler 固有の回復 ---
        if site.upper() == "MONCLER_OFFICIAL" and not likely_plp:
            try:
                gate_links_count = await page.evaluate("() => document.querySelectorAll(\"a[href*='/en-']\").length")
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
                    with contextlib.suppress(Exception):
                        await page.wait_for_load_state("networkidle", timeout=2000)
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
                    self.logger.warning(f"[Moncler] Bounced to corporate. Forcing back to PLP: {fixed_url}")
                    await page.goto(url=fixed_url, wait_until="domcontentloaded")
                    await self._accept_cookies_if_present(page, site_config)
                    await self._click_continue_shopping_if_present(page, site_config)
            except Exception:
                pass

        return page

    def _build_pdp_prepare_hook(
        self,
        *,
        site_config: dict[str, Any],
        settings: dict[str, Any],
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
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
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
        run_context: RunContext | None,
        settings: dict[str, Any],
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

    async def run_with_repair(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
        max_steps: int = 5,
        repair_budget_ms: int = 60000,
    ) -> DiscoveryResult:
        """Atlas型の「自己修復つきスクレイピング」フロー。"""
        settings = self._resolve_run_settings(site_config)
        self.run_context = run_context
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        _merge_learned_selectors(site, site_config, run_context)

        base_result = await self._repair_initial_run(
            site=site, query=query, site_config=site_config,
            run_context=run_context, target_url=target_url,
            likely_plp=likely_plp, settings=settings,
        )

        if getattr(base_result, "ok", False):
            try:
                await self._close_session(run_context, settings)
            finally:
                if hasattr(self, "run_context"):
                    del self.run_context
            return base_result

        return await self._run_interactive_repair(
            base_result=base_result, site=site, query=query,
            site_config=site_config, run_context=run_context,
            settings=settings, max_steps=max_steps,
            repair_budget_ms=repair_budget_ms, target_url=target_url,
        )

    async def _repair_initial_run(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
        settings: dict[str, Any],
    ) -> DiscoveryResult:
        """Phase 1: セッション開始 → PLP/PDP抽出の初回試行。"""
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        try:
            page = await self._open_session(
                site=site, site_config=site_config, run_context=run_context,
                settings=settings, target_url=target_url,
                timeout_ms=timeout_ms, likely_plp=likely_plp,
            )

            await run_context.take_screenshot(page, "20_pre_vrt_and_extraction")
            if settings.get("enable_visual_regression_check") and "plp" in (settings.get("vrt_scope") or ""):
                await self._perform_vrt(page, "plp", settings)

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

            if mode == "learn":
                raise ValueError("mode='learn' is not supported in run_with_repair. Use run() instead.")

            if likely_plp:
                return await self._run_plp_flow(
                    page, context, site, query, site_config,
                    settings, run_context, target_url=target_url,
                    start_t=start_t, budget_ms=budget_ms,
                )
            else:
                return await self._run_pdp_flow(page, site, query, settings, run_context, site_config)

        except Exception as e:
            return await self._handle_run_failure(e, site, query, site_config, run_context, self._page)

    async def _run_interactive_repair(
        self,
        *,
        base_result: DiscoveryResult,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
        max_steps: int,
        repair_budget_ms: int,
        target_url: str,
    ) -> DiscoveryResult:
        """Phase 3: インタラクティブ修復ループ + 成果物評価。"""
        self.logger.warning(
            f"[run_with_repair] Initial run failed. Entering guided repair loop. Reason: {base_result.message}"
        )

        if InteractiveRepairSession is None:
            self.logger.error("[run_with_repair] InteractiveRepairSession not available, cannot self-heal.")
            await self._close_session_safely(run_context, settings)
            return base_result

        try:
            from app.utils.ai_llm_controller import AiLlmController
            llm_ctrl = AiLlmController(mode="Chat/Default")
        except Exception as e:
            self.logger.error(f"[run_with_repair] Failed to instantiate AiLlmController: {e}. Aborting repair.")
            await self._close_session_safely(run_context, settings)
            return base_result

        failure_ctx = self._build_repair_failure_context(base_result, site_config)

        repair_out, repair_status, healed_result = await self._execute_repair_loop(
            site=site, query=query, run_context=run_context,
            max_steps=max_steps, repair_budget_ms=repair_budget_ms,
            base_result=base_result, failure_ctx=failure_ctx, llm_ctrl=llm_ctrl,
        )

        if healed_result is None:
            healed_result = self._evaluate_repair_result(
                repair_out=repair_out, repair_status=repair_status,
                site=site, query=query, site_config=site_config,
                run_context=run_context, target_url=target_url,
                base_result=base_result,
            )

        await self._close_session_safely(run_context, settings)
        return healed_result

    async def _close_session_safely(self, run_context: RunContext, settings: dict[str, Any]) -> None:
        """セッションを閉じ、self.run_context を削除する。"""
        try:
            await self._close_session(run_context, settings)
        finally:
            if hasattr(self, "run_context"):
                del self.run_context

    @staticmethod
    def _build_repair_failure_context(base_result: DiscoveryResult, site_config: dict[str, Any]) -> dict[str, Any]:
        """InteractiveRepairSession に渡す failure_context を構築。"""
        failure_ev = base_result.evidence or {}
        failure_ctx_from_run = failure_ev.get("failure_context", {})
        return {
            "final_url": failure_ev.get("final_url"),
            "page_html_path": failure_ctx_from_run.get("dom_snapshot_path"),
            "screenshot_path": (failure_ctx_from_run.get("screenshots") or [None])[0],
            "exception_message": (failure_ctx_from_run.get("errors") or [""])[0],
            "selectors_used": {
                "pdp": (site_config.get("selectors", {}) or {}).get("pdp", {}),
            },
            "intent_description": failure_ctx_from_run.get(
                "intent_description", "Goal: Extract PLP items or PDP price. Initial attempt failed."
            ),
        }

    async def _execute_repair_loop(
        self,
        *,
        site: str,
        query: str,
        run_context: RunContext,
        max_steps: int,
        repair_budget_ms: int,
        base_result: DiscoveryResult,
        failure_ctx: dict[str, Any],
        llm_ctrl: Any,
    ) -> tuple[Any, str, DiscoveryResult | None]:
        """Atlas修復ループを実行し、(repair_out, repair_status, healed_result_on_error) を返す。"""
        repair_out = None
        repair_status = "unknown_error"
        healed_result: DiscoveryResult | None = None

        try:
            repair_session = InteractiveRepairSession(
                ai_controller=llm_ctrl, run_context=run_context, max_steps=max_steps,
            )
            maybe_coro = repair_session.run_repair_loop(
                page=self._page, site_key=site,
                intent="Collect PLP items and PDP prices",
                initial_failure=failure_ctx,
            )
            repair_budget_sec = max(5.0, float(repair_budget_ms) / 1000.0)

            if asyncio.iscoroutine(maybe_coro):
                self.logger.info(f"[run_with_repair] Waiting for async repair loop (budget: {repair_budget_sec}s)")
                repair_out = await asyncio.wait_for(maybe_coro, timeout=repair_budget_sec)
            else:
                self.logger.warning("[run_with_repair] Running sync repair loop (cannot enforce budget)")
                repair_out = maybe_coro
            repair_status = "completed"

        except asyncio.TimeoutError:
            self.logger.error(f"[run_with_repair] InteractiveRepairSession timed out after {repair_budget_sec}s.")
            repair_status = "timeout_exceeded"
            healed_result = DiscoveryResult(
                ok=False, site=site, query=query,
                message=f"Repair loop timed out after {repair_budget_sec}s",
                evidence={"status": "timeout_exceeded", "initial_failure": base_result.evidence},
            )
        except Exception as repair_e:
            self.logger.error(f"[run_with_repair] InteractiveRepairSession failed: {repair_e}", exc_info=True)
            repair_status = "catastrophic_failure"
            healed_result = DiscoveryResult(
                ok=False, site=site, query=query,
                message=f"Repair loop failed: {repair_e}",
                evidence={"status": "catastrophic_failure", "initial_failure": base_result.evidence},
            )

        return repair_out, repair_status, healed_result

    def _evaluate_repair_result(
        self,
        *,
        repair_out: Any,
        repair_status: str,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        base_result: DiscoveryResult,
    ) -> DiscoveryResult:
        """修復ループ結果を評価し、セレクタマージ + DiscoveryResult を構築。"""
        if repair_status == "completed" and repair_out:
            selectors_update = repair_out.get("selectors_update")
            code_patch = repair_out.get("code_patch", "")
            self._merge_repair_selectors(selectors_update, site, run_context)

            if code_patch:
                self.logger.info("[run_with_repair] Received code patch suggestion. Saving to artifacts.")
                run_context.save_text("browser_use_agent.patch", code_patch)

            if selectors_update or code_patch:
                self.logger.info("[run_with_repair] Repair loop completed and produced artifacts.")
                return DiscoveryResult(
                    ok=True, site=site, query=query,
                    message="Recovered via InteractiveRepairSession",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "selectors_update": selectors_update, "code_patch": code_patch,
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []), "status": "recovered",
                    },
                )
            else:
                self.logger.warning("[run_with_repair] Repair completed but produced no artifacts.")
                status_from_repair = repair_out.get("status", "exhausted_steps")
                return DiscoveryResult(
                    ok=False, site=site, query=query,
                    message=f"Repair loop finished without artifacts (Status: {status_from_repair})",
                    evidence={
                        "final_url": repair_out.get("final_url") or (self._page.url if self._page else target_url),
                        "steps_taken": repair_out.get("steps_taken"),
                        "repair_log": repair_out.get("log", []),
                        "status": status_from_repair, "initial_failure": base_result.evidence,
                    },
                )

        self.logger.error(
            f"[run_with_repair] Repair loop finished abnormally (Status: {repair_status}, repair_out: {repair_out})"
        )
        return DiscoveryResult(
            ok=False, site=site, query=query,
            message=f"Repair loop failed with unknown status: {repair_status}",
            evidence={"status": repair_status, "initial_failure": base_result.evidence},
        )

    def _merge_repair_selectors(self, selectors_update: dict | None, site: str, run_context: RunContext) -> None:
        """修復されたセレクタを overrides.local.json にマージ。"""
        if not selectors_update:
            return
        try:
            from app.utils.overrides_store import update_site_selectors
            site_block = selectors_update.get(site.upper()) or selectors_update.get(site) or {}
            new_sels = site_block.get("selectors") or {}
            if new_sels:
                overrides_path = "app/config/sites/overrides.local.json"
                updated, diff_txt = update_site_selectors(
                    site=site.upper(), new_selectors=new_sels, overrides_path=overrides_path,
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

    # --- Settings Resolution ---
    def _resolve_run_settings(self, site_config: dict[str, Any]) -> dict[str, Any]:
        return settings_resolve_run_settings(site_config, self.runtime_kwargs, self.logger)

    # --- Time Budget Helpers ---
    @staticmethod
    def _start_watchdog(budget_ms: int) -> tuple[float, int]:
        return time.monotonic(), int(budget_ms)

    @staticmethod
    def _time_left_ms(start_t: float, budget_ms: int) -> int:
        return settings_time_left_ms(start_t, budget_ms)

    @staticmethod
    def _slice_timeout_ms(left_ms: int, cap_ms: int) -> int:
        return max(500, min(left_ms, cap_ms))

    # --- Safe Wait ---
    async def safe_wait_selector(self, page: Page, selector: str, *, timeout_ms: int, state: str = "visible") -> bool:
        return await ui_safe_wait_selector(page, selector, timeout_ms=timeout_ms, state=state)

    # --- UI Helpers ---
    async def _kill_overlays(self, page: Page) -> None:
        await ui_kill_overlays(page)

    async def _click_continue_shopping_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        return await ui_click_continue_shopping_if_present(page, site_config)

    async def _pause_for_operator(self, page: Page | None, run_context: RunContext | None, label: str) -> None:
        """Headful 実行中に人間が介入して操作できるよう一時停止する。"""
        await ui_pause_for_operator(page, run_context, label, self.runtime_kwargs, self.logger)

    async def _accept_cookies_if_present(self, page: Page, site_config: dict[str, Any]) -> bool:
        return await ui_accept_cookies_if_present(page, site_config)

    async def _dismiss_geo_modal(self, page: Page) -> None:
        await ui_dismiss_geo_modal(page, self.logger)

    # --- Locale Helpers (Moncler specific but could be generalized) ---
    def _normalize_to_en_int_url(self, url: str) -> str:
        u = urlparse(url)
        path = (u.path or "/").replace("//", "/")
        path = path.replace("/en-gb/", "/en-int/")
        seg = [s for s in path.split("/") if s]
        i = 0
        while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""):
            i += 1
        seg = [s for s in seg[i:] if s.lower() != "en-int"]
        norm = "/en-int/" + "/".join(seg)
        if not norm.endswith("/"):
            norm += "/"
        q = dict(parse_qsl(u.query))
        q["forceLocale"] = "en-int"
        q.setdefault("shipToCountry", "GB")
        return urlunparse((u.scheme, u.netloc, norm, u.params, urlencode(q), u.fragment))

    async def _force_en_int(self, page: Page) -> None:
        try:
            if page.context:
                await page.context.add_cookies(
                    [
                        {"name": "moncler-shipping-country", "value": "GB", "domain": ".moncler.com", "path": "/"},
                        {"name": "moncler-shipping-language", "value": "en", "domain": ".moncler.com", "path": "/"},
                    ]
                )
        except Exception:
            pass
        try:
            fixed = self._normalize_to_en_int_url(page.url)
            if fixed != page.url:
                # V88.5.3: (BugFix) `url=` キーワード引数を明示的に指定
                await page.goto(url=fixed, wait_until="domcontentloaded")
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=1500)
        except Exception:
            pass

    # --- PLP Materialize ---
    async def _ensure_plp_materialized(
        self,
        page: Page,
        site_config: dict[str, Any],
        settings: dict[str, Any],
        *,
        start_t: float,
        budget_ms: int,
        target_url: str | None = None,
    ) -> bool:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        tile_selectors = _dedupe_keep_order(
            (pdp_cfg.get("pdp_link_selectors") or [])
            + (pdp_cfg.get("plp_container_selectors") or [])
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
        run_ctx = getattr(self, "run_context", None)

        locale_recover_attempts = 0
        locale_recover_max = int(settings.get("locale_recover_max", 5))

        for attempt in range(max_scroll_attempts):
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms <= 0:
                self.logger.warning("[Materialize] Timed out.")
                return False

            # v88.6.x: Attemptごとに遅延表示ゲート/バナーを掃除する
            with contextlib.suppress(Exception):
                await self._accept_cookies_if_present(page, site_config)
            with contextlib.suppress(Exception):
                await self._dismiss_geo_modal(page)
            with contextlib.suppress(Exception):
                await self._kill_overlays(page)

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
                    await run_ctx.take_screenshot(page, f"30_plp_materialize_attempt_{attempt + 1:02d}")
                except Exception as ss_e:
                    self.logger.warning(f"[Materialize] Screenshot failed on attempt {attempt + 1}: {ss_e}")

            try:
                for _ in range(6):
                    await page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.6))")
                    await page.wait_for_timeout(160)
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=800)
            except Exception as e:
                self.logger.warning(f"[Materialize] Scroll failed on attempt {attempt + 1}: {e}")
                break

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
                    self.logger.info(f"[Materialize] Success: Found {count} tiles (>= {target_min_tiles}).")
                    return True
                if count < 4 and attempt >= 1:
                    self.logger.warning(
                        f"[Materialize] Low tiles ({count}) after {attempt + 1} attempts, forcing recovery hop."
                    )
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
            except Exception as e:
                self.logger.warning(f"[Materialize] Could not count tiles on attempt {attempt + 1}: {e}")

        final_count = await page.locator(tile_selector_str).count()
        if final_count > 0:
            self.logger.warning(
                f"[Materialize] Finished attempts, found {final_count} tiles (< {target_min_tiles}), but proceeding as non-empty."
            )
            return True
        self.logger.error("[Materialize] Failed: No product tiles found after all scroll attempts.")
        return False

    # --- Price / Size Selection ---
    async def _read_price_or_none(self, page: Page) -> str | None:
        try:
            for sel in PRICE_SELECTORS:
                loc = page.locator(sel)
                count = await loc.count()
                if count == 0:
                    continue
                for i in range(count):
                    el = loc.nth(i)
                    try:
                        tag = (await el.evaluate("e => e && e.tagName") or "").lower()
                        if not tag:
                            continue
                        content = await (el.get_attribute("content") if tag == "meta" else el.inner_text())
                    except Exception:
                        continue
                    if content:
                        s = content.strip()
                        if s and re.search(r"\d", s):
                            self.logger.debug(f"Price found via selector '{sel}' (nth={i}): {s}")
                            return s
        except Exception as e:
            self.logger.warning(f"Error: Exception during _read_price_or_none (outer loop): {e}")
        self.logger.debug("Price string not found (_read_price_or_none).")
        return None

    # --- PDP Extraction ---
    # --- PLP -> PDP Link Collection (Robust v85.5) ---
    def _normalize_abs_url(self, base_url: str, href: str) -> str:
        """Phase 2: navigation_helpers.normalize_abs_url に委譲"""
        from app.agents.browser.navigation_helpers import normalize_abs_url

        return normalize_abs_url(base_url, href)

    async def _collect_pdp_links(
        self, page: Page, site_config: dict, settings: dict, run_context: RunContext
    ) -> list[str]:
        target_url = page.url
        found_links: set[str] = set()

        # Phase 1a: Global <a href> sweep + Regex Filter
        try:
            raw_hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)"
            )
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
        selectors_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
        PLP_PDP_LINK_SELECTORS = _dedupe_keep_order(
            (selectors_cfg.get("pdp_link_selectors", []) or [])
            + [
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
        )
        for sel in PLP_PDP_LINK_SELECTORS:
            try:
                nodes = await page.query_selector_all(sel)
                if not nodes:
                    continue
                matched_count = 0
                for n in nodes:
                    href = (
                        await n.get_attribute("href")
                        or await n.get_attribute("data-href")
                        or await n.get_attribute("data-product-url")
                        or await n.get_attribute("data-url")
                    )
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
        cleaned: list[str] = []
        noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
        for u in links:
            if not noise_rx.search(u):
                cleaned.append(u)
        logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")
        try:
            sample = cleaned[:20]
            logger.debug(f"[PLP→PDP] sample={sample}")
            if self.run_context:
                self.run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
            # V87.0: Robust save_raw_hrefs call
            try:
                if callable(save_raw_hrefs) and run_context:
                    res = save_raw_hrefs(run_context, cleaned, name="raw_hrefs_final_cleaned")
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass
        except Exception:
            pass
        return cleaned

    # --- V88.3.0J: _run_deep_extraction_phase2 Safer Fallback Evaluate ---
    # --- V88.6.2J: (BugFix) SyntaxError on container_sels ---
    async def _run_deep_extraction_phase2(self, page: Page, site_config: dict) -> list[str]:
        from app.agents.browser.deep_extraction import run_deep_extraction_phase2

        return await run_deep_extraction_phase2(page, site_config, self.safe_wait_selector)

    # --- VRT ---
    async def _perform_vrt(self, page: Page, scope: str, settings: dict[str, Any]):
        from app.utils.visual_regression import perform_vrt

        site_name = self.runtime_kwargs.get("site") or "GENERIC"
        await perform_vrt(page, scope, settings, site_name, self.logger)

    # --- Browser Context Setup ---
    def _build_context_options(
        self,
        settings: dict[str, Any],
        run_context: RunContext,
    ) -> dict[str, Any]:
        from app.agents.browser.session_config import build_context_options

        return build_context_options(settings, run_context.get_path, self.logger)

    async def _setup_routes(self, context: BrowserContext, settings: dict[str, Any]):
        from app.agents.browser.route_setup import setup_routes

        await setup_routes(context, settings, self._normalize_to_en_int_url, self.logger)

    def _get_session_file(self, site: str, site_config: dict[str, Any]) -> Path:
        from app.agents.browser.session_config import get_session_file

        return get_session_file(site, site_config)

    async def _apply_saved_session(
        self, context: BrowserContext, page: Page, site: str, site_config: dict[str, Any]
    ) -> None:
        from app.agents.browser.session_config import apply_saved_session, get_session_file

        sess_file = get_session_file(site, site_config)
        await apply_saved_session(context, sess_file, self.logger)

    # --- Human-like interaction helpers ---
    async def _human_like_pause(self, page: Page, *, min_ms: int = 400, max_ms: int = 900):
        await ui_human_like_pause(page, min_ms=min_ms, max_ms=max_ms)

    async def _human_like_mouse_move(self, page: Page):
        await ui_human_like_mouse_move(page)

    async def _human_like_scroll(self, page: Page):
        await ui_human_like_scroll(page)

    async def _setup_init_scripts(self, context: BrowserContext):
        from app.agents.browser.stealth import setup_stealth_init_scripts

        await setup_stealth_init_scripts(context)

    # --- Main Run Logic (V88.5.0: Refactored for session management) ---
    async def run(
        self,
        *,
        site: str,
        query: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        target_url: str,
        likely_plp: bool,
    ) -> DiscoveryResult:
        settings = self._resolve_run_settings(site_config)
        timeout_ms = int(settings.get("timeout_sec", 60)) * 1000
        self.run_context = run_context  # Attach for downstream helpers
        self.runtime_kwargs["site_config"] = site_config
        self.runtime_kwargs["site"] = site
        mode = (self.runtime_kwargs or {}).get("mode", "run").lower()

        from app.agents.browser.plugins import get_plugin_registry

        registry = get_plugin_registry()
        plugin = registry.get(site.upper())
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

        page: Page | None = None

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

                if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None and plugin is None:
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
                    if site.upper() == "MONCLER_OFFICIAL" and moncler_plp_recovery is not None and plugin is None:
                        try:
                            await moncler_plp_recovery(page, site_config, query)
                        except Exception as _e:
                            self.logger.warning(f"[MonclerPatch] skipped: {_e}")
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

                    nav_ctx = NavigationContext(
                        site=site,
                        query=query,
                        site_config=site_config,
                        settings=settings,
                        run_context=run_context,
                        start_t=start_t,
                        budget_ms=budget_ms,
                        entry_url=nav_url,
                    )
                    navigation_driver = NavigationDriver(
                        page=page,
                        ensure_plp_materialized=lambda pg, scfg, stg, s_t, b_ms: self._ensure_plp_materialized(
                            pg,
                            scfg,
                            stg,
                            start_t=s_t,
                            budget_ms=b_ms,
                            target_url=nav_url,
                        ),
                        trap_checker=None,
                        telemetry=None,
                        strategy=plugin,
                    )
                    try:
                        nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
                    except Exception as nav_e:
                        self.logger.debug(f"[NavigationDriver] Stage3A-2 failed (fallback to legacy): {nav_e}")
                        nav_outcome = None

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
    async def _run_plp_flow(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        query: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        target_url: str,
        *,
        start_t: float,
        budget_ms: int,
        skip_materialize: bool = False,
    ) -> DiscoveryResult:
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

        # --- Stage 3A-2: NavigationDriver が materialize 済みならスキップ ---
        if skip_materialize:
            ok_materialized = True
        else:
            ok_materialized = await self._ensure_plp_materialized(
                page, site_config, settings, start_t=start_t, budget_ms=budget_ms, target_url=target_url
            )

        # ★ V88.5.7: (Fail Fast) まともなPLP(タイル0枚)が出なかったらすぐ諦める
        if not ok_materialized:
            raise ValueError(f"PLP did not materialize (no product tiles). URL={page.url}")

        # ★ V88.6.0: 呼び出しを self._looks_like_trap_or_legal に修正
        # ★ V88.5.9: スクロール後のURL再チェック (v88.5.6ロジック)
        if self._looks_like_trap_or_legal(page.url):
            # V88.5.9: まだ回復トライしてなければ、ここで試す
            if not attempted_recover:
                self.logger.warning("[_looks_like_trap] trap-like url after materialize: %s", page.url)
                await self._force_plp_recover(page, site_config, target_url)
                if self._looks_like_trap_or_legal(page.url):
                    raise ValueError(
                        f"After materialize still on legal/trap page (even after recovery attempt): {page.url}"
                    )
                self.logger.info("[_looks_like_trap] Recovery navigation (post-materialize) seems successful.")
            else:
                # 既に回復試行済みで、スクロールしたらまたトラップに戻った場合
                raise ValueError(f"After materialize, bounced back to legal/trap page: {page.url}")

        try:
            await save_dom(run_context, page, "plp_dom_initial_materialized")
            pdp_cfg_a = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            await count_selectors(
                run_context,
                page,
                (pdp_cfg_a.get("pdp_link_selectors") or []) + (pdp_cfg_a.get("plp_container_selectors") or []),
                name="selector_counts_plp_initial",
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
                raise ValueError(f"No PDP links and URL looks like trap/legal page: {page.url}")

            try:
                self.logger.debug("[Fallback] Trying header search UI...")
                did_search = await self._plp_header_search_fallback(
                    page, query, site_config, settings, run_context, context, start_t=start_t, budget_ms=budget_ms
                )
                if did_search:
                    await self._click_continue_shopping_if_present(page, site_config)
                    try:
                        anchors = await page.locator("a[href*='/p/'], a[href*='/product/']").count()
                    except Exception:
                        anchors = 0
                    if anchors < 6:
                        self.logger.debug(f"[Fallback] Materializing after search (anchors={anchors}<6)")
                        await self._ensure_plp_materialized(
                            page, site_config, settings, start_t=start_t, budget_ms=budget_ms, target_url=target_url
                        )
                    try:
                        await save_dom(run_context, page, "plp_dom_search_fallback")
                        pdp_cfg_a2 = (site_config.get("selectors") or {}).get("pdp", {}) or {}
                        await count_selectors(
                            run_context,
                            page,
                            (pdp_cfg_a2.get("pdp_link_selectors") or [])
                            + (pdp_cfg_a2.get("plp_container_selectors") or []),
                            name="selector_counts_after_search_fallback",
                        )
                    except Exception as e:
                        logger.warning(f"[Hook A3] Failed: {e}")
                    pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)

                    # --- V88.5.5: 早期失敗ロジック ---
                    if not pdp_links:
                        self.logger.warning("[Fallback] No hrefs after search. Clicking first card...")
                        new_page = await self._click_first_card_or_link(page, site_config, settings, context)
                        if new_page:
                            return await self._run_pdp_flow(
                                new_page or page, site, query, settings, run_context, site_config
                            )
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
            prepare_hook = self._build_pdp_prepare_hook(
                site_config=site_config, settings=settings, run_context=run_context
            )
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

    async def _plp_header_search_fallback(
        self,
        page,
        query: str,
        site_config,
        settings,
        run_context,
        context: BrowserContext,
        *,
        start_t: float,
        budget_ms: int,
    ) -> bool:
        ui = (site_config.get("selectors") or {}).get("ui") or {}
        sel_open = _dedupe_keep_order(
            ui.get("search_open", []) + ["button[aria-label='Search']", "[aria-label*='Search' i]"]
        )
        sel_input = _dedupe_keep_order(
            ui.get("search_input", [])
            + [
                "form[role='search'] input",
                "input[type='search']",
                "input[name='q']",
                "[data-testid*='search' i] input",
                "[role='search'] input",
                "dialog input[type='search']",
            ]
        )
        sel_submit = _dedupe_keep_order(ui.get("search_submit", []) + ["form[role='search'] button[type='submit']"])
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
                    self.logger.debug(f"[Fallback] opened search with '{s}'")
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
                    await el.fill(query, timeout=8000)
                    found_input = True
                    self.logger.debug(f"[Fallback] filled '{query}' into '{s}'")
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
                    self.logger.debug(f"[Fallback] submitted with '{s}'")
                    break
            if not submitted:
                await page.keyboard.press("Enter")
                self.logger.debug("[Fallback] submitted with Enter key.")
            left_ms = self._time_left_ms(start_t, budget_ms)
            if left_ms > 1000:
                await page.wait_for_load_state("domcontentloaded", timeout=min(left_ms, 15000))
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main wait timed out.")
            return True
        except Exception:
            self.logger.warning("[Fallback] UI search failed. Trying direct search URL.")
            try:
                search_url = f"https://www.moncler.com/en-int/search?q={quote_plus(query)}&forceLocale=en-int"
                # V88.5.3: (BugFix) `url=` キーワード引数を明示的に指定
                await page.goto(url=search_url, wait_until="domcontentloaded", timeout=30000)
                await self._click_continue_shopping_if_present(page, site_config)
                try:
                    await page.wait_for_selector("main, #main, [role='main']", state="visible", timeout=800)
                except Exception:
                    self.logger.debug("[Fallback] Optional main wait (URL) timed out.")
                return True
            except Exception as final_e:
                self.logger.error(f"[Fallback] Direct search URL failed: {final_e}")
                return False

    # ★ V88.5.5: タイムアウトを 15000ms -> 5000ms に短縮
    async def _click_and_capture_navigation(
        self,
        click_coro,
        page: Page,
        context: BrowserContext,
        *,
        url_regex: re.Pattern | None = re.compile(r"/product[s]?/|/p/|/pp/", re.I),
        wait_state: str = "domcontentloaded",
        timeout_ms: int = 5000,
    ) -> Page | None:
        popup_task = asyncio.create_task(context.wait_for_event("page", timeout=timeout_ms))
        same_tab_nav_task = asyncio.create_task(page.wait_for_event("framenavigated", timeout=timeout_ms))
        spa_url_task = asyncio.create_task(page.wait_for_url(url_regex, timeout=timeout_ms)) if url_regex else None
        sel_spa = ", ".join(VISIBLE_PRICE_SELECTORS) or "[itemprop=price],[class*=price],[data-testid*=price]"
        spa_price_task = asyncio.create_task(page.wait_for_selector(sel_spa, state="visible", timeout=timeout_ms))
        try:
            await click_coro()
        except Exception:
            [t.cancel() for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task) if t and not t.done()]
            return None
        tasks = {popup_task, same_tab_nav_task, spa_price_task}
        tasks.add(spa_url_task) if spa_url_task else None
        try:
            # V88.5.5: timeout_ms が 5000 になったため、 asyncio.wait のタイムアウトも 5.0 秒になる
            done, pending = await asyncio.wait(tasks, timeout=timeout_ms / 1000, return_when=asyncio.FIRST_COMPLETED)
            [t.cancel() for t in pending]
            if not done:
                return None
            winner = next(iter(done))
            new_page = winner.result() if winner is popup_task else page
            log_msg = f"Winner: {'popup' if winner is popup_task else 'framenav' if winner is same_tab_nav_task else 'SPA URL' if winner is spa_url_task else 'SPA Price'}"
            self.logger.debug(f"[_click_and_capture] {log_msg}")
            try:
                if new_page.url == "about:blank":
                    await new_page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Exception as e_blank:
                self.logger.debug(f"[_click_and_capture] Wait for about:blank failed: {e_blank}")
            with contextlib.suppress(Exception):
                await new_page.wait_for_load_state(wait_state, timeout=max(500, timeout_ms // 10))  # 500ms
            if url_regex:
                try:
                    await new_page.wait_for_url(url_regex, timeout=max(1000, timeout_ms // 4))  # 1250ms
                except Exception as e_url_final:
                    self.logger.debug(f"[_click_and_capture] Final wait_for_url failed: {e_url_final}")
            return new_page
        except Exception as e_wait:
            self.logger.warning(f"[_click_and_capture] Nav race failed: {e_wait}")
            return None
        finally:
            [t.cancel() for t in (popup_task, same_tab_nav_task, spa_url_task, spa_price_task) if t and not t.done()]

    async def _click_first_card_or_link(
        self, page: Page, site_config: dict, settings: dict, context: BrowserContext
    ) -> Page | None:
        pdp = site_config.get("selectors", {}).get("pdp") or {}
        link_sel = pdp.get("pdp_link_selectors", [])
        plp_boxes = pdp.get(
            "plp_container_selectors",
            ["main", "section[role='main']", "#main", "[id*='product' i]", "[class*='product' i]"],
        )
        block_ng = set(pdp.get("blocklist_href_substrings", ["/cart", "/wishlist", "javascript:void"]))
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
                                lambda el=el: el.click(timeout=5000), page, context, url_regex=url_pat
                            )
                            if newp:
                                return newp
                except Exception:
                    continue
        # ★ 88.6.2: (Refactor) 可読性のため整形
        tile_selectors = [
            "[data-qa='product-tile']",
            ".c-product-tile",
            ".product-card",
            "[data-testid*='product-card']",
            "article[data-product-id]",
        ]
        for box in plp_boxes:
            for tile_sel in tile_selectors:
                try:
                    card = page.locator(f"{box} {tile_sel}").first
                    await card.scroll_into_view_if_needed()
                    if await card.count() > 0:
                        newp = await self._click_and_capture_navigation(
                            lambda card=card: card.click(timeout=5000), page, context, url_regex=url_pat
                        )
                        if newp:
                            return newp
                except Exception:
                    continue
        self.logger.warning("[Fallback:click-card] Could not find any clickable link or card.")
        return None

    # --- Flow Logic: PDP ---
    async def _run_pdp_flow(
        self,
        page: Page,
        site: str,
        query: str,
        settings: dict,
        run_context: RunContext,
        site_config: dict[str, Any],
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
            plp = self._normalize_to_en_int_url(plp)  # V88.6.1: 修正
            logger.info("[recover] Forcing PLP navigation: %s", plp)
            await page.goto(url=plp, wait_until="domcontentloaded")  # V88.6.1: 修正
        except Exception as e:
            logger.debug("[recover] force PLP failed: %r", e)

    # --- V88.1.0: Refined Failure Handling ---
    # --- V88.4.0: Add intent context ---
    async def _handle_run_failure(
        self, e: Exception, site: str, query: str, site_config: dict, run_context: RunContext, page: Page | None
    ) -> DiscoveryResult:
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
                screenshots_list = [str(p) for p in recent_pngs[:6]]  # Take last 6
            except Exception:
                pass  # Ignore if path finding fails

        failure_context = {
            "final_url": final_url_on_fail,
            "dom_snapshot_path": dom_path_str,  # Use path obtained from write_fail_snapshot result
            "errors": [str(e)],
            "screenshots": screenshots_list,  # Use dynamically obtained list
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
                "failure_context": failure_context,  # Pass context to Orchestrator
            },
        )

    # --- Flow Logic: Learning ---
    async def _run_learning_flow(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        site_config: dict,
        settings: dict,
        run_context: RunContext,
        *,
        start_t: float,
        budget_ms: int,
    ) -> DiscoveryResult:
        self.logger.info(f"[LEARN] Starting learning flow for site: {site}")
        await self._ensure_plp_materialized(page, site_config, settings, start_t=start_t, budget_ms=budget_ms)
        await save_dom(run_context, page, "learn_plp_dom_for_discovery")
        try:
            discovered_selectors = await self.discovery_agent.discover(
                page=page, context=context, run_context=run_context
            )
            if not discovered_selectors:
                raise ValueError("SelectorDiscoveryAgent returned no selectors.")
            self.logger.info(f"[LEARN] Discovered selectors: {json.dumps(discovered_selectors, indent=2)}")
        except Exception as e:
            self.logger.error(f"[LEARN] Selector discovery failed: {e}", exc_info=True)
            return await self._handle_run_failure(
                e, site, "(learning)", site_config, run_context, page
            )  # Use failure handler
        try:
            await self._save_learned_selectors(site, discovered_selectors, run_context)
        except Exception as e:
            self.logger.error(f"[LEARN] Failed to save learned selectors: {e}", exc_info=True)
            # Saving failed, but discovery succeeded, so still return ok=False but with learned selectors
            return DiscoveryResult(
                ok=False,
                site=site,
                query="(learning)",
                message=f"Failed to save: {e}",
                evidence={"learned_selectors": discovered_selectors},
            )
        return DiscoveryResult(
            ok=True,
            site=site,
            query="(learning)",
            message="Successfully learned and saved.",
            evidence={"learned_selectors": discovered_selectors},
        )

    async def _save_learned_selectors(
        self, site: str, new_selectors: dict[str, list[str]], run_context: RunContext
    ) -> None:
        try:
            instance_dir = Path(run_context.run_path).parent.parent
            site_dir = instance_dir / "sites" / site.upper()
            site_dir.mkdir(parents=True, exist_ok=True)
            learned_path = site_dir / "learned_selectors.json"
        except Exception:
            learned_path = Path(f"instance/sites/{site.upper()}/learned_selectors.json")
        learned_path.parent.mkdir(parents=True, exist_ok=True)
        existing_selectors = {}
        if learned_path.exists():
            with contextlib.suppress(Exception):
                existing_selectors = json.loads(learned_path.read_text(encoding="utf-8")) or {}
        merged_selectors = {}
        all_keys = set(existing_selectors.keys()) | set(new_selectors.keys())
        for key in all_keys:
            merged_list = _dedupe_keep_order(new_selectors.get(key, []) + existing_selectors.get(key, []))
            if merged_list:
                merged_selectors[key] = merged_list
        learned_path.write_text(json.dumps(merged_selectors, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info(f"[LEARN] Saved merged selectors to: {learned_path}")
